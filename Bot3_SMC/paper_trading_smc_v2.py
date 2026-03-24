"""
paper_trading_smc_v2.py
=======================
Бот #3 v2 — SMC переработан полностью
Биржа: Bitget | TF: 1h, 4h

Новая логика:
  1. MSS (Market Structure Shift) — слом структуры с объёмом
  2. Order Block — последняя свеча перед MSS
  3. Дивергенция RSI — подтверждение разворота
  4. HTF тренд — торгуем только по тренду 4h/1d
  5. Только Лондон + Нью-Йорк (07-18 UTC)
  6. RR 1:4, стоп за тело OB

Журнал: smc2_journal.json
"""

import sys, time, os, json
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from divergence_module import calc_rsi, find_divergence
    HAS_DIV = True
except ImportError:
    HAS_DIV = False

CONFIG = {
    "timeframes":       ["15m", "1h", "4h"],  # добавлен 15m — больше MSS сигналов
    "candles":          200,
    "vol_period":       20,
    "min_vol":          2.0,
    "ob_lookback":      30,
    "mss_min_pct":      0.005,
    "max_entry_miss":   0.03,
    "min_usdt_vol":     5_000_000,
    "rr":               4.0,
    "commission":       0.001,
    "buf":              0.001,
    "max_wait":         6,
    "deposit":          50.0,
    "risk_pct":         0.05,
    "max_trades":       3,
    "cooldown_h":       3,
    "max_stop_pct":     0.005,           # стоп >0.5% — не берём (WR 5% vs 11%)
    "interval_min":     15,              # снижено с 30 — не пропускаем 15m свечи
    "session_filter":   True,
    "session_hours":    [(7, 18)],
    "require_div":      False,  # дивергенция опциональна — даёт бонус но не блокирует
    "journal":          "smc2_journal.json",
    "logfile":          "smc2_log.txt",
}


def is_active_session(cfg):
    if not cfg.get("session_filter", True):
        return True, "все сессии"
    hour = datetime.now().hour
    for start, end in cfg.get("session_hours", [(7,18)]):
        if start <= hour < end:
            if 7 <= hour < 12:    return True, "🇬🇧 Лондон"
            elif 13 <= hour < 18: return True, "🇺🇸 Нью-Йорк"
            else:                 return True, "активная"
    return False, f"🌙 ночь ({hour}:00 UTC)"


def log(msg, cfg, show=True):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    try:
        if bool(show):
            print(line)
    except Exception:
        print(line)
    try:
        logfile = cfg["logfile"] if isinstance(cfg, dict) else cfg
        with open(logfile, "a", encoding="utf-8", errors="replace") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_journal(cfg):
    if os.path.exists(cfg["journal"]):
        with open(cfg["journal"], "r", encoding="utf-8") as f:
            return json.load(f)
    return {"created": datetime.now().isoformat(),
            "deposit": cfg["deposit"], "balance": cfg["deposit"],
            "trades": [], "pending": [], "open": [],
            "cycles": 0, "refills": []}


def save_journal(j, cfg):
    with open(cfg["journal"], "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False, indent=2)


def get_exchange():
    return ccxt.bitget({"enableRateLimit": True,
                        "options": {"defaultType": "spot"}})


def get_symbols(ex, min_vol):
    try:
        tickers = ex.fetch_tickers()
        syms = [(s, t.get("quoteVolume") or 0)
                for s, t in tickers.items()
                if s.endswith("/USDT") and s != "BTC/USDT"
                and (t.get("quoteVolume") or 0) >= min_vol]
        syms.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in syms]
    except Exception:
        return []


def fetch_df(ex, symbol, tf, limit, cfg):
    try:
        ms_map = {"15m":900000,"30m":1800000,"1h":3600000,"4h":14400000,"1d":86400000}
        ms = ms_map.get(tf, 3600000)
        since = ex.milliseconds() - limit * ms - ms * 10
        raw = ex.fetch_ohlcv(symbol, tf, since=since, limit=limit)
        if not raw or len(raw) < 50:
            return None
        df = pd.DataFrame(raw,
             columns=["ts","open","high","low","close","vol"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        p = cfg["vol_period"]
        df["vol_avg"]   = df["vol"].rolling(p).mean()
        df["vol_ratio"] = df["vol"] / df["vol_avg"].replace(0, 1e-9)
        return df.reset_index(drop=True)
    except Exception:
        return None


def get_price(ex, symbol):
    try:
        return ex.fetch_ticker(symbol)["last"]
    except Exception:
        return None


def get_btc_trend(ex, cfg):
    """Глобальный тренд по BTC/USDT EMA50"""
    try:
        df = fetch_df(ex, "BTC/USDT", "1h", 60, cfg)
        if df is None:
            return "neutral"
        ema = df["close"].ewm(span=50, adjust=False).mean()
        lc, le = df["close"].iloc[-1], ema.iloc[-1]
        if lc > le * 1.005: return "bull"
        if lc < le * 0.995: return "bear"
        return "neutral"
    except Exception:
        return "neutral"


def get_htf_trend(ex, symbol, cfg):
    try:
        tf_map = {"15m": "1h", "1h": "4h", "4h": "1d"}
        htf = tf_map.get(cfg["timeframes"][0], "4h")
        df  = fetch_df(ex, symbol, htf, 60, cfg)
        if df is None:
            return "neutral"
        ema = df["close"].ewm(span=50, adjust=False).mean()
        lc, le = df["close"].iloc[-1], ema.iloc[-1]
        if lc > le * 1.005: return "bull"
        if lc < le * 0.995: return "bear"
        return "neutral"
    except Exception:
        return "neutral"


def find_mss_ob(df, cfg, htf_trend):
    """
    Market Structure Shift + Order Block + Дивергенция RSI
    """
    signals   = []
    lookback  = cfg["ob_lookback"]
    min_vol   = cfg["min_vol"]
    mss_pct   = cfg["mss_min_pct"]
    buf       = cfg["buf"]
    rr        = cfg["rr"]
    miss      = cfg["max_entry_miss"]
    close_now = df.iloc[-1]["close"]
    req_div   = cfg.get("require_div", True)

    # Дивергенция на всём датафрейме
    div_type, div_strength = (None, 0)
    if HAS_DIV:
        div_type, div_strength = find_divergence(df)

    start = max(5, len(df) - lookback - 5)

    for i in range(start, len(df) - 2):
        mss = df.iloc[i+1]
        ob  = df.iloc[i]

        mss_vol  = mss["vol_ratio"]
        if mss_vol < min_vol:
            continue

        mss_move = (mss["close"] - mss["open"]) / mss["open"]
        if abs(mss_move) < mss_pct:
            continue

        # ── Медвежий MSS+OB (шорт) ───────────
        if htf_trend in ("bear", "neutral") and mss_move < 0:
            if ob["close"] <= ob["open"]:
                continue

            # Стоп за ТЕЛО OB (не за хай свечи — короче!)
            ob_top    = ob["close"]
            ob_bottom = ob["open"]
            ob_mid    = (ob_top + ob_bottom) / 2

            if not (ob_bottom <= close_now <= ob_top * (1 + miss)):
                continue

            # Проверка дивергенции
            if req_div and HAS_DIV:
                if div_type != "bearish":
                    continue  # нет медвежьей дивергенции — пропускаем

            entry = ob_mid
            stop  = ob_top * (1 + buf)      # стоп за тело OB
            risk  = stop - entry
            if risk <= 0:
                continue
            if entry > 0 and (risk / entry) > cfg.get("max_stop_pct", 1):
                continue  # стоп слишком большой
            take = entry - risk * rr
            if take <= 0:
                continue

            signals.append({
                "dir":         -1,
                "entry":       round(entry, 6),
                "stop":        round(stop,  6),
                "take":        round(take,  6),
                "type":        "MSS_OB_ШОРТ",
                "ob_top":      round(ob_top, 6),
                "ob_bottom":   round(ob_bottom, 6),
                "mss_vol":     round(mss_vol, 1),
                "mss_pct":     round(mss_move * 100, 2),
                "div_type":    div_type or "none",
                "div_strength":div_strength,
                "rr":          round((entry - take) / risk, 2),
            })

        # ── Бычий MSS+OB (лонг) ──────────────
        elif htf_trend in ("bull", "neutral") and mss_move > 0:
            if ob["close"] >= ob["open"]:
                continue

            ob_top    = ob["open"]
            ob_bottom = ob["close"]
            ob_mid    = (ob_top + ob_bottom) / 2

            if not (ob_bottom * (1 - miss) <= close_now <= ob_top):
                continue

            if req_div and HAS_DIV:
                if div_type != "bullish":
                    continue

            entry = ob_mid
            stop  = ob_bottom * (1 - buf)   # стоп за тело OB
            risk  = entry - stop
            if risk <= 0:
                continue
            if entry > 0 and (risk / entry) > cfg.get("max_stop_pct", 1):
                continue  # стоп слишком большой
            take = entry + risk * rr

            signals.append({
                "dir":         1,
                "entry":       round(entry, 6),
                "stop":        round(stop,  6),
                "take":        round(take,  6),
                "type":        "MSS_OB_ЛОНГ",
                "ob_top":      round(ob_top, 6),
                "ob_bottom":   round(ob_bottom, 6),
                "mss_vol":     round(mss_vol, 1),
                "mss_pct":     round(mss_move * 100, 2),
                "div_type":    div_type or "none",
                "div_strength":div_strength,
                "rr":          round((take - entry) / risk, 2),
            })

    return signals


def run_cycle(ex, journal, cfg):
    now     = datetime.now().isoformat()
    balance = journal["balance"]
    comm    = cfg["commission"]
    journal["cycles"] = journal.get("cycles", 0) + 1

    # Автопополнение
    if 0 < balance < journal["deposit"] * cfg["risk_pct"]:
        old_bal = balance
        balance = journal["deposit"]
        journal["balance"] = balance
        journal.setdefault("refills", []).append(
            {"date": now, "from": round(old_bal,4), "to": balance})
        log(f"💰 АВТОПОПОЛНЕНИЕ ${old_bal:.2f} → ${balance:.2f}", cfg)

    # Pending
    new_pending = []
    new_open    = list(journal["open"])

    for p in journal["pending"]:
        price = get_price(ex, p["symbol"])
        time.sleep(0.1)
        if price is None:
            new_pending.append(p); continue

        p["waited"] = p.get("waited", 0) + 1
        d, entry    = p["dir"], p["entry"]
        hit = (d == 1 and price <= entry * 1.005) or \
              (d == -1 and price >= entry * 0.995)

        if hit:
            risk = abs(entry - p["stop"])
            qty  = (balance * cfg["risk_pct"]) / risk if risk > 0 else 0
            if qty > 0:
                new_open.append({**p, "qty": round(qty,6), "opened": now})
                log(f"✅ ОТКРЫТА {p['symbol']} "
                    f"{'ЛОНГ' if d==1 else 'ШОРТ'} [{p['type']}] "
                    f"div={p.get('div_type','?')} "
                    f"вход={entry} стоп={p['stop']} тейк={p['take']}", cfg)
        elif p["waited"] >= cfg["max_wait"]:
            log(f"⏱️  ОТМЕНЕНА {p['symbol']} — OB устарел", cfg)
        else:
            new_pending.append(p)

    # Open
    still_open = []
    for pos in new_open:
        price = get_price(ex, pos["symbol"])
        time.sleep(0.1)
        if price is None:
            still_open.append(pos); continue

        d, entry         = pos["dir"], pos["entry"]
        stop, take, qty  = pos["stop"], pos["take"], pos["qty"]
        result = exit_p  = None

        if d == 1:
            if price <= stop:   result, exit_p = "LOSS", stop
            elif price >= take: result, exit_p = "WIN",  take
        else:
            if price >= stop:   result, exit_p = "LOSS", stop
            elif price <= take: result, exit_p = "WIN",  take

        if result:
            pnl = (exit_p-entry)*qty*d - (entry+exit_p)*qty*comm
            balance += pnl
            journal["trades"].append({
                "symbol":  pos["symbol"], "tf": pos["tf"],
                "dir":     d, "type": pos.get("type","?"),
                "entry":   round(entry,6), "exit": round(exit_p,6),
                "stop":    round(stop,6),  "take": round(take,6),
                "mss_vol": pos.get("mss_vol",0),
                "div_type":pos.get("div_type","none"),
                "pnl":     round(pnl,4), "result": result,
                "balance": round(balance,4), "closed": now,
            })
            em = "🟢 WIN" if result=="WIN" else "🔴 LOSS"
            log(f"{em} {pos['symbol']} "
                f"{'ЛОНГ' if d==1 else 'ШОРТ'} "
                f"PnL=${round(pnl,2):+.2f} "
                f"баланс=${round(balance,2)}", cfg)
        else:
            still_open.append(pos)

    journal["open"]    = still_open
    journal["pending"] = new_pending
    journal["balance"] = round(balance, 4)

    # Новые сигналы
    open_cnt = len(journal["open"]) + len(journal["pending"])
    if open_cnt >= cfg["max_trades"]:
        return journal, 0

    active, session = is_active_session(cfg)
    if not active:
        log(f"⏸️  {session}", cfg)
        return journal, 0

    existing = set(f"{p['symbol']}_{p['dir']}"
                   for p in journal["pending"] + journal["open"])
    cooling  = set()
    for t in journal["trades"]:
        if t["result"] == "LOSS":
            closed = datetime.fromisoformat(t["closed"])
            if (datetime.now()-closed).total_seconds()/3600 < cfg["cooldown_h"]:
                cooling.add(t["symbol"])

    # Глобальный фильтр BTC тренда
    btc_trend = get_btc_trend(ex, cfg)
    allow_long  = btc_trend in ("bull", "neutral")
    allow_short = btc_trend in ("bear", "neutral")
    log(f"BTC тренд: {btc_trend}  allow_long={allow_long}  "
        f"allow_short={allow_short}", cfg, show=True)

    symbols  = get_symbols(ex, cfg["min_usdt_vol"])
    new_sigs = 0

    for tf in cfg["timeframes"]:
        if open_cnt >= cfg["max_trades"]:
            break
        for sym in symbols[:60]:
            if open_cnt >= cfg["max_trades"]:
                break
            if sym in cooling:
                continue

            df = fetch_df(ex, sym, tf, cfg["candles"], cfg)
            time.sleep(0.12)
            if df is None:
                continue

            htf  = get_htf_trend(ex, sym, cfg)
            time.sleep(0.1)
            sigs = find_mss_ob(df, cfg, htf)
            if not sigs:
                continue

            sig = sigs[-1]

            # Проверяем соответствие BTC тренду
            if sig["dir"] == 1 and not allow_long:
                continue  # BTC медвежий — не берём лонги
            if sig["dir"] == -1 and not allow_short:
                continue  # BTC бычий — не берём шорты

            key = f"{sym}_{sig['dir']}"
            if key in existing:
                continue

            journal["pending"].append({
                "symbol":   sym, "tf": tf,
                "dir":      sig["dir"],
                "entry":    sig["entry"],
                "stop":     sig["stop"],
                "take":     sig["take"],
                "type":     sig["type"],
                "mss_vol":  sig["mss_vol"],
                "div_type": sig["div_type"],
                "htf":      htf,
                "added":    now, "waited": 0,
            })
            existing.add(key)
            open_cnt += 1
            new_sigs += 1
            d_ru = "ЛОНГ" if sig["dir"]==1 else "ШОРТ"
            log(f"➕ MSS+OB {sym} [{tf}] {d_ru} "
                f"HTF={htf} div={sig['div_type']} "
                f"vol={sig['mss_vol']}x "
                f"лимит={sig['entry']} стоп={sig['stop']} "
                f"тейк={sig['take']} RR=1:{sig['rr']}", cfg)

    return journal, new_sigs


def print_stats(j, cfg):
    trades  = j["trades"]
    balance = j["balance"]
    deposit = j["deposit"]
    pnl     = balance - deposit
    pnl_pct = round(pnl / deposit * 100, 2)

    print(f"\n{'='*62}")
    print(f"  📊 БОТ #3 v2 — SMC MSS+OB+DIV | Bitget")
    print(f"  Старт: {j['created'][:10]}  Циклов: {j.get('cycles',0)}")
    print(f"{'='*62}")
    s = "+" if pnl >= 0 else ""
    print(f"  Депозит: ${deposit:.2f}  Баланс: ${balance:.2f}")
    print(f"  P&L: {s}${pnl:.2f}  ({s}{pnl_pct}%)")

    if trades:
        wins   = [t for t in trades if t["result"] == "WIN"]
        losses = [t for t in trades if t["result"] == "LOSS"]
        wr     = round(len(wins) / len(trades) * 100, 1)
        longs  = [t for t in trades if t["dir"] ==  1]
        shorts = [t for t in trades if t["dir"] == -1]
        eq     = [deposit] + [t["balance"] for t in trades]
        peak   = np.maximum.accumulate(eq)
        dd     = round(((np.array(eq) - peak) / peak * 100).min(), 2)

        # По дивергенции
        with_div = [t for t in trades if t.get("div_type") != "none"]
        if with_div:
            div_wr = round(
                len([t for t in with_div if t["result"]=="WIN"]) /
                len(with_div) * 100, 1)
            print(f"\n  С дивергенцией: {len(with_div)} сделок  WR={div_wr}%")

        print(f"\n  Сделок: {len(trades)}  "
              f"(🟢{len(wins)} побед  🔴{len(losses)} потерь)")
        print(f"  WR: {wr}%  DD: {dd}%")
        print(f"  Лонгов: {len(longs)}  Шортов: {len(shorts)}")

        print(f"\n  Последние 5 сделок:")
        for t in trades[-5:]:
            em = "🟢" if t["result"]=="WIN" else "🔴"
            d  = "ЛОНГ" if t["dir"]==1 else "ШОРТ"
            print(f"  {em} {t['symbol']:<14} [{t['tf']}] {d} "
                  f"div={t.get('div_type','?')} "
                  f"{t['result']} {t['pnl']:>+.2f}$")

        pd.DataFrame(trades).to_csv("smc2_trades.csv", index=False)

    if j["pending"]:
        print(f"\n  ⏳ Ожидают ({len(j['pending'])}):")
        for p in j["pending"]:
            d = "ЛОНГ" if p["dir"]==1 else "ШОРТ"
            print(f"     {p['symbol']:<14} [{p['tf']}] {d} "
                  f"div={p.get('div_type','?')} лимит={p['entry']}")

    print(f"\n  🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*62}")


def main():
    cfg  = CONFIG.copy()
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"

    if mode == "reset":
        for f in [cfg["journal"], cfg["logfile"], "smc2_trades.csv"]:
            if os.path.exists(f): os.remove(f)
        print("[+] Сброшен.")
        return

    j = load_journal(cfg)

    if mode == "status":
        print_stats(j, cfg)
        return

    if mode == "refill":
        old = j["balance"]
        j["balance"] = cfg["deposit"]
        save_journal(j, cfg)
        print(f"[+] ${old:.2f} → ${cfg['deposit']:.2f}")
        return

    print("=" * 62)
    print(f"  🧠 БОТ #3 v2 — SMC MSS+OB+Дивергенция | Bitget")
    print(f"  TF: {', '.join(cfg['timeframes'])}  "
          f"RR 1:{cfg['rr']}  Сессия: Лондон+НьюЙорк")
    print(f"  Дивергенция RSI: {'ВКЛ' if cfg['require_div'] else 'ВЫКЛ'}")
    print(f"  Остановка: Ctrl+C")
    print("=" * 62)

    log(f"СТАРТ БОТ#3v2 SMC  Bitget  "
        f"депозит=${cfg['deposit']}  RR=1:{cfg['rr']}", cfg)

    ex = get_exchange()

    while True:
        try:
            log(f"--- ЦИКЛ #{j.get('cycles',0)+1} ---", cfg)
            j, new_sigs = run_cycle(ex, j, cfg)
            save_journal(j, cfg)

            trades = j["trades"]
            wins   = len([t for t in trades if t["result"]=="WIN"])
            wr     = round(wins/len(trades)*100,1) if trades else 0
            s      = "+" if j["balance"] >= cfg["deposit"] else ""
            log(f"Баланс=${j['balance']}  Сделок={len(trades)}  "
                f"WR={wr}%  "
                f"P&L={s}{round(j['balance']-cfg['deposit'],2)}$  "
                f"Новых={new_sigs}", cfg)
            log(f"Следующий цикл через {cfg['interval_min']} минут...", cfg)
            time.sleep(cfg["interval_min"] * 60)

        except KeyboardInterrupt:
            log("ОСТАНОВЛЕН (Ctrl+C)", cfg)
            print_stats(j, cfg)
            break
        except Exception as e:
            import traceback
            log(f"ОШИБКА: {e}", cfg)
            log(f"ДЕТАЛИ: {traceback.format_exc()}", cfg)
            log("Повтор через 1 минуту...", cfg)
            time.sleep(60)


if __name__ == "__main__":
    main()
