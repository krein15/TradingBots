"""
paper_trading_funding.py
========================
Бот #3 v4 — Funding Rate Contrarian
Биржа: Bitget (фьючерсы) | TF: 1h

Логика:
  Funding Rate — плата за удержание позиции на фьючерсах.
  Положительный (+) = лонгисты платят шортистам → рынок перегрет лонгами
  Отрицательный (-) = шортисты платят лонгистам → рынок перегрет шортами

  Когда все в одну сторону → рынок разворачивается!

  Высокий FR (+0.05%+) → толпа лонгует → мы ШОРТИМ
  Низкий FR (-0.05%-) → толпа шортит → мы ЛОНГУЕМ

Подтверждения:
  RSI перекуплен/перепродан
  Цена у BB верхней/нижней границы
  Объём без аномалий

Биржа Bitget — одна из лучших для funding rate стратегий.
Данные по FR доступны через ccxt.

Журнал: funding_journal.json
"""

import sys, time, os, json
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime

CONFIG = {
    "timeframes":        ["1h"],
    "candles":           100,
    "vol_period":        20,
    "rsi_period":        14,
    "bb_period":         20,
    "bb_std":            2.0,
    "fr_long_threshold": 0.0003,   # FR > 0.03% → все лонгуют → шортим
    "fr_short_threshold":-0.0003,  # FR < -0.03% → все шортят → лонгуем
    "rsi_upper":         60,       # RSI > 60 подтверждает перекупленность
    "rsi_lower":         40,       # RSI < 40 подтверждает перепроданность
    "max_vol_mult":      3.0,      # исключаем новостные движения
    "min_usdt_vol":      5_000_000,
    "rr":                3.0,
    "commission":        0.001,
    "buf":               0.002,
    "max_wait":          6,
    "deposit":           50.0,
    "risk_pct":          0.05,
    "max_trades":        4,
    "cooldown_h":        3,
    "interval_min":      15,       # сканируем часто — цена меняется постоянно
    "journal":           "funding_journal.json",
    "logfile":           "funding_log.txt",
}


def log(msg, cfg, show=True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if show:
        print(line)
    with open(cfg["logfile"], "a", encoding="utf-8", errors="replace") as f:
        f.write(line + "\n")


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
    return ccxt.bitget({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},  # фьючерсы
    })


def get_symbols(ex, min_vol):
    try:
        # Для фьючерсов Bitget
        markets = ex.load_markets()
        tickers = ex.fetch_tickers()
        syms = []
        for s, t in tickers.items():
            if not s.endswith("/USDT:USDT"):
                continue
            vol = t.get("quoteVolume") or 0
            if vol >= min_vol:
                syms.append((s, vol))
        syms.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in syms[:50]]
    except Exception as e:
        return []


def get_funding_rate(ex, symbol):
    """Получаем текущий Funding Rate"""
    try:
        fr = ex.fetch_funding_rate(symbol)
        return fr.get("fundingRate", 0) or 0
    except Exception:
        return 0


def fetch_df(ex, symbol, tf, limit, cfg):
    try:
        ms_map = {"15m":900000,"1h":3600000,"4h":14400000}
        ms = ms_map.get(tf, 3600000)
        since = ex.milliseconds() - limit * ms - ms * 5
        raw = ex.fetch_ohlcv(symbol, tf, since=since, limit=limit)
        if not raw or len(raw) < 30:
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


def calc_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def calc_bb(close, period=20, std=2.0):
    mid = close.rolling(period).mean()
    s   = close.rolling(period).std()
    return mid + std*s, mid, mid - std*s


def find_signals(df, cfg, funding_rate, symbol):
    """
    Funding Rate Contrarian сигнал:

    ШОРТ: FR высокий (все лонгуют) + RSI > 60 + цена у верх BB
    ЛОНГ: FR низкий (все шортят)  + RSI < 40 + цена у ниж BB
    """
    signals   = []
    close     = df["close"]
    close_now = close.iloc[-1]
    vol_now   = df["vol_ratio"].iloc[-1]

    # Фильтр аномального объёма
    if vol_now >= cfg["max_vol_mult"]:
        return signals

    # Индикаторы
    rsi     = calc_rsi(close, cfg["rsi_period"]).iloc[-1]
    bb_up, bb_mid, bb_dn = calc_bb(close, cfg["bb_period"], cfg["bb_std"])
    bb_up_now  = bb_up.iloc[-1]
    bb_dn_now  = bb_dn.iloc[-1]
    bb_mid_now = bb_mid.iloc[-1]

    buf = cfg["buf"]
    rr  = cfg["rr"]
    fr  = funding_rate
    fr_pct = round(fr * 100, 4)

    # ШОРТ: FR высокий → все лонгуют → разворот вниз
    if fr >= cfg["fr_long_threshold"]:
        if rsi >= cfg["rsi_upper"] and close_now >= bb_mid_now:
            entry = close_now
            stop  = bb_up_now * (1 + buf)
            risk  = stop - entry
            if risk > 0:
                take = entry - risk * rr
                if take > 0:
                    signals.append({
                        "dir":     -1,
                        "entry":   round(entry, 6),
                        "stop":    round(stop, 6),
                        "take":    round(take, 6),
                        "type":    "FR_ШОРТ",
                        "fr":      fr_pct,
                        "rsi":     round(rsi, 1),
                        "vol":     round(vol_now, 2),
                        "rr":      rr,
                    })

    # ЛОНГ: FR низкий → все шортят → разворот вверх
    if fr <= cfg["fr_short_threshold"]:
        if rsi <= cfg["rsi_lower"] and close_now <= bb_mid_now:
            entry = close_now
            stop  = bb_dn_now * (1 - buf)
            risk  = entry - stop
            if risk > 0:
                take = entry + risk * rr
                signals.append({
                    "dir":     1,
                    "entry":   round(entry, 6),
                    "stop":    round(stop, 6),
                    "take":    round(take, 6),
                    "type":    "FR_ЛОНГ",
                    "fr":      fr_pct,
                    "rsi":     round(rsi, 1),
                    "vol":     round(vol_now, 2),
                    "rr":      rr,
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
        d, entry = p["dir"], p["entry"]
        # Лимитный ордер: цена должна быть близко к уровню входа
        # Funding Rate — входим по рынку при первом цикле
        # Цена FR сигнала = текущая цена, поэтому ±1% допуск
        hit = (d == 1 and entry * 0.990 <= price <= entry * 1.010) or \
              (d == -1 and entry * 0.990 <= price <= entry * 1.010)
        if hit:
            risk = abs(entry - p["stop"])
            qty  = (balance * cfg["risk_pct"]) / risk if risk > 0 else 0
            if qty > 0:
                new_open.append({**p, "qty": round(qty,6), "opened": now})
                log(f"✅ ОТКРЫТА {p['symbol']} "
                    f"{'ЛОНГ' if d==1 else 'ШОРТ'} [{p['type']}] "
                    f"FR={p.get('fr','?')}% RSI={p.get('rsi','?')} "
                    f"вход={entry}", cfg)
        elif p["waited"] >= cfg["max_wait"]:
            log(f"⏱️  ОТМЕНЕНА {p['symbol']}", cfg)
        else:
            new_pending.append(p)

    # Open
    still_open = []
    for pos in new_open:
        price = get_price(ex, pos["symbol"])
        time.sleep(0.1)
        if price is None:
            still_open.append(pos); continue
        d, entry = pos["dir"], pos["entry"]
        stop, take, qty = pos["stop"], pos["take"], pos["qty"]
        result = exit_p = None
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
                "symbol": pos["symbol"], "tf": pos["tf"],
                "dir": d, "type": pos.get("type","?"),
                "entry": round(entry,6), "exit": round(exit_p,6),
                "fr": pos.get("fr", 0),
                "rsi": pos.get("rsi", 0),
                "vol": pos.get("vol", 0),
                "pnl": round(pnl,4), "result": result,
                "balance": round(balance,4), "closed": now,
            })
            em = "🟢 WIN" if result=="WIN" else "🔴 LOSS"
            log(f"{em} {pos['symbol']} "
                f"FR={pos.get('fr','?')}% "
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

    existing = set(f"{p['symbol']}_{p['dir']}"
                   for p in journal["pending"] + journal["open"])
    cooling  = set()
    for t in journal["trades"]:
        if t["result"] == "LOSS":
            closed = datetime.fromisoformat(t["closed"])
            if (datetime.now()-closed).total_seconds()/3600 < cfg["cooldown_h"]:
                cooling.add(t["symbol"])

    try:
        symbols = get_symbols(ex, cfg["min_usdt_vol"])
    except Exception:
        symbols = []

    if not symbols:
        log("Не удалось получить список монет", cfg)
        return journal, 0

    new_sigs = 0
    fr_stats = []

    for sym in symbols:
        if open_cnt >= cfg["max_trades"]: break
        if sym in cooling: continue

        # Получаем funding rate
        fr = get_funding_rate(ex, sym)
        time.sleep(0.1)
        fr_stats.append((sym, fr))

        # Проверяем только экстремальные значения
        if abs(fr) < cfg["fr_long_threshold"] * 0.8:
            continue

        df = fetch_df(ex, sym, "1h", cfg["candles"], cfg)
        time.sleep(0.1)
        if df is None:
            continue

        sigs = find_signals(df, cfg, fr, sym)
        if not sigs:
            continue

        sig = sigs[0]
        key = f"{sym}_{sig['dir']}"
        if key in existing:
            continue

        journal["pending"].append({
            "symbol": sym, "tf": "1h",
            "dir": sig["dir"], "entry": sig["entry"],
            "stop": sig["stop"], "take": sig["take"],
            "type": sig["type"], "fr": sig["fr"],
            "rsi": sig["rsi"], "vol": sig["vol"],
            "rr": sig["rr"], "added": now, "waited": 0,
        })
        existing.add(key)
        open_cnt += 1
        new_sigs += 1
        d_ru = "ЛОНГ" if sig["dir"]==1 else "ШОРТ"
        log(f"➕ FR {sym} {d_ru} [{sig['type']}] "
            f"FR={sig['fr']}% RSI={sig['rsi']} "
            f"вход={sig['entry']} стоп={sig['stop']} "
            f"тейк={sig['take']}", cfg)

    # Логируем топ FR
    if fr_stats:
        fr_stats.sort(key=lambda x: abs(x[1]), reverse=True)
        top = fr_stats[:5]
        log(f"Топ FR: " +
            "  ".join([f"{s}={round(f*100,4)}%" for s,f in top]), cfg)

    return journal, new_sigs


def print_stats(j, cfg):
    trades  = j["trades"]
    balance = j["balance"]
    deposit = j["deposit"]
    pnl     = balance - deposit
    print(f"\n{'='*62}")
    print(f"  💰 БОТ #3 v4 — FUNDING RATE CONTRARIAN | Bitget")
    print(f"  Циклов: {j.get('cycles',0)}")
    s = "+" if pnl >= 0 else ""
    print(f"  Баланс: ${balance:.2f}  ({s}${pnl:.2f})")
    if trades:
        wins = [t for t in trades if t["result"]=="WIN"]
        wr   = round(len(wins)/len(trades)*100,1)
        eq   = [deposit] + [t["balance"] for t in trades]
        dd   = round(((np.array(eq)-np.maximum.accumulate(eq))/
                      np.maximum.accumulate(eq)*100).min(), 2)
        print(f"  Сделок: {len(trades)}  WR: {wr}%  DD: {dd}%")

        # По типу
        for tp in ["FR_ШОРТ", "FR_ЛОНГ"]:
            ts = [t for t in trades if t.get("type")==tp]
            if ts:
                tw = round(len([t for t in ts if t["result"]=="WIN"])/len(ts)*100,1)
                avg_fr = round(np.mean([abs(t.get("fr",0)) for t in ts]), 4)
                print(f"  {tp}: {len(ts)} сд.  WR={tw}%  ср.FR={avg_fr}%")

        print(f"\n  Последние 5:")
        for t in trades[-5:]:
            em = "🟢" if t["result"]=="WIN" else "🔴"
            d  = "ЛОНГ" if t["dir"]==1 else "ШОРТ"
            print(f"  {em} {t['symbol']:<18} {d} "
                  f"FR={t.get('fr','?')}% "
                  f"{t['result']} {t['pnl']:>+.2f}$")
        pd.DataFrame(trades).to_csv("funding_trades.csv", index=False)
    else:
        print(f"\n  Сделок пока нет — ждём экстремальных FR")
        print(f"  FR > {cfg['fr_long_threshold']*100}% → шорт")
        print(f"  FR < {cfg['fr_short_threshold']*100}% → лонг")

    if j["pending"]:
        print(f"\n  ⏳ Ожидают ({len(j['pending'])}):")
        for p in j["pending"]:
            d = "ЛОНГ" if p["dir"]==1 else "ШОРТ"
            print(f"     {p['symbol']:<20} {d} FR={p.get('fr','?')}%")
    print(f"\n  🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*62}")


def main():
    cfg  = CONFIG.copy()
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"

    if mode == "reset":
        for f in [cfg["journal"], cfg["logfile"], "funding_trades.csv"]:
            if os.path.exists(f): os.remove(f)
        print("[+] Сброшен."); return

    j = load_journal(cfg)

    if mode == "status":
        print_stats(j, cfg); return

    if mode == "refill":
        old = j["balance"]
        j["balance"] = cfg["deposit"]
        save_journal(j, cfg)
        print(f"[+] ${old:.2f} → ${cfg['deposit']:.2f}"); return

    print("="*62)
    print(f"  💰 БОТ #3 v4 — FUNDING RATE CONTRARIAN")
    print(f"  Стратегия: торгуем против толпы по Funding Rate")
    print(f"  FR > {cfg['fr_long_threshold']*100}% → ШОРТ (все лонгуют)")
    print(f"  FR < {cfg['fr_short_threshold']*100}% → ЛОНГ (все шортят)")
    print(f"  Биржа: Bitget фьючерсы  RR 1:{cfg['rr']}")
    print(f"  Интервал: {cfg['interval_min']} мин  (FR обновляется каждые 8ч)")
    print("="*62)

    log("="*50, cfg)
    log(f"СТАРТ FundingRate  депозит=${cfg['deposit']}", cfg)

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
                f"WR={wr}%  Новых={new_sigs}", cfg)
            log(f"Следующий цикл через {cfg['interval_min']} мин...", cfg)
            time.sleep(cfg["interval_min"] * 60)

        except KeyboardInterrupt:
            log("ОСТАНОВЛЕН", cfg)
            print_stats(j, cfg); break
        except Exception as e:
            import traceback
            log(f"ОШИБКА: {e}", cfg)
            log(traceback.format_exc(), cfg)
            time.sleep(60)


if __name__ == "__main__":
    main()
