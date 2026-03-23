"""
paper_trading_structure_v2.py
=============================
Бот #2 v2 — Структура рынка. Переписан с нуля.

Логика:
  1. Определяем тренд через свинг-хаи и свинг-лои (5 свечей)
  2. Медвежья структура (LH+LL) → ищем шорт
  3. Бычья структура (HH+HL)   → ищем лонг
  4. Вход: откат к уровню 50% последнего импульса
  5. Стоп: за последний свинг
  6. Тейк: риск × 3

Биржа: Binance
Журнал: structure2_journal.json

Запуск:
  python paper_trading_structure_v2.py
  python paper_trading_structure_v2.py status
  python paper_trading_structure_v2.py reset
"""

import sys, time, os, json
import ccxt
try:
    from divergence_module import calc_rsi, find_divergence
    HAS_DIV = True
except ImportError:
    HAS_DIV = False
import pandas as pd
import numpy as np
from datetime import datetime

CONFIG = {
    "timeframes":       ["5m", "15m"],
    "candles":          300,
    "swing_bars":       5,
    "vol_avg_period":   20,
    "min_vol_mult":     1.5,
    "min_swing_pct":    0.015,   # минимальный диапазон свингов 1.5%
    "entry_fibo":       0.5,     # вход на 50% отката
    "stop_buffer":      0.001,
    "rr_ratio":          4.0,
    "max_entry_miss":   0.03,
    "max_wait_bars":    8,
    "min_usdt_vol":     1_000_000,
    "initial_deposit":  50.0,
    "risk_pct":         0.05,
    "max_open_trades":  5,
    "session_filter":    False,       # 24/7 — сбор статистики (WR по сессиям одинаковый)
    "session_hours":     [(7,18)],    # UTC 7-18 (Лондон + Нью-Йорк)
    "cooldown_hours":   2,
    "scan_interval":    10,
    "journal_file":     "structure2_journal.json",
    "log_file":         "structure2_log.txt",
}


# ─────────────────────────────────────────────
#  ФИЛЬТР ТОРГОВОЙ СЕССИИ
# ─────────────────────────────────────────────
def is_active_session(cfg):
    """
    Торгуем только в активные сессии (UTC):
    Лондон:   07:00 - 12:00
    Нью-Йорк: 13:00 - 18:00
    Пересечение: 13:00 - 17:00 — самое активное время
    Ночь Азии: пропускаем (низкий объём, много ложных сигналов)
    """
    if not cfg.get("session_filter", True):
        return True, "все сессии"
    hour = datetime.now().hour
    sessions = cfg.get("session_hours", [(7, 18)])
    for start, end in sessions:
        if start <= hour < end:
            if 7 <= hour < 12:
                return True, "🇬🇧 Лондон"
            elif 13 <= hour < 18:
                return True, "🇺🇸 Нью-Йорк"
            else:
                return True, "активная"
    return False, f"🌙 ночь/Азия ({hour}:00 UTC)"


# ─────────────────────────────────────────────
#  ЛОГ / ЖУРНАЛ
# ─────────────────────────────────────────────
def log(msg, cfg, show=True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if show:
        print(line)
    with open(cfg["log_file"], "a", encoding="utf-8", errors="replace") as f:
        f.write(line + "\n")


def load_journal(cfg):
    if os.path.exists(cfg["journal_file"]):
        with open(cfg["journal_file"], "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "created":         datetime.now().isoformat(),
        "initial_deposit": cfg["initial_deposit"],
        "balance":         cfg["initial_deposit"],
        "trades":          [],
        "pending":         [],
        "open":            [],
        "scan_count":      0,
        "refills":         [],
    }


def save_journal(journal, cfg):
    with open(cfg["journal_file"], "w", encoding="utf-8") as f:
        json.dump(journal, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
#  БИРЖА
# ─────────────────────────────────────────────
def get_exchange():
    return ccxt.binance({"enableRateLimit": True})


def get_symbols(exchange, min_vol):
    tickers = exchange.fetch_tickers()
    syms = [(s, t.get("quoteVolume") or 0)
            for s, t in tickers.items()
            if s.endswith("/USDT") and s != "BTC/USDT"
            and (t.get("quoteVolume") or 0) >= min_vol]
    syms.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in syms]


def fetch_candles(exchange, symbol, timeframe, limit, cfg):
    try:
        ms_map = {"1m":60000,"5m":300000,"15m":900000,"1h":3600000}
        ms    = ms_map.get(timeframe, 300000)
        since = exchange.milliseconds() - limit * ms - ms * 20
        raw   = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not raw or len(raw) < 60:
            return None
        df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        p = cfg["vol_avg_period"]
        df["vol_avg"]   = df["volume"].rolling(p).mean()
        df["vol_ratio"] = df["volume"] / df["vol_avg"].replace(0, 1e-9)
        return df.reset_index(drop=True)
    except Exception:
        return None


def get_price(exchange, symbol):
    try:
        return exchange.fetch_ticker(symbol)["last"]
    except Exception:
        return None


# ─────────────────────────────────────────────
#  СВИНГИ
# ─────────────────────────────────────────────
def get_swings(df, n=5):
    """Находим свинг-хаи и свинг-лои."""
    highs = df["high"].values
    lows  = df["low"].values
    sh_list = []  # (индекс, цена)
    sl_list = []

    for i in range(n, len(df) - n):
        if all(highs[i] > highs[i-j] for j in range(1, n+1)) and \
           all(highs[i] > highs[i+j] for j in range(1, n+1)):
            sh_list.append((i, highs[i]))
        if all(lows[i] < lows[i-j] for j in range(1, n+1)) and \
           all(lows[i] < lows[i+j] for j in range(1, n+1)):
            sl_list.append((i, lows[i]))

    return sh_list, sl_list


# ─────────────────────────────────────────────
#  СТРУКТУРА И СИГНАЛЫ
# ─────────────────────────────────────────────
def find_signals(df, cfg):
    n        = cfg["swing_bars"]
    min_sw   = cfg["min_swing_pct"]
    fibo     = cfg["entry_fibo"]
    buf      = cfg["stop_buffer"]
    rr       = cfg["rr_ratio"]
    min_vol  = cfg["min_vol_mult"]
    miss     = cfg["max_entry_miss"]
    signals  = []

    sh_list, sl_list = get_swings(df, n)

    if len(sh_list) < 2 or len(sl_list) < 2:
        return signals

    # Последние два свинга
    sh1_i, sh1 = sh_list[-1]
    sh2_i, sh2 = sh_list[-2]
    sl1_i, sl1 = sl_list[-1]
    sl2_i, sl2 = sl_list[-2]

    spread = sh1 - sl1
    if spread / sl1 < min_sw:
        return signals  # диапазон слишком маленький — флет

    # Определяем структуру
    hh = sh1 > sh2
    hl = sl1 > sl2
    lh = sh1 < sh2
    ll = sl1 < sl2

    close_now = df.iloc[-1]["close"]
    vol_now   = df.iloc[-1]["vol_ratio"]

    if lh and ll:
        # ── МЕДВЕЖЬЯ СТРУКТУРА → шорт ─────────
        # Вход на откате к 50% последнего импульса вниз
        # Импульс: от sh1 до sl1
        entry = sh1 - spread * fibo   # 50% отката вверх от sl1
        stop  = sh1 * (1 + buf)       # стоп за последний LH
        risk  = stop - entry
        if risk <= 0:
            return signals
        take = entry - risk * rr
        if take <= 0:
            return signals

        # Проверяем что цена около уровня входа
        diff = abs(close_now - entry) / entry
        if diff > miss:
            return signals

        if vol_now < min_vol:
            return signals

        signals.append({
            "dir":         -1,
            "entry_limit": round(entry, 6),
            "stop":        round(stop, 6),
            "take":        round(take, 6),
            "structure":   "LH+LL",
            "sh1":         round(sh1, 6),
            "sl1":         round(sl1, 6),
            "spread_pct":  round(spread / sl1 * 100, 2),
            "vol_ratio":   round(vol_now, 1),
            "rr":          round((entry - take) / risk, 2),
        })

    elif hh and hl:
        # ── БЫЧЬЯ СТРУКТУРА → лонг ────────────
        entry = sl1 + spread * fibo
        stop  = sl1 * (1 - buf)
        risk  = entry - stop
        if risk <= 0:
            return signals
        take = entry + risk * rr

        diff = abs(close_now - entry) / entry
        if diff > miss:
            return signals

        if vol_now < min_vol:
            return signals

        signals.append({
            "dir":         1,
            "entry_limit": round(entry, 6),
            "stop":        round(stop, 6),
            "take":        round(take, 6),
            "structure":   "HH+HL",
            "sh1":         round(sh1, 6),
            "sl1":         round(sl1, 6),
            "spread_pct":  round(spread / sl1 * 100, 2),
            "vol_ratio":   round(vol_now, 1),
            "rr":          round((take - entry) / risk, 2),
        })

    return signals


# ─────────────────────────────────────────────
#  ЦИКЛ
# ─────────────────────────────────────────────
def run_cycle(exchange, journal, cfg):
    now     = datetime.now().isoformat()
    journal["scan_count"] = journal.get("scan_count", 0) + 1
    balance = journal["balance"]
    comm    = cfg["commission"] if "commission" in cfg else 0.001

    # Автопополнение
    min_trade = cfg["initial_deposit"] * cfg["risk_pct"]
    if balance < min_trade:
        old_bal = balance
        journal["balance"] = cfg["initial_deposit"]
        balance = cfg["initial_deposit"]
        journal.setdefault("refills", []).append({"date": now, "from": round(old_bal,4), "to": cfg["initial_deposit"]})
        log(f"💰 АВТОПОПОЛНЕНИЕ: ${old_bal:.2f} → ${cfg['initial_deposit']:.2f}", cfg)

    # ── Обновляем pending ─────────────────────
    new_pending = []
    for p in journal["pending"]:
        price = get_price(exchange, p["symbol"])
        time.sleep(0.08)
        if price is None:
            new_pending.append(p)
            continue

        p["waited"] = p.get("waited", 0) + 1
        d     = p["dir"]
        entry = p["entry_limit"]
        filled  = (d == 1 and price <= entry * 1.005) or \
                  (d == -1 and price >= entry * 0.995)
        expired = p["waited"] >= cfg["max_wait_bars"]

        if filled:
            risk_usd = balance * cfg["risk_pct"]
            risk_pp  = abs(entry - p["stop"])
            qty = risk_usd / risk_pp if risk_pp > 0 else 0
            if qty > 0:
                journal["open"].append({**p, "qty": round(qty,6),
                                        "risk_usd": round(risk_usd,4),
                                        "opened_at": now})
                log(f"✅ ОТКРЫТА  {p['symbol']} {'ЛОНГ' if d==1 else 'ШОРТ'} "
                    f"[{p['structure']}] вход={entry} стоп={p['stop']} "
                    f"тейк={p['take']} риск=${round(risk_usd,2)}", cfg)
        elif not expired:
            new_pending.append(p)
        else:
            log(f"⏱️  ОТМЕНЕНА {p['symbol']} — истекло", cfg)

    journal["pending"] = new_pending

    # ── Обновляем open ────────────────────────
    still_open = []
    for pos in journal["open"]:
        price = get_price(exchange, pos["symbol"])
        time.sleep(0.08)
        if price is None:
            still_open.append(pos)
            continue

        d     = pos["dir"]
        entry = pos["entry_limit"]
        result = exit_p = None

        if d == 1:
            if price <= pos["stop"]:   result, exit_p = "LOSS", pos["stop"]
            elif price >= pos["take"]: result, exit_p = "WIN",  pos["take"]
        else:
            if price >= pos["stop"]:   result, exit_p = "LOSS", pos["stop"]
            elif price <= pos["take"]: result, exit_p = "WIN",  pos["take"]

        if result:
            pnl = (exit_p - entry) * pos["qty"] * d - \
                  (entry + exit_p) * pos["qty"] * 0.001
            balance += pnl
            journal["trades"].append({
                **pos,
                "result":        result,
                "exit_price":    exit_p,
                "pnl_usd":       round(pnl, 4),
                "pnl_pct":       round(pnl / cfg["initial_deposit"] * 100, 2),
                "closed_at":     now,
                "balance_after": round(balance, 4),
            })
            em = "🟢 WIN" if result == "WIN" else "🔴 LOSS"
            log(f"{em}  {pos['symbol']} {'ЛОНГ' if d==1 else 'ШОРТ'} "
                f"[{pos['structure']}] "
                f"PnL=${round(pnl,2):+.2f}  баланс=${round(balance,2)}", cfg)
        else:
            still_open.append(pos)

    journal["open"]    = still_open
    journal["balance"] = round(balance, 4)

    # ── Новые сигналы ─────────────────────────
    open_count = len(journal["open"]) + len(journal["pending"])
    new_sigs   = 0

    if open_count >= cfg["max_open_trades"]:
        return journal, 0

    # Проверяем торговую сессию
    active, session_name = is_active_session(cfg)
    if not active:
        log(f"⏸️  {session_name} — пропускаем сканирование", cfg)
        return journal, 0
    log(f"✅ Сессия: {session_name}", cfg)

    existing = set(f"{p['symbol']}_{p['dir']}"
                   for p in journal["pending"] + journal["open"])

    # Cooldown
    on_cooldown = set()
    for t in journal["trades"]:
        if t["result"] == "LOSS":
            age_h = (datetime.now() -
                     datetime.fromisoformat(t["closed_at"])).total_seconds() / 3600
            if age_h < cfg["cooldown_hours"]:
                on_cooldown.add(t["symbol"])

    try:
        symbols = get_symbols(exchange, cfg["min_usdt_vol"])
    except Exception:
        return journal, 0

    for tf in cfg["timeframes"]:
        if open_count >= cfg["max_open_trades"]:
            break
        for sym in symbols[:100]:
            if open_count >= cfg["max_open_trades"]:
                break
            if sym in on_cooldown:
                continue

            df = fetch_candles(exchange, sym, tf, cfg["candles"], cfg)
            time.sleep(0.08)
            if df is None:
                continue

            sigs = find_signals(df, cfg)
            if not sigs:
                continue

            sig = sigs[-1]
            key = f"{sym}_{sig['dir']}"
            if key in existing:
                continue

            journal["pending"].append({
                "symbol":      sym,
                "tf":          tf,
                **sig,
                "added_at":    now,
                "waited":      0,
            })
            existing.add(key)
            open_count += 1
            new_sigs   += 1
            d_ru = "ЛОНГ" if sig["dir"] == 1 else "ШОРТ"
            log(f"➕ СИГНАЛ  {sym} [{tf}] {d_ru} [{sig['structure']}] "
                f"диапазон={sig['spread_pct']}%  объём={sig['vol_ratio']}x  "
                f"лимит={sig['entry_limit']}  стоп={sig['stop']}  "
                f"тейк={sig['take']}", cfg)

    return journal, new_sigs


# ─────────────────────────────────────────────
#  СТАТИСТИКА
# ─────────────────────────────────────────────
def print_stats(journal, cfg):
    trades  = journal["trades"]
    balance = journal["balance"]
    init    = journal["initial_deposit"]
    pnl     = balance - init
    pnl_pct = round(pnl / init * 100, 2)
    scans   = journal.get("scan_count", 0)
    refills = len(journal.get("refills", []))

    print(f"\n{'='*62}")
    print(f"  📊 БОТ #2 v2 — СТРУКТУРА РЫНКА HH/HL/LH/LL")
    print(f"  Старт: {journal['created'][:10]}  Сканирований: {scans}")
    print(f"{'='*62}")
    s = "+" if pnl >= 0 else ""
    print(f"  Депозит:  ${init:.2f}  (пополнений: {refills})")
    print(f"  Баланс:   ${balance:.2f}  ({s}${pnl:.2f}  {s}{pnl_pct}%)")

    if trades:
        wins   = [t for t in trades if t["result"] == "WIN"]
        losses = [t for t in trades if t["result"] == "LOSS"]
        longs  = [t for t in trades if t["dir"] == 1]
        shorts = [t for t in trades if t["dir"] == -1]
        wr     = round(len(wins) / len(trades) * 100, 1)
        avg_w  = round(np.mean([t["pnl_usd"] for t in wins]),   2) if wins   else 0
        avg_l  = round(np.mean([t["pnl_usd"] for t in losses]), 2) if losses else 0
        eq     = [init] + [t["balance_after"] for t in trades]
        peak   = np.maximum.accumulate(eq)
        dd     = round(((np.array(eq) - peak) / peak * 100).min(), 2)

        # По структурам
        for st in ["HH+HL", "LH+LL"]:
            st_trades = [t for t in trades if t.get("structure") == st]
            if st_trades:
                st_wr = round(len([t for t in st_trades
                                   if t["result"]=="WIN"]) / len(st_trades) * 100, 1)
                print(f"  {st}: {len(st_trades)} сделок  WR={st_wr}%")

        print(f"\n  Всего: {len(trades)}  (🟢{len(wins)}  🔴{len(losses)})")
        print(f"  WR: {wr}%  Просадка: {dd}%")
        print(f"  Ср. профит: +${avg_w}  Ср. убыток: ${avg_l}")
        print(f"  Лонгов: {len(longs)}  Шортов: {len(shorts)}")

        print(f"\n  📋 Последние 5 сделок:")
        for t in trades[-5:]:
            em = "🟢" if t["result"] == "WIN" else "🔴"
            d  = "ЛОНГ" if t["dir"] == 1 else "ШОРТ"
            print(f"    {em} {t['symbol']:<14} [{t['tf']}] {d} "
                  f"[{t.get('structure','?')}] "
                  f"{t['result']} {t['pnl_usd']:>+.2f}$  "
                  f"баланс=${t['balance_after']:.2f}")
        pd.DataFrame(trades).to_csv("structure2_trades.csv", index=False)
    else:
        print(f"\n  Сделок пока нет")

    if journal["pending"]:
        print(f"\n  ⏳ Ожидают ({len(journal['pending'])}):")
        for p in journal["pending"]:
            d = "ЛОНГ" if p["dir"] == 1 else "ШОРТ"
            print(f"    {p['symbol']:<14} [{p['tf']}] {d} "
                  f"[{p.get('structure','?')}] "
                  f"лимит={p['entry_limit']}  ждём {p.get('waited',0)} цикл.")
    if journal["open"]:
        print(f"\n  🔓 Открыто ({len(journal['open'])}):")
        for p in journal["open"]:
            d = "ЛОНГ" if p["dir"] == 1 else "ШОРТ"
            print(f"    {p['symbol']:<14} [{p['tf']}] {d} "
                  f"[{p.get('structure','?')}] "
                  f"вход={p['entry_limit']}  стоп={p['stop']}  тейк={p['take']}")

    print(f"\n  🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*62}")


# ─────────────────────────────────────────────
#  ГЛАВНЫЙ ЦИКЛ
# ─────────────────────────────────────────────
def main():
    cfg  = CONFIG.copy()
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"

    if mode == "reset":
        for f in [cfg["journal_file"], cfg["log_file"], "structure2_trades.csv"]:
            if os.path.exists(f):
                os.remove(f)
        print("[+] Журнал сброшен.")
        return

    journal = load_journal(cfg)

    if mode == "status":
        print_stats(journal, cfg)
        return

    if mode == "refill":
        old_bal = journal["balance"]
        journal["balance"] = cfg["initial_deposit"]
        save_journal(journal, cfg)
        print(f"[+] Баланс: ${old_bal:.2f} → ${cfg['initial_deposit']:.2f}")
        return

    interval = cfg["scan_interval"] * 60

    print("=" * 62)
    print(f"  🏗️  БОТ #2 v2 — СТРУКТУРА РЫНКА")
    print(f"  TF: {', '.join(cfg['timeframes'])}  "
          f"Свинг: {cfg['swing_bars']}св  "
          f"Вход: Фибо {int(cfg['entry_fibo']*100)}%")
    print(f"  Депозит: ${cfg['initial_deposit']}  "
          f"Риск: {int(cfg['risk_pct']*100)}%  "
          f"RR 1:{cfg['rr_ratio']}")
    print(f"  Интервал: {cfg['scan_interval']}мин  |  Ctrl+C остановка")
    print("=" * 62)

    log("="*50, cfg)
    log(f"СТАРТ БОТ#2 v2  депозит=${cfg['initial_deposit']}  "
        f"риск={int(cfg['risk_pct']*100)}%  "
        f"свинг={cfg['swing_bars']}св", cfg)

    ex = get_exchange()

    while True:
        try:
            log(f"--- ЦИКЛ #{journal.get('scan_count',0)+1} ---", cfg)
            journal, new_sigs = run_cycle(ex, journal, cfg)
            save_journal(journal, cfg)

            trades = journal["trades"]
            wins   = len([t for t in trades if t["result"] == "WIN"])
            wr     = round(wins/len(trades)*100,1) if trades else 0
            s      = "+" if journal["balance"] >= cfg["initial_deposit"] else ""
            log(f"Баланс=${journal['balance']}  Сделок={len(trades)}  "
                f"WR={wr}%  "
                f"P&L={s}{round(journal['balance']-cfg['initial_deposit'],2)}$  "
                f"Новых={new_sigs}", cfg)
            log(f"Следующий цикл через {cfg['scan_interval']} минут...", cfg)
            time.sleep(interval)

        except KeyboardInterrupt:
            log("ОСТАНОВЛЕН (Ctrl+C)", cfg)
            print_stats(journal, cfg)
            break
        except Exception as e:
            import traceback
            log(f"ОШИБКА: {e}", cfg)
            log(f"ДЕТАЛИ: {traceback.format_exc()}", cfg)
            log("Повтор через 1 минуту...", cfg)
            time.sleep(60)


if __name__ == "__main__":
    main()
