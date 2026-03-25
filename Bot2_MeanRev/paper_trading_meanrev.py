"""
paper_trading_meanrev.py
========================
Бот #2 v3 — Mean Reversion (Боковик)
Биржа: Binance | TF: 15m, 1h

Логика:
  Рынок в боковике (ADX < 25) →
  RSI перепродан (<30) → лонг от нижней границы BB
  RSI перекуплен (>70) → шорт от верхней границы BB
  Стоп за BB band + буфер
  Тейк: средняя линия BB (возврат к среднему)

Фильтры:
  ADX < 25 (флет, не тренд)
  BB squeeze (полосы сужены)
  Объём не аномальный (<3x — нет новостного движения)
  BTC тренд нейтральный или слабый
"""

import sys, time, os, json
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime

CONFIG = {
    "timeframes":       ["15m", "1h"],
    "candles":          200,
    "vol_period":       20,
    "rsi_period":       14,
    "bb_period":        20,
    "bb_std":           2.0,
    "rsi_oversold":     30,
    "rsi_overbought":   70,
    "adx_period":       14,
    "adx_max":          25,        # торгуем только в боковике
    "max_vol_mult":     3.0,       # исключаем новостные движения
    "min_usdt_vol":     1_000_000,
    "rr":               2.0,       # RR 1:2 (тейк = средняя BB)
    "commission":       0.001,
    "buf":              0.002,
    "max_wait":         8,
    "deposit":          50.0,
    "risk_pct":         0.05,
    "max_trades":       5,
    "cooldown_h":       2,
    "interval_min":     10,
    "journal":          "meanrev_journal.json",
    "logfile":          "meanrev_log.txt",
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
    return ccxt.binance({"enableRateLimit": True})


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
        ms_map = {"5m":300000,"15m":900000,"1h":3600000,"4h":14400000}
        ms = ms_map.get(tf, 900000)
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


def calc_indicators(df, cfg):
    """RSI + Bollinger Bands + ADX"""
    close = df["close"]

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(span=cfg["rsi_period"], adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(span=cfg["rsi_period"], adjust=False).mean()
    rs    = gain / loss.replace(0, 1e-9)
    rsi   = 100 - (100 / (1 + rs))

    # Bollinger Bands
    bb_mid = close.rolling(cfg["bb_period"]).mean()
    bb_std = close.rolling(cfg["bb_period"]).std()
    bb_up  = bb_mid + cfg["bb_std"] * bb_std
    bb_dn  = bb_mid - cfg["bb_std"] * bb_std
    bb_width = (bb_up - bb_dn) / bb_mid  # ширина полос

    # ADX
    high, low = df["high"], df["low"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    up   = high - high.shift()
    down = low.shift() - low
    plus_dm  = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr14    = tr.ewm(span=14, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr14.replace(0,1e-9)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr14.replace(0,1e-9)
    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0,1e-9)
    adx = dx.ewm(span=14, adjust=False).mean()

    return {
        "rsi":      round(rsi.iloc[-1], 1),
        "bb_mid":   round(bb_mid.iloc[-1], 6),
        "bb_up":    round(bb_up.iloc[-1], 6),
        "bb_dn":    round(bb_dn.iloc[-1], 6),
        "bb_width": round(bb_width.iloc[-1], 4),
        "adx":      round(adx.iloc[-1], 1),
    }


def find_signals(df, cfg):
    """
    Сигнал Mean Reversion:

    ЛОНГ: RSI < 30 + цена у нижней BB + ADX < 25
    ШОРТ: RSI > 70 + цена у верхней BB + ADX < 25
    Тейк: средняя линия BB
    Стоп: за BB + буфер
    """
    ind       = calc_indicators(df, cfg)
    close_now = df.iloc[-1]["close"]
    vol_now   = df.iloc[-1]["vol_ratio"]
    signals   = []

    # Фильтр: боковик
    if ind["adx"] >= cfg["adx_max"]:
        return signals

    # Фильтр: нет новостного движения
    if vol_now >= cfg["max_vol_mult"]:
        return signals

    buf = cfg["buf"]
    rr  = cfg["rr"]

    # ЛОНГ: RSI перепродан + цена у нижней BB
    if ind["rsi"] <= cfg["rsi_oversold"]:
        near_dn = close_now <= ind["bb_dn"] * 1.005
        if near_dn:
            entry = close_now
            stop  = ind["bb_dn"] * (1 - buf)
            risk  = entry - stop
            if risk <= 0:
                return signals
            take  = ind["bb_mid"]  # тейк = средняя BB
            if take <= entry:
                return signals
            signals.append({
                "dir":      1,
                "entry":    round(entry, 6),
                "stop":     round(stop, 6),
                "take":     round(take, 6),
                "type":     "MR_ЛОНГ",
                "rsi":      ind["rsi"],
                "adx":      ind["adx"],
                "bb_width": ind["bb_width"],
                "vol":      round(vol_now, 2),
                "rr":       round((take - entry) / risk, 2),
            })

    # ШОРТ: RSI перекуплен + цена у верхней BB
    if ind["rsi"] >= cfg["rsi_overbought"]:
        near_up = close_now >= ind["bb_up"] * 0.995
        if near_up:
            entry = close_now
            stop  = ind["bb_up"] * (1 + buf)
            risk  = stop - entry
            if risk <= 0:
                return signals
            take  = ind["bb_mid"]
            if take >= entry:
                return signals
            signals.append({
                "dir":      -1,
                "entry":    round(entry, 6),
                "stop":     round(stop, 6),
                "take":     round(take, 6),
                "type":     "MR_ШОРТ",
                "rsi":      ind["rsi"],
                "adx":      ind["adx"],
                "bb_width": ind["bb_width"],
                "vol":      round(vol_now, 2),
                "rr":       round((entry - take) / risk, 2),
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
        time.sleep(0.08)
        if price is None:
            new_pending.append(p); continue
        p["waited"] = p.get("waited", 0) + 1
        d, entry = p["dir"], p["entry"]
        hit = (d == 1 and price <= entry * 1.003) or \
              (d == -1 and price >= entry * 0.997)
        if hit:
            risk = abs(entry - p["stop"])
            qty  = (balance * cfg["risk_pct"]) / risk if risk > 0 else 0
            if qty > 0:
                new_open.append({**p, "qty": round(qty,6), "opened": now})
                log(f"✅ ОТКРЫТА {p['symbol']} "
                    f"{'ЛОНГ' if d==1 else 'ШОРТ'} [{p['type']}] "
                    f"RSI={p.get('rsi','?')} ADX={p.get('adx','?')} "
                    f"вход={entry}", cfg)
        elif p["waited"] >= cfg["max_wait"]:
            log(f"⏱️  ОТМЕНЕНА {p['symbol']}", cfg)
        else:
            new_pending.append(p)

    # Open
    still_open = []
    for pos in new_open:
        price = get_price(ex, pos["symbol"])
        time.sleep(0.08)
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
                "rsi": pos.get("rsi",0), "adx": pos.get("adx",0),
                "bb_width": pos.get("bb_width",0),
                "vol": pos.get("vol",0),
                "pnl": round(pnl,4), "result": result,
                "balance": round(balance,4), "closed": now,
            })
            em = "🟢 WIN" if result=="WIN" else "🔴 LOSS"
            log(f"{em} {pos['symbol']} "
                f"PnL=${round(pnl,2):+.2f} баланс=${round(balance,2)}", cfg)
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

    symbols  = get_symbols(ex, cfg["min_usdt_vol"])
    new_sigs = 0

    for tf in cfg["timeframes"]:
        if open_cnt >= cfg["max_trades"]: break
        for sym in symbols[:100]:
            if open_cnt >= cfg["max_trades"]: break
            if sym in cooling: continue
            df = fetch_df(ex, sym, tf, cfg["candles"], cfg)
            time.sleep(0.08)
            if df is None: continue
            sigs = find_signals(df, cfg)
            if not sigs: continue
            sig = sigs[0]
            key = f"{sym}_{sig['dir']}"
            if key in existing: continue
            journal["pending"].append({
                "symbol": sym, "tf": tf,
                "dir": sig["dir"], "entry": sig["entry"],
                "stop": sig["stop"], "take": sig["take"],
                "type": sig["type"], "rsi": sig["rsi"],
                "adx": sig["adx"], "bb_width": sig["bb_width"],
                "vol": sig["vol"], "rr": sig["rr"],
                "added": now, "waited": 0,
            })
            existing.add(key)
            open_cnt += 1
            new_sigs += 1
            d_ru = "ЛОНГ" if sig["dir"]==1 else "ШОРТ"
            log(f"➕ {sym} [{tf}] {d_ru} [{sig['type']}] "
                f"RSI={sig['rsi']} ADX={sig['adx']} "
                f"BB_width={sig['bb_width']} "
                f"вход={sig['entry']} стоп={sig['stop']} "
                f"тейк={sig['take']}", cfg)

    return journal, new_sigs


def print_stats(j, cfg):
    trades  = j["trades"]
    balance = j["balance"]
    deposit = j["deposit"]
    pnl     = balance - deposit
    print(f"\n{'='*60}")
    print(f"  📊 БОТ #2 v3 — MEAN REVERSION | Binance")
    print(f"  Циклов: {j.get('cycles',0)}")
    s = "+" if pnl >= 0 else ""
    print(f"  Баланс: ${balance:.2f}  ({s}${pnl:.2f})")
    if trades:
        wins = [t for t in trades if t["result"]=="WIN"]
        wr   = round(len(wins)/len(trades)*100,1)
        eq   = [deposit] + [t["balance"] for t in trades]
        dd   = round(((np.array(eq)-np.maximum.accumulate(eq))/
                      np.maximum.accumulate(eq)*100).min(),2)
        print(f"  Сделок: {len(trades)}  WR: {wr}%  DD: {dd}%")
        print(f"\n  Последние 5:")
        for t in trades[-5:]:
            em = "🟢" if t["result"]=="WIN" else "🔴"
            d  = "ЛОНГ" if t["dir"]==1 else "ШОРТ"
            print(f"  {em} {t['symbol']:<14} {d} "
                  f"RSI={t.get('rsi','?')} ADX={t.get('adx','?')} "
                  f"{t['result']} {t['pnl']:>+.2f}$")
        pd.DataFrame(trades).to_csv("meanrev_trades.csv", index=False)
    print(f"{'='*60}")


def main():
    cfg  = CONFIG.copy()
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "reset":
        for f in [cfg["journal"], cfg["logfile"], "meanrev_trades.csv"]:
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

    print("="*60)
    print(f"  📊 БОТ #2 v3 — MEAN REVERSION")
    print(f"  Стратегия: RSI + Bollinger Bands в боковике")
    print(f"  ADX < {cfg['adx_max']} (только флет)  RR 1:{cfg['rr']}")
    print(f"  TF: {', '.join(cfg['timeframes'])}  Binance")
    print("="*60)

    log("="*50, cfg)
    log(f"СТАРТ МeanRev  депозит=${cfg['deposit']}", cfg)
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
