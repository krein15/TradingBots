"""
paper_trading_breakout.py
=========================
Бот #4 v2 — Breakout (Волатильность/Пробой)
Биржа: Binance | TF: 15m, 1h

Логика:
  Цена консолидируется N свечей (ATR сжат) →
  Резкий пробой уровня с объёмом →
  Вход в направлении пробоя
  
  Это противоположность Mean Reversion —
  работает когда рынок выходит из боковика

Фильтры:
  ATR сжат последние 10 свечей (консолидация)
  Пробой на объёме > 2x
  BTC в том же направлении
  Свеча закрывается за уровнем (не ложный пробой)
"""

import sys, time, os, json
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime

CONFIG = {
    "timeframes":       ["15m", "1h"],
    "candles":          150,
    "vol_period":       20,
    "atr_period":       14,
    "consol_bars":      10,        # свечей консолидации
    "atr_squeeze":      0.9,       # ATR < 90% от среднего = сжатие
    "breakout_vol":     1.5,       # снижено до 1.5x
    "min_breakout_pct": 0.003,     # минимальный пробой 0.3%
    "max_entry_miss":   0.005,     # цена не ушла более 0.5% от входа
    "min_usdt_vol":     2_000_000,
    "rr":               3.0,
    "commission":       0.001,
    "buf":              0.001,
    "max_wait":         5,
    "deposit":          50.0,
    "risk_pct":         0.05,
    "max_trades":       4,
    "cooldown_h":       2,
    "interval_min":     15,
    "journal":          "breakout_journal.json",
    "logfile":          "breakout_log.txt",
}


def log(msg, cfg, show=True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if show: print(line)
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


def get_btc_trend(ex, cfg):
    try:
        df = fetch_df(ex, "BTC/USDT", "1h", 60, cfg)
        if df is None: return "neutral"
        ema = df["close"].ewm(span=50, adjust=False).mean()
        lc, le = df["close"].iloc[-1], ema.iloc[-1]
        if lc > le * 1.005: return "bull"
        if lc < le * 0.995: return "bear"
        return "neutral"
    except Exception:
        return "neutral"


def find_signals(df, cfg):
    """
    Ищем пробой консолидации:
    1. ATR последние N свечей ниже среднего (сжатие)
    2. Последняя свеча пробивает max/min диапазона
    3. Объём высокий на пробое
    4. Свеча закрывается ЗА уровнем
    """
    signals = []
    n     = cfg["consol_bars"]
    buf   = cfg["buf"]
    rr    = cfg["rr"]

    # ATR
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr     = tr.ewm(span=cfg["atr_period"], adjust=False).mean()
    atr_avg = atr.rolling(20).mean()

    # Последняя свеча
    last     = df.iloc[-1]
    prev_n   = df.iloc[-(n+1):-1]  # N свечей до последней

    # Проверка сжатия ATR — последние N свечей ATR должен быть сжат
    atr_last  = atr.iloc[-2]
    atr_mean  = atr_avg.iloc[-2]
    # Проверяем что ATR последних N свечей в среднем ниже нормы
    atr_consol = atr.iloc[-(n+2):-2].mean()
    if atr_mean <= 0:
        return signals
    if atr_consol / atr_mean > cfg["atr_squeeze"]:
        return signals  # нет сжатия ATR в зоне консолидации

    # Уровни консолидации
    consol_high = prev_n["high"].max()
    consol_low  = prev_n["low"].min()
    consol_range = consol_high - consol_low
    if consol_range <= 0:
        return signals

    vol_now = last["vol_ratio"]
    if vol_now < cfg["breakout_vol"]:
        return signals  # нет объёма

    # Проверяем что пробой произошёл на ПОСЛЕДНЕЙ свече
    # (не на исторической — иначе момент упущен)
    prev_close = df.iloc[-2]["close"]
    if prev_close > consol_high or prev_close < consol_low:
        return signals  # пробой уже произошёл раньше — опоздали

    # Пробой вверх
    if last["close"] > consol_high and \
       (last["close"] - consol_high) / consol_high >= cfg["min_breakout_pct"]:
        entry = last["close"]
        stop  = consol_low * (1 - buf)   # стоп за низ консолидации
        risk  = entry - stop
        if risk > 0:
            take = entry + risk * rr
            signals.append({
                "dir":           1,
                "entry":         round(entry, 6),
                "stop":          round(stop, 6),
                "take":          round(take, 6),
                "type":          "BO_ЛОНГ",
                "consol_high":   round(consol_high, 6),
                "consol_low":    round(consol_low, 6),
                "consol_bars":   n,
                "atr_ratio":     round(atr_last/atr_mean, 2),
                "vol":           round(vol_now, 2),
                "rr":            round(risk * rr / risk, 2),
            })

    # Пробой вниз
    if last["close"] < consol_low and \
       (consol_low - last["close"]) / consol_low >= cfg["min_breakout_pct"]:
        entry = last["close"]
        stop  = consol_high * (1 + buf)
        risk  = stop - entry
        if risk > 0:
            take = entry - risk * rr
            if take > 0:
                signals.append({
                    "dir":           -1,
                    "entry":         round(entry, 6),
                    "stop":          round(stop, 6),
                    "take":          round(take, 6),
                    "type":          "BO_ШОРТ",
                    "consol_high":   round(consol_high, 6),
                    "consol_low":    round(consol_low, 6),
                    "consol_bars":   n,
                    "atr_ratio":     round(atr_last/atr_mean, 2),
                    "vol":           round(vol_now, 2),
                    "rr":            rr,
                })

    return signals


def run_cycle(ex, journal, cfg):
    now     = datetime.now().isoformat()
    balance = journal["balance"]
    comm    = cfg["commission"]
    journal["cycles"] = journal.get("cycles", 0) + 1

    if 0 < balance < journal["deposit"] * cfg["risk_pct"]:
        old_bal = balance
        balance = journal["deposit"]
        journal["balance"] = balance
        journal.setdefault("refills",[]).append(
            {"date":now,"from":round(old_bal,4),"to":balance})
        log(f"💰 АВТОПОПОЛНЕНИЕ ${old_bal:.2f} → ${balance:.2f}", cfg)

    new_pending = []
    new_open    = list(journal["open"])
    for p in journal["pending"]:
        price = get_price(ex, p["symbol"])
        time.sleep(0.08)
        if price is None:
            new_pending.append(p); continue
        p["waited"] = p.get("waited",0) + 1
        d, entry = p["dir"], p["entry"]
        # Лимитный ордер: цена должна быть близко к уровню входа
        # Breakout — вход по рынку, широкая зона
        hit = (d == 1 and entry * 0.993 <= price <= entry * 1.007) or \
              (d == -1 and entry * 0.993 <= price <= entry * 1.007)
        if hit:
            risk = abs(entry - p["stop"])
            qty  = (balance * cfg["risk_pct"]) / risk if risk > 0 else 0
            if qty > 0:
                new_open.append({**p, "qty":round(qty,6), "opened":now})
                log(f"✅ ОТКРЫТА {p['symbol']} "
                    f"{'ЛОНГ' if d==1 else 'ШОРТ'} [{p['type']}] "
                    f"vol={p.get('vol','?')}x вход={entry}", cfg)
        elif p["waited"] >= cfg["max_wait"]:
            log(f"⏱️  ОТМЕНЕНА {p['symbol']}", cfg)
        else:
            new_pending.append(p)

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
            if price <= stop:   result,exit_p = "LOSS",stop
            elif price >= take: result,exit_p = "WIN",take
        else:
            if price >= stop:   result,exit_p = "LOSS",stop
            elif price <= take: result,exit_p = "WIN",take
        if result:
            pnl = (exit_p-entry)*qty*d - (entry+exit_p)*qty*comm
            balance += pnl
            journal["trades"].append({
                "symbol":pos["symbol"],"tf":pos["tf"],
                "dir":d,"type":pos.get("type","?"),
                "entry":round(entry,6),"exit":round(exit_p,6),
                "vol":pos.get("vol",0),
                "atr_ratio":pos.get("atr_ratio",0),
                "pnl":round(pnl,4),"result":result,
                "balance":round(balance,4),"closed":now,
            })
            em = "🟢 WIN" if result=="WIN" else "🔴 LOSS"
            log(f"{em} {pos['symbol']} "
                f"PnL=${round(pnl,2):+.2f} баланс=${round(balance,2)}", cfg)
        else:
            still_open.append(pos)

    journal["open"]    = still_open
    journal["pending"] = new_pending
    journal["balance"] = round(balance,4)

    open_cnt = len(journal["open"]) + len(journal["pending"])
    if open_cnt >= cfg["max_trades"]:
        return journal, 0

    # BTC тренд
    btc = get_btc_trend(ex, cfg)

    existing = set(f"{p['symbol']}_{p['dir']}"
                   for p in journal["pending"]+journal["open"])
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
        for sym in symbols[:80]:
            if open_cnt >= cfg["max_trades"]: break
            if sym in cooling: continue
            df = fetch_df(ex, sym, tf, cfg["candles"], cfg)
            time.sleep(0.08)
            if df is None: continue
            sigs = find_signals(df, cfg)
            if not sigs: continue
            sig = sigs[0]
            # BTC фильтр
            if sig["dir"] == 1 and btc == "bear": continue
            if sig["dir"] == -1 and btc == "bull": continue
            key = f"{sym}_{sig['dir']}"
            if key in existing: continue
            journal["pending"].append({
                "symbol":sym,"tf":tf,
                "dir":sig["dir"],"entry":sig["entry"],
                "stop":sig["stop"],"take":sig["take"],
                "type":sig["type"],"vol":sig["vol"],
                "atr_ratio":sig["atr_ratio"],
                "rr":sig["rr"],"added":now,"waited":0,
            })
            existing.add(key)
            open_cnt += 1
            new_sigs += 1
            d_ru = "ЛОНГ" if sig["dir"]==1 else "ШОРТ"
            log(f"➕ {sym} [{tf}] {d_ru} [{sig['type']}] "
                f"ATR_ratio={sig['atr_ratio']} vol={sig['vol']}x "
                f"вход={sig['entry']} стоп={sig['stop']} "
                f"тейк={sig['take']}", cfg)

    return journal, new_sigs


def print_stats(j, cfg):
    trades  = j["trades"]
    balance = j["balance"]
    deposit = j["deposit"]
    pnl     = balance - deposit
    print(f"\n{'='*60}")
    print(f"  📊 БОТ #4 v2 — BREAKOUT | Binance")
    print(f"  Циклов: {j.get('cycles',0)}")
    s = "+" if pnl >= 0 else ""
    print(f"  Баланс: ${balance:.2f}  ({s}${pnl:.2f})")
    if trades:
        wins = [t for t in trades if t["result"]=="WIN"]
        wr   = round(len(wins)/len(trades)*100,1)
        eq   = [deposit]+[t["balance"] for t in trades]
        dd   = round(((np.array(eq)-np.maximum.accumulate(eq))/
                      np.maximum.accumulate(eq)*100).min(),2)
        print(f"  Сделок: {len(trades)}  WR: {wr}%  DD: {dd}%")
        for t in trades[-5:]:
            em = "🟢" if t["result"]=="WIN" else "🔴"
            d  = "ЛОНГ" if t["dir"]==1 else "ШОРТ"
            print(f"  {em} {t['symbol']:<14} {d} "
                  f"vol={t.get('vol','?')}x "
                  f"{t['result']} {t['pnl']:>+.2f}$")
        pd.DataFrame(trades).to_csv("breakout_trades.csv", index=False)
    print(f"{'='*60}")


def main():
    cfg  = CONFIG.copy()
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "reset":
        for f in [cfg["journal"],cfg["logfile"],"breakout_trades.csv"]:
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
    print(f"  🚀 БОТ #4 v2 — BREAKOUT")
    print(f"  Стратегия: ATR squeeze + пробой с объёмом")
    print(f"  TF: {', '.join(cfg['timeframes'])}  RR 1:{cfg['rr']}")
    print("="*60)

    log("="*50, cfg)
    log(f"СТАРТ Breakout  депозит=${cfg['deposit']}", cfg)
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
