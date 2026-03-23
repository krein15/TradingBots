"""
debug_scanner.py
================
Диагностика — показывает ПОЧЕМУ нет сигналов.
Запускать когда основной бот не находит сигналов.

Запуск:
  python debug_scanner.py
"""

import time
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime

CONFIG = {
    "timeframes":       ["5m", "15m"],
    "candles":          300,
    "impulse_candles":  3,
    "min_price_chg":    0.01,        # снижено до 1%
    "min_vol_mult":     1.8,
    "vol_avg_period":   20,
    "fibo_entry":       0.236,
    "max_entry_miss":   0.05,
    "min_usdt_vol":     1_000_000,
    "btc_ema_period":   50,
    "btc_neutral_zone": 0.005,
    "stop_buffer":      0.001,
    "rr_ratio":         3.0,
}


def get_exchange():
    return ccxt.binance({"enableRateLimit": True})


def fetch_candles(exchange, symbol, timeframe, limit):
    try:
        ms_map = {"1m":60000,"5m":300000,"15m":900000}
        ms    = ms_map.get(timeframe, 300000)
        since = exchange.milliseconds() - limit * ms - ms * 20
        raw   = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not raw or len(raw) < 50:
            return None
        df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.reset_index(drop=True)
    except Exception as e:
        return None


def get_btc_trend(exchange, tf, cfg):
    try:
        df    = fetch_candles(exchange, "BTC/USDT", tf, cfg["btc_ema_period"] + 20)
        close = df["close"]
        ema50 = close.ewm(span=cfg["btc_ema_period"], adjust=False).mean()
        lc, le = close.iloc[-1], ema50.iloc[-1]
        diff  = (lc - le) / le
        # Нет нейтральной зоны — строго выше или ниже EMA
        trend = "bull" if diff > 0 else "bear"
        return trend, round(lc, 2), round(le, 2), round(diff*100, 3)
    except Exception as e:
        return "neutral", 0, 0, 0


def main():
    cfg = CONFIG.copy()
    ex  = get_exchange()

    print("=" * 65)
    print(f"  🔍 ДИАГНОСТИКА СКАНЕРА")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── ШАГ 1: Проверяем BTC тренд ────────────
    print("\n📊 ШАГ 1 — Тренд BTC:")
    for tf in cfg["timeframes"]:
        trend, btc_p, ema_p, diff = get_btc_trend(ex, tf, cfg)
        trend_ru = {"bull":"🟢 БЫЧИЙ","bear":"🔴 МЕДВЕЖИЙ",
                    "neutral":"🟡 БОКОВИК"}.get(trend)
        allow_l = trend in ("bull","neutral")
        allow_s = trend in ("bear","neutral")
        print(f"  [{tf}]  BTC=${btc_p}  EMA50=${ema_p}  "
              f"разница={diff:+}%  {trend_ru}")
        print(f"         allow_long={allow_l}  allow_short={allow_s}")

    # ── ШАГ 2: Берём топ-10 монет и смотрим детально ──
    print("\n📊 ШАГ 2 — Детальный анализ топ-10 монет:")
    tickers = ex.fetch_tickers()
    symbols = [(s, t.get("quoteVolume") or 0)
               for s, t in tickers.items()
               if s.endswith("/USDT") and s != "BTC/USDT"
               and (t.get("quoteVolume") or 0) >= cfg["min_usdt_vol"]]
    symbols.sort(key=lambda x: x[1], reverse=True)
    top10 = [s[0] for s in symbols[:10]]

    for tf in cfg["timeframes"]:
        trend, _, _, _ = get_btc_trend(ex, tf, cfg)
        allow_long  = trend in ("bull", "neutral")
        allow_short = trend in ("bear", "neutral")

        print(f"\n  ── {tf} (тренд: {trend}) ──────────────────────")
        total_sigs = 0
        total_recent = 0
        total_not_missed = 0

        for sym in top10:
            df = fetch_candles(ex, sym, tf, cfg["candles"])
            time.sleep(0.1)
            if df is None:
                print(f"  {sym:<16} нет данных")
                continue

            # Индикаторы
            c, n, p = df["close"], cfg["impulse_candles"], cfg["vol_avg_period"]
            df["vol_avg"]      = df["volume"].rolling(p).mean()
            df["vol_ratio"]    = df["volume"] / df["vol_avg"].replace(0, 1e-9)
            df["price_chg"]    = (c - c.shift(n)) / c.shift(n)
            df["impulse_high"] = df["high"].rolling(n).max()
            df["impulse_low"]  = df["low"].rolling(n).min()

            start_i = n + p
            all_sigs = 0
            all_recent = 0
            not_missed = 0
            reject_reasons = []

            for i in range(start_i, len(df)):
                row = df.iloc[i]
                chg, vol = row["price_chg"], row["vol_ratio"]

                # Считаем все потенциальные сигналы
                if abs(chg) >= cfg["min_price_chg"] and vol >= cfg["min_vol_mult"]:
                    all_sigs += 1
                    imp_high = row["impulse_high"]
                    imp_low  = row["impulse_low"]
                    spread   = imp_high - imp_low

                    if chg > 0 and allow_long:
                        entry = imp_low + spread * cfg["fibo_entry"]
                        stop  = imp_low * (1 - cfg["stop_buffer"])
                        risk  = entry - stop
                        if risk > 0 and entry < imp_high:
                            if i >= len(df) - 5:
                                all_recent += 1
                                current = df.iloc[-1]["close"]
                                diff_pct = (current - entry) / entry
                                if diff_pct <= cfg["max_entry_miss"]:
                                    not_missed += 1

                    elif chg < 0 and allow_short:
                        entry = imp_high - spread * cfg["fibo_entry"]
                        stop  = imp_high * (1 + cfg["stop_buffer"])
                        risk  = stop - entry
                        if risk > 0 and entry > imp_low:
                            take = entry - risk * cfg["rr_ratio"]
                            if take > 0:
                                if i >= len(df) - 5:
                                    all_recent += 1
                                    current = df.iloc[-1]["close"]
                                    diff_pct = (entry - current) / entry
                                    if diff_pct <= cfg["max_entry_miss"]:
                                        not_missed += 1

                # Причины отсева на последних свечах
                elif i >= len(df) - 5:
                    if abs(chg) < cfg["min_price_chg"]:
                        reject_reasons.append(
                            f"чг={round(chg*100,2)}%<{cfg['min_price_chg']*100}%")
                    elif vol < cfg["min_vol_mult"]:
                        reject_reasons.append(
                            f"объём={round(vol,1)}x<{cfg['min_vol_mult']}x")

            total_sigs     += all_sigs
            total_recent   += all_recent
            total_not_missed += not_missed

            status = "✅ СИГНАЛ!" if not_missed > 0 else "❌"
            print(f"  {status} {sym:<16} "
                  f"всего_сигн={all_sigs:>4}  "
                  f"свежих={all_recent}  "
                  f"не_упущено={not_missed}"
                  + (f"  причины_отсева: {', '.join(reject_reasons[-2:])}"
                     if reject_reasons and all_sigs == 0 else ""))

        print(f"\n  ИТОГО [{tf}]: сигналов={total_sigs}  "
              f"свежих={total_recent}  "
              f"активных={total_not_missed}")

    # ── ШАГ 3: Проверяем пороги ────────────────
    print(f"\n📊 ШАГ 3 — Текущие пороги фильтрации:")
    print(f"  min_price_chg:  {cfg['min_price_chg']*100}%  "
          f"(импульс за {cfg['impulse_candles']} свечи)")
    print(f"  min_vol_mult:   {cfg['min_vol_mult']}x  "
          f"(объём выше среднего за {cfg['vol_avg_period']} свечей)")
    print(f"  max_entry_miss: {cfg['max_entry_miss']*100}%  "
          f"(максимум как далеко цена ушла от лимитки)")
    print(f"  fibo_entry:     {cfg['fibo_entry']*100}%  "
          f"(уровень отката для лимитного входа)")

    # ── ШАГ 4: Рекомендации ────────────────────
    print(f"\n📊 ШАГ 4 — Рекомендации:")
    print(f"  Если всего_сигналов=0  → рынок флет, снизить min_price_chg до 1.5%")
    print(f"  Если свежих=0          → сигналы есть но не на последних свечах")
    print(f"  Если не_упущено=0      → цена уже ушла от лимитки, увеличить max_entry_miss")
    print(f"\n[✓] Диагностика завершена!")


if __name__ == "__main__":
    main()
