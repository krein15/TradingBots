"""
divergence_module.py
====================
Модуль дивергенции RSI для всех ботов.
Импортируй в любой бот: from divergence_module import calc_rsi, find_divergence
"""

import numpy as np
import pandas as pd


def calc_rsi(close, period=14):
    """Считаем RSI."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(span=period, adjust=False).mean()
    avg_l = loss.ewm(span=period, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def find_swing_highs(series, n=3):
    """Находим локальные максимумы."""
    vals = series.values
    result = []
    for i in range(n, len(vals) - n):
        if all(vals[i] > vals[i-j] for j in range(1, n+1)) and \
           all(vals[i] > vals[i+j] for j in range(1, n+1)):
            result.append((i, vals[i]))
    return result


def find_swing_lows(series, n=3):
    """Находим локальные минимумы."""
    vals = series.values
    result = []
    for i in range(n, len(vals) - n):
        if all(vals[i] < vals[i-j] for j in range(1, n+1)) and \
           all(vals[i] < vals[i+j] for j in range(1, n+1)):
            result.append((i, vals[i]))
    return result


def find_divergence(df, lookback=50, rsi_period=14, swing_n=3):
    """
    Ищем дивергенцию между ценой и RSI.

    Медвежья дивергенция (для шорта):
      Цена: Higher High (новый максимум)
      RSI:  Lower High  (RSI не подтверждает)
      → сигнал на шорт

    Бычья дивергенция (для лонга):
      Цена: Lower Low  (новый минимум)
      RSI:  Higher Low (RSI не подтверждает)
      → сигнал на лонг

    Возвращает:
      "bearish" — медвежья дивергенция
      "bullish" — бычья дивергенция
      None      — нет дивергенции
    """
    if len(df) < rsi_period + lookback:
        return None, 0

    close = df["close"]
    rsi   = calc_rsi(close, rsi_period)

    # Смотрим последние N свечей
    start = max(rsi_period + swing_n, len(df) - lookback)
    price_slice = close.iloc[start:]
    rsi_slice   = rsi.iloc[start:]

    # ── Медвежья дивергенция ──────────────────
    price_highs = find_swing_highs(price_slice, swing_n)
    rsi_highs   = find_swing_highs(rsi_slice,   swing_n)

    if len(price_highs) >= 2 and len(rsi_highs) >= 2:
        # Последние два хая цены и RSI
        ph1_i, ph1 = price_highs[-1]
        ph2_i, ph2 = price_highs[-2]
        rh1_i, rh1 = rsi_highs[-1]
        rh2_i, rh2 = rsi_highs[-2]

        # Цена делает новый хай, RSI нет
        price_hh = ph1 > ph2  # Higher High на цене
        rsi_lh   = rh1 < rh2  # Lower High на RSI
        recent   = ph1_i >= len(price_slice) - 10  # свежий сигнал

        if price_hh and rsi_lh and recent:
            strength = round(abs(ph1/ph2 - 1) * 100, 2)
            return "bearish", strength

    # ── Бычья дивергенция ────────────────────
    price_lows = find_swing_lows(price_slice, swing_n)
    rsi_lows   = find_swing_lows(rsi_slice,   swing_n)

    if len(price_lows) >= 2 and len(rsi_lows) >= 2:
        pl1_i, pl1 = price_lows[-1]
        pl2_i, pl2 = price_lows[-2]
        rl1_i, rl1 = rsi_lows[-1]
        rl2_i, rl2 = rsi_lows[-2]

        price_ll = pl1 < pl2  # Lower Low на цене
        rsi_hl   = rl1 > rl2  # Higher Low на RSI
        recent   = pl1_i >= len(price_slice) - 10

        if price_ll and rsi_hl and recent:
            strength = round(abs(pl1/pl2 - 1) * 100, 2)
            return "bullish", strength

    return None, 0


def calc_volume_profile(df, bins=20):
    """
    Volume Profile — распределение объёма по уровням цены.
    Возвращает Point of Control (POC) — уровень с максимальным объёмом.
    """
    if len(df) < 20:
        return None, None, None

    price_min = df["low"].min()
    price_max = df["high"].max()

    if price_max <= price_min:
        return None, None, None

    # Делим диапазон на bins уровней
    bin_size = (price_max - price_min) / bins
    vol_by_level = np.zeros(bins)

    for _, row in df.iterrows():
        # Для каждой свечи добавляем объём в соответствующие уровни
        low_bin  = int((row["low"]  - price_min) / bin_size)
        high_bin = int((row["high"] - price_min) / bin_size)
        low_bin  = max(0, min(low_bin,  bins-1))
        high_bin = max(0, min(high_bin, bins-1))

        for b in range(low_bin, high_bin + 1):
            vol_by_level[b] += row["vol"] / max(1, high_bin - low_bin + 1)

    # POC — уровень с максимальным объёмом
    poc_bin   = np.argmax(vol_by_level)
    poc_price = price_min + (poc_bin + 0.5) * bin_size

    # Value Area (70% объёма вокруг POC)
    total_vol = vol_by_level.sum()
    target    = total_vol * 0.7
    accumulated = vol_by_level[poc_bin]
    low_bin_va  = poc_bin
    high_bin_va = poc_bin

    while accumulated < target:
        expand_low  = low_bin_va  > 0
        expand_high = high_bin_va < bins - 1
        if not expand_low and not expand_high:
            break
        add_low  = vol_by_level[low_bin_va-1]  if expand_low  else 0
        add_high = vol_by_level[high_bin_va+1] if expand_high else 0
        if add_low >= add_high and expand_low:
            low_bin_va -= 1
            accumulated += add_low
        elif expand_high:
            high_bin_va += 1
            accumulated += add_high
        else:
            break

    vah = price_min + (high_bin_va + 1) * bin_size  # Value Area High
    val = price_min + low_bin_va * bin_size           # Value Area Low

    return round(poc_price, 6), round(val, 6), round(vah, 6)
