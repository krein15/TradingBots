"""
Backtest/context.py
===================
Исторический рыночный контекст: режим рынка и тренд BTC.

Боты в бою берут режим из shared_state.json, который пишет
market_regime.py по последним 100 часовым свечам BTC. Чтобы бэктест
проверял ТУ ЖЕ стратегию, а не её пересказ, режим здесь считается
вызовом тех же самых функций из market_regime.py — на скользящем
окне той же длины.

Защита от подглядывания в будущее:
  Часовая свеча с меткой t закрывается в t+1ч. Значит в момент
  5-минутной свечи T последняя ИЗВЕСТНАЯ часовая — это та, чья
  метка <= T - 1ч. Если брать свечу с меткой <= T, стратегия
  получит цену закрытия часа, который ещё не закончился.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import market_regime as mr
from Backtest.data import TF_MS

# Столько свечей market_regime.py запрашивает у биржи в бою
REGIME_WINDOW = 100


def regime_series(btc_1h, window=REGIME_WINDOW):
    """
    Режим рынка для каждой часовой свечи BTC.

    Возвращает DataFrame: timestamp (метка часовой свечи),
    regime, confidence. Значение относится к состоянию рынка
    НА МОМЕНТ ЗАКРЫТИЯ этой свечи.
    """
    rows = []
    cols = ["ts", "open", "high", "low", "close", "vol"]

    for i in range(len(btc_1h)):
        if i + 1 < window:
            rows.append((int(btc_1h.timestamp.iloc[i]), "?", 0))
            continue
        win = btc_1h.iloc[i + 1 - window: i + 1]
        df = pd.DataFrame({
            "ts":    win["timestamp"].values,
            "open":  win["open"].values,
            "high":  win["high"].values,
            "low":   win["low"].values,
            "close": win["close"].values,
            "vol":   win["volume"].values,
        })[cols]

        adx, plus_di, minus_di = mr.calc_adx(df)
        atr_ratio              = mr.calc_atr_ratio(df)
        bb_width, bb_avg       = mr.calc_bb_width(df)
        bb_squeeze   = bb_width < bb_avg * 0.8
        bb_expansion = bb_width > bb_avg * 1.3

        close = df["close"].iloc[-1]
        ema50 = df["close"].ewm(span=50, adjust=False).mean().iloc[-1]

        # Порядок проверок повторяет market_regime.get_market_regime
        if atr_ratio > 1.8 and bb_expansion:
            regime, conf = "VOLATILE", 80
        elif adx >= 30 and close > ema50 and plus_di > minus_di:
            regime, conf = "TREND_UP", min(100, int(adx * 2))
        elif adx >= 30 and close < ema50 and minus_di > plus_di:
            regime, conf = "TREND_DOWN", min(100, int(adx * 2))
        elif bb_squeeze and atr_ratio > 1.1:
            regime, conf = "BREAKOUT", 70
        elif adx < 25 and atr_ratio < 1.3:
            regime, conf = "SIDEWAYS", 75
        else:
            regime, conf = "SIDEWAYS", 50

        rows.append((int(btc_1h.timestamp.iloc[i]), regime, conf))

    return pd.DataFrame(rows, columns=["timestamp", "regime", "confidence"])


def btc_trend_series(btc_tf, ema_period=50, impulse_candles=3):
    """
    Тренд BTC для каждой свечи выбранного таймфрейма — повторяет
    get_btc_trend из Bot1: строго выше/ниже EMA, без нейтральной зоны.

    Возвращает DataFrame: timestamp, trend (bull/bear), btc_chg.
    """
    close = btc_tf["close"]
    ema   = close.ewm(span=ema_period, adjust=False).mean()
    trend = np.where(close > ema, "bull", "bear")
    chg   = close.pct_change(impulse_candles).fillna(0.0)
    return pd.DataFrame({
        "timestamp": btc_tf["timestamp"].astype("int64").values,
        "trend":     trend,
        "btc_chg":   chg.values,
    })


class ContextLookup:
    """
    Отдаёт значение контекста, известное на момент времени ts.

    Свеча с меткой t закрывается в t + длительность_тф, поэтому
    ищем последнюю запись с меткой <= ts - длительность_тф.
    Без этого сдвига бэктест читал бы будущее.
    """

    def __init__(self, frame, timeframe, value_cols):
        self.ts = frame["timestamp"].to_numpy(dtype="int64")
        self.values = {c: frame[c].to_numpy() for c in value_cols}
        self.close_lag = TF_MS[timeframe]
        self.value_cols = value_cols

    def at(self, ts_ms, default=None):
        idx = np.searchsorted(self.ts, ts_ms - self.close_lag, side="right") - 1
        if idx < 0:
            return default
        return tuple(self.values[c][idx] for c in self.value_cols)
