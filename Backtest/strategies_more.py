"""
Backtest/strategies_more.py
===========================
Дополнительные семейства стратегий для широкого поиска.

Первый перебор проверил три семейства — пробой Дончиана, возврат к
среднему и откат к скользящей. Здесь добавлены стратегии с ДРУГОЙ
механикой, а не вариации тех же:

  тренд:          ma_cross, tsmom, keltner, supertrend, macd, rsi_momo
  волатильность:  bb_squeeze, nr7
  возврат:        zscore

Все сигналы причинные: каждое значение на свече i считается только
по свечам 0..i. Уровень пробоя всегда берётся со сдвигом на одну
свечу, чтобы текущая свеча не сравнивалась сама с собой.

Выход у всех одинаковый: стоп в ATR и тейк в R. Если rr = 0, тейка
нет — позиция живёт до стопа, трейлинг-стопа или выхода по времени.
Это нужно трендовым стратегиям: фиксированный тейк обрезает ровно те
сделки, ради которых тренд и торгуют.
"""

import numpy as np
import pandas as pd

from Backtest.strategies import _attach, adx, atr, rsi


def _finish(d, long_sig, short_sig, a, p, name):
    """
    Общий хвост: фильтр тренда, запрет шортов, стоп и тейк.

    ema     — лонги только выше EMA(ema), шорты только ниже
    adx_min — только при выраженном тренде
    rr = 0  — без тейка (далёкий недостижимый уровень)
    """
    c = d["close"]
    long_sig = long_sig.fillna(False).astype(bool)
    short_sig = short_sig.fillna(False).astype(bool)

    if p.get("ema"):
        e = c.ewm(span=p["ema"], adjust=False).mean()
        long_sig &= c > e
        short_sig &= c < e
    if p.get("adx_min"):
        ax = adx(d, 14)
        long_sig &= ax >= p["adx_min"]
        short_sig &= ax >= p["adx_min"]
    if not p.get("allow_short", True):
        short_sig &= False
    if not p.get("allow_long", True):
        long_sig &= False

    risk = (a * p["atr_mult"]).values
    entry = c.values
    direction = np.where(long_sig.values, 1, np.where(short_sig.values, -1, 0))
    stop = np.where(direction == 1, entry - risk, entry + risk)

    rr = p.get("rr", 3.0)
    if rr and rr > 0:
        take = np.where(direction == 1, entry + risk * rr, entry - risk * rr)
    else:
        # Тейка нет: уровень, до которого цена не дойдёт никогда.
        # Геометрия при этом остаётся валидной для _attach.
        take = np.where(direction == 1, entry * 100.0, entry * 0.001)
    return _attach(d, direction, entry, stop, take, name)


def _prep(df, p):
    d = df.copy()
    a = atr(d, p.get("atr_period", 14))
    d["atr"] = a
    return d, a


# ── Тренд ─────────────────────────────────────────────────────
def ma_cross(df, p):
    """Быстрая EMA пересекает медленную."""
    d, a = _prep(df, p)
    c = d["close"]
    fast = c.ewm(span=p["fast"], adjust=False).mean()
    slow = c.ewm(span=p["slow"], adjust=False).mean()
    diff = fast - slow
    long_sig = (diff > 0) & (diff.shift(1) <= 0)
    short_sig = (diff < 0) & (diff.shift(1) >= 0)
    return _finish(d, long_sig, short_sig, a, p, "MA")


def tsmom(df, p):
    """
    Моментум по собственной доходности (time-series momentum).

    Доходность за lookback свечей пересекает порог. Один из самых
    документированных эффектов на фьючерсных рынках вообще.
    """
    d, a = _prep(df, p)
    ret = d["close"].pct_change(p["lookback"])
    thr = p.get("threshold", 0.0)
    long_sig = (ret > thr) & (ret.shift(1) <= thr)
    short_sig = (ret < -thr) & (ret.shift(1) >= -thr)
    return _finish(d, long_sig, short_sig, a, p, "TSM")


def keltner(df, p):
    """Закрытие за каналом EMA ± k·ATR."""
    d, a = _prep(df, p)
    c = d["close"]
    mid = c.ewm(span=p["period"], adjust=False).mean()
    upper = (mid + p["k"] * a).shift(1)
    lower = (mid - p["k"] * a).shift(1)
    long_sig = (c > upper) & (c.shift(1) <= upper.shift(1))
    short_sig = (c < lower) & (c.shift(1) >= lower.shift(1))
    return _finish(d, long_sig, short_sig, a, p, "KC")


def supertrend(df, p):
    """
    Разворот индикатора Supertrend.

    Считается циклом — векторно он не выражается, потому что каждая
    граница зависит от предыдущей. Все обращения только к прошлому.
    """
    d, a = _prep(df, p)
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    av = a.values
    n = len(c)
    hl2 = (h + l) / 2
    ub = hl2 + p["mult"] * av
    lb = hl2 - p["mult"] * av
    fub, flb = ub.copy(), lb.copy()
    direction = np.ones(n, dtype=int)
    for i in range(1, n):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if direction[i - 1] == 1:
            direction[i] = -1 if c[i] < flb[i] else 1
        else:
            direction[i] = 1 if c[i] > fub[i] else -1
    dir_s = pd.Series(direction, index=d.index)
    long_sig = (dir_s == 1) & (dir_s.shift(1) == -1)
    short_sig = (dir_s == -1) & (dir_s.shift(1) == 1)
    return _finish(d, long_sig, short_sig, a, p, "ST")


def macd(df, p):
    """Линия MACD пересекает сигнальную."""
    d, a = _prep(df, p)
    c = d["close"]
    line = (c.ewm(span=p["fast"], adjust=False).mean()
            - c.ewm(span=p["slow"], adjust=False).mean())
    signal = line.ewm(span=p["signal"], adjust=False).mean()
    hist = line - signal
    long_sig = (hist > 0) & (hist.shift(1) <= 0)
    short_sig = (hist < 0) & (hist.shift(1) >= 0)
    return _finish(d, long_sig, short_sig, a, p, "MACD")


def rsi_momo(df, p):
    """
    RSI как индикатор СИЛЫ, а не перекупленности: вход, когда RSI
    уходит выше 50+level (лонг) или ниже 50-level (шорт).
    Противоположно тому, как RSI использовался в Bot1.
    """
    d, a = _prep(df, p)
    r = rsi(d["close"], p.get("rsi_period", 14))
    up, dn = 50 + p["level"], 50 - p["level"]
    long_sig = (r > up) & (r.shift(1) <= up)
    short_sig = (r < dn) & (r.shift(1) >= dn)
    return _finish(d, long_sig, short_sig, a, p, "RSIM")


# ── Волатильность ─────────────────────────────────────────────
def bb_squeeze(df, p):
    """
    Сжатие полос Боллинджера, затем пробой.

    Сжатие — ширина полос в нижнем квантиле за lookback свечей.
    Квантиль скользящий, то есть считается только по прошлому.
    Сигнал — закрытие за полосой, если сжатие было в последние
    `within` свечей до текущей.
    """
    d, a = _prep(df, p)
    c = d["close"]
    per = p.get("bb_period", 20)
    mid = c.rolling(per).mean()
    sd = c.rolling(per).std()
    up, dn = mid + 2 * sd, mid - 2 * sd
    width = (up - dn) / mid
    thr = width.rolling(p["lookback"]).quantile(p["q"])
    squeezed = (width <= thr).astype(float)
    recent = squeezed.shift(1).rolling(p.get("within", 5)).max() == 1
    long_sig = recent & (c > up.shift(1))
    short_sig = recent & (c < dn.shift(1))
    return _finish(d, long_sig, short_sig, a, p, "BBS")


def nr7(df, p):
    """
    Самая узкая свеча за n последних, затем пробой её границ.
    Классика из рабочих тетрадей Тоби Крабела.
    """
    d, a = _prep(df, p)
    h, l, c = d["high"], d["low"], d["close"]
    rng = h - l
    narrow = rng <= rng.rolling(p["n"]).min()
    prev_narrow = narrow.shift(1).fillna(False).astype(bool)
    long_sig = prev_narrow & (c > h.shift(1))
    short_sig = prev_narrow & (c < l.shift(1))
    return _finish(d, long_sig, short_sig, a, p, "NR")


# ── Возврат к среднему ────────────────────────────────────────
def zscore(df, p):
    """
    Цена ушла от скользящей средней больше чем на thr стандартных
    отклонений — ставим на возврат. Вход на пересечении порога.
    """
    d, a = _prep(df, p)
    c = d["close"]
    per = p["period"]
    mean = c.rolling(per).mean()
    sd = c.rolling(per).std().replace(0, np.nan)
    z = (c - mean) / sd
    thr = p["thr"]
    long_sig = (z < -thr) & (z.shift(1) >= -thr)
    short_sig = (z > thr) & (z.shift(1) <= thr)
    return _finish(d, long_sig, short_sig, a, p, "Z")


REGISTRY = {
    "ma_cross": ma_cross, "tsmom": tsmom, "keltner": keltner,
    "supertrend": supertrend, "macd": macd, "rsi_momo": rsi_momo,
    "bb_squeeze": bb_squeeze, "nr7": nr7, "zscore": zscore,
}
