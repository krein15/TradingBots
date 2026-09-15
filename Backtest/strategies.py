"""
Backtest/strategies.py
======================
Параметризуемые стратегии для перебора.

Зачем отдельно от бота: у Bot1 правила зашиты в код, и подобрать
дистанцию стопа там нельзя — она вычисляется из границ канала или
полос Боллинджера. А первый бэктест показал, что дело именно в
дистанции: комиссия в единицах риска равна 2*0.1%/стоп, и при стопе
1.2% она съедает 0.207R при валовом крае всего +0.048R.

Значит нужны стратегии, у которых стоп задаётся ЯВНО (в единицах
ATR) и может быть сколь угодно широким.

Сигналы считаются векторно один раз на инструмент и кладутся в
numpy-массивы в df.attrs. Движок потом только читает строку по
индексу. Без этого перебор сотен конфигураций упирался бы в
нарезку DataFrame на каждом баре.
"""

import numpy as np
import pandas as pd

SIG_COLS = ("dir", "entry", "stop", "take", "type")


# ── Индикаторы ────────────────────────────────────────────────
def atr(df, period=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(span=period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, 1e-9))


def adx(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    a = tr.ewm(span=period, adjust=False).mean().replace(0, 1e-9)
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / a
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / a
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
    return dx.ewm(span=period, adjust=False).mean()


# ── Упаковка сигналов ─────────────────────────────────────────
def _attach(df, direction, entry, stop, take, type_name):
    """
    Кладём сигналы в df.attrs numpy-массивами.

    Проверяем геометрию: стоп обязан быть по правильную сторону от
    входа, тейк — по другую. Сигнал с вывернутой геометрией дал бы
    отрицательный риск и бессмысленный размер позиции.
    """
    direction = np.asarray(direction, dtype="int8")
    entry = np.asarray(entry, dtype="float64")
    stop = np.asarray(stop, dtype="float64")
    take = np.asarray(take, dtype="float64")

    bad = np.zeros(len(direction), dtype=bool)
    long_m, short_m = direction == 1, direction == -1
    bad |= long_m & ~((stop < entry) & (take > entry))
    bad |= short_m & ~((stop > entry) & (take < entry))
    bad |= ~np.isfinite(entry) | ~np.isfinite(stop) | ~np.isfinite(take)
    bad |= (entry <= 0) | (stop <= 0) | (take <= 0)
    direction = np.where(bad, 0, direction).astype("int8")

    df.attrs["sig"] = {"dir": direction, "entry": entry,
                       "stop": stop, "take": take, "type": type_name}
    return df


def signal_fn(df, i, cfg, btc_trend, btc_chg, regime):
    """Чтение предрассчитанного сигнала — то, что вызывает движок."""
    sig = df.attrs.get("sig")
    if sig is None or sig["dir"][i] == 0:
        return []
    return [{
        "bar": i,
        "dir": int(sig["dir"][i]),
        "entry_limit": float(sig["entry"][i]),
        "stop": float(sig["stop"][i]),
        "take": float(sig["take"][i]),
        "type": f"{sig['type']}_{'ЛОНГ' if sig['dir'][i] == 1 else 'ШОРТ'}",
    }]


# ── Стратегия A: пробой канала Дончиана ───────────────────────
def donchian(df, p):
    """
    Классическое следование за трендом.

    Вход: закрытие выше максимума последних N свечей (для лонга).
    Стоп: atr_mult * ATR от входа — дистанция задаётся явно.
    Тейк: rr * дистанция стопа.

    Фильтры (каждый отключаемый):
      ema     — торгуем только по сторону длинной скользящей
      vol     — объём выше среднего в vol_mult раз
      adx_min — сила тренда
    """
    d = df.copy()
    n = p["channel"]
    a = atr(d, p.get("atr_period", 14))
    d["atr"] = a

    hi = d["high"].rolling(n).max().shift(1)
    lo = d["low"].rolling(n).min().shift(1)
    c = d["close"]

    long_sig = c > hi
    short_sig = c < lo

    if p.get("ema"):
        e = c.ewm(span=p["ema"], adjust=False).mean()
        long_sig &= c > e
        short_sig &= c < e
    if p.get("vol_mult"):
        vr = d["volume"] / d["volume"].rolling(20).mean().replace(0, 1e-9)
        long_sig &= vr >= p["vol_mult"]
        short_sig &= vr >= p["vol_mult"]
    if p.get("adx_min"):
        ax = adx(d, 14)
        long_sig &= ax >= p["adx_min"]
        short_sig &= ax >= p["adx_min"]
    if not p.get("allow_short", True):
        short_sig &= False

    risk = a * p["atr_mult"]
    direction = np.where(long_sig.fillna(False), 1,
                         np.where(short_sig.fillna(False), -1, 0))
    entry = c.values
    stop = np.where(direction == 1, entry - risk.values, entry + risk.values)
    take = np.where(direction == 1, entry + risk.values * p["rr"],
                    entry - risk.values * p["rr"])
    return _attach(d, direction, entry, stop, take, "DC")


# ── Стратегия B: возврат к среднему ───────────────────────────
def meanrev(df, p):
    """
    Вход против движения на перепроданности/перекупленности.

    Стоп и тейк — в ATR, а не по границам полос: именно это и
    позволяет управлять дистанцией стопа, чего в Bot1 не было.
    Фильтр adx_max держит стратегию в боковике.
    """
    d = df.copy()
    a = atr(d, p.get("atr_period", 14))
    d["atr"] = a
    c = d["close"]

    r = rsi(c, p.get("rsi_period", 14))
    bb_p, bb_s = p.get("bb_period", 20), p.get("bb_std", 2.0)
    mid = c.rolling(bb_p).mean()
    sd = c.rolling(bb_p).std()
    up, dn = mid + bb_s * sd, mid - bb_s * sd

    long_sig = (r <= p["rsi_low"]) & (c <= dn)
    short_sig = (r >= p["rsi_high"]) & (c >= up)

    if p.get("adx_max"):
        ax = adx(d, 14)
        long_sig &= ax <= p["adx_max"]
        short_sig &= ax <= p["adx_max"]
    if p.get("ema"):
        e = c.ewm(span=p["ema"], adjust=False).mean()
        long_sig &= c > e            # не ловим ножи против тренда
        short_sig &= c < e
    if not p.get("allow_short", True):
        short_sig &= False
    if not p.get("allow_long", True):
        long_sig &= False

    risk = a * p["atr_mult"]
    direction = np.where(long_sig.fillna(False), 1,
                         np.where(short_sig.fillna(False), -1, 0))
    entry = c.values
    stop = np.where(direction == 1, entry - risk.values, entry + risk.values)
    take = np.where(direction == 1, entry + risk.values * p["rr"],
                    entry - risk.values * p["rr"])
    return _attach(d, direction, entry, stop, take, "MR")


# ── Стратегия C: откат к скользящей по тренду ─────────────────
def pullback(df, p):
    """
    Тренд задан парой скользящих, вход — на возврате цены к быстрой
    после отката. Сделок заметно меньше, чем у пробоя, что при
    высокой комиссии само по себе плюс.
    """
    d = df.copy()
    a = atr(d, p.get("atr_period", 14))
    d["atr"] = a
    c = d["close"]

    fast = c.ewm(span=p["ema_fast"], adjust=False).mean()
    slow = c.ewm(span=p["ema_slow"], adjust=False).mean()

    up_trend, dn_trend = fast > slow, fast < slow
    # Откат: предыдущая свеча ушла за быструю, текущая вернулась
    long_sig = up_trend & (d["low"] <= fast) & (c > fast)
    short_sig = dn_trend & (d["high"] >= fast) & (c < fast)

    if p.get("adx_min"):
        ax = adx(d, 14)
        long_sig &= ax >= p["adx_min"]
        short_sig &= ax >= p["adx_min"]
    if not p.get("allow_short", True):
        short_sig &= False

    risk = a * p["atr_mult"]
    direction = np.where(long_sig.fillna(False), 1,
                         np.where(short_sig.fillna(False), -1, 0))
    entry = c.values
    stop = np.where(direction == 1, entry - risk.values, entry + risk.values)
    take = np.where(direction == 1, entry + risk.values * p["rr"],
                    entry - risk.values * p["rr"])
    return _attach(d, direction, entry, stop, take, "PB")


REGISTRY = {"donchian": donchian, "meanrev": meanrev, "pullback": pullback}
