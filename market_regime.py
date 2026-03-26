"""
market_regime.py
================
ML мета-модель — определяет режим рынка и
выбирает лучшую стратегию.

Режимы рынка:
  TREND_UP   → Бот #1 EMA (только лонги)
  TREND_DOWN → Бот #1 EMA (только шорты)
  SIDEWAYS   → Бот #2 MeanReversion
  BREAKOUT   → Бот #4 Breakout
  VOLATILE   → пропускаем (высокий риск)

Запуск (информационный — логирует режим):
  python market_regime.py

Интеграция с ботами — импортируй:
  from market_regime import get_market_regime
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import time
import json
import os

# ── Пути к файлам состояния ──────────────────────────────────
SHARED_STATE_PATH  = "C:\\TradingBots\\shared_state.json"
REGIME_HISTORY_PATH = "C:\\TradingBots\\ML\\regime_history.jsonl"


def save_regime(regime, confidence, details):
    """
    Сохраняем текущий режим в shared_state.json (для ботов)
    и дописываем строку в regime_history.jsonl (для ML).
    """
    now = datetime.now().isoformat()
    bot, desc = regime_to_strategy(regime)

    # shared_state.json — текущий режим, боты читают перед сканированием
    state = {
        "regime":      regime,
        "confidence":  confidence,
        "recommended_bot": bot,
        "description": desc,
        "updated_at":  now,
        "details":     details,
    }
    try:
        os.makedirs(os.path.dirname(SHARED_STATE_PATH), exist_ok=True)
        with open(SHARED_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  [!] Ошибка записи shared_state: {e}")

    # regime_history.jsonl — история режимов, одна строка = одно измерение
    record = {
        "ts":         now,
        "regime":     regime,
        "confidence": confidence,
        "adx":        details.get("adx"),
        "atr_ratio":  details.get("atr_ratio"),
        "bb_width":   details.get("bb_width"),
        "plus_di":    details.get("plus_di"),
        "minus_di":   details.get("minus_di"),
        "btc_close":  details.get("close"),
        "ema50":      details.get("ema50"),
    }
    try:
        os.makedirs(os.path.dirname(REGIME_HISTORY_PATH), exist_ok=True)
        with open(REGIME_HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  [!] Ошибка записи regime_history: {e}")


def read_regime():
    """
    Читаем текущий режим из shared_state.json.
    Боты вызывают эту функцию перед сканированием.
    Возвращает dict или None если файл не найден/устарел.
    """
    try:
        with open(SHARED_STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        # Проверяем свежесть — не старше 60 минут
        updated = datetime.fromisoformat(state["updated_at"])
        age_min = (datetime.now() - updated).total_seconds() / 60
        if age_min > 60:
            return None  # данные устарели
        return state
    except Exception:
        return None


def get_exchange():
    return ccxt.binance({"enableRateLimit": True})


def fetch_btc(ex, tf="1h", limit=100):
    try:
        raw = ex.fetch_ohlcv("BTC/USDT", tf, limit=limit)
        df  = pd.DataFrame(raw,
              columns=["ts","open","high","low","close","vol"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        return df
    except Exception:
        return None


def calc_adx(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    up   = high - high.shift()
    down = low.shift() - low
    plus_dm  = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr      = tr.ewm(span=period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0,1e-9)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0,1e-9)
    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0,1e-9)
    adx = dx.ewm(span=period, adjust=False).mean()
    return round(adx.iloc[-1], 1), round(plus_di.iloc[-1], 1), round(minus_di.iloc[-1], 1)


def calc_atr_ratio(df, period=14):
    """ATR текущий / ATR средний за 20 периодов"""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr     = tr.ewm(span=period, adjust=False).mean()
    atr_avg = atr.rolling(20).mean()
    return round(atr.iloc[-1] / atr_avg.iloc[-1], 2)


def calc_bb_width(df, period=20):
    """Ширина Bollinger Bands — показатель волатильности"""
    close   = df["close"]
    mid     = close.rolling(period).mean()
    std     = close.rolling(period).std()
    width   = (2 * std / mid)
    # Текущая vs средняя
    return round(width.iloc[-1], 4), round(width.rolling(20).mean().iloc[-1], 4)


def get_market_regime(ex=None):
    """
    Определяем режим рынка по BTC 1h.

    Возвращает:
      regime: TREND_UP / TREND_DOWN / SIDEWAYS / BREAKOUT / VOLATILE
      confidence: 0-100
      details: dict с метриками
    """
    if ex is None:
        ex = get_exchange()

    df = fetch_btc(ex, "1h", 100)
    if df is None:
        return "SIDEWAYS", 50, {}

    # Метрики
    adx, plus_di, minus_di = calc_adx(df)
    atr_ratio               = calc_atr_ratio(df)
    bb_width, bb_avg        = calc_bb_width(df)
    bb_squeeze              = bb_width < bb_avg * 0.8  # полосы сужены
    bb_expansion            = bb_width > bb_avg * 1.3  # полосы расширяются

    # EMA тренд
    ema20  = df["close"].ewm(span=20,  adjust=False).mean().iloc[-1]
    ema50  = df["close"].ewm(span=50,  adjust=False).mean().iloc[-1]
    ema200 = df["close"].ewm(span=200, adjust=False).mean().iloc[-1]
    close  = df["close"].iloc[-1]

    # Волатильность
    returns   = df["close"].pct_change().dropna()
    volatility= round(returns.std() * 100, 3)

    details = {
        "adx":        adx,
        "plus_di":    plus_di,
        "minus_di":   minus_di,
        "atr_ratio":  atr_ratio,
        "bb_width":   bb_width,
        "bb_avg":     bb_avg,
        "bb_squeeze": bool(bb_squeeze),     # явный Python bool для JSON
        "bb_expansion": bool(bb_expansion),  # явный Python bool для JSON
        "ema20":      round(ema20, 2),
        "ema50":      round(ema50, 2),
        "close":      round(close, 2),
        "volatility": volatility,
    }

    # ── Определяем режим ─────────────────────

    # VOLATILE: ATR резко вырос + широкие BB
    if atr_ratio > 1.8 and bb_expansion:
        return "VOLATILE", 80, details

    # TREND_UP: сильный ADX + цена выше EMA + plus_di > minus_di
    if adx >= 30 and close > ema50 and plus_di > minus_di:
        confidence = min(100, int(adx * 2))
        return "TREND_UP", confidence, details

    # TREND_DOWN: сильный ADX + цена ниже EMA + minus_di > plus_di
    if adx >= 30 and close < ema50 and minus_di > plus_di:
        confidence = min(100, int(adx * 2))
        return "TREND_DOWN", confidence, details

    # BREAKOUT: BB squeeze + ATR начинает расти
    if bb_squeeze and atr_ratio > 1.1:
        return "BREAKOUT", 70, details

    # SIDEWAYS: ADX низкий + ATR нормальный
    if adx < 25 and atr_ratio < 1.3:
        return "SIDEWAYS", 75, details

    # По умолчанию
    return "SIDEWAYS", 50, details


def regime_to_strategy(regime):
    """Какой бот запускать при данном режиме"""
    mapping = {
        "TREND_UP":   ("Bot1_EMA",        "лонги по тренду"),
        "TREND_DOWN": ("Bot1_EMA",        "шорты по тренду"),
        "SIDEWAYS":   ("Bot2_MeanRev",    "возврат к среднему"),
        "BREAKOUT":   ("Bot4_Breakout",   "пробой консолидации"),
        "VOLATILE":   ("ПАУЗА",           "высокий риск — ждём"),
    }
    return mapping.get(regime, ("Bot1_EMA", "по умолчанию"))


def print_regime_report(regime, confidence, details):
    bot, desc = regime_to_strategy(regime)
    emoji = {
        "TREND_UP":   "📈",
        "TREND_DOWN": "📉",
        "SIDEWAYS":   "➡️",
        "BREAKOUT":   "🚀",
        "VOLATILE":   "⚡",
    }.get(regime, "❓")

    print(f"\n{'='*55}")
    print(f"  {emoji} РЕЖИМ РЫНКА: {regime}  (уверенность: {confidence}%)")
    print(f"  Рекомендуемый бот: {bot} — {desc}")
    print(f"{'='*55}")
    print(f"  ADX:        {details.get('adx','?')}  "
          f"(>30=тренд, <25=боковик)")
    print(f"  +DI/-DI:    {details.get('plus_di','?')} / "
          f"{details.get('minus_di','?')}")
    print(f"  ATR ratio:  {details.get('atr_ratio','?')}  "
          f"(>1.5=высокая волатильность)")
    print(f"  BB width:   {details.get('bb_width','?')} / "
          f"avg {details.get('bb_avg','?')}")
    print(f"  BB squeeze: {details.get('bb_squeeze','?')}")
    print(f"  BTC close:  ${details.get('close','?')}  "
          f"EMA50: ${details.get('ema50','?')}")
    print(f"  Волатильность: {details.get('volatility','?')}%/час")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    print("Определяем режим рынка по BTC/USDT 1h...")
    ex = get_exchange()

    # Мониторинг каждые 30 минут
    while True:
        try:
            regime, confidence, details = get_market_regime(ex)
            print_regime_report(regime, confidence, details)
            save_regime(regime, confidence, details)

            bot, desc = regime_to_strategy(regime)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            print(f"  [{ts}] {regime} → запускай {bot}")
            print(f"  [+] Сохранено в shared_state.json и regime_history.jsonl")
            print(f"  Следующая проверка через 30 минут...")
            time.sleep(30 * 60)

        except KeyboardInterrupt:
            print("\nОстановлен.")
            break
        except Exception as e:
            print(f"Ошибка: {e}")
            time.sleep(60)
