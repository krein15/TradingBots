"""
Backtest/test_engine.py
=======================
Регрессионные тесты движка на синтетике с заранее известным ответом.

Запуск:
  python Backtest/test_engine.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from Backtest import search, strategies
from Backtest.engine import FILL_MARKET, Engine

WARMUP = 60      # движок пропускает первые свечи на прогрев индикаторов


def _synthetic_trend(n=140, s0=70):
    """Цена 100, сигнал лонг на s0, рост до 130, откат до 110."""
    close = np.full(n, 100.0)
    close[s0 + 1:s0 + 20] = np.linspace(100, 130, 19)
    close[s0 + 20:] = np.linspace(130, 110, n - s0 - 20)
    df = pd.DataFrame({"timestamp": np.arange(n) * 14_400_000, "open": close,
                       "high": close + 0.5, "low": close - 0.5,
                       "close": close, "volume": 1.0})
    df["atr"] = 2.0
    d = np.zeros(n, dtype="int8")
    d[s0] = 1
    df.attrs["sig"] = strategies.Signals(d, close.copy(), close - 6.0,
                                         close * 100, "TEST")
    return df


def test_r_uses_initial_stop_with_trailing():
    """
    R обязан считаться от ИСХОДНОГО риска, а не от подтянутого стопа.

    Баг, который этот тест ловит: сделка записывала финальный стоп, а
    r_metrics делил PnL на |вход - стоп|. У прибыльного трейлингового
    выхода финальный стоп и есть цена выхода, поэтому КАЖДАЯ такая
    сделка получала ровно +1R, сколько бы ни заработала. А при стопе,
    подтянутом вплотную к входу, знаменатель стремился к нулю и R
    улетал до +7 и выше. Все трейлинговые результаты перебора были
    искажены в обе стороны.
    """
    eng = Engine(cfg={}, signal_fn=strategies.signal_fn, deposit=1e6,
                 risk_pct=0.01, max_open=5, commission=0.0, slippage=0.0,
                 fill_model=FILL_MARKET, max_wait_bars=1, cooldown_minutes=0,
                 max_leverage=1000, trail_atr=3.0,
                 max_hold_bars=200).run({"X": _synthetic_trend()})
    tdf = eng.trades_df()
    assert len(tdf) == 1, f"ожидалась 1 сделка, получено {len(tdf)}"
    t = tdf.iloc[0]
    assert abs(t.initial_stop - 94.0) < 1e-9, f"исходный стоп {t.initial_stop}, ожидался 94"
    assert t.exit_reason == "trail", f"причина выхода {t.exit_reason}"
    true_r = (t.exit_price - t.entry_price) / (t.entry_price - t.initial_stop)
    got = search.r_metrics(eng)["mean_R"]
    assert abs(got - true_r) < 1e-3, f"R={got}, правильный {true_r:.3f}"
    assert true_r > 3.5, f"правильный R {true_r:.3f} подозрительно мал"
    return f"R={got:+.3f} (без исправления было бы +1.000)"


def test_fixed_stop_r_unchanged():
    """Без трейлинга исходный и финальный стоп совпадают — R как раньше."""
    df = _synthetic_trend()
    n = len(df)
    df.attrs["sig"] = strategies.Signals(
        df.attrs["sig"]["dir"], df["close"].values.copy(),
        df["close"].values - 6.0, df["close"].values + 18.0, "TEST")
    eng = Engine(cfg={}, signal_fn=strategies.signal_fn, deposit=1e6,
                 risk_pct=0.01, max_open=5, commission=0.0, slippage=0.0,
                 fill_model=FILL_MARKET, max_wait_bars=1, cooldown_minutes=0,
                 max_leverage=1000, trail_atr=None,
                 max_hold_bars=200).run({"X": df})
    t = eng.trades_df().iloc[0]
    assert t.exit_reason == "take", f"причина выхода {t.exit_reason}"
    assert abs(t.initial_stop - t["stop"]) < 1e-9
    got = search.r_metrics(eng)["mean_R"]
    assert abs(got - 3.0) < 1e-3, f"R={got}, ожидалось +3.0"
    return f"R={got:+.3f}"


def main():
    tests = [test_r_uses_initial_stop_with_trailing, test_fixed_stop_r_unchanged]
    failed = 0
    print("=" * 62)
    print("  РЕГРЕССИОННЫЕ ТЕСТЫ ДВИЖКА")
    print("=" * 62)
    for fn in tests:
        try:
            msg = fn()
            print(f"  ✔  {fn.__name__}: {msg}")
        except AssertionError as e:
            failed += 1
            print(f"  ✘  {fn.__name__}: {e}")
    print("=" * 62)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
