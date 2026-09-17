"""
Bot5_Donchian/test_exits.py
===========================
Регрессионные тесты выходов бумажного бота.

Проверяют ровно ту ошибку, из-за которой они появились: бот входит
внутри свечи, а стоп проверял по всему её размаху — включая движение
до входа. На истории это выглядело бы как честный стоп, и заметить
подмену по журналу невозможно.

Запуск:
  python Bot5_Donchian/test_exits.py
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import paper_trading_donchian as core  # noqa: E402

STEP = 14_400_000
SIG = 1_700_000_000_000          # свеча сигнала
BAR = SIG + STEP                 # свеча, внутри которой вошли


def bars(rows):
    return pd.DataFrame(rows, columns=["timestamp", "open", "high",
                                       "low", "close", "volume"])


def long_pos():
    return {"symbol": "TEST/USDT:USDT", "dir": 1, "entry": 100.0,
            "stop": 94.0, "take": 118.0, "qty": 1.0, "entry_ts": SIG}


def test_pre_entry_wick_ignored():
    """Прокол до входа не должен становиться стопом."""
    pos = long_pos()
    df = bars([[BAR, 100, 104, 90, 103, 1]])        # low 90 — ниже стопа 94
    reason, _, _ = core.check_exit(pos, df)
    assert reason == "stop", "без уточнения свеча входа даёт стоп"

    # Тот же бар, но размах с момента входа: прокола не было
    reason, _, _ = core.check_exit(pos, df, {"ts": BAR, "high": 104, "low": 98})
    assert reason is None, f"прокол до входа засчитан как стоп: {reason}"
    print("  ✔  test_pre_entry_wick_ignored: прокол до входа не стал стопом")


def test_real_stop_still_fires():
    """Уточнение не должно прятать настоящий стоп."""
    pos = long_pos()
    df = bars([[BAR, 100, 104, 90, 92, 1]])
    reason, ts, _ = core.check_exit(pos, df, {"ts": BAR, "high": 104, "low": 93})
    assert reason == "stop" and ts == BAR, f"настоящий стоп потерян: {reason}"
    print("  ✔  test_real_stop_still_fires: стоп после входа сработал")


def test_later_bars_untouched():
    """Уточнение применяется только к свече входа."""
    pos = long_pos()
    df = bars([[BAR, 100, 104, 98, 103, 1],
               [BAR + STEP, 103, 105, 90, 95, 1]])   # стоп на следующей свече
    reason, ts, _ = core.check_exit(pos, df, {"ts": BAR, "high": 104, "low": 98})
    assert reason == "stop" and ts == BAR + STEP, f"не та свеча: {reason} {ts}"
    print("  ✔  test_later_bars_untouched: следующие свечи считаются целиком")


def test_stop_wins_ties():
    """Стоп и тейк на одной свече — считаем стопом."""
    pos = long_pos()
    df = bars([[BAR, 100, 120, 93, 110, 1]])
    reason, _, _ = core.check_exit(pos, df, {"ts": BAR, "high": 120, "low": 93})
    assert reason == "stop", f"при двойном касании выбран {reason}"
    print("  ✔  test_stop_wins_ties: при двойном касании выбран стоп")


def test_short_side():
    """То же самое для шорта — знаки не перепутаны."""
    pos = {"symbol": "TEST/USDT:USDT", "dir": -1, "entry": 100.0,
           "stop": 106.0, "take": 82.0, "qty": 1.0, "entry_ts": SIG}
    df = bars([[BAR, 100, 110, 96, 97, 1]])          # high 110 — выше стопа
    assert core.check_exit(pos, df)[0] == "stop"
    reason, _, _ = core.check_exit(pos, df, {"ts": BAR, "high": 102, "low": 96})
    assert reason is None, f"шорт: прокол до входа засчитан ({reason})"
    print("  ✔  test_short_side: для шорта уточнение работает так же")


if __name__ == "__main__":
    print("=" * 62)
    print("  РЕГРЕССИОННЫЕ ТЕСТЫ ВЫХОДОВ БОТА")
    print("=" * 62)
    test_pre_entry_wick_ignored()
    test_real_stop_still_fires()
    test_later_bars_untouched()
    test_stop_wins_ties()
    test_short_side()
    print("=" * 62)
