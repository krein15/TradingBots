"""
Bot5_Donchian/test_cycle.py
===========================
Сквозной тест цикла бота на подставной бирже.

Зачем. test_exits.py проверяет check_exit по отдельности и прошёл —
а бот при этом сутки не работал: код вокруг вызывал функцию, которой
нет, и каждый цикл падал с NameError. Отдельные функции были верны,
сломано было их соединение. Этот тест гоняет настоящий run_cycle
целиком, поэтому ловит и такое.

Три сценария за один цикл:
  OLD  — позиция, открытая до правки со свечой входа (нет
         entry_wall_ms). На свече входа прокол ниже стопа случился
         ДО входа, после входа цена стопа не касалась. Позиция должна
         остаться открытой. Это ровно тот путь, который падал.
  STOP — стоп задет на более поздней свече. Позиция закрывается по
         стопу, баланс меняется ровно на PnL сделки.
  UP   — свежий пробой канала. Бот открывает позицию и записывает
         момент входа.

Запуск:
  python Bot5_Donchian/test_cycle.py
"""

import os
import sys
import tempfile
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import paper_trading_donchian as core  # noqa: E402

core.time.sleep = lambda s: None          # пауза между запросами тесту не нужна

STEP = core.TF_MS["4h"]
MIN = 60_000
BASE = 1_780_000_000_000 // STEP * STEP  # начало истории, кратно 4ч
N = 400                                  # закрытых свечей в истории
LAST = BASE + (N - 1) * STEP             # последняя закрытая свеча
NOW = LAST + STEP + 10 * MIN             # идёт 10-я минута новой свечи

SIG = BASE + 390 * STEP                  # сигнальная свеча у OLD и STOP
ENTRY_BAR = SIG + STEP                   # свеча, внутри которой вошли


def flat(wick_bar=None, wick_low=None):
    """Спокойная цена ~100; на одной свече — прокол вниз."""
    rows = []
    for k in range(N + 1):               # +1 — незакрытая свеча
        ts = BASE + k * STEP
        low = wick_low if ts == wick_bar else 99.0
        rows.append([ts, 100.0, 102.0, low, 100.5, 1000.0])
    return rows


def breakout():
    """Плавный рост, на последней закрытой свече — пробой вверх."""
    rows = []
    for k in range(N + 1):
        ts = BASE + k * STEP
        p = 50 + k * 0.1 + (0.4 if k % 2 else -0.4)
        if k == N - 1:
            rows.append([ts, p, p * 1.06, p * 0.995, p * 1.05, 5000.0])
        else:
            rows.append([ts, p, p + 0.5, p - 0.5, p, 1000.0])
    return rows


class FakeExchange:
    def __init__(self):
        self.h4 = {
            "OLD/USDT:USDT": flat(ENTRY_BAR, 90.0),         # прокол ДО входа
            "STOP/USDT:USDT": flat(SIG + 5 * STEP, 90.0),   # прокол позже
            "UP/USDT:USDT": breakout(),
        }
        self.calls = []

    def milliseconds(self):
        return NOW

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        self.calls.append((symbol, timeframe))
        if timeframe == "1m":
            # Минутки после входа: цена держится выше стопа
            out, t = [], since
            while t < NOW and len(out) < (limit or 200):
                out.append([t, 100.0, 101.0, 98.5, 100.2, 10.0])
                t += MIN
            return out
        rows = self.h4[symbol]
        return rows[-limit:] if limit else rows

    def fetch_ticker(self, symbol):
        return {"last": self.h4[symbol][-1][4]}


def journal_with_positions():
    old_wall = ENTRY_BAR + 15 * MIN
    return {
        "created": "2026-09-01T00:00:00", "deposit": 50.0, "balance": 50.0,
        "trades": [], "cooldown": {}, "cycles": 0, "acted": {},
        "open": [
            {   # старая позиция: момент входа только в поле opened
                "symbol": "OLD/USDT:USDT", "dir": 1, "type": "test",
                "entry": 100.0, "stop": 94.0, "take": 118.0,
                "qty": 0.2, "notional": 20.0, "atr": 2.4, "stop_pct": 0.06,
                "opened": datetime.fromtimestamp(old_wall / 1000).isoformat(),
                "entry_ts": SIG, "signal_age_min": 15.0, "bars_held": 0,
            },
            {   # новая позиция, стоп задет на 5-й свече после сигнала
                "symbol": "STOP/USDT:USDT", "dir": 1, "type": "test",
                "entry": 100.0, "stop": 94.0, "take": 118.0,
                "qty": 0.2, "notional": 20.0, "atr": 2.4, "stop_pct": 0.06,
                "opened": datetime.fromtimestamp((ENTRY_BAR + 5 * MIN) / 1000).isoformat(),
                "entry_ts": SIG, "entry_wall_ms": ENTRY_BAR + 5 * MIN,
                "signal_age_min": 5.0, "bars_held": 0,
            },
        ],
    }


def main():
    print("=" * 62)
    print("  СКВОЗНОЙ ТЕСТ ЦИКЛА БОТА (подставная биржа)")
    print("=" * 62)

    tmp = tempfile.mkdtemp()
    cfg = dict(core.CONFIG)
    cfg["journal"] = os.path.join(tmp, "j.json")
    cfg["logfile"] = os.path.join(tmp, "log.txt")

    ex = FakeExchange()
    j = journal_with_positions()

    # Главное: цикл проходит целиком, без исключений
    opened = core.run_cycle(ex, j, cfg, ["UP/USDT:USDT"])
    print("  ✔  цикл прошёл целиком, без исключений")

    by_sym = {p["symbol"]: p for p in j["open"]}

    # OLD: прокол до входа не стал стопом, размах уточнён по минуткам
    old = by_sym.get("OLD/USDT:USDT")
    assert old is not None, "OLD закрыта по проколу, случившемуся ДО входа"
    assert old.get("entry_bar", {}).get("low") == 98.5, f"размах не уточнён: {old.get('entry_bar')}"
    assert ("OLD/USDT:USDT", "1m") in ex.calls, "минутки для свечи входа не запрашивались"
    print("  ✔  OLD: старая позиция без entry_wall_ms — прокол до входа не стал стопом")

    # STOP: закрыта по стопу на нужной свече, баланс = депозит + PnL
    assert "STOP/USDT:USDT" not in by_sym, "STOP не закрыта, хотя стоп задет"
    t = j["trades"][-1]
    assert t["symbol"] == "STOP/USDT:USDT" and t["exit_reason"] == "stop", t
    assert t["exit_ts"] == SIG + 5 * STEP, f"стоп не на той свече: {t['exit_ts']}"
    assert t["entry_bar_checked"] is True
    assert abs(j["balance"] - (50.0 + t["pnl"])) < 1e-6, "баланс не сошёлся с PnL"
    assert -1.2 < t["r_multiple"] < -1.0, f"R стопа вне ожидаемого: {t['r_multiple']}"
    assert "STOP/USDT:USDT" in j["cooldown"], "после убытка нет паузы"
    print(f"  ✔  STOP: закрыта по стопу, {t['r_multiple']:+.3f}R, баланс ${j['balance']:.4f}")

    # UP: новый вход по свежему пробою, момент входа записан
    up = by_sym.get("UP/USDT:USDT")
    assert opened == 1 and up is not None, f"пробой не открыт (opened={opened})"
    assert up["entry_wall_ms"] == NOW
    assert up["entry_ts"] == LAST, "сигнал взят не с последней закрытой свечи"
    assert abs(up["signal_age_min"] - 10.0) < 0.01
    assert up["stop"] < up["entry"] < up["take"]
    print(f"  ✔  UP: открыт лонг по пробою, возраст сигнала {up['signal_age_min']} мин")

    # Повторный цикл: тот же сигнал второй раз не открывается
    opened2 = core.run_cycle(ex, j, cfg, ["UP/USDT:USDT"])
    assert opened2 == 0, "по одному сигналу открыто дважды"
    print("  ✔  повторный цикл: тот же сигнал второй раз не открыт")

    # Биржа молчит: цикл обязан упасть с понятной ошибкой, а не
    # отчитаться «Баланс=...», ничего не проверив (так было 18.09)
    class DeadExchange(FakeExchange):
        def fetch_ohlcv(self, *a, **k):
            raise ConnectionError("Remote end closed connection")

    j2 = journal_with_positions()
    try:
        core.run_cycle(DeadExchange(), j2, cfg, ["UP/USDT:USDT"])
    except core.DataUnavailable as e:
        assert "ConnectionError" in str(e), f"в ошибке нет причины: {e}"
        assert len(j2["open"]) == 2, "позиции потеряны при сбое связи"
        print("  ✔  биржа молчит: цикл падает с понятной ошибкой, позиции на месте")
    else:
        raise AssertionError("цикл вслепую прошёл как успешный")
    print("=" * 62)


if __name__ == "__main__":
    main()
