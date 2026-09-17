"""
Backtest/test_strategies.py
===========================
Проверка всех стратегий до того, как им верить.

Две ошибки опаснее всего, и обе выглядят как результат:

1. Стратегия падает с исключением. search.prepare глотает его молча,
   и падение выглядит как «сделок нет» — то есть «края нет». Хорошая
   идея будет отброшена из-за опечатки.

2. Стратегия подглядывает в будущее. Тогда перебор находит
   «прибыльную» конфигурацию, которая в бою не заработает ничего.

Проверка причинности: история обрезается сразу после проверяемой
свечи, и сигнал на этой последней свече сравнивается с сигналом по
полной истории. На последней свече будущего нет физически — если
стратегия им пользуется, сигнал изменится. Подробнее в check().

Сам тест тоже проверен: две нарочно жульнические стратегии (по
следующему закрытию и по центрированному скользящему окну) он
обязан ловить. Первая версия теста их пропускала.

Запуск:
  python Backtest/test_strategies.py
"""

import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from Backtest import data
from Backtest.search import REGISTRY

# Разумные параметры по умолчанию для каждой стратегии
DEFAULTS = {
    "donchian":   dict(channel=20, atr_mult=2.5, rr=3.0, ema=200),
    "meanrev":    dict(rsi_low=30, rsi_high=70, bb_std=2.0, atr_mult=2.5, rr=2.0, adx_max=25),
    "pullback":   dict(ema_fast=20, ema_slow=100, atr_mult=2.5, rr=3.0),
    "ma_cross":   dict(fast=20, slow=100, atr_mult=2.5, rr=3.0),
    "tsmom":      dict(lookback=30, threshold=0.05, atr_mult=2.5, rr=3.0),
    "keltner":    dict(period=20, k=2.0, atr_mult=2.5, rr=3.0),
    "supertrend": dict(mult=3.0, atr_mult=2.5, rr=3.0),
    "macd":       dict(fast=12, slow=26, signal=9, atr_mult=2.5, rr=3.0, ema=200),
    "rsi_momo":   dict(level=10, atr_mult=2.5, rr=3.0),
    "bb_squeeze": dict(lookback=100, q=0.2, within=5, atr_mult=2.5, rr=3.0),
    "nr7":        dict(n=7, atr_mult=2.5, rr=3.0),
    "zscore":     dict(period=50, thr=2.0, atr_mult=2.5, rr=2.0),
}


def check(name, fn, df, params, n_random=40, seed=0):
    """
    Возвращает (ok, сообщение).

    Причинность проверяется ОБРЕЗКОЙ ИСТОРИИ, а не сравнением двух
    префиксов. Первая версия теста сравнивала сигналы на истории до
    свечи N и на удлинённой истории — и пропустила обе нарочно
    жульнические стратегии: для всех свечей, кроме последней,
    «будущая» свеча существует в обоих вариантах, и расхождение
    возможно только на самой границе.

    Правильно так: для свечи k посчитать стратегию на истории,
    обрезанной сразу после k, и сравнить сигнал на этой последней
    свече с сигналом на полной истории. На последней свече будущего
    нет физически — если стратегия им пользуется, сигнал изменится.
    Проверяются все свечи с сигналом плюс случайная выборка без него.
    """
    warm = 260
    try:
        full = fn(df.reset_index(drop=True), params)
    except Exception:
        return False, "ИСКЛЮЧЕНИЕ: " + traceback.format_exc(limit=3)

    sig = full.attrs["sig"]
    d = sig["dir"]
    n = len(d)

    # Сигналы есть, но не на каждой свече
    n_sig = int((d != 0).sum())
    frac = n_sig / n
    if n_sig == 0:
        return False, "сигналов нет ни одного"
    if frac > 0.5:
        return False, f"сигнал на {frac:.0%} свечей — похоже на ошибку условия"

    # Геометрия
    e, st, tk = sig["entry"], sig["stop"], sig["take"]
    L, S = d == 1, d == -1
    if np.any(st[L] >= e[L]) or np.any(tk[L] <= e[L]):
        return False, "у лонга стоп или тейк не с той стороны"
    if np.any(st[S] <= e[S]) or np.any(tk[S] >= e[S]):
        return False, "у шорта стоп или тейк не с той стороны"

    # Причинность: обрезаем историю сразу после проверяемой свечи
    rng = np.random.default_rng(seed)
    with_sig = [k for k in np.nonzero(d)[0] if k >= warm]
    without = [k for k in range(warm, n) if d[k] == 0]
    sample = list(with_sig)
    if without:
        sample += list(rng.choice(without, size=min(n_random, len(without)),
                                  replace=False))

    for k in sorted(sample):
        try:
            cut = fn(df.iloc[:k + 1].reset_index(drop=True), params)
        except Exception:
            return False, f"ИСКЛЮЧЕНИЕ при обрезке на свече {k}"
        cs = cut.attrs["sig"]
        if cs["dir"][k] != d[k]:
            return False, (f"ПОДГЛЯДЫВАНИЕ В БУДУЩЕЕ: на свече {k} сигнал "
                           f"{int(d[k])} по полной истории и {int(cs['dir'][k])} "
                           f"по истории, обрезанной на ней")
        if d[k] != 0:
            for key in ("stop", "take"):
                if not np.isclose(cs[key][k], sig[key][k], rtol=1e-9):
                    return False, (f"ПОДГЛЯДЫВАНИЕ В БУДУЩЕЕ: на свече {k} "
                                   f"уровень {key} зависит от будущих свечей")

    return True, (f"сигналов {n_sig} ({frac:.1%}), лонг {int(L.sum())} / "
                  f"шорт {int(S.sum())}, обрезок проверено {len(sample)}")


def main():
    ex = data.get_exchange("bitget")
    df = data.load("bitget", "BTC/USDT:USDT", "4h", "2023-06-01", "2026-09-01",
                   exchange=ex)
    if df is None:
        print("[!] Нет данных для проверки")
        sys.exit(2)

    print("=" * 70)
    print("  ПРОВЕРКА СТРАТЕГИЙ: не падает / есть сигналы / без подглядывания")
    print("=" * 70)
    failed = 0
    for name, fn in REGISTRY.items():
        params = DEFAULTS.get(name)
        if params is None:
            print(f"  ?  {name:<11} нет параметров по умолчанию — пропуск")
            failed += 1
            continue
        ok, msg = check(name, fn, df, params)
        print(f"  {'✔' if ok else '✘'}  {name:<11} {msg}")
        if not ok:
            failed += 1
    print("=" * 70)
    print(f"  Итог: {len(REGISTRY) - failed} из {len(REGISTRY)} прошли")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
