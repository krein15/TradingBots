"""
Backtest/validate.py
====================
Проверка движка против сырых свечей.

Бэктест легко заставить показать прибыль — достаточно одной ошибки
в модели исполнения, и результат становится художественным
вымыслом. Поэтому каждая сделка перепроверяется по исходным
данным: вход был возможен, выход случился ровно там, где цена
впервые задела уровень, и ни секундой раньше.

Запуск (после Backtest/run.py):
  python Backtest/validate.py
  python Backtest/validate.py --trades путь.csv --start ... --end ...
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from Backtest import data


def validate(trades, exchange_id, timeframe, start, end, fill_model="limit",
             verbose=True):
    """
    Возвращает список найденных проблем. Пустой список = движок
    отработал добросовестно.
    """
    ex = data.get_exchange(exchange_id)
    problems = []
    cache = {}

    for _, t in trades.iterrows():
        sym = t["symbol"]
        if sym not in cache:
            cache[sym] = data.load(exchange_id, sym, timeframe, start, end,
                                   exchange=ex)
        df = cache[sym]
        if df is None:
            problems.append((sym, "нет данных для проверки"))
            continue

        d = int(t["direction"])
        stop, take = float(t["stop"]), float(t["take"])
        entry_ts, exit_ts = int(t["entry_ts"]), int(t["exit_ts"])

        # ── 1. Вход: цена действительно доходила до уровня заявки ──
        bar = df[df.timestamp == entry_ts]
        if bar.empty:
            problems.append((sym, f"свечи входа {entry_ts} нет в данных"))
            continue
        bar = bar.iloc[0]
        entry = float(t["entry_price"])
        if fill_model == "limit":
            reached = (bar.low <= entry) if d == 1 else (bar.high >= entry)
            if not reached:
                problems.append((sym, f"вход по {entry} невозможен: "
                                      f"свеча {bar.low}..{bar.high}"))

        # ── 2. Выход: первое касание уровня совпадает с моментом выхода ──
        seg = df[(df.timestamp > entry_ts) & (df.timestamp <= exit_ts)]
        if seg.empty:
            # Вышли на той же свече, где вошли — проверяем её саму
            seg = df[df.timestamp == entry_ts]
        if d == 1:
            breach = seg[(seg.low <= stop) | (seg.high >= take)]
        else:
            breach = seg[(seg.high >= stop) | (seg.low <= take)]

        if breach.empty:
            problems.append((sym, f"выход в {exit_ts}, но уровень не задет"))
        elif int(breach.timestamp.iloc[0]) != exit_ts:
            first = pd.to_datetime(breach.timestamp.iloc[0], unit="ms", utc=True)
            problems.append((sym, f"вышли {t['exit_dt']}, а уровень задет "
                                  f"раньше — {first}"))

        # ── 3. Цена выхода не лучше уровня ──────────────────────
        exit_price = float(t["exit_price"])
        if t["result"] == "WIN" and abs(exit_price - take) > take * 1e-6:
            problems.append((sym, f"тейк {take}, а выход по {exit_price}"))
        if t["result"] == "LOSS":
            worse = exit_price <= stop if d == 1 else exit_price >= stop
            if not worse:
                problems.append((sym, f"стоп {stop}, а выход по {exit_price} — "
                                      f"лучше уровня, проскальзывание не учтено"))

    if verbose:
        print(f"  Проверено сделок: {len(trades)}")
        if problems:
            print(f"  [!] ПРОБЛЕМ: {len(problems)}")
            for sym, msg in problems[:20]:
                print(f"      {sym}: {msg}")
        else:
            print("  Расхождений нет — движок соответствует сырым данным.")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default=None)
    ap.add_argument("--exchange", default="bitget")
    ap.add_argument("--timeframe", default="5m")
    ap.add_argument("--start", default="2026-08-01")
    ap.add_argument("--end", default="2026-08-21")
    ap.add_argument("--fill", default="limit")
    args = ap.parse_args()

    path = args.trades or (data.ROOT / "Backtest" / "backtest_trades.csv")
    if not os.path.exists(path):
        print(f"[!] Нет файла со сделками: {path}")
        print("    Сначала запусти Backtest/run.py")
        return

    trades = pd.read_csv(path)
    print("=" * 64)
    print("  ПРОВЕРКА ДВИЖКА ПРОТИВ СЫРЫХ СВЕЧЕЙ")
    print("=" * 64)
    problems = validate(trades, args.exchange, args.timeframe,
                        args.start, args.end, args.fill)
    print("=" * 64)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
