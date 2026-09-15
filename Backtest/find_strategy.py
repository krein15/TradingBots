"""
Backtest/find_strategy.py
=========================
Поиск конфигурации с положительным матожиданием ПОСЛЕ комиссий.

Порядок честного поиска:
  обучение -> отбор лучших -> проверка на данных, которых отбор не видел.

Проверочный период не участвует в отборе ни на одном шаге. Если
конфигурация хороша только на обучении — значит найдена подгонка,
и это тоже результат, который надо показать, а не спрятать.

Запуск:
  python Backtest/find_strategy.py --tf 4h
  python Backtest/find_strategy.py --tf 1h --strategy donchian
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from Backtest import data, search

# Комиссия Bitget spot: 0.1% за сторону (тейкер)
COMMISSION = 0.001
SLIPPAGE = 0.0005

# Сколько сделок минимум, чтобы вообще рассматривать конфигурацию.
# Ниже этого t-статистика ничего не значит.
MIN_TRADES = 60


SPACES = {
    "donchian": {
        "channel":   [20, 30, 50, 80],
        "atr_mult":  [1.5, 2.5, 4.0],
        "rr":        [2.0, 3.0, 5.0],
        "ema":       [0, 200],
        "allow_short": [True, False],
        "atr_period": [14],
        "trail_atr": [None],
        "max_hold_bars": [200],
    },
    "meanrev": {
        "rsi_low":   [20, 30],
        "rsi_high":  [70, 80],
        "bb_std":    [2.0, 2.5],
        "atr_mult":  [1.5, 2.5, 4.0],
        "rr":        [1.0, 2.0, 3.0],
        "adx_max":   [0, 25],
        "allow_short": [True, False],
        "atr_period": [14],
        "max_hold_bars": [200],
    },
    "pullback": {
        "ema_fast":  [20, 50],
        "ema_slow":  [100, 200],
        "atr_mult":  [1.5, 2.5, 4.0],
        "rr":        [2.0, 3.0, 5.0],
        "adx_min":   [0, 20],
        "allow_short": [True, False],
        "atr_period": [14],
        "max_hold_bars": [200],
    },
}


def load_all(exchange_id, timeframe, start, end, limit, min_vol):
    ex = data.get_exchange(exchange_id)
    syms = data.top_symbols(exchange_id, limit=limit,
                            min_quote_vol=min_vol, exchange=ex)
    out = {}
    for s in syms:
        df = data.load(exchange_id, s, timeframe, start, end, exchange=ex)
        if df is not None and len(df) > 500:
            out[s] = df
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--exchange", default="bitget")
    ap.add_argument("--strategy", default=None,
                    help="donchian | meanrev | pullback (по умолчанию все)")
    ap.add_argument("--data-start", default="2023-06-01")
    ap.add_argument("--split", default="2025-06-01",
                    help="граница обучение/проверка")
    ap.add_argument("--data-end", default="2026-09-01")
    ap.add_argument("--symbols", type=int, default=30)
    ap.add_argument("--min-vol", type=float, default=5_000_000)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--max-open", type=int, default=5)
    args = ap.parse_args()

    print("=" * 70)
    print("  ПОИСК СТРАТЕГИИ С ПОЛОЖИТЕЛЬНЫМ КРАЕМ ПОСЛЕ КОМИССИЙ")
    print("=" * 70)
    print(f"  Таймфрейм: {args.tf}   биржа: {args.exchange}")
    print(f"  Обучение:  {args.data_start} .. {args.split}")
    print(f"  Проверка:  {args.split} .. {args.data_end}  (в отборе НЕ участвует)")
    print(f"  Комиссия:  {COMMISSION:.2%} за сторону, "
          f"проскальзывание {SLIPPAGE:.2%}")
    print()

    print("  Загрузка данных...")
    t0 = time.time()
    sd = load_all(args.exchange, args.tf, args.data_start, args.data_end,
                  args.symbols, args.min_vol)
    print(f"  Инструментов: {len(sd)}  за {time.time()-t0:.0f}с")
    if not sd:
        print("[!] Нет данных")
        return
    print()

    names = [args.strategy] if args.strategy else list(SPACES)
    train_period = (args.data_start, args.split)
    test_period = (args.split, args.data_end)

    # Инструмент должен присутствовать В ОБОИХ периодах. Иначе набор
    # на обучении и на проверке разный, и сравнивать их нельзя:
    # монета вроде ZEC с листингом в декабре 2025 попала бы только
    # в проверку и перекосила бы её.
    tr_part = search.slice_period(sd, *train_period)
    te_part = search.slice_period(sd, *test_period)
    both = set(tr_part) & set(te_part)
    dropped = sorted(set(sd) - both)
    sd = {k: v for k, v in sd.items() if k in both}
    if dropped:
        print(f"  Отброшено (нет истории в обоих периодах): {', '.join(dropped)}")
    print(f"  В работе инструментов: {len(sd)}")
    if not sd:
        print("[!] Не осталось инструментов")
        return
    print()

    merged = []

    for name in names:
        space = SPACES[name]
        n_combos = int(np.prod([len(v) for v in space.values()]))
        print(f"  ── {name}: {n_combos} конфигураций ──", flush=True)

        t0 = time.time()
        tr = search.evaluate(sd, name, space, train_period, COMMISSION,
                             SLIPPAGE, max_open=args.max_open, verbose=False)
        te = search.evaluate(sd, name, space, test_period, COMMISSION,
                             SLIPPAGE, max_open=args.max_open, verbose=False)
        print(f"     посчитано за {time.time()-t0:.0f}с", flush=True)
        if tr.empty or te.empty:
            print("     сделок нет" + '\n')
            continue

        keys = [k for k in space if k in tr.columns]
        m = tr.merge(te, on=keys, suffixes=("_train", "_test"))
        m["strategy"] = name
        m = m[(m.trades_train >= MIN_TRADES) & (m.trades_test >= MIN_TRADES // 2)]
        if m.empty:
            print("     ни одна конфигурация не набрала сделок" + '\n')
            continue
        merged.append((name, m, keys))

        pos_train = (m.mean_R_train > 0).mean() * 100
        pos_test = (m.mean_R_test > 0).mean() * 100
        corr = m[["t_stat_train", "mean_R_test"]].corr().iloc[0, 1]
        print(f"     конфигураций в зачёте: {len(m)}")
        print(f"     положительных на обучении: {pos_train:.0f}%   "
              f"на проверке: {pos_test:.0f}%")
        print(f"     связь обучение->проверка (корреляция t и mean_R): {corr:+.2f}")

        best = m.sort_values("t_stat_train", ascending=False).head(args.top)
        cols = keys + ["trades_train", "mean_R_train", "t_stat_train",
                       "trades_test", "wr_test", "mean_R_test",
                       "mean_R_gross_test", "fee_R_test", "t_stat_test",
                       "sum_R_test", "pf_test"]
        cols = [c for c in cols if c in best.columns]
        print('\n' + "     Отобрано по ОБУЧЕНИЮ, показано поведение на ПРОВЕРКЕ:")
        print("       " + best[cols].to_string(index=False).replace('\n', '\n       '))
        print()

    if not merged:
        print("Нечего сравнивать.")
        return

    print("=" * 70)
    print("  ИТОГ")
    print("=" * 70)

    total = sum(len(m) for _, m, _ in merged)
    print(f"  Перебрано конфигураций (в зачёте): {total}")
    print()

    verdicts = []
    for name, m, keys in merged:
        best = m.sort_values("t_stat_train", ascending=False).head(args.top)
        survived = best[best.mean_R_test > 0]
        med_test = m.mean_R_test.median()
        verdicts.append({
            "стратегия": name,
            "конфигураций": len(m),
            "медиана_mean_R_проверка": round(med_test, 4),
            "доля_плюсовых_проверка": f"{(m.mean_R_test > 0).mean()*100:.0f}%",
            "из_топ5_выжило": f"{len(survived)}/{len(best)}",
            "лучший_mean_R_проверка": round(best.mean_R_test.max(), 4),
        })
    print(pd.DataFrame(verdicts).to_string(index=False))
    print()

    everything = pd.concat([m.assign(strategy=n) for n, m, _ in merged],
                           ignore_index=True)
    out = data.ROOT / "Backtest" / f"search_{args.tf}.csv"
    everything.to_csv(out, index=False)
    print(f"  Полные результаты: {out}")
    print()
    print("  Как читать: если медиана по ВСЕМ конфигурациям на проверке")
    print("  положительна, край скорее всего настоящий — он не зависит от")
    print("  выбора параметров. Если положителен только отобранный топ,")
    print("  а медиана около нуля — это подгонка.")


if __name__ == "__main__":
    main()
