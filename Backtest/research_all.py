"""
Backtest/research_all.py
========================
Широкий поиск: все семейства стратегий одной методикой.

  12 семейств по отдельным монетам (strategies + strategies_more)
  + кросс-секционный моментум (xsmom), портфельный

Для каждого семейства вся сетка считается на обучении и на проверке,
отбор — только по обучению. Главный показатель — не лучшая настройка,
а МЕДИАНА по всем настройкам на проверке и доля прибыльных: если край
есть только у отобранного топа, а медиана около нуля, это подгонка.

Перед тем как что-либо сюда добавлять, стратегия обязана пройти
Backtest/test_strategies.py — иначе подглядывание в будущее будет
выглядеть как найденный край.

Запуск:
  python Backtest/research_all.py
  python Backtest/research_all.py --families donchian,tsmom,xsmom
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

from Backtest import data, search, xsmom

FEE, SLIP = 0.0006, 0.0005          # Bitget фьючерсы, тейкер

FROZEN = ["BTC", "ETH", "XRP", "SOL", "ZEC", "LSK", "SUI", "DOGE", "ARB", "PEPE",
          "ENA", "UNI", "ADA", "XLM", "FIL", "LINK", "NEAR", "VTHO", "TRUMP",
          "ONDO", "TAO", "BNB", "WLD", "DOT", "INJ", "PUMP", "APT", "AVAX",
          "BCH", "AAVE", "LTC", "ASTR", "FET", "ETHFI", "PENGU"]

# Общие варианты выхода для трендовых стратегий: фиксированный тейк
# против трейлинга без тейка — второе и есть «дать тренду ехать»
TREND_EXIT = {"atr_mult": [2.0, 3.0], "rr": [3.0, 0], "trail_atr": [None, 3.0],
              "allow_short": [True], "atr_period": [14], "max_hold_bars": [200]}
MR_EXIT = {"atr_mult": [1.5, 3.0], "rr": [1.0, 2.0], "trail_atr": [None],
           "allow_short": [True], "atr_period": [14], "max_hold_bars": [60]}

SPACES = {
    "donchian":   {"channel": [20, 55], "ema": [0, 200], **TREND_EXIT},
    "ma_cross":   {"fast": [10, 20], "slow": [50, 100], "ema": [0], **TREND_EXIT},
    "tsmom":      {"lookback": [30, 90], "threshold": [0.0, 0.1], "ema": [0], **TREND_EXIT},
    "keltner":    {"period": [20, 50], "k": [1.5, 2.5], "ema": [0], **TREND_EXIT},
    "supertrend": {"mult": [2.0, 3.0], "ema": [0, 200], **TREND_EXIT},
    "macd":       {"fast": [12], "slow": [26], "signal": [9], "ema": [0, 200], **TREND_EXIT},
    "rsi_momo":   {"level": [10, 20], "ema": [0, 200], **TREND_EXIT},
    "bb_squeeze": {"lookback": [100, 300], "q": [0.1, 0.2], "within": [5], "ema": [0], **TREND_EXIT},
    "nr7":        {"n": [7, 14], "ema": [0, 200], **TREND_EXIT},
    "pullback":   {"ema_fast": [20, 50], "ema_slow": [200], "adx_min": [0, 20], **TREND_EXIT},
    "meanrev":    {"rsi_low": [25], "rsi_high": [75], "bb_std": [2.0, 2.5],
                   "adx_max": [0, 25], **MR_EXIT},
    "zscore":     {"period": [20, 50], "thr": [2.0, 3.0], "ema": [0, 200], **MR_EXIT},
}

XSMOM_SPACE = {
    "lookback":  [6, 42, 180],     # сутки, неделя, месяц (в 4ч свечах)
    "rebalance": [6, 18, 42],      # раз в сутки, в трое суток, в неделю
    "top_k":     [3, 5, 8],
    "mode":      ["long_short", "long_only"],
    "skip":      [0, 6],
}


def load(tf, d0, d1):
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    out = {}
    for b in FROZEN:
        df = data.load("bitget", f"{b}/USDT:USDT", tf, d0, d1, exchange=ex)
        if df is not None and len(df) > 500:
            out[f"{b}/USDT:USDT"] = df
    return out


def single_family(sd, name, space, train, test, min_tr=60, min_te=30):
    a = search.evaluate(sd, name, space, train, FEE, SLIP, max_open=5, verbose=False)
    b = search.evaluate(sd, name, space, test, FEE, SLIP, max_open=5, verbose=False)
    if a.empty or b.empty:
        return None
    keys = [k for k in space if k in a.columns]
    # trail_atr=None ломает merge по NaN — сравниваем строковым видом.
    # str() по каждому значению явно: в pandas 3 astype(str) оставляет
    # пропуски как NaN, и склейка падает.
    for df in (a, b):
        df["_key"] = df[keys].apply(lambda r: "|".join(str(v) for v in r), axis=1)
    m = a.merge(b, on="_key", suffixes=("_tr", "_te"))
    m = m[(m.trades_tr >= min_tr) & (m.trades_te >= min_te)]
    if m.empty:
        return None
    best = m.sort_values("t_stat_tr", ascending=False).head(3)
    return {
        "семейство": name, "тип": "монета",
        "настроек": len(m),
        "медиана_проверка": round(float(m.mean_R_te.median()), 3),
        "плюсовых": round(float((m.mean_R_te > 0).mean() * 100)),
        "медиана_обучение": round(float(m.mean_R_tr.median()), 3),
        "топ3_проверка": ", ".join(f"{x:+.3f}" for x in best.mean_R_te),
        "единица": "R/сделку",
    }, m


def xsmom_family(sd, train, test):
    O = xsmom.build_panel(sd, "open")
    C = xsmom.build_panel(sd, "close")
    t_split = int(pd.Timestamp(test[0], tz="UTC").timestamp() * 1000)
    t_start = int(pd.Timestamp(train[0], tz="UTC").timestamp() * 1000)
    rows = []
    for params in search.grid(XSMOM_SPACE):
        res = xsmom.run(O, C, fee=FEE, slippage=SLIP, **params)
        if res.empty:
            continue
        # Периоды относим к обучению или проверке по моменту ВХОДА.
        # Разгон берётся из более ранней истории — сигнал причинный,
        # утечки через это нет.
        tr = res[(res.ts >= t_start) & (res.ts < t_split)]
        te = res[res.ts >= t_split]
        mt = xsmom.metrics(tr, "4h", params["rebalance"])
        me = xsmom.metrics(te, "4h", params["rebalance"])
        rows.append({**params,
                     **{f"{k}_tr": v for k, v in mt.items()},
                     **{f"{k}_te": v for k, v in me.items()}})
    m = pd.DataFrame(rows)
    m = m[(m.periods_tr >= 20) & (m.periods_te >= 10)]
    if m.empty:
        return None
    best = m.sort_values("sharpe_tr", ascending=False).head(3)
    return {
        "семейство": "xsmom", "тип": "портфель",
        "настроек": len(m),
        "медиана_проверка": round(float(m.ann_ret_te.median()), 1),
        "плюсовых": round(float((m.ann_ret_te > 0).mean() * 100)),
        "медиана_обучение": round(float(m.ann_ret_tr.median()), 1),
        "топ3_проверка": ", ".join(f"{x:+.1f}%" for x in best.ann_ret_te),
        "единица": "%/год",
    }, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=None)
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--data-start", default="2023-06-01")
    ap.add_argument("--split", default="2025-06-01")
    ap.add_argument("--data-end", default="2026-09-01")
    args = ap.parse_args()

    train = (args.data_start, args.split)
    test = (args.split, args.data_end)
    wanted = args.families.split(",") if args.families else list(SPACES) + ["xsmom"]

    print("=" * 74)
    print("  ШИРОКИЙ ПОИСК СТРАТЕГИЙ")
    print("=" * 74)
    print(f"  ТФ {args.tf}   обучение {train[0]}..{train[1]}   "
          f"проверка {test[0]}..{test[1]}")
    print(f"  Комиссия {FEE:.2%} + проскальзывание {SLIP:.2%} за сторону")
    t0 = time.time()
    sd = load(args.tf, args.data_start, args.data_end)
    a = search.slice_period(sd, *train)
    b = search.slice_period(sd, *test)
    sd = {k: v for k, v in sd.items() if k in set(a) & set(b)}
    print(f"  Инструментов с историей в обоих периодах: {len(sd)}  "
          f"({time.time() - t0:.0f}с)\n", flush=True)

    summary, details = [], {}
    total_configs = 0
    for name in wanted:
        t1 = time.time()
        if name == "xsmom":
            out = xsmom_family(sd, train, test)
        else:
            out = single_family(sd, name, SPACES[name], train, test)
        if out is None:
            print(f"  {name:<11} недостаточно сделок", flush=True)
            continue
        row, full = out
        summary.append(row)
        details[name] = full
        total_configs += row["настроек"]
        print(f"  {name:<11} настроек {row['настроек']:>3}  "
              f"медиана на проверке {row['медиана_проверка']:>+8} {row['единица']:<9} "
              f"плюсовых {row['плюсовых']:>3}%   [{time.time() - t1:.0f}с]", flush=True)

    print()
    print("=" * 74)
    print(f"  СВОДКА   (всего настроек в зачёте: {total_configs})")
    print("=" * 74)
    df = pd.DataFrame(summary).sort_values(["тип", "плюсовых"], ascending=[False, False])
    print(df.to_string(index=False))

    out_dir = data.ROOT / "Backtest"
    df.to_csv(out_dir / f"research_summary_{args.tf}.csv", index=False)
    for name, full in details.items():
        full.to_csv(out_dir / f"research_{name}_{args.tf}.csv", index=False)
    print(f"\n  Сохранено: Backtest/research_summary_{args.tf}.csv и по семействам")


if __name__ == "__main__":
    main()
