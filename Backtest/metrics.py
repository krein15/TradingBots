"""
Backtest/metrics.py
===================
Статистика по результатам прогона.

Главное отличие от того, что печатал бумажный бот: просадка
считается по непрерывной эквити-кривой. Раньше баланс молча
возвращался к стартовому при сливе, из-за чего кривая рвалась,
и максимальная просадка выходила заниженной.

Отдельно считается фактический RR: планировали 1:3, а реально
получается другое — из-за комиссий, проскальзывания и того, что
часть выходов происходит не ровно по уровню.
"""

import numpy as np
import pandas as pd


def max_drawdown(equity):
    """Максимальная просадка в процентах по кривой баланса."""
    if len(equity) == 0:
        return 0.0
    eq = np.asarray(equity, dtype="float64")
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak == 0, 1, peak) * 100
    return float(dd.min())


def summarize(engine, label=""):
    """Сводка одного прогона -> dict."""
    trades = engine.trades_df()
    eq = engine.equity_df()

    out = {
        "label": label,
        "trades": len(trades),
        "deposit": engine.deposit,
        "final_balance": round(engine.balance, 2),
        "pnl": round(engine.balance - engine.deposit, 2),
        "pnl_pct": round((engine.balance / engine.deposit - 1) * 100, 2),
        "broke": engine.broke_at is not None,
    }

    if len(trades) == 0:
        out.update({"wr": 0.0, "max_dd": 0.0, "profit_factor": 0.0,
                    "expectancy": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                    "rr_real": 0.0, "fees": 0.0})
        return out

    wins = trades[trades.result == "WIN"]
    losses = trades[trades.result == "LOSS"]
    gross_win = wins.pnl.sum()
    gross_loss = abs(losses.pnl.sum())

    avg_win = wins.pnl.mean() if len(wins) else 0.0
    avg_loss = losses.pnl.mean() if len(losses) else 0.0

    out.update({
        "wr": round(len(wins) / len(trades) * 100, 1),
        "wins": len(wins),
        "losses": len(losses),
        "max_dd": round(max_drawdown(eq.balance.values), 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "expectancy": round(trades.pnl.mean(), 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        # Фактический RR: во сколько раз средний выигрыш больше
        # среднего проигрыша. Плановый RR почти всегда выше.
        "rr_real": round(abs(avg_win / avg_loss), 2) if avg_loss else 0.0,
        "fees": round(trades.fees.sum(), 2),
        "capped_pct": round(trades.capped.mean() * 100, 1),
        "avg_bars_held": round(trades.bars_held.mean(), 1),
    })

    # Точка безубытка: какой WR нужен при таком фактическом RR
    rr = out["rr_real"]
    out["breakeven_wr"] = round(100 / (1 + rr), 1) if rr > 0 else 100.0
    out["edge"] = round(out["wr"] - out["breakeven_wr"], 1)
    return out


def breakdown(trades, by):
    """Разрез статистики по колонке (тип сигнала, режим, направление)."""
    if len(trades) == 0 or by not in trades.columns:
        return pd.DataFrame()
    g = trades.groupby(by).agg(
        сделок=("pnl", "size"),
        побед=("result", lambda s: (s == "WIN").sum()),
        pnl=("pnl", "sum"),
    )
    g["WR%"] = (g["побед"] / g["сделок"] * 100).round(1)
    g["pnl"] = g["pnl"].round(2)
    return g.sort_values("сделок", ascending=False)


def print_report(engine, label="", show_breakdowns=True):
    s = summarize(engine, label)
    trades = engine.trades_df()

    w = 64
    print("=" * w)
    print(f"  {label}")
    print("=" * w)
    print(f"  Депозит:          ${s['deposit']:.2f}")
    print(f"  Итоговый баланс:  ${s['final_balance']:.2f}   "
          f"({s['pnl']:+.2f}$ / {s['pnl_pct']:+.1f}%)")
    if s["broke"]:
        print("  [!] ДЕПОЗИТ СЛИТ — торговля остановлена до конца периода")

    if s["trades"] == 0:
        print("\n  Сделок нет.")
        print("=" * w)
        return s

    print(f"\n  Сделок:           {s['trades']}  "
          f"({s['wins']} побед / {s['losses']} потерь)")
    print(f"  WR:               {s['wr']}%")
    print(f"  Точка безубытка:  {s['breakeven_wr']}%  "
          f"(при фактическом RR {s['rr_real']})")
    edge = s["edge"]
    verdict = "преимущество есть" if edge > 0 else "преимущества нет"
    print(f"  Запас:            {edge:+.1f}%  -> {verdict}")
    print(f"\n  Средний выигрыш:  ${s['avg_win']:+.4f}")
    print(f"  Средний проигрыш: ${s['avg_loss']:+.4f}")
    print(f"  Матожидание:      ${s['expectancy']:+.4f} на сделку")
    print(f"  Profit factor:    {s['profit_factor']}")
    print(f"  Макс. просадка:   {s['max_dd']}%")
    print(f"  Комиссии всего:   ${s['fees']}")
    print(f"  Урезано плечом:   {s['capped_pct']}% сделок")
    print(f"  Держим в среднем: {s['avg_bars_held']} свечей")

    c = engine.counters
    print(f"\n  Сигналов: {c['signals']}  заявок: {c['orders']}  "
          f"исполнено: {c['filled']}  истекло: {c['expired']}")
    fill_rate = c["filled"] / c["orders"] * 100 if c["orders"] else 0
    print(f"  Доля исполнения заявок: {fill_rate:.1f}%")
    print(f"  Пропущено: кулдаун={c['skip_cooldown']} "
          f"плохой час={c['skip_bad_hour']} лимит позиций={c['skip_max_open']} "
          f"дубль={c['skip_duplicate']}")

    if show_breakdowns:
        for col, title in (("signal_type", "По типу сигнала"),
                           ("regime", "По режиму рынка"),
                           ("direction", "По направлению")):
            b = breakdown(trades, col)
            if len(b):
                print(f"\n  {title}:")
                print("    " + b.to_string().replace("\n", "\n    "))

    print("=" * w)
    return s


def compare(summaries):
    """Таблица сравнения нескольких прогонов."""
    df = pd.DataFrame(summaries)
    cols = ["label", "trades", "wr", "breakeven_wr", "edge",
            "pnl", "pnl_pct", "max_dd", "profit_factor", "rr_real"]
    cols = [c for c in cols if c in df.columns]
    return df[cols]
