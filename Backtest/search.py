"""
Backtest/search.py
==================
Перебор параметров с честной проверкой на невиданных данных.

Главная опасность такого перебора — подгонка. Если гонять сотни
конфигураций по одному куску истории и брать лучшую, она почти
наверняка окажется случайностью: при 300 попытках и пороге «лучшее
из» вы найдёте красивую кривую даже на случайных числах.

Поэтому здесь:
  1. История делится по времени на ОБУЧЕНИЕ и ПРОВЕРКУ. Проверочный
     кусок при отборе не участвует вообще.
  2. Конфигурации ранжируются по t-статистике края (среднее R,
     делённое на стандартную ошибку), а не по итоговой прибыли.
     Прибыль растёт вместе с числом сделок и любит переподгонку,
     t-статистика — нет.
  3. Отбирается несколько лучших, и только они прогоняются по
     проверочному куску. Итог засчитывается по нему.
  4. Печатается, сколько конфигураций было проверено — чтобы
     поправку на множественность можно было держать в голове.

Размер депозита намеренно большой, а риск малый: тогда лимит плеча
не срабатывает и результат не искажается тем, что счёт на $50 не
может открыть позицию нужного объёма. Качество стратегии меряется
в R, а не в долларах.
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Backtest import data, strategies
from Backtest.engine import Engine, FILL_MARKET

# Счёт заведомо больше нужного: лимит плеча не должен вмешиваться
BIG_DEPOSIT = 1_000_000.0
SMALL_RISK = 0.002


def r_metrics(engine):
    """Метрики в единицах риска — не зависят от размера счёта."""
    tr = engine.trades_df()
    if len(tr) == 0:
        return {"trades": 0, "mean_R": 0.0, "t_stat": 0.0, "sum_R": 0.0,
                "wr": 0.0, "pf": 0.0, "mean_R_gross": 0.0, "fee_R": 0.0,
                "max_dd": 0.0}

    risk_usd = (tr.entry_price - tr["stop"]).abs() * tr.qty
    risk_usd = risk_usd.replace(0, np.nan)
    r_net = (tr.pnl / risk_usd).dropna()
    r_gross = ((tr.pnl + tr.fees) / risk_usd).dropna()
    fee_r = (tr.fees / risk_usd).dropna()

    n = len(r_net)
    sd = r_net.std(ddof=1) if n > 1 else 0.0
    t = float(r_net.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else 0.0

    wins = tr[tr.result == "WIN"]
    losses = tr[tr.result == "LOSS"]
    gl = abs(losses.pnl.sum())

    eq = engine.equity_df()
    from Backtest.metrics import max_drawdown
    dd = max_drawdown(eq.balance.values) if len(eq) else 0.0

    return {
        "trades": n,
        "mean_R": round(float(r_net.mean()), 4),
        "mean_R_gross": round(float(r_gross.mean()), 4),
        "fee_R": round(float(fee_r.mean()), 4),
        "sum_R": round(float(r_net.sum()), 1),
        "t_stat": round(t, 2),
        "wr": round(len(wins) / n * 100, 1),
        "pf": round(wins.pnl.sum() / gl, 2) if gl > 0 else float("inf"),
        "max_dd": round(dd, 1),
    }


def prepare(symbols_data, strategy, params):
    """Считаем сигналы для всех инструментов один раз."""
    fn = strategies.REGISTRY[strategy]
    out = {}
    for sym, df in symbols_data.items():
        if df is None or len(df) < 200:
            continue
        try:
            out[sym] = fn(df, params)
        except Exception:
            continue
    return out


def slice_period(symbols_data, start, end):
    a = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    b = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    out = {}
    for sym, df in symbols_data.items():
        part = df[(df.timestamp >= a) & (df.timestamp < b)]
        if len(part) > 200:
            out[sym] = part.reset_index(drop=True)
    return out


def run_one(prepared, commission, slippage, max_open, cooldown_min,
            trail_atr=None, max_hold_bars=None, deposit=BIG_DEPOSIT,
            risk=SMALL_RISK):
    eng = Engine(
        cfg={}, signal_fn=strategies.signal_fn,
        deposit=deposit, risk_pct=risk, max_open=max_open,
        commission=commission, slippage=slippage,
        fill_model=FILL_MARKET,          # пробой берём рынком по следующему открытию
        max_wait_bars=1,                 # заявка живёт один бар
        cooldown_minutes=cooldown_min,
        bad_hours=(), max_leverage=1000.0,
        trail_atr=trail_atr, max_hold_bars=max_hold_bars,
    ).run(prepared)
    return eng


def grid(space):
    """Декартово произведение словаря списков -> список словарей."""
    keys = list(space)
    return [dict(zip(keys, combo))
            for combo in itertools.product(*(space[k] for k in keys))]


def evaluate(symbols_data, strategy, space, period, commission, slippage,
             max_open=5, cooldown_min=0, verbose=True, label=""):
    """Прогоняем всю сетку по одному периоду."""
    part = slice_period(symbols_data, *period)
    combos = grid(space)
    rows = []
    for k, params in enumerate(combos, 1):
        prepared = prepare(part, strategy, params)
        if not prepared:
            continue
        eng = run_one(prepared, commission, slippage, max_open, cooldown_min,
                      trail_atr=params.get("trail_atr"),
                      max_hold_bars=params.get("max_hold_bars"))
        m = r_metrics(eng)
        m.update({"strategy": strategy, **params})
        rows.append(m)
        if verbose and k % 10 == 0:
            print(f"    {label} {k}/{len(combos)}", flush=True)
    return pd.DataFrame(rows)
