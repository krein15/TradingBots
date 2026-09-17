"""
Backtest/xsmom.py
=================
Кросс-секционный моментум — портфельная стратегия.

Чем принципиально отличается от всего остального в проекте:
остальные стратегии ловят событие на ОДНОЙ монете (пробой, пересечение)
и держат позицию до стопа или тейка. Здесь сигнала на отдельной монете
нет вообще. Раз в `rebalance` свечей все монеты ранжируются по
доходности за `lookback` свечей, и портфель перестраивается:

  long_short — лонг top_k сильнейших, шорт top_k слабейших,
               вложения поровну, чистая позиция по рынку около нуля;
  long_only  — только лонг top_k сильнейших.

Ставка не на направление рынка, а на то, что сильные монеты
относительно продолжат обгонять слабые. В исследованиях по крипте это
один из самых устойчиво воспроизводимых эффектов.

Строгость расчёта:
  ранжирование по закрытию свечи t — известно к её концу;
  вход по ОТКРЫТИЮ свечи t+1, выход по открытию свечи
  t+1+rebalance — никакого исполнения по цене, по которой считали сигнал;
  комиссия и проскальзывание списываются с ОБОРОТА: перестроили
  портфель сильнее — заплатили больше;
  монета без данных в момент t в ранжировании не участвует, поэтому
  листинги в середине периода обрабатываются честно.

Не смоделировано: funding. Для лонг-шорт портфеля он в среднем
взаимно гасится (лонги платят, шорты получают), для long_only это
реальная статья расходов — отмечено в отчёте.
"""

import numpy as np
import pandas as pd

TF_HOURS = {"15m": 0.25, "1h": 1, "4h": 4, "1d": 24}


def build_panel(symbols_data, field):
    """Широкая таблица: строки — время, столбцы — монеты."""
    cols = {}
    for sym, df in symbols_data.items():
        s = df.set_index("timestamp")[field]
        cols[sym] = s[~s.index.duplicated(keep="last")]
    return pd.DataFrame(cols).sort_index()


def run(opens, closes, lookback, rebalance, top_k, mode="long_short",
        skip=0, fee=0.0006, slippage=0.0005, min_coins=None, reverse=False):
    """
    Возвращает DataFrame по периодам: ts, gross, cost, net, turnover, n.

    skip — сколько последних свечей не учитывать в доходности. На
    дневках классика пропускает последний месяц из-за краткосрочного
    разворота; здесь это параметр перебора.

    reverse — лонг слабых и шорт сильных. Контрольный прогон: если
    моментум и его разворот прибыльны одновременно, расчёт врёт.
    """
    O = opens.to_numpy(dtype="float64")
    C = closes.to_numpy(dtype="float64")
    idx = opens.index.to_numpy()
    n_bars, n_sym = O.shape
    min_coins = min_coins or 2 * top_k

    w_prev = np.zeros(n_sym)
    rows = []
    start = lookback + skip
    for t in range(start, n_bars - rebalance - 1, rebalance):
        # Сигнал — только прошлое, по закрытию свечи t
        now = C[t - skip]
        past = C[t - skip - lookback]
        mom = now / past - 1.0

        # Исполнение — будущее, по открытиям
        entry = O[t + 1]
        exit_ = O[t + 1 + rebalance]
        ok = (np.isfinite(mom) & np.isfinite(entry) & np.isfinite(exit_)
              & (entry > 0) & (past > 0))
        if ok.sum() < min_coins:
            w_prev = np.zeros(n_sym)
            continue

        cand = np.nonzero(ok)[0]
        order = cand[np.argsort(mom[cand])]          # по возрастанию
        if reverse:
            order = order[::-1]
        w = np.zeros(n_sym)
        if mode == "long_short":
            w[order[-top_k:]] = 0.5 / top_k
            w[order[:top_k]] = -0.5 / top_k
        else:
            w[order[-top_k:]] = 1.0 / top_k

        ret = np.where(ok, exit_ / entry - 1.0, 0.0)
        gross = float(np.sum(w * ret))
        turnover = float(np.sum(np.abs(w - w_prev)))
        cost = turnover * (fee + slippage)
        rows.append((idx[t + 1], gross, cost, gross - cost, turnover, int(ok.sum())))

        # Вес к концу периода смещается вместе с ценами — учитываем
        # дрейф, иначе оборот при перестройке был бы занижен
        drift = w * (1.0 + ret)
        gross_exposure = np.sum(np.abs(drift))
        w_prev = drift / gross_exposure * np.sum(np.abs(w)) if gross_exposure > 0 else drift

    return pd.DataFrame(rows, columns=["ts", "gross", "cost", "net", "turnover", "n"])


def metrics(res, timeframe, rebalance):
    """Годовая доходность, Шарп, просадка — по чистому результату."""
    if res is None or len(res) < 4:
        return {"periods": 0, "ann_ret": 0.0, "sharpe": 0.0, "max_dd": 0.0,
                "t_stat": 0.0, "hit": 0.0, "cost_share": 0.0,
                "ann_gross": 0.0, "turnover": 0.0}
    per_year = 365 * 24 / (TF_HOURS[timeframe] * rebalance)
    r = res["net"].to_numpy()
    g = res["gross"].to_numpy()
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min() * 100)
    sd = r.std(ddof=1)
    years = len(r) / per_year
    ann = float(eq[-1] ** (1 / years) - 1) * 100 if years > 0 and eq[-1] > 0 else -100.0
    ann_g = float(np.prod(1 + g) ** (1 / years) - 1) * 100 if years > 0 else 0.0
    gross_sum = np.abs(g).sum()
    return {
        "periods": len(r),
        "ann_ret": round(ann, 1),
        "ann_gross": round(ann_g, 1),
        "sharpe": round(float(r.mean() / sd * np.sqrt(per_year)), 2) if sd > 0 else 0.0,
        "t_stat": round(float(r.mean() / (sd / np.sqrt(len(r)))), 2) if sd > 0 else 0.0,
        "max_dd": round(dd, 1),
        "hit": round(float((r > 0).mean() * 100), 1),
        "turnover": round(float(res["turnover"].mean()), 2),
        "cost_share": round(float(res["cost"].sum() / gross_sum * 100), 1) if gross_sum > 0 else 0.0,
    }
