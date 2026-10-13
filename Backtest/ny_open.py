"""
Backtest/ny_open.py
===================
Пробой диапазона открытия Нью-Йорка с возвратом — проверка стратегии.

Правило, как его описывают:
  1. Дождаться открытия нью-йоркской сессии (9:30 по Нью-Йорку).
  2. Взять первые 5 минутных свечей — это и есть первая пятиминутка
     сессии. Её максимум и минимум образуют диапазон.
  3. Дождаться ПРОБОЯ одной из границ.
  4. Дождаться ПОВТОРНОГО ТЕСТИРОВАНИЯ пробитой границы — теперь она
     работает как поддержка (для лонга) или сопротивление (для шорта).
  5. Дождаться СИЛЬНОЙ СВЕЧИ в сторону пробоя: зелёной, закрывшейся
     вблизи своего максимума.
  6. Вход выше её максимума, стоп под её минимумом, цель 3 к 1.

── Чего эта проверка не делает ────────────────────────────────
Стратегию показывают на индексе (на видео — фьючерс или индекс США), а
у нас есть минутные данные только по крипто-фьючерсам Bitget. Поэтому
проверяется МЕХАНИКА правила на доступном рынке, а не исходное
утверждение про 14 лет на индексе. Это разные вещи, и выводы отсюда
переносить на индекс нельзя.

Разница существенная ещё и по издержкам: у нас круг стоит 0.174%
(измерено демо-стендом), а стоп здесь — размер одной минутной свечи,
обычно 0.1-0.3% от цены. То есть издержки съедают заметную долю риска,
и это надо смотреть в пересчёте на R, а не в процентах.

Нью-Йорк открывается в 13:30 UTC с марта по ноябрь (летнее время) и в
14:30 UTC зимой. Данные охватывают август-октябрь 2026 — всё летнее.

Запуск:
  python Backtest/ny_open.py
  python Backtest/ny_open.py --rr 2 --strong 0.6
"""

import argparse
import glob
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

CACHE = "Backtest/cache/bitget/1m"
FEE, SLIP = 0.0006, 0.0003        # тейкер Bitget + измеренное проскальзывание
NY_OPEN_UTC_MIN = 13 * 60 + 30    # 9:30 Нью-Йорка = 13:30 UTC летом

CFG = {
    "range_bars": 5,       # сколько минутных свечей образуют диапазон
    "wait_break": 60,      # сколько минут ждём пробоя
    "wait_retest": 30,     # сколько минут ждём возврата после пробоя
    "wait_strong": 15,     # сколько минут ждём сильную свечу после возврата
    "strong": 0.7,         # «закрылась вблизи максимума»: доля размаха свечи
    "rr": 3.0,             # цель к риску
    "max_hold": 240,       # минут на сделку, дальше выход по рынку
    "tf": 5,            # минут в рабочей свече: на видео график пятиминутный
    "min_risk": 0.1,    # минимальный стоп, % от цены
}


def to_tf(df, k):
    """Минутки -> рабочие свечи. Границы кратны часу, чтобы первая
    свеча сессии начиналась ровно в 13:30 UTC."""
    if k == 1:
        return df
    g = (df.timestamp // (k * 60_000)).values
    out = pd.DataFrame({
        "timestamp": df.timestamp.groupby(g).first(),
        "open": df.open.groupby(g).first(),
        "high": df.high.groupby(g).max(),
        "low": df.low.groupby(g).min(),
        "close": df.close.groupby(g).last()})
    return out.reset_index(drop=True)


def sessions(df):
    """Разбивка на дни по открытию Нью-Йорка."""
    t = pd.to_datetime(df.timestamp, unit="ms", utc=True)
    minute = t.dt.hour * 60 + t.dt.minute
    day = t.dt.date
    return day.values, minute.values


def run_symbol(df, cfg):
    """Одна сделка на день: первая, которая сложилась по правилам."""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    day, minute = sessions(df)
    out = []
    for d in pd.unique(day):
        sel = np.where((day == d) & (minute >= NY_OPEN_UTC_MIN))[0]
        if len(sel) < cfg["range_bars"] + 30:
            continue
        i0 = sel[0]
        if minute[i0] != NY_OPEN_UTC_MIN:      # нет свечи ровно на открытии
            continue
        rng = slice(i0, i0 + max(1, cfg["range_bars"] * 1 // cfg["tf"] or 1))
        hi, lo = h[rng].max(), l[rng].min()
        if hi <= lo:
            continue
        start = i0 + max(1, cfg["range_bars"] // cfg["tf"])
        end_day = sel[-1]

        # 1. Пробой: закрытие за границей
        brk = None
        for i in range(start, min(start + max(1, cfg["wait_break"] // cfg["tf"]), end_day)):
            if c[i] > hi:
                brk = (i, 1, hi); break
            if c[i] < lo:
                brk = (i, -1, lo); break
        if brk is None:
            continue
        bi, direction, level = brk

        # 2. Возврат к пробитой границе
        ret = None
        for i in range(bi + 1, min(bi + 1 + max(1, cfg["wait_retest"] // cfg["tf"]), end_day)):
            if (direction == 1 and l[i] <= level) or (direction == -1 and h[i] >= level):
                ret = i; break
        if ret is None:
            continue

        # 3. Сильная свеча в сторону пробоя: закрытие вблизи экстремума
        strong = None
        for i in range(ret, min(ret + max(1, cfg["wait_strong"] // cfg["tf"]), end_day)):
            span = h[i] - l[i]
            if span <= 0:
                continue
            pos = (c[i] - l[i]) / span            # 1 = закрылась на максимуме
            if direction == 1 and c[i] > o[i] and pos >= cfg["strong"]:
                strong = i; break
            if direction == -1 and c[i] < o[i] and (1 - pos) >= cfg["strong"]:
                strong = i; break
        if strong is None:
            continue

        # 4. Вход за экстремумом сильной свечи, стоп за противоположным
        trigger = h[strong] if direction == 1 else l[strong]
        stop_px = l[strong] if direction == 1 else h[strong]
        risk = abs(trigger - stop_px)
        if risk <= 0 or risk / trigger * 100 < cfg["min_risk"]:
            continue

        entry_i = None
        for i in range(strong + 1, min(strong + 11, end_day)):
            if (direction == 1 and h[i] >= trigger) or (direction == -1 and l[i] <= trigger):
                entry_i = i; break
        if entry_i is None:
            continue

        entry = trigger * (1 + SLIP * direction)
        stop = stop_px
        target = entry + cfg["rr"] * risk * direction

        exit_px, reason = None, None
        for i in range(entry_i, min(entry_i + max(1, cfg["max_hold"] // cfg["tf"]), end_day + 1)):
            hit_stop = (l[i] <= stop) if direction == 1 else (h[i] >= stop)
            hit_take = (h[i] >= target) if direction == 1 else (l[i] <= target)
            if hit_stop:                       # оба задеты — считаем стопом
                exit_px, reason = stop * (1 - SLIP * direction), "стоп"; break
            if hit_take:
                exit_px, reason = target, "цель"; break
        if exit_px is None:
            j = min(entry_i + max(1, cfg["max_hold"] // cfg["tf"]), end_day)
            exit_px, reason = c[j], "время"

        gross = (exit_px - entry) * direction
        fees = (entry + abs(exit_px)) * FEE
        r = (gross - fees) / risk
        out.append({"день": str(d), "сторона": direction, "причина": reason,
                    "R": r, "риск_%": risk / entry * 100,
                    "издержки_R": fees / risk})
    return out


def main():
    ap = argparse.ArgumentParser()
    for k, v in CFG.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args = vars(ap.parse_args())
    cfg = {k: args[k] for k in CFG}

    files = sorted(glob.glob(f"{CACHE}/*.csv"))
    print("=" * 74)
    print("  ПРОБОЙ ДИАПАЗОНА ОТКРЫТИЯ НЬЮ-ЙОРКА, возврат, сильная свеча")
    print(f"  диапазон {cfg['range_bars']} мин, цель {cfg['rr']}:1, "
          f"сильная свеча: закрытие в {cfg['strong']:.0%} размаха")
    print(f"  издержки: комиссия {FEE:.2%} + проскальзывание {SLIP:.2%} за сторону")
    print("=" * 74)

    rows = []
    for f in files:
        sym = os.path.basename(f).replace("-USDT_USDT.csv", "")
        df = pd.read_csv(f).sort_values("timestamp").reset_index(drop=True)
        tr = run_symbol(to_tf(df, cfg["tf"]), cfg)
        for t in tr:
            t["монета"] = sym
        rows += tr
        if tr:
            r = np.array([t["R"] for t in tr])
            print(f"  {sym:<6} сделок {len(tr):>4}  среднее {r.mean():>+7.3f}R  "
                  f"прибыльных {(r > 0).mean()*100:>5.1f}%")
    if not rows:
        print("  ни одной сделки по правилам не сложилось")
        return
    d = pd.DataFrame(rows)
    r = d.R.values
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
    print("-" * 74)
    print(f"  ВСЕГО {len(d)} сделок на {d['монета'].nunique()} монетах, "
          f"{d['день'].nunique()} торговых дней")
    print(f"  среднее {r.mean():+.3f}R   медиана {np.median(r):+.3f}R   t = {t:.2f}")
    print(f"  прибыльных {(r > 0).mean()*100:.1f}%   сумма {r.sum():+.1f}R")
    print(f"  средний риск на сделку: {d['риск_%'].median():.3f}% от цены")
    print(f"  издержки съедают {d['издержки_R'].median():.3f}R на сделку")
    print("\n  по причине выхода:")
    for reason, g in d.groupby("причина"):
        print(f"    {reason:<6} {len(g):>4} ({len(g)/len(d)*100:4.1f}%)  "
              f"среднее {g.R.mean():+.3f}R")
    print("\n  по стороне:")
    for side, g in d.groupby("сторона"):
        print(f"    {'лонг' if side == 1 else 'шорт':<6} {len(g):>4}  "
              f"среднее {g.R.mean():+.3f}R  прибыльных {(g.R > 0).mean()*100:.0f}%")
    return d


if __name__ == "__main__":
    main()
