"""
Backtest/tape_flow.py
=====================
Поиск «робота» в ленте сделок и проверка, можно ли на нём заработать.

Идея, как её описывают скальперы: смотришь в ленту и видишь, что
кто-то покупает по рынку одинаковыми кусками — ровно по $10 000, снова
и снова. Это робот набирает позицию и толкает цену. Встаёшь вместе с
ним, а когда он перестаёт покупать — выходишь.

Проверяем ровно это, тремя шагами.

1. ВОССТАНОВЛЕНИЕ ЗАЯВОК. Лента отдаёт исполнения, а не заявки. Одна
   рыночная покупка сносит несколько уровней стакана и появляется как
   десяток сделок по разным ценам в одну миллисекунду. Поэтому сначала
   склеиваем исполнения по (время, сторона) — получаем заявки. На DOGE
   19 000 исполнений дают 8 246 заявок, в крупных бывает до 49 кусков.

2. ПОИСК ПОВТОРА. Робот — это несколько заявок подряд одной стороны,
   близких по размеру, в коротком окне. Все три порога (сколько
   повторов, какой разброс размера, какое окно) задаются, потому что
   угадывать их бессмысленно — надо смотреть, как меняется ответ.

3. ЧЕСТНАЯ ОЦЕНКА. Две ловушки, которые надо обойти:
     сравнение с фоном. Если цена и так росла весь период, любой
       «сигнал на покупку» покажет плюс. Поэтому рядом всегда стоит
       средний ход за то же время из случайного момента;
     издержки. Вход за роботом — рыночный, то есть тейкер. Круг стоит
       0.174% (комиссия 0.06% на сторону плюс измеренное на демо-счёте
       проскальзывание 0.027%). Это вычитается, а не упоминается.

Запуск:
  python Backtest/tape_flow.py
  python Backtest/tape_flow.py --min-usd 20000 --repeats 4
"""

import argparse
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

from Backtest import tape  # noqa: E402

COST = 0.174          # круг тейкером, % — комиссия + измеренное проскальзывание
# Секунды после входа. Короткие горизонты обязательны: в описании
# стратегии сделка живёт секунды, а не минуты.
HORIZONS = (5, 15, 30, 60, 300, 900)


def orders(df):
    """Исполнения -> заявки. Склейка по (время, сторона)."""
    d = df.copy()
    d["usd"] = d["price"] * d["size"]
    g = d.groupby(["ts", "side"], sort=True).agg(
        usd=("usd", "sum"), price=("price", "last"), fills=("usd", "size"))
    g = g.reset_index().sort_values("ts").reset_index(drop=True)
    return g


def find_robots(o, min_usd, repeats, window_s, tol):
    """
    Моменты, когда подряд идут repeats заявок одной стороны, каждая
    крупнее min_usd, близких по размеру (разброс не больше tol), и все
    укладываются в окно window_s секунд.

    Возвращает индексы ПОСЛЕДНЕЙ заявки серии — момент, когда скальпер
    уже видит закономерность и может входить. Раньше он её не видит,
    поэтому брать начало серии было бы подглядыванием.
    """
    big = o[o.usd >= min_usd].reset_index()
    out = []
    for i in range(repeats - 1, len(big)):
        w = big.iloc[i - repeats + 1: i + 1]
        if w.side.nunique() != 1:
            continue
        if w.ts.iloc[-1] - w.ts.iloc[0] > window_s * 1000:
            continue
        lo, hi = w.usd.min(), w.usd.max()
        if hi / lo - 1 > tol:
            continue
        out.append((int(big["index"].iloc[i]), int(w.side.iloc[0])))
    return out


def forward(o, idx, horizons):
    """Ход цены после момента idx, в процентах, по каждому горизонту."""
    ts, px = o.ts.values, o.price.values
    t0, p0 = ts[idx], px[idx]
    res = []
    for h in horizons:
        j = np.searchsorted(ts, t0 + h * 1000)
        res.append((px[min(j, len(px) - 1)] / p0 - 1) * 100 if j < len(px) else np.nan)
    return res


def baseline(o, n, horizons, obs_sides, seed=7):
    """
    Тот же замер из случайных моментов — фон, с которым сравниваем.

    Сторона берётся из НАБЛЮДЁННОГО распределения серий: сигнал
    считается в сторону робота, и сравнивать его с ненаправленным
    дрейфом рынка нельзя — при росте рынка любая покупка в плюсе.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(o) - 1, size=min(n * 5, 20000))
    rows = [forward(o, int(j), horizons) for j in idx]
    sides = rng.choice(obs_sides, size=len(idx))
    return np.nanmean(np.array(rows) * sides[:, None], axis=0)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-usd", type=float, default=10000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--window", type=float, default=120, help="секунд на серию")
    ap.add_argument("--tol", type=float, default=0.25, help="допустимый разброс размера")
    args = ap.parse_args()

    coins = tape.available()
    if not coins:
        print("ленты нет — сначала: python Backtest/tape.py DOGE SUI --days 7")
        return
    print("=" * 84)
    print(f"  РОБОТ В ЛЕНТЕ: {args.repeats} заявки подряд крупнее ${args.min_usd:,.0f}, "
          f"разброс до {args.tol:.0%}, окно {args.window:.0f}с")
    print("=" * 84)

    all_sig, all_base, total = [], [], 0
    for c in coins:
        df = tape.read(c)
        if df is None or len(df) < 10000:
            continue
        o = orders(df)
        sig = find_robots(o, args.min_usd, args.repeats, args.window, args.tol)
        days = (o.ts.iloc[-1] - o.ts.iloc[0]) / 86400_000
        if len(sig) < 20:
            print(f"  {c:<6} заявок {len(o):>7,}  за {days:>4.1f} сут  "
                  f"серий {len(sig):>4} — мало для выводов")
            continue
        rows = np.array([forward(o, i, HORIZONS) for i, _ in sig])
        side = np.array([s for _, s in sig])
        # доход, если встать В СТОРОНУ робота
        prof = rows * side[:, None]
        base = baseline(o, len(sig), HORIZONS, side)
        all_sig.append((c, prof, base, len(o), days))
        total += len(sig)
        print(f"  {c:<6} заявок {len(o):>7,}  за {days:>4.1f} сут  серий {len(sig):>4}  "
              f"(в покупку {int((side == 1).sum())}, в продажу {int((side == -1).sum())})")

    if not all_sig:
        print("\n  ни по одной монете не набралось серий")
        return

    print(f"\n{'монета':<8}" + "".join(f"{h//60 if h>=60 else h:>9}м" for h in HORIZONS)
          + "   <- ход в сторону робота, %")
    print("-" * 84)
    for c, prof, base, _, _ in all_sig:
        m = np.nanmean(prof, axis=0)
        print(f"{c:<8}" + "".join(f"{v:>+10.4f}" for v in m))
        print(f"{'  фон':<8}" + "".join(f"{v:>+10.4f}" for v in base)
              + "   (случайный момент)")

    pooled = np.vstack([p for _, p, _, _, _ in all_sig])
    m = np.nanmean(pooled, axis=0)
    sd = np.nanstd(pooled, axis=0, ddof=1)
    t = m / (sd / np.sqrt(len(pooled)))
    print("-" * 84)
    print(f"{'ВСЕ':<8}" + "".join(f"{v:>+10.4f}" for v in m) + f"   n = {len(pooled)}")
    print(f"{'  t':<8}" + "".join(f"{v:>10.2f}" for v in t))
    print(f"{'  минус':<8}" + "".join(f"{v - COST:>+10.4f}" for v in m)
          + f"   после издержек {COST}%")
    best = np.nanmax(m - COST)
    print("\n" + "=" * 84)
    if best > 0:
        h = HORIZONS[int(np.nanargmax(m - COST))]
        print(f"  Лучший горизонт: {h}с, {best:+.4f}% после издержек. "
              f"Проверять дальше: разбить на обучение и проверку.")
    else:
        print(f"  Ни на одном горизонте ход не покрывает издержки в {COST}%.")
    print("=" * 84)


if __name__ == "__main__":
    main()
