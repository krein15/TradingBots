"""
Backtest/spike_ladder.py
========================
Лимитная лестница против прострелов: честная проверка скальперской идеи.

Идея (как её описывают скальперы). Прострел — это когда кто-то сносит
стакан рыночной заявкой: цена на секунды выстреливает и возвращается.
Торгуют это не вдогонку, а заранее: ставят лимитки на продажу выше
цены — на +1, +2, +3, +4 и +5 процентов, — и вынос сам их собирает.
Потом цена падает обратно, позиция закрывается в плюс.

Почему это НЕ то же самое, что «войти после прострела». Вход лимиткой
даёт две вещи, которых нет у рыночного входа:
  цену пика, а не цену после отката — лимитка исполняется в вершине
    выноса, потому что она там стояла;
  комиссию мейкера 0.02% вместо тейкера 0.06%, и нулевое
    проскальзывание: исполнение ровно по своей цене.
Проверка «вход на открытии следующей свечи» меряет совсем другую
стратегию и даёт совсем другой ответ.

── Чего эта проверка НЕ умеет ─────────────────────────────────
Главное ограничение честно: по свечам нельзя узнать, исполнилась бы
лимитка на самом деле. Известно лишь, что цена там побывала. Реально
же нужно, чтобы вынос дошёл до вашего уровня И съел объём, стоящий в
очереди перед вами. Поэтому результат здесь — ОПТИМИСТИЧНАЯ граница:
хуже будет наверняка, лучше — нет.

Чтобы понять, насколько это важно, есть запас `edge_pct`: уровень
считается исполненным, только если цена прошла его насквозь на
заданную долю процента, а не коснулась.

Выход считается тейкером (0.06%) — по рынку, как и бывает, когда
позицию закрывают быстро.

Запуск:
  python Backtest/spike_ladder.py            — 1м, все монеты из кэша
  python Backtest/spike_ladder.py 5m         — то же на 5-минутках
"""

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

MAKER = 0.0002      # комиссия мейкера Bitget, вход лимиткой
TAKER = 0.0006      # комиссия тейкера, выход по рынку

CFG = {
    "levels": (0.01, 0.02, 0.03, 0.04, 0.05),   # лестница от опорной цены
    "size_per_level": 2000.0,                   # долларов на уровень
    "edge_pct": 0.0,      # насколько цена должна ПРОЙТИ уровень, чтобы считать исполнение
    "max_hold": 60,       # свечей на удержание, дальше выход по рынку
    "stop_pct": 0.08,     # аварийный выход, если цена ушла выше опорной на столько
    "target": "reference",  # цель — возврат к опорной цене
}


def episodes(df, cfg, direction=1):
    """
    Один проход по свечам. direction=1 — лестница на продажу выше цены
    (ловим вынос вверх), -1 — зеркально вниз.

    Опорная цена — закрытие ПРЕДЫДУЩЕЙ свечи: она известна до того, как
    свеча началась, и лимитки стоят исходя из неё. Подглядывания нет.
    """
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    n = len(df)
    out = []
    i = 1
    while i < n - 1:
        ref = c[i - 1]
        levels = [ref * (1 + lv * direction) for lv in cfg["levels"]]
        # Исполнились те уровни, сквозь которые цена прошла
        if direction == 1:
            hit = [lv for lv in levels if h[i] >= lv * (1 + cfg["edge_pct"])]
        else:
            hit = [lv for lv in levels if l[i] <= lv * (1 - cfg["edge_pct"])]
        if not hit:
            i += 1
            continue

        qty_usd = cfg["size_per_level"] * len(hit)
        entry = float(np.mean(hit))                  # средняя цена входа
        # Стоп — от ЦЕНЫ ВХОДА, а не от опорной. Если считать от
        # опорной, при лестнице до +5% и стопе 3% «стоп» оказывается
        # НИЖЕ входа, то есть работает как тейк и приносит прибыль.
        # Перебор именно такие варианты и вытащил наверх: t = 53 на
        # обучении — это была ошибка в тесте, а не край.
        stop = entry * (1 + cfg["stop_pct"] * direction)
        target = ref                                  # возврат к опорной

        exit_px, reason, bars = None, None, 0
        for j in range(i + 1, min(i + 1 + cfg["max_hold"], n)):
            bars = j - i
            if direction == 1:
                if h[j] >= stop:
                    exit_px, reason = stop, "стоп"; break
                if l[j] <= target:
                    exit_px, reason = target, "цель"; break
            else:
                if l[j] <= stop:
                    exit_px, reason = stop, "стоп"; break
                if h[j] >= target:
                    exit_px, reason = target, "цель"; break
        if exit_px is None:
            j = min(i + cfg["max_hold"], n - 1)
            exit_px, reason, bars = c[j], "время", j - i

        gross = (entry - exit_px) / entry * direction * qty_usd
        fees = qty_usd * MAKER + qty_usd * TAKER
        out.append({"i": i, "уровней": len(hit), "вход": entry, "выход": exit_px,
                    "причина": reason, "свечей": bars,
                    "объём": qty_usd, "pnl": gross - fees,
                    "pnl_pct": (gross - fees) / qty_usd * 100})
        i = j + 1          # пока в позиции — новых входов нет
    return out


def run(tf, cfg=None, quiet=False):
    cfg = {**CFG, **(cfg or {})}
    files = sorted(glob.glob(f"Backtest/cache/bitget/{tf}/*.csv"))
    rows = []
    for f in files:
        sym = os.path.basename(f).replace(".csv", "").replace("-USDT_USDT", "").replace("-USDT", "")
        df = pd.read_csv(f).sort_values("timestamp").reset_index(drop=True)
        if len(df) < 2000:
            continue
        ep = episodes(df, cfg, 1) + episodes(df, cfg, -1)
        for e in ep:
            e["монета"] = sym
        rows += ep
    if not rows:
        print("событий нет")
        return pd.DataFrame()
    r = pd.DataFrame(rows)
    if not quiet:
        report(r, tf, cfg, len(files))
    return r


def report(r, tf, cfg, n_files):
    pnl = r.pnl.values
    t = pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl))) if len(pnl) > 1 else 0
    print("=" * 70)
    print(f"  ЛИМИТНАЯ ЛЕСТНИЦА ПРОТИВ ПРОСТРЕЛОВ — {tf}, файлов {n_files}")
    print(f"  уровни {[f'{x:.0%}' for x in cfg['levels']]}, "
          f"${cfg['size_per_level']:.0f} на уровень, запас {cfg['edge_pct']:.2%}")
    print("=" * 70)
    print(f"  эпизодов: {len(r)}   средний результат: {pnl.mean():+.2f}$ "
          f"({r.pnl_pct.mean():+.3f}% от объёма)   t = {t:.2f}")
    print(f"  итого: {pnl.sum():+,.0f}$   прибыльных: {(pnl > 0).mean()*100:.1f}%")
    print(f"  лучший: {pnl.max():+,.0f}$   ХУДШИЙ: {pnl.min():+,.0f}$")
    print(f"\n  по причине выхода:")
    for reason, g in r.groupby("причина"):
        print(f"    {reason:<7} {len(g):>6} ({len(g)/len(r)*100:4.1f}%)  "
              f"средний {g.pnl.mean():+8.2f}$   медиана свечей {g['свечей'].median():.0f}")
    print(f"\n  по числу собранных уровней:")
    for k, g in r.groupby("уровней"):
        print(f"    {k} шт  {len(g):>6} эпизодов   средний {g.pnl.mean():+8.2f}$ "
              f"({g.pnl_pct.mean():+.3f}%)")
    per = r.groupby("монета").pnl.agg(["count", "mean", "sum"]).sort_values("mean", ascending=False)
    print(f"\n  монет в плюсе: {(per['mean'] > 0).sum()} из {len(per)}")
    print(per.head(5).to_string(float_format=lambda v: f"{v:,.1f}"))
    print("  ...")
    print(per.tail(3).to_string(header=False, float_format=lambda v: f"{v:,.1f}"))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "1m")
