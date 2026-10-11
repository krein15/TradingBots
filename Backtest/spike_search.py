"""
Backtest/spike_search.py
========================
Перебор настроек лимитной лестницы с разделением обучение / проверка.

Зачем отдельный файл. Соблазн подобрать «оптимальное число уровней» по
всем данным сразу — главный способ обмануть себя: при нескольких
десятках вариантов лучший найдётся даже в случайных числах. Поэтому
здесь то же, что в search.py для основных стратегий:

  1. История делится по ВРЕМЕНИ: первые 60% — отбор, последние 40% в
     отборе не участвуют вообще;
  2. варианты ранжируются по t-статистике на обучении, а не по сумме:
     сумма растёт вместе с числом сделок и любит подгонку;
  3. лучшие прогоняются по проверке, и засчитывается именно она;
  4. печатается, сколько вариантов вообще положительны на проверке —
     если из 48 положительны 3, это похоже на случайность, а не на край.

Отдельная ловушка этой стратегии. В разборе по числу собранных уровней
«1 уровень» выглядит прибыльным. Это отбор задним числом: заранее
неизвестно, остановится вынос на +1% или пройдёт все пять. Торгуемый
вариант — «ставлю лестницу из одного уровня», и он совсем другой.

Запуск:
  python Backtest/spike_search.py          — 5-минутки из кэша
  python Backtest/spike_search.py 1m       — минутки
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Backtest import data, spike_ladder as sl   # noqa: E402

try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

LADDERS = {
    "+1":                (0.01,),
    "+2":                (0.02,),
    "+3":                (0.03,),
    "+5":                (0.05,),
    "+1,+2":             (0.01, 0.02),
    "+2,+3":             (0.02, 0.03),
    "+3,+5":             (0.03, 0.05),
    "+1..+3":            (0.01, 0.02, 0.03),
    "+3,+4,+5":          (0.03, 0.04, 0.05),
    "+1..+5 (как в ролике)": (0.01, 0.02, 0.03, 0.04, 0.05),
    "+2,+4,+6":          (0.02, 0.04, 0.06),
    "+5,+7,+10":         (0.05, 0.07, 0.10),
}
STOPS = (0.03, 0.08, 0.15)
HOLDS_MIN = (60, 300)        # минут удержания, независимо от таймфрейма


def crypto_files(tf):
    """Только то, что сейчас торгуется как крипто-фьючерс на Bitget."""
    ex = data.get_exchange("bitget")
    markets = ex.load_markets()
    out = []
    for f in sorted(glob.glob(f"Backtest/cache/bitget/{tf}/*.csv")):
        sym = os.path.basename(f).replace(".csv", "").replace("-USDT_USDT", "").replace("-USDT", "")
        m = markets.get(f"{sym}/USDT:USDT")
        if not m or data.is_stock_token(m):
            continue
        out.append((sym, f))
    return out


def metrics(rows):
    if len(rows) < 10:
        return None
    p = np.array([r["pnl"] for r in rows])
    return {"n": len(p), "среднее": p.mean(), "сумма": p.sum(),
            "t": p.mean() / (p.std(ddof=1) / np.sqrt(len(p))),
            "плюс%": (p > 0).mean() * 100}


def main(tf="5m"):
    step_min = int(tf.replace("m", ""))
    files = crypto_files(tf)
    print(f"инструментов: {len(files)} — {', '.join(s for s, _ in files)}")
    frames = {}
    for sym, f in files:
        d = pd.read_csv(f).sort_values("timestamp").reset_index(drop=True)
        if len(d) < 2000:
            continue
        cut = d.timestamp.quantile(0.6)
        frames[sym] = (d[d.timestamp < cut].reset_index(drop=True),
                       d[d.timestamp >= cut].reset_index(drop=True))
    print(f"обучение/проверка разделены по времени 60/40, "
          f"свечей на инструмент ~{len(d):,}\n")

    rows = []
    for name, lv in LADDERS.items():
        for stop in STOPS:
            for hold in HOLDS_MIN:
                cfg = {**sl.CFG, "levels": lv, "stop_pct": stop,
                       "max_hold": max(1, hold // step_min)}
                tr, te = [], []
                for sym, (a, b) in frames.items():
                    tr += sl.episodes(a, cfg, 1) + sl.episodes(a, cfg, -1)
                    te += sl.episodes(b, cfg, 1) + sl.episodes(b, cfg, -1)
                m_tr, m_te = metrics(tr), metrics(te)
                if not m_tr or not m_te:
                    continue
                rows.append({"лестница": name, "стоп": stop, "держим_мин": hold,
                             "обуч_n": m_tr["n"], "обуч_средн": m_tr["среднее"], "обуч_t": m_tr["t"],
                             "пров_n": m_te["n"], "пров_средн": m_te["среднее"], "пров_t": m_te["t"],
                             "пров_плюс%": m_te["плюс%"]})
    r = pd.DataFrame(rows)
    if r.empty:
        print("событий не набралось"); return

    print(f"проверено вариантов: {len(r)}")
    pos = (r["пров_средн"] > 0).sum()
    print(f"положительны на ПРОВЕРКЕ: {pos} из {len(r)} ({pos/len(r)*100:.0f}%)")
    print(f"значимы на проверке (t > 2): {(r['пров_t'] > 2).sum()}\n")

    top = r.sort_values("обуч_t", ascending=False).head(8)
    print("Лучшие 8 по обучению — и что они дали на проверке:")
    print(f"{'лестница':<24}{'стоп':>6}{'мин':>6}{'обуч n':>8}{'обуч $':>9}{'обуч t':>8}"
          f"{'пров n':>8}{'пров $':>9}{'пров t':>8}")
    for _, x in top.iterrows():
        print(f"{x['лестница']:<24}{x['стоп']:>6.0%}{x['держим_мин']:>6}"
              f"{x['обуч_n']:>8}{x['обуч_средн']:>+9.2f}{x['обуч_t']:>8.2f}"
              f"{x['пров_n']:>8}{x['пров_средн']:>+9.2f}{x['пров_t']:>8.2f}")

    best = r.sort_values("пров_средн", ascending=False).head(3)
    print("\nЛучшие на проверке (для сравнения — это уже подглядывание):")
    for _, x in best.iterrows():
        print(f"  {x['лестница']:<24} стоп {x['стоп']:.0%} держим {x['держим_мин']} мин: "
              f"{x['пров_средн']:+.2f}$ при t={x['пров_t']:.2f}, {x['пров_n']} эпизодов")
    return r


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "5m")
