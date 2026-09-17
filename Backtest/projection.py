"""
Backtest/projection.py
======================
Сколько будет на счёте через год: $100, риск 5%, два бота по $50.

Моделирует ровно то, как торгуют боты: риск от текущего баланса,
плечо до 3x, пауза 8 часов после убытка, комиссия и проскальзывание
фьючерсов. Сигналы считаются по всей истории, а торговля начинается
с начала года (trade_from) — чтобы EMA200 была прогрета, а не
досчитывалась первый месяц по неполному окну. Незакрытые на конец
года позиции оцениваются по последней цене.

Одного числа не даёт намеренно: год — это одна выборка, и результат
сильно зависит от состава монет. Поэтому кроме ровно того набора, что
у ботов, считаются 20 случайных наборов по 25 монет, и показывается
распределение: медиана, 10-й и 90-й перцентили, худший набор, доля
убыточных лет и типичная просадка.

Результат пишется в Backtest/projection_100.json — его показывает
панель на вкладке «Стратегии».

Запуск из корня проекта:
  python Backtest/projection.py
"""
import sys, os, random, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from Backtest import data, strategies, strategies_more
from Backtest.engine import Engine, FILL_MARKET
from Backtest.research_all import FROZEN

FEE, SLIP = 0.0006, 0.0005
COMMON = dict(atr_period=14, atr_mult=2.5, rr=3.0, ema=200, allow_short=True, max_hold_bars=200)
STRATS = {"donchian": (strategies.donchian, dict(channel=20, **COMMON)),
          "supertrend": (strategies_more.supertrend, dict(mult=3.0, **COMMON))}

def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)

raw = {}
for b in FROZEN:
    c = data.read_cache("bitget", f"{b}/USDT:USDT", "4h")
    if c is not None and len(c) > 2000:
        raw[b] = c.sort_values("timestamp").reset_index(drop=True)

def run_sleeve(strat, coins, y0, y1, deposit):
    fn, p = STRATS[strat]
    frames = {}
    for b in coins:
        df = raw[b]
        df = df[df.timestamp < ms(y1)].reset_index(drop=True)
        if len(df) < 400 or df.timestamp.iloc[-1] < ms(y0): continue
        frames[b] = fn(data._finalize(df), p)
    eng = Engine(cfg={}, signal_fn=strategies.signal_fn, deposit=deposit, risk_pct=0.05,
                 max_open=5, commission=FEE, slippage=SLIP, fill_model=FILL_MARKET,
                 max_wait_bars=1, cooldown_minutes=480, max_leverage=3.0,
                 max_hold_bars=200).run(frames, trade_from=ms(y0))
    # Незакрытые позиции оцениваем по последнему закрытию года
    unreal = 0.0
    for pos in eng.positions:
        last = frames[pos.order.symbol]["close"].iloc[-1]
        d = pos.order.direction
        unreal += (last - pos.entry_price) * pos.qty * d - pos.notional * FEE
    eq = np.array([b for _, b in eng.equity])
    return eng.balance + unreal, eq, len(eng.trades)

def dd(eq):
    if len(eq) == 0: return 0.0
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min() * 100)

YEARS = [("2023-09-01", "2024-09-01", "обучение"),
         ("2024-09-01", "2025-09-01", "обучение"),
         ("2025-09-01", "2026-09-01", "ПРОВЕРКА")]
coins_all = sorted(raw)
rng = random.Random(2026)
out = {"years": []}

print(f"монет: {len(coins_all)}\n")
for y0, y1, tag in YEARS:
    print(f"=== {y0[:7]} .. {y1[:7]}  [{tag}] ===", flush=True)
    row = {"from": y0, "to": y1, "tag": tag}
    # 1) Ровно тот набор, что у ботов
    d100, e_d, n_d = run_sleeve("donchian", coins_all, y0, y1, 100.0)
    s100, e_s, n_s = run_sleeve("supertrend", coins_all, y0, y1, 100.0)
    d50, e_d50, _ = run_sleeve("donchian", coins_all, y0, y1, 50.0)
    s50, e_s50, _ = run_sleeve("supertrend", coins_all, y0, y1, 50.0)
    m = min(len(e_d50), len(e_s50)); both = e_d50[:m] + e_s50[:m]
    row["full"] = {"donchian": [round(d100, 2), round(dd(e_d), 1), n_d],
                   "supertrend": [round(s100, 2), round(dd(e_s), 1), n_s],
                   "both": [round(d50 + s50, 2), round(dd(both), 1), n_d + n_s]}
    print(f"  весь набор:  только Дончиан ${d100:>8.2f} (просадка {dd(e_d):>5.1f}%, {n_d} сд.)")
    print(f"               только Supertrend ${s100:>6.2f} (просадка {dd(e_s):>5.1f}%, {n_s} сд.)")
    print(f"               ОБА по $50  ${d50+s50:>8.2f} (просадка {dd(both):>5.1f}%)", flush=True)
    # 2) Удача состава: 20 случайных наборов по 25 монет
    dist = {"donchian": [], "supertrend": [], "both": []}
    ddist = {"donchian": [], "supertrend": [], "both": []}
    for _ in range(20):
        sub = rng.sample(coins_all, 25)
        a, ea, _ = run_sleeve("donchian", sub, y0, y1, 100.0)
        b, eb, _ = run_sleeve("supertrend", sub, y0, y1, 100.0)
        a5, ea5, _ = run_sleeve("donchian", sub, y0, y1, 50.0)
        b5, eb5, _ = run_sleeve("supertrend", sub, y0, y1, 50.0)
        m = min(len(ea5), len(eb5))
        for k, v, e in (("donchian", a, ea), ("supertrend", b, eb),
                        ("both", a5 + b5, ea5[:m] + eb5[:m])):
            dist[k].append(v); ddist[k].append(dd(e))
    row["subsets"] = {}
    for k in dist:
        v = np.array(dist[k]); d_ = np.array(ddist[k])
        row["subsets"][k] = {"median": round(float(np.median(v)), 2), "p10": round(float(np.percentile(v, 10)), 2),
                             "p90": round(float(np.percentile(v, 90)), 2), "worst": round(float(v.min()), 2),
                             "loss_share": round(float((v < 100).mean() * 100)), "dd_median": round(float(np.median(d_)), 1)}
        r = row["subsets"][k]
        name = {"donchian": "Дончиан", "supertrend": "Supertrend", "both": "ОБА по $50"}[k]
        print(f"  наборы {name:<11} медиана ${r['median']:>7.2f}  10%..90%: ${r['p10']:>6.2f}..${r['p90']:>7.2f}  "
              f"худший ${r['worst']:>6.2f}  в минусе {r['loss_share']:>3}%  просадка ~{r['dd_median']}%", flush=True)
    out["years"].append(row)
    print()

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projection_100.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("ГОТОВО")
