"""
Backtest/research_fast.py
=========================
Тот же широкий поиск, что на 4 часах, но на младших таймфреймах.

Зачем. На 4ч перебор 13 семейств нашёл два работающих: Дончиан и
Supertrend. На 15м из 72 настроек трёх семейств не выжила ни одна, но
трёх семейств мало, чтобы закрывать вопрос. Скальперы существуют и
кто-то на этом зарабатывает — значит, проверить стоит все семейства, и
той же методикой, иначе ответ не будет значить ничего.

Отличия от research_all.py ровно три:

  данные берутся из локального кэша минутных свечей (фьючерсы, те же
    инструменты, что торгуют боты), а 5м собираются из них же — чтобы
    сравнение между таймфреймами шло на одном и том же рынке;
  проскальзывание взято ИЗМЕРЕННОЕ на бирже демо-стендом (0.027% за
    сторону), а не предполагаемое 0.05%. На 4ч разница безразлична, на
    минутках она решает;
  деление на обучение и проверку — по времени, 60/40.

Арифметика, ради которой всё это затевается. Размах минутной свечи у
DOGE — 0.086% от цены. Круг тейкером стоит 0.174%, то есть ДВА размаха.
Чтобы стратегия на минутках работала на рыночных заявках, она должна
забирать больше двух полных свечей на сделку. Это и проверяем.

Запуск:
  python Backtest/research_fast.py --tf 5m
  python Backtest/research_fast.py --tf 1m --days 30
"""

import argparse
import glob
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

from Backtest import data, search, research_all   # noqa: E402

FEE = 0.0006        # тейкер Bitget
SLIP = 0.0003       # ИЗМЕРЕНО демо-стендом: 0.027% за сторону
MAKER_FEE = 0.0002  # мейкер Bitget
MAKER_SLIP = 0.0    # у лимитки проскальзывания нет: своя цена
CACHE_1M = "Backtest/cache/bitget/1m"


def resample(df, k):
    """k минутных свечей в одну. Тот же рынок, крупнее масштаб."""
    if k == 1:
        return df
    g = np.arange(len(df)) // k
    out = pd.DataFrame({
        "timestamp": df.timestamp.groupby(g).first(),
        "open": df.open.groupby(g).first(),
        "high": df.high.groupby(g).max(),
        "low": df.low.groupby(g).min(),
        "close": df.close.groupby(g).last(),
        "volume": df.volume.groupby(g).sum() if "volume" in df else 0,
    }).reset_index(drop=True)
    return out


def load_cached(tf, days=None, limit_symbols=None):
    """Минутный кэш -> нужный таймфрейм. Ничего не скачивает."""
    k = {"1m": 1, "5m": 5, "15m": 15}[tf]
    out = {}
    for f in sorted(glob.glob(f"{CACHE_1M}/*.csv")):
        sym = os.path.basename(f).replace("-USDT_USDT.csv", "")
        d = pd.read_csv(f).sort_values("timestamp").reset_index(drop=True)
        if days:
            d = d[d.timestamp >= d.timestamp.max() - days * 86400_000].reset_index(drop=True)
        d = resample(d, k)
        if len(d) < 500:
            continue
        out[f"{sym}/USDT:USDT"] = data._finalize(d)
        if limit_symbols and len(out) >= limit_symbols:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="5m")
    ap.add_argument("--days", type=int, default=None, help="сколько последних дней брать")
    ap.add_argument("--symbols", type=int, default=None)
    ap.add_argument("--families", default=None)
    ap.add_argument("--maker", action="store_true",
                    help="вход лимиткой: комиссия мейкера, без проскальзывания")
    ap.add_argument("--queue", action="store_true",
                    help="мейкер с поправкой на очередь: цена должна пройти уровень")
    args = ap.parse_args()

    # Мейкерский вариант честнее к скальпингу: так и зарабатывают те,
    # кто торгует на минутах. Но лимитка исполняется не всегда и не
    # тогда, когда хочется — движок требует, чтобы цена дошла до уровня.
    fee, slip = (MAKER_FEE, MAKER_SLIP) if args.maker else (FEE, SLIP)
    fill = ("limit_queue" if args.queue else "limit") if args.maker else "market"
    wait = 3 if args.maker else 1

    print("=" * 76)
    print(f"  ШИРОКИЙ ПОИСК НА МЛАДШЕМ ТФ — {args.tf}")
    print("=" * 76)
    sd = load_cached(args.tf, args.days, args.symbols)
    if not sd:
        print("  минутного кэша нет — сначала скачайте свечи"); return
    any_df = next(iter(sd.values()))
    t0, t1 = int(any_df.timestamp.iloc[0]), int(any_df.timestamp.iloc[-1])
    split = t0 + int((t1 - t0) * 0.6)
    fmt = lambda ms: pd.Timestamp(ms, unit="ms").strftime("%Y-%m-%d")
    train, test = (fmt(t0), fmt(split)), (fmt(split), fmt(t1))
    print(f"  инструментов {len(sd)}, свечей на каждый ~{len(any_df):,}")
    print(f"  обучение {train[0]}..{train[1]}   проверка {test[0]}..{test[1]} (60/40)")
    режим = ('МЕЙКЕР с очередью' if args.queue else 'МЕЙКЕР: лимитный вход') if args.maker else 'ТЕЙКЕР: рыночный вход'
    print(f"  {режим}, "
          f"комиссия {fee:.2%} + проскальзывание {slip:.3%} за сторону")


    research_all.FEE, research_all.SLIP = fee, slip
    wanted = args.families.split(",") if args.families else list(research_all.SPACES)
    rows = []
    for name in wanted:
        t = time.time()
        out = research_all.single_family(sd, name, research_all.SPACES[name],
                                         train, test, min_tr=40, min_te=20,
                                         fill_model=fill, max_wait_bars=wait)
        if out is None:
            print(f"  {name:<11} сделок не хватило", flush=True)
            continue
        row, _ = out
        rows.append(row)
        print(f"  {name:<11} настроек {row['настроек']:>3}  "
              f"медиана на проверке {row['медиана_проверка']:>+7} R/сделку  "
              f"плюсовых {row['плюсовых']:>3}%   [{time.time()-t:.0f}с]", flush=True)

    if not rows:
        print("\n  ни одно семейство не набрало сделок"); return
    r = pd.DataFrame(rows)
    print("\n" + "=" * 76)
    print(f"  ИТОГ по {args.tf} ({'мейкер+очередь' if args.queue else 'мейкер' if args.maker else 'тейкер'}): "
          f"семейств {len(r)}, настроек {r['настроек'].sum()}")
    print(f"  медиана медиан на проверке: {r['медиана_проверка'].median():+.3f} R/сделку")
    print(f"  семейств с положительной медианой: {(r['медиана_проверка'] > 0).sum()} из {len(r)}")
    good = r[r["медиана_проверка"] > 0].sort_values("медиана_проверка", ascending=False)
    if len(good):
        print("\n  в плюсе на проверке:")
        for _, x in good.iterrows():
            print(f"    {x['семейство']:<11} {x['медиана_проверка']:+.3f} R/сделку, "
                  f"плюсовых настроек {x['плюсовых']}%, топ-3: {x['топ3_проверка']}")
    print("=" * 76)
    return r


if __name__ == "__main__":
    main()
