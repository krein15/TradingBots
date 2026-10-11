"""
Backtest/tape.py
================
Лента сделок с Bitget: загрузка истории и чтение.

Зачем. Перебор 1800 настроек двенадцати семейств на трёх младших
таймфреймах не дал ни одной стратегии, переживающей реалистичное
исполнение. Вывод из этого узкий и точный: на СВЕЧНЫХ закономерностях
и розничных комиссиях младшие таймфреймы закрыты. Про другие источники
информации он не говорит ничего.

Лента — другой источник. В свече видно, что цена сходила вверх. В
ленте видно, КТО её туда отправил: одна крупная заявка или сто мелких,
покупали по рынку или продавали, повторяется ли это. Скальперы,
которые смотрят «кластеры», работают именно с этим.

Что отдаёт биржа:
  ts     — время сделки, миллисекунды
  price  — цена
  size   — объём в монетах
  side   — сторона АГРЕССОРА: Buy = кто-то купил по рынку

Чего НЕ отдаёт: стакан. Его исторических снимков не существует, их
можно только записывать вперёд. Поэтому проверить можно идеи про
поток сделок, но не про плотности в стакане.

Техника: публичный метод отдаёт 1000 сделок за запрос и листается
НАЗАД по tradeId. Вперёд по времени листать нельзя — since биржа
игнорирует. Поэтому идём от свежих к старым, пока не наберём нужную
глубину.

Запуск:
  python Backtest/tape.py DOGE SUI WLD ONDO PEPE --days 7
"""

import argparse
import gzip
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import ccxt  # noqa: E402

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "tape")
COLS = ["ts", "price", "size", "side"]


def path(coin):
    return os.path.join(DIR, f"{coin}.csv.gz")


def download(coin, days, ex, quiet=False):
    """
    Тянет ленту назад от текущего момента на days суток.

    Если файл уже есть, докачивает только недостающее старое: начинаем
    с самой старой известной сделки и уходим глубже. Повторная загрузка
    того, что уже лежит на диске, не нужна — лента не меняется задним
    числом.
    """
    os.makedirs(DIR, exist_ok=True)
    sym = f"{coin}USDT"
    until = ex.milliseconds() - days * 86400_000

    have = read(coin) if os.path.exists(path(coin)) else None
    cursor = None
    if have is not None and len(have):
        oldest = have.iloc[0]
        if int(oldest.ts) <= until:
            if not quiet:
                print(f"  {coin}: уже загружено до {ts_str(int(oldest.ts))}, пропускаю")
            return have
        cursor = str(int(oldest.trade_id)) if "trade_id" in have.columns else None

    rows, t0, pages = [], time.time(), 0
    while True:
        params = {"symbol": sym, "productType": "USDT-FUTURES", "limit": "1000"}
        if cursor:
            params["idLessThan"] = cursor
        try:
            d = ex.publicMixGetV2MixMarketFillsHistory(params).get("data") or []
        except Exception as e:
            print(f"  {coin}: обрыв на {pages} странице — {type(e).__name__}: {str(e)[:70]}")
            break
        if not d:
            break
        rows += [(int(x["ts"]), float(x["price"]), float(x["size"]),
                  1 if x["side"].lower().startswith("b") else -1, int(x["tradeId"]))
                 for x in d]
        pages += 1
        cursor = d[-1]["tradeId"]
        oldest_ts = int(d[-1]["ts"])
        if oldest_ts <= until:
            break
        if pages % 50 == 0 and not quiet:
            print(f"  {coin}: {len(rows):>7} сделок, дошли до {ts_str(oldest_ts)} "
                  f"({time.time()-t0:.0f}с)", flush=True)

    new = pd.DataFrame(rows, columns=COLS + ["trade_id"]).sort_values("ts")
    if have is not None:
        new = pd.concat([new, have]).drop_duplicates("trade_id").sort_values("ts")
    new = new.reset_index(drop=True)
    with gzip.open(path(coin), "wt", encoding="utf-8", newline="") as f:
        new.to_csv(f, index=False)
    if not quiet:
        span = (new.ts.iloc[-1] - new.ts.iloc[0]) / 86400_000
        print(f"  {coin}: {len(new):,} сделок за {span:.1f} сут "
              f"({ts_str(int(new.ts.iloc[0]))} … {ts_str(int(new.ts.iloc[-1]))}), "
              f"{os.path.getsize(path(coin))/1e6:.0f} МБ, {time.time()-t0:.0f}с")
    return new


def read(coin):
    """Лента с диска. Возвращает None, если её не качали."""
    if not os.path.exists(path(coin)):
        return None
    with gzip.open(path(coin), "rt", encoding="utf-8") as f:
        return pd.read_csv(f)


def available():
    if not os.path.isdir(DIR):
        return []
    return sorted(f.replace(".csv.gz", "") for f in os.listdir(DIR) if f.endswith(".csv.gz"))


def ts_str(ms):
    return pd.Timestamp(ms, unit="ms").strftime("%d.%m %H:%M")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coins", nargs="+")
    ap.add_argument("--days", type=float, default=7)
    args = ap.parse_args()
    ex = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    print(f"Лента сделок Bitget, глубина {args.days} сут\n")
    for c in args.coins:
        download(c.upper(), args.days, ex)


if __name__ == "__main__":
    main()
