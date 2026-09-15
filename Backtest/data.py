"""
Backtest/data.py
================
Загрузка исторических свечей с кэшем на диск.

Биржа отдаёт максимум ~1000 свечей за запрос, поэтому длинные
периоды качаются постранично. Скачанное кладём в Backtest/cache/,
чтобы повторный прогон бэктеста не ходил в сеть вообще — это
разница между «минуты» и «мгновенно» при подборе параметров.
"""

import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT

CACHE_DIR = ROOT / "Backtest" / "cache"

# Длительность свечи в миллисекундах
TF_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _cache_path(exchange_id, symbol, timeframe):
    safe = symbol.replace("/", "-").replace(":", "_")
    return CACHE_DIR / exchange_id / timeframe / f"{safe}.csv"


def _to_ms(dt):
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def get_exchange(exchange_id):
    import ccxt
    if exchange_id == "bitget":
        return ccxt.bitget({"enableRateLimit": True,
                            "options": {"defaultType": "spot"}})
    if exchange_id == "binance":
        return ccxt.binance({"enableRateLimit": True})
    raise ValueError(f"неизвестная биржа: {exchange_id}")


def read_cache(exchange_id, symbol, timeframe):
    """Кэш или None. Пустой/битый файл считаем отсутствующим."""
    path = _cache_path(exchange_id, symbol, timeframe)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or not set(COLUMNS).issubset(df.columns):
            return None
        return df[COLUMNS]
    except Exception:
        return None


def write_cache(exchange_id, symbol, timeframe, df):
    path = _cache_path(exchange_id, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(path, index=False)


# Сколько свечей биржа реально отдаёт за один запрос.
# Bitget всегда возвращает ровно 200 и трактует since + limit*tf как
# КОНЕЦ окна: запросив limit=1000 от 1 августа, получаешь 200 свечей
# от 3 августа. Просить у него больше 200 — значит молча потерять
# начало периода, поэтому размер страницы фиксирован по бирже.
PAGE_LIMIT = {"bitget": 200, "binance": 1000}
DEFAULT_PAGE_LIMIT = 200


def page_limit_for(exchange):
    return PAGE_LIMIT.get(getattr(exchange, "id", ""), DEFAULT_PAGE_LIMIT)


def fetch_ohlcv(exchange, symbol, timeframe, since_ms, until_ms,
                page_limit=None, verbose=False):
    """
    Постраничная выкачка свечей [since_ms, until_ms).

    Сдвигаемся от ПОСЛЕДНЕЙ полученной свечи, а не на фиксированный
    шаг: при пропусках в истории фиксированный шаг либо зациклил бы
    цикл, либо перескочил кусок данных. Выходим, когда биржа
    перестала продвигаться вперёд.
    """
    if page_limit is None:
        page_limit = page_limit_for(exchange)

    step = TF_MS[timeframe]
    rows, cursor = [], since_ms
    errors = 0
    empty_pages = 0
    # Сколько пустых окон подряд готовы промотать в поисках листинга
    max_empty = 200
    max_pages = max(8, (until_ms - since_ms) // (step * page_limit) * 2 + 16)
    pages = 0

    while cursor < until_ms and pages < max_pages:
        pages += 1
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe,
                                         since=cursor, limit=page_limit)
        except Exception as e:
            errors += 1
            if verbose:
                print(f"    [!] {symbol} {timeframe}: {type(e).__name__} — пауза 3с")
            if errors > 3:
                break
            time.sleep(3)
            continue

        if not batch:
            # Пусто — это чаще всего участок ДО листинга монеты, а не
            # конец истории. Раньше здесь стоял break, и символы вроде
            # ZEC (листинг 2026) или HYPE (2025) молча возвращали ноль
            # свечей при запросе с 2023 года. Промотаем окно вперёд.
            empty_pages += 1
            if empty_pages > max_empty:
                break
            cursor += step * page_limit
            continue
        empty_pages = 0

        last_raw = batch[-1][0]
        batch = [c for c in batch if since_ms <= c[0] < until_ms]
        if batch:
            rows.extend(batch)

        # Биржа не продвинулась вперёд — дальше данных нет
        if last_raw < cursor + step:
            break
        cursor = last_raw + step

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    return df.reset_index(drop=True)


def load(exchange_id, symbol, timeframe, start, end,
         exchange=None, use_cache=True, verbose=False):
    """
    Свечи [start, end). Возвращает DataFrame с UTC-временем в
    колонке `dt` или None, если данных нет.

    Кэш дополняется: если в нём уже есть часть периода, из сети
    тянем только недостающий хвост.
    """
    since_ms, until_ms = _to_ms(start), _to_ms(end)
    cached = read_cache(exchange_id, symbol, timeframe) if use_cache else None

    need_from = since_ms
    if cached is not None and not cached.empty:
        have_from, have_to = int(cached.timestamp.min()), int(cached.timestamp.max())
        covered = have_from <= since_ms and have_to >= until_ms - TF_MS[timeframe]
        if covered:
            df = cached
            df = df[(df.timestamp >= since_ms) & (df.timestamp < until_ms)]
            return _finalize(df)
        need_from = max(since_ms, have_to + TF_MS[timeframe]) if have_from <= since_ms else since_ms

    if exchange is None:
        exchange = get_exchange(exchange_id)

    fresh = fetch_ohlcv(exchange, symbol, timeframe, need_from, until_ms,
                        verbose=verbose)

    if cached is not None and fresh is not None:
        merged = pd.concat([cached, fresh], ignore_index=True)
    elif fresh is not None:
        merged = fresh
    elif cached is not None:
        merged = cached
    else:
        return None

    merged = (merged.drop_duplicates(subset="timestamp")
                    .sort_values("timestamp").reset_index(drop=True))
    if use_cache:
        write_cache(exchange_id, symbol, timeframe, merged)

    out = merged[(merged.timestamp >= since_ms) & (merged.timestamp < until_ms)]
    return _finalize(out)


def _finalize(df):
    if df is None or df.empty:
        return None
    df = df.reset_index(drop=True).copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


# Стейблкоины: пара USDC/USDT почти не движется, её "пробои" —
# это шум на третьем знаке. Торговать их бессмысленно, а в топ по
# обороту они попадают всегда.
STABLE_BASES = {
    "USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDD", "USDP", "PYUSD",
    "EURT", "EURS", "USDE", "USDS", "USD1", "RLUSD", "XAUT", "PAXG",
}


def is_stable_pair(symbol):
    base = symbol.split("/")[0].upper()
    return base in STABLE_BASES


def is_stock_token(market):
    """
    Токенизированная акция, а не криптовалюта.

    Bitget листит 1178 таких пар (RNVDA = NVIDIA, RQQQ = ETF QQQ,
    RTSLA = Tesla) против 513 настоящих криптовалютных. По обороту
    они занимают ВЕСЬ топ-20, поэтому отбор "самые ликвидные пары"
    без этого фильтра даёт портфель из акций.
    """
    return (market or {}).get("info", {}).get("areaSymbol") == "yes"


def top_symbols(exchange_id, limit=30, min_quote_vol=1_000_000, exchange=None,
                exclude_stock_tokens=True, exclude_stables=True):
    """
    Самые ликвидные пары к USDT — по текущему обороту.

    Осторожно: это survivorship bias. Список строится по сегодняшним
    объёмам, а бэктест гоняется по прошлому, где часть этих монет
    ещё не торговалась, а вылетевшие в ноль сюда не попадут.
    Для честной оценки список нужно фиксировать на начало периода.
    """
    if exchange is None:
        exchange = get_exchange(exchange_id)

    markets = {}
    if exclude_stock_tokens:
        try:
            markets = exchange.load_markets()
        except Exception:
            markets = {}

    tickers = exchange.fetch_tickers()
    out = []
    for s, t in tickers.items():
        if not s.endswith("/USDT") or ":" in s:
            continue
        if exclude_stock_tokens and is_stock_token(markets.get(s)):
            continue
        if exclude_stables and is_stable_pair(s):
            continue
        vol = t.get("quoteVolume") or 0
        if vol >= min_quote_vol:
            out.append((s, vol))
    out.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in out[:limit]]
