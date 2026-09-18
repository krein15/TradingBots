"""
paper_trading_donchian.py
=========================
Бот #5 — пробой канала Дончиана. Bitget перпетуальные фьючерсы, 4ч.

Правила не придуманы заново, а найдены перебором 522 конфигураций с
разделением истории на обучение и проверку (см. README, раздел
«Найденная стратегия»).

Край подтверждён на ТОМ ЖЕ фьючерсном наборе, который берёт этот
бот (34 пары), на периоде, в отборе не участвовавшем:
  обучение 2023-06..2025-06:  340 сделок, WR 38.5%, +0.359R
  проверка 2025-06..2026-09:  211 сделок, WR 35.5%, +0.247R  (t=2.01)
  последний год:              159 сделок, WR 37.1%, +0.301R

Почему фьючерсы, а не спот:
  на проверочном (медвежьем) периоде шорты дали больше лонгов, а на
  споте шортить нельзя. Заодно комиссия ниже: 0.06% тейкер против
  0.1% на споте.

Почему сигналы берутся из Backtest/strategies.py, а не переписаны
здесь: чтобы бот и бэктест не могли разойтись. Любая правка правил
автоматически меняет обе стороны.

Главное отличие от прежних ботов проекта:
  стоп задаётся явно в ATR (в среднем 6.5% от цены), а не выводится
  из границ канала. Это и есть причина, по которой стратегия
  переживает комиссию: на фьючерсах она съедает 0.02R вместо 0.207R
  у Bot1 на пятиминутках.

Запуск:
  python paper_trading_donchian.py           — торговля
  python paper_trading_donchian.py status    — статистика
  python paper_trading_donchian.py symbols   — показать набор инструментов
  python paper_trading_donchian.py reset     — сбросить журнал
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import ccxt
import numpy as np
import pandas as pd

# Корень проекта в путях импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Консоль Windows не UTF-8, а в логах эмодзи — без этого print падает
try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from Backtest import strategies, strategies_more

HERE = os.path.dirname(os.path.abspath(__file__))

# Этот файл — общее ядро бумажных ботов: исполнение, журнал, стопы и
# отбор инструментов одни на всех. Бот #6 (Supertrend) импортирует его
# и подменяет только правила входа. Так два бота гарантированно
# исполняют сделки одинаково и различаются ровно тем, что мы сравниваем.
STRATEGY_FUNCS = {
    "donchian":   strategies.donchian,
    "supertrend": strategies_more.supertrend,
}

# Какие ключи CONFIG являются параметрами правил у каждой стратегии
STRATEGY_KEYS = {
    "donchian":   ("channel", "atr_period", "atr_mult", "rr", "ema",
                   "allow_short", "max_hold_bars"),
    "supertrend": ("mult", "atr_period", "atr_mult", "rr", "ema",
                   "allow_short", "max_hold_bars"),
}

CONFIG = {
    "bot_id":         "bot5",
    "bot_name":       "Бот #5 — Дончиан",
    "strategy":       "donchian",

    # ── Правила (найдены перебором, менять только вместе с бэктестом) ──
    "timeframe":      "4h",
    "channel":        20,      # пробой максимума/минимума 20 свечей
    "atr_period":     14,
    "atr_mult":       2.5,     # стоп = 2.5 ATR
    "rr":             3.0,     # тейк = 3 дистанции стопа
    "ema":            200,     # лонги выше EMA200, шорты ниже
    "allow_short":    True,
    "max_hold_bars":  200,     # 33 дня, дальше выходим по рынку

    # ── Деньги ────────────────────────────────────────────────
    "deposit":        50.0,
    # 5% — осознанный выбор для форвард-теста.
    # Статистика в R от размера риска не зависит: он масштабирует
    # кривую баланса, но не матожидание на сделку. Зато на 5%
    # сразу видно настоящую просадку.
    # Чего ждать: на 20 случайных наборах монет за год медиана
    # была $57 из $50, худший исход $2, просадка около -80%, и
    # 40% наборов закончили год в минусе. Максимальная серия
    # убытков подряд — 11.
    "risk_pct":       0.05,
    "max_open":       5,
    "max_leverage":   3.0,     # перпетуалы позволяют, но без фанатизма
    "commission":     0.0006,  # тейкер Bitget futures
    "slippage":       0.0005,
    "min_notional":   5.0,     # минимальный объём заявки на бирже

    # ── Отбор инструментов ────────────────────────────────────
    "min_usdt_vol":   2_000_000,
    "max_symbols":    40,

    # ── Работа ────────────────────────────────────────────────
    "scan_interval_min": 20,
    # Сигнал действителен, только пока свеча закрылась недавно.
    # Бэктест входит на открытии СЛЕДУЮЩЕЙ свечи, то есть сразу после
    # закрытия сигнальной. Без этого ограничения бот при первом
    # запуске и при каждом перезапуске подхватывал свечу, закрывшуюся
    # до 4 часов назад, и входил по совсем другой цене. Так и
    # случилось при первом запуске на 5%: три входа по свече,
    # закрывшейся 190 минут назад. Порог = интервал опроса + запас.
    "max_signal_age_min": 30,
    # Пауза по инструменту после убытка — ровно как в движке бэктеста.
    # На проверке длина паузы почти ни на что не влияет: 0, 4, 8 и 24
    # часа дают от +0.193R до +0.226R, разброс меньше погрешности.
    # 8 часов взяты как середина, менять смысла нет.
    "cooldown_hours": 8,
    # Чего ждать от бота: показывается и в статистике, и в панели
    # рядом с фактом, чтобы расхождение было видно сразу. Диапазон,
    # а не точка: нижняя граница — медиана по случайным наборам монет
    # на проверке, верхняя — среднее за весь период. Реальный набор
    # монет может оказаться любым из них.
    "expect": {"wr": 35, "r_lo": 0.10, "r_hi": 0.20},
    "candles":        400,     # хватает на EMA200 с запасом
    "journal":        os.path.join(HERE, "donchian_journal.json"),
    "logfile":        os.path.join(HERE, "donchian_log.txt"),
}

TF_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


# ─────────────────────────────────────────────────────────────
#  Служебное
# ─────────────────────────────────────────────────────────────
def log(msg, cfg, show=True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if show:
        print(line, flush=True)
    with open(cfg["logfile"], "a", encoding="utf-8", errors="replace") as f:
        f.write(line + "\n")


def load_journal(cfg):
    if os.path.exists(cfg["journal"]):
        with open(cfg["journal"], "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "created": datetime.now().isoformat(),
        "deposit": cfg["deposit"],
        "balance": cfg["deposit"],
        "trades": [], "open": [], "cooldown": {}, "cycles": 0,
    }


def save_journal(j, cfg):
    """
    Атомарная запись: во временный файл, затем подмена.

    Раньше json.dump писал прямо поверх журнала. Остановка процесса или
    пропадание питания посреди записи оставляли обрезанный JSON — и
    вся история сделок форвард-теста терялась. os.replace на одном
    томе атомарен: журнал всегда либо старый целиком, либо новый.
    """
    tmp = cfg["journal"] + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, cfg["journal"])


# ─────────────────────────────────────────────────────────────
#  Единственный экземпляр, остановка по запросу, запрет сна
# ─────────────────────────────────────────────────────────────
def lock_path(cfg):
    return cfg["journal"] + ".lock"


def pid_path(cfg):
    return cfg["journal"] + ".pid"


def stop_path(cfg):
    return cfg["journal"] + ".stop"


def acquire_single_instance(cfg):
    """
    Блокировка на уровне ОС: второй экземпляр того же бота не стартует.

    Два процесса с одним журналом перезаписывают друг другу сделки. Это
    легко получить, запустив бота и через .bat, и из панели. Блокировку
    держит сама ОС и снимает её, когда процесс завершается любым
    способом — даже аварийно, поэтому «зависших» блокировок не бывает.

    Возвращает дескриптор (его нужно держать открытым) или None, если
    бот уже запущен.
    """
    fd = os.open(lock_path(cfg), os.O_RDWR | os.O_CREAT)
    try:
        if os.name == "nt":
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    with open(pid_path(cfg), "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    return fd


def keep_awake(enable):
    """
    Просим Windows не засыпать, пока бот работает.

    Во время форвард-теста компьютер уходил в сон: цикл, занимающий
    секунды, растягивался на 2.5 часа. Это не настройка системы —
    запрос действует только пока жив процесс, экран гаснуть может,
    ручной сон и выключение работают как обычно.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


def sleep_or_stop(cfg, seconds):
    """
    Сон между циклами, прерываемый файлом-флагом остановки.

    Панель останавливает бота, создавая файл .stop. Бот замечает его
    в течение секунды — и выходит между циклами, а не посреди записи
    журнала или обработки позиций. Возвращает True, если пора выходить.
    """
    end = time.time() + seconds
    while time.time() < end:
        if os.path.exists(stop_path(cfg)):
            try:
                os.remove(stop_path(cfg))
            except OSError:
                pass
            return True
        time.sleep(1)
    return False


def get_exchange():
    return ccxt.bitget({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })


# ─────────────────────────────────────────────────────────────
#  Инструменты
# ─────────────────────────────────────────────────────────────
def crypto_bases():
    """
    Белый список базовых активов: то, что листится спотом на Binance.

    Почему именно так. Bitget листит на перпетуалах не только крипту:
    токенизированные акции (SKHYNIX, SAMSUNG, LGELECTRONICS),
    плечевые ETF (KORU, SOXL), драгметаллы (XAU, XAG) и разного рода
    экзотику. Чёрный список тут проигрывает — их сотни и появляются
    новые. Binance спотом акции не листит, поэтому пересечение с ним
    даёт крипту и только крипту, без ручного ведения списков.

    Стратегия проверялась на криптовалютных парах. У акций и металлов
    другая механика, в том числе выходные на базовом рынке, и
    переносить на них результат бэктеста нет оснований.

    Плата за это — отсеются мелкие альты, которых на Binance нет.
    Для стратегии на 4 часах с удержанием в дни это приемлемо:
    ликвидность там всё равно нужна.
    """
    bases = set()
    try:
        bn = ccxt.binance({"enableRateLimit": True})
        for sym, m in bn.load_markets().items():
            if m.get("spot") and m.get("quote") == "USDT" and m.get("active"):
                b = (m.get("base") or "").upper()
                if b:
                    bases.add(b)
    except Exception:
        pass
    return bases


STABLE_BASES = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDD", "USDP",
                "PYUSD", "EURT", "EURS", "USDE", "USDS", "USD1", "RLUSD",
                "XAUT", "PAXG"}

# Сюда можно дописать что угодно руками, не трогая остальной код
EXCLUDE_BASES = set()


# ─────────────────────────────────────────────────────────────
#  Зафиксированный список инструментов
# ─────────────────────────────────────────────────────────────
# Отобран 16.09.2026 по ДВУМ признакам, и оба объективные:
#   оборот на перпетуалах Bitget >= $5М в сутки;
#   не меньше 1800 четырёхчасовых свечей истории (год с лишним),
#   иначе EMA200 и канал не успевают прогреться.
# Плюс базовый актив обязан листиться спотом на Binance — это
# отсекает токенизированные акции и металлы.
#
# Чего в отборе НЕТ намеренно: результатов бэктеста. Выбирать монеты
# по тому, как они отработали на истории, значит подгонять состав
# под ответ, и форвард-тест перестанет что-либо доказывать.
#
# Зачем вообще фиксировать. Динамический список строится по текущим
# оборотам и дрейфует вместе с рынком. А проверка показала, что край
# сильно зависит от состава: на 30 случайных наборах по 40 монет
# медиана на проверочном периоде +0.144R при разбросе 0.105 и
# диапазоне от -0.089R до +0.386R. С плавающим списком форвард-тест
# мерил бы удачу состава пополам со стратегией.
#
# Обновлять список руками и только осознанно:
#   python paper_trading_donchian.py refresh-symbols
# и вставить вывод сюда. Каждое обновление обнуляет чистоту
# накопленной форвард-статистики, так что без нужды не стоит.
SYMBOLS = (
    "BTC/USDT:USDT", "ETH/USDT:USDT", "XRP/USDT:USDT", "SOL/USDT:USDT",
    "ZEC/USDT:USDT", "LSK/USDT:USDT", "SUI/USDT:USDT", "DOGE/USDT:USDT",
    "ARB/USDT:USDT", "PEPE/USDT:USDT", "ENA/USDT:USDT", "UNI/USDT:USDT",
    "ADA/USDT:USDT", "XLM/USDT:USDT", "FIL/USDT:USDT", "LINK/USDT:USDT",
    "NEAR/USDT:USDT", "VTHO/USDT:USDT", "TRUMP/USDT:USDT", "ONDO/USDT:USDT",
    "TAO/USDT:USDT", "BNB/USDT:USDT", "WLD/USDT:USDT", "DOT/USDT:USDT",
    "INJ/USDT:USDT", "PUMP/USDT:USDT", "APT/USDT:USDT", "AVAX/USDT:USDT",
    "BCH/USDT:USDT", "AAVE/USDT:USDT", "LTC/USDT:USDT", "ASTR/USDT:USDT",
    "FET/USDT:USDT", "ETHFI/USDT:USDT", "PENGU/USDT:USDT",
)


def active_symbols(exchange, cfg):
    """
    Зафиксированный список, очищенный от того, что биржа больше
    не торгует. Делистинг монеты не должен ронять бота, но и
    молча подменять состав чем-то новым тоже нельзя — поэтому
    только убираем, никогда не добавляем.
    """
    try:
        markets = exchange.load_markets()
    except Exception as e:
        return [], f"биржа недоступна: {e}"
    live, gone = [], []
    for sym in SYMBOLS:
        m = markets.get(sym)
        (live if (m and m.get('active')) else gone).append(sym)
    return live, (f"больше не торгуются: {', '.join(gone)}" if gone else None)


def discover_symbols(exchange, cfg, allowed=None):
    """
    Динамический подбор по обороту — используется только для
    режима refresh-symbols, в торговле список зафиксирован.
    """
    allowed = allowed if allowed is not None else crypto_bases()
    try:
        markets = exchange.load_markets()
        tickers = exchange.fetch_tickers()
    except Exception as e:
        return [], f"биржа недоступна: {e}"
    if not allowed:
        return [], "не удалось построить список криптоактивов"

    rows = []
    for sym, m in markets.items():
        if not (m.get("swap") and m.get("settle") == "USDT" and m.get("active")):
            continue
        base = (m.get("base") or "").upper()
        if base not in allowed:
            continue
        if base in STABLE_BASES or base in EXCLUDE_BASES:
            continue
        vol = (tickers.get(sym) or {}).get("quoteVolume") or 0
        if vol >= cfg["min_usdt_vol"]:
            rows.append((sym, vol))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in rows[:cfg["max_symbols"]]], None


class DataUnavailable(Exception):
    """Биржа не отдала данные ни по одному инструменту — цикл был бы вслепую."""


def fetch_candles(exchange, symbol, timeframe, limit, errors=None):
    """
    Свечи БЕЗ последней, ещё формирующейся.

    Биржа отдаёт текущую незакрытую свечу последним элементом. Если
    считать по ней сигнал, бот увидит «пробой», которого к закрытию
    может не остаться — это подглядывание в несостоявшееся будущее
    и главный источник расхождения бота с бэктестом.
    """
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not raw or len(raw) < 60:
            return None
        step = TF_MS.get(timeframe, 14_400_000)
        now_ms = exchange.milliseconds()
        if raw and raw[-1][0] + step > now_ms:
            raw = raw[:-1]          # отбрасываем незакрытую
        if len(raw) < 60:
            return None
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high",
                                        "low", "close", "volume"])
        df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.reset_index(drop=True)
    except Exception as e:
        # Раньше ошибка глоталась молча. 18.09 после выхода ноутбука из
        # режима ожидания биржа 7 часов не отвечала, бот в каждом цикле
        # получал пустоту, писал «Баланс=...» как при успехе — и никто
        # об этом не знал. Теперь причина уходит вызывающему коду.
        if errors is not None:
            errors.append(f"{symbol.split('/')[0]}: {type(e).__name__}: {str(e)[:120]}")
        return None


def last_price(exchange, symbol):
    try:
        return exchange.fetch_ticker(symbol)["last"]
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
#  Позиции
# ─────────────────────────────────────────────────────────────
def position_size(balance, entry, stop, cfg):
    """
    Объём по риску, урезанный плечом. Возвращает (qty, notional, note).

    Проверяем минимальный размер заявки: на счёте $50 при риске 1% и
    стопе 6.5% объём выходит около $7.7 — уже близко к биржевому
    минимуму $5, и при просадке легко уйти ниже.
    """
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0 or entry <= 0:
        return 0.0, 0.0, "нулевой риск"
    qty = (balance * cfg["risk_pct"]) / risk_per_unit
    notional = qty * entry
    cap = balance * cfg["max_leverage"]
    note = ""
    if notional > cap:
        qty, notional, note = cap / entry, cap, "урезано плечом"
    if notional < cfg["min_notional"]:
        return 0.0, 0.0, (f"объём ${notional:.2f} меньше минимума "
                          f"${cfg['min_notional']:.2f}")
    return qty, notional, note


def entry_bar_range(exchange, pos, cfg):
    """
    Размах свечи, ВНУТРИ которой бот вошёл, — считая только с момента входа.

    Сигнал приходит по закрытой свече, а вход происходит уже внутри
    следующей. К моменту входа эта свеча успевает прожить до
    max_signal_age_min минут, и её high/low включают движение, которого
    в сделке бота не было. Судить по ним нельзя: бот зафиксировал бы
    стоп по проколу, случившемуся до его входа (или тейк — так же зря).

    Поэтому для этой одной свечи берём минутные свечи с момента входа
    и считаем размах по ним. Считаем один раз: свеча уже закрыта,
    пересчитывать её каждый цикл незачем.

    Вернуть None значит «уточнить не вышло» — тогда вызывающий код
    работает по исходной свече, как раньше.
    """
    if "entry_bar" in pos:
        return pos["entry_bar"]

    step = TF_MS.get(cfg["timeframe"], 14_400_000)
    bar_ts = pos["entry_ts"] + step              # свеча, в которой вошли
    if exchange.milliseconds() < bar_ts + step:
        return None                              # ещё формируется

    wall = pos.get("entry_wall_ms")
    if wall is None:                             # позиции до этой правки
        dt = _safe_dt(pos.get("opened", ""))
        wall = int(dt.timestamp() * 1000) if dt else None
    if wall is None or wall <= bar_ts:
        return None                              # вход в самом начале свечи

    end = bar_ts + step
    mins, since = [], wall
    for _ in range(3):                           # 1м-страницы по 200 штук
        try:
            page = exchange.fetch_ohlcv(pos["symbol"], "1m", since=since, limit=200)
        except Exception:
            return None
        if not page:
            break
        mins += [r for r in page if wall <= r[0] < end]
        last = page[-1][0]
        if last + 60_000 >= end:
            break
        since = last + 60_000
    if not mins or max(r[0] for r in mins) + 60_000 < end:
        return None                              # данных на всю свечу нет

    pos["entry_bar"] = {"ts": bar_ts,
                        "high": max(r[2] for r in mins),
                        "low": min(r[3] for r in mins)}
    return pos["entry_bar"]


def check_exit(pos, df, entry_bar=None):
    """
    Задет ли стоп или тейк свечами, закрывшимися ПОСЛЕ входа.

    Смотрим high/low свечей, а не текущую цену: прежние боты
    опрашивали цену раз в N минут и пропускали касания между
    опросами. Здесь проверка совпадает с бэктестом свеча в свечу.

    Оба уровня задеты одной свечой — считаем стопом: порядок
    движения цены внутри свечи неизвестен.

    entry_bar — уточнённый размах свечи входа (см. entry_bar_range).
    """
    seg = df[df.timestamp > pos["entry_ts"]]
    if seg.empty:
        return None, None, 0
    d, stop, take = pos["dir"], pos["stop"], pos["take"]
    for _, bar in seg.iterrows():
        high, low = bar.high, bar.low
        if entry_bar and int(bar.timestamp) == entry_bar["ts"]:
            high, low = entry_bar["high"], entry_bar["low"]
        if d == 1:
            hit_stop, hit_take = low <= stop, high >= take
        else:
            hit_stop, hit_take = high >= stop, low <= take
        if hit_stop:
            return "stop", int(bar.timestamp), len(seg)
        if hit_take:
            return "take", int(bar.timestamp), len(seg)
    return None, None, len(seg)


def exit_price(pos, reason, cfg, market_price=None):
    """Тейк — лимиткой по уровню, остальное рыночно и с проскальзыванием."""
    if reason == "take":
        return pos["take"]
    level = pos["stop"] if reason == "stop" else market_price
    slip = level * cfg["slippage"]
    return level - slip if pos["dir"] == 1 else level + slip


def close_position(journal, pos, reason, price, ts, cfg):
    d = pos["dir"]
    gross = (price - pos["entry"]) * pos["qty"] * d
    fees = (pos["notional"] + price * pos["qty"]) * cfg["commission"]
    pnl = gross - fees
    journal["balance"] = round(journal["balance"] + pnl, 6)

    journal["trades"].append({
        "symbol": pos["symbol"], "dir": d, "type": pos["type"],
        "entry": pos["entry"], "exit": round(price, 8),
        "stop": pos["stop"], "take": pos["take"],
        "qty": pos["qty"], "notional": pos["notional"],
        "atr": pos.get("atr"), "stop_pct": pos.get("stop_pct"),
        "result": "WIN" if pnl > 0 else "LOSS",
        "exit_reason": reason,
        "pnl": round(pnl, 6), "fees": round(fees, 6),
        "r_multiple": round(pnl / (abs(pos["entry"] - pos["stop"]) * pos["qty"]), 3)
                      if pos["qty"] > 0 else 0,
        "balance": round(journal["balance"], 4),
        "opened": pos["opened"], "closed": datetime.now().isoformat(),
        "entry_ts": pos["entry_ts"], "exit_ts": ts,
        "bars_held": pos.get("bars_held", 0),
        # Уточнялся ли размах свечи входа по минуткам. Нужно проверке
        # журналов: выход на самой свече входа без уточнения — повод
        # не доверять сделке.
        "entry_bar_checked": "entry_bar" in pos,
    })
    if pnl <= 0:
        unblock = datetime.now() + timedelta(hours=cfg.get("cooldown_hours", 8))
        journal.setdefault("cooldown", {})[pos["symbol"]] = unblock.isoformat()

    em = "🟢 WIN " if pnl > 0 else "🔴 LOSS"
    log(f"{em} {pos['symbol']} {'ЛОНГ' if d == 1 else 'ШОРТ'} "
        f"[{reason}] вход={pos['entry']} выход={round(price, 6)} "
        f"PnL=${pnl:+.2f} ({pnl / (abs(pos['entry'] - pos['stop']) * pos['qty']):+.2f}R) "
        f"баланс=${journal['balance']:.2f}", cfg)


# ─────────────────────────────────────────────────────────────
#  Цикл
# ─────────────────────────────────────────────────────────────
def strategy_params(cfg):
    name = cfg.get("strategy", "donchian")
    return {k: cfg[k] for k in STRATEGY_KEYS[name]}


def describe_rules(cfg):
    """Правила одной строкой — для лога и панели."""
    name = cfg.get("strategy", "donchian")
    if name == "supertrend":
        entry = f"Supertrend ×{cfg['mult']}"
    else:
        entry = f"Дончиан {cfg['channel']} свечей"
    return (f"{entry}, стоп {cfg['atr_mult']} ATR, тейк {cfg['rr']}R, "
            f"фильтр EMA{cfg['ema']}, ТФ {cfg['timeframe']}")


def prune_acted(journal, days=10):
    """Отметки об отработанных сигналах старше N дней не нужны."""
    acted = journal.get("acted", {})
    cutoff = datetime.now() - timedelta(days=days)
    for k in [k for k, v in acted.items()
              if _safe_dt(v) and _safe_dt(v) < cutoff]:
        del acted[k]


def _safe_dt(iso):
    try:
        return datetime.fromisoformat(iso)
    except Exception:
        return None


def run_cycle(exchange, journal, cfg, symbols):
    journal["cycles"] = journal.get("cycles", 0) + 1
    prune_acted(journal)
    p = strategy_params(cfg)

    # ── 1. Открытые позиции ───────────────────────────────────
    still_open, blind = [], []
    for pos in journal["open"]:
        df = fetch_candles(exchange, pos["symbol"], cfg["timeframe"], cfg["candles"], blind)
        time.sleep(0.1)
        if df is None:
            still_open.append(pos)
            continue

        reason, ts, bars = check_exit(pos, df, entry_bar_range(exchange, pos, cfg))
        pos["bars_held"] = bars
        if reason:
            close_position(journal, pos, reason,
                           exit_price(pos, reason, cfg), ts, cfg)
            continue
        if bars >= cfg["max_hold_bars"]:
            price = last_price(exchange, pos["symbol"]) or df.close.iloc[-1]
            close_position(journal, pos, "time",
                           exit_price(pos, "time", cfg, price),
                           int(df.timestamp.iloc[-1]), cfg)
            continue
        still_open.append(pos)
    journal["open"] = still_open

    # Стоп, пропущенный в этом цикле, не потерян: check_exit смотрит все
    # свечи с момента входа и найдёт касание позже, по той же цене. Но
    # если не проверена НИ ОДНА позиция, цикл прошёл вслепую, и это
    # должно быть видно как ошибка, а не как «Баланс=...».
    if blind and len(blind) == len(journal["open"]):
        raise DataUnavailable(f"биржа не отдала свечи ни по одной из "
                              f"{len(blind)} позиций, стопы не проверены — {blind[0]}")
    if blind:
        log(f"[!] Нет свечей по {len(blind)} поз., проверю в следующем цикле: "
            + "; ".join(blind[:3]), cfg)

    # ── 2. Хватит ли денег ────────────────────────────────────
    if journal["balance"] < cfg["deposit"] * cfg["risk_pct"]:
        log(f"🛑 ДЕПОЗИТ СЛИТ: ${journal['balance']:.2f}. Новых входов нет, "
            f"открытые позиции доводим.", cfg)
        return 0

    if len(journal["open"]) >= cfg["max_open"]:
        log(f"Лимит позиций занят ({len(journal['open'])}/{cfg['max_open']})", cfg)
        return 0

    # ── 3. Поиск сигналов ─────────────────────────────────────
    busy = {(o["symbol"], o["dir"]) for o in journal["open"]}
    on_cooldown = set()
    for sym, until in journal.get("cooldown", {}).items():
        try:
            if datetime.now() < datetime.fromisoformat(until):
                on_cooldown.add(sym)
        except Exception:
            pass

    opened, scanned, missed = 0, 0, []
    for sym in symbols:
        if len(journal["open"]) >= cfg["max_open"]:
            break
        if sym in on_cooldown:
            continue

        scanned += 1
        df = fetch_candles(exchange, sym, cfg["timeframe"], cfg["candles"], missed)
        time.sleep(0.1)
        if df is None or len(df) < cfg["ema"] + 30:
            continue

        # Тот же код сигналов, что и в бэктесте
        prepared = STRATEGY_FUNCS[cfg.get("strategy", "donchian")](df, p)
        i = len(prepared) - 1            # последняя ЗАКРЫТАЯ свеча
        sigs = strategies.signal_fn(prepared, i, cfg, None, None, None)
        if not sigs:
            continue
        sig = sigs[0]

        # Возраст сигнала: сколько прошло с ЗАКРЫТИЯ сигнальной свечи
        signal_ts = int(prepared.timestamp.iloc[i])
        closed_ms = signal_ts + TF_MS.get(cfg["timeframe"], 14_400_000)
        age_min = (exchange.milliseconds() - closed_ms) / 60000
        if age_min > cfg["max_signal_age_min"]:
            continue                     # устарел — бэктест так не входит

        # Один сигнал — один вход. Без этого перезапуск бота после
        # закрытия позиции снова открыл бы её по той же свече.
        acted_key = f"{sym}|{signal_ts}"
        if acted_key in journal.setdefault("acted", {}):
            continue

        if (sym, sig["dir"]) in busy:
            continue

        price = last_price(exchange, sym)
        if price is None:
            continue
        # Вход рыночный — в бэктесте это открытие следующей свечи
        entry = price * (1 + cfg["slippage"] * sig["dir"])

        # Стоп и тейк пересчитываем от фактического входа, сохраняя
        # ту же дистанцию в ATR: иначе реальный риск разъедется с
        # заложенным в сигнал.
        atr_val = float(prepared["atr"].iloc[i])
        risk = atr_val * cfg["atr_mult"]
        stop = entry - risk if sig["dir"] == 1 else entry + risk
        take = entry + risk * cfg["rr"] if sig["dir"] == 1 else entry - risk * cfg["rr"]

        qty, notional, note = position_size(journal["balance"], entry, stop, cfg)
        if qty <= 0:
            log(f"⏭️  {sym} пропуск: {note}", cfg)
            continue

        pos = {
            "symbol": sym, "dir": sig["dir"], "type": sig["type"],
            "entry": round(entry, 8), "stop": round(stop, 8),
            "take": round(take, 8), "qty": round(qty, 8),
            "notional": round(notional, 4),
            "atr": round(atr_val, 8),
            "stop_pct": round(risk / entry, 5),
            "opened": datetime.now().isoformat(),
            "entry_ts": signal_ts,
            # Момент входа по часам биржи: по нему отсекается движение
            # свечи входа, случившееся до нас (см. entry_bar_range)
            "entry_wall_ms": exchange.milliseconds(),
            "signal_age_min": round(age_min, 1),
            "bars_held": 0,
        }
        journal["open"].append(pos)
        journal["acted"][acted_key] = datetime.now().isoformat()
        busy.add((sym, sig["dir"]))
        opened += 1
        log(f"✅ ОТКРЫТА {sym} {'ЛОНГ' if sig['dir'] == 1 else 'ШОРТ'} "
            f"вход={round(entry, 6)} стоп={round(stop, 6)} ({risk/entry:.2%}) "
            f"тейк={round(take, 6)} объём=${notional:.2f} "
            f"риск=${journal['balance']*cfg['risk_pct']:.2f}"
            + (f" [{note}]" if note else ""), cfg)

    if scanned and len(missed) == scanned:
        raise DataUnavailable(f"биржа не отдала свечи ни по одному из {scanned} "
                              f"инструментов, сигналы не проверены — {missed[0]}")
    return opened


# ─────────────────────────────────────────────────────────────
#  Статистика
# ─────────────────────────────────────────────────────────────
def print_stats(journal, cfg):
    t = journal["trades"]
    bal, dep = journal["balance"], journal["deposit"]
    print("=" * 62)
    print(f"  {cfg.get('bot_name', 'Бот')} — Bitget перпетуалы")
    print("=" * 62)
    print(f"  Старт: {journal['created'][:16]}   циклов: {journal.get('cycles', 0)}")
    print(f"  Депозит ${dep:.2f} → баланс ${bal:.2f}  "
          f"({bal - dep:+.2f}$ / {(bal/dep - 1)*100:+.1f}%)")

    if not t:
        print("\n  Закрытых сделок пока нет.")
    else:
        df = pd.DataFrame(t)
        wins = df[df.result == "WIN"]
        wr = len(wins) / len(df) * 100
        eq = [dep] + df.balance.tolist()
        peak = np.maximum.accumulate(eq)
        dd = ((np.array(eq) - peak) / peak * 100).min()
        print(f"\n  Сделок: {len(df)}   WR: {wr:.1f}%   "
              f"среднее: {df.r_multiple.mean():+.3f}R")
        print(f"  Макс. просадка: {dd:.1f}%   комиссии: ${df.fees.sum():.2f}")
        e = cfg.get("expect")
        if e:
            print(f"  Ожидание по бэктесту: WR ~{e['wr']}%, "
                  f"{e['r_lo']:+.2f}R … {e['r_hi']:+.2f}R на сделку")
        print("\n  По причине выхода:")
        for r, g in df.groupby("exit_reason"):
            print(f"    {r:<6} {len(g):>4} шт   среднее {g.r_multiple.mean():+.3f}R")
        print("\n  Последние 8 сделок:")
        for x in t[-8:]:
            em = "🟢" if x["result"] == "WIN" else "🔴"
            print(f"    {em} {x['symbol']:<20} {'ЛОНГ' if x['dir']==1 else 'ШОРТ':<5} "
                  f"{x['exit_reason']:<5} {x['r_multiple']:>+6.2f}R  ${x['balance']:.2f}")

    if journal["open"]:
        print(f"\n  Открытые позиции ({len(journal['open'])}/{cfg['max_open']}):")
        for o in journal["open"]:
            print(f"    {o['symbol']:<20} {'ЛОНГ' if o['dir']==1 else 'ШОРТ':<5} "
                  f"вход={o['entry']} стоп={o['stop']} тейк={o['take']} "
                  f"({o.get('bars_held',0)} свечей)")
    print("=" * 62)


# ─────────────────────────────────────────────────────────────
#  Точка входа
# ─────────────────────────────────────────────────────────────
def main(cfg=None):
    cfg = dict(cfg or CONFIG)
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"

    if mode == "reset":
        # Сброс на ходу ничего бы не дал: работающий бот держит журнал
        # в памяти и в ближайшем цикле записал бы его обратно. Хуже
        # того, человек считал бы, что история обнулена.
        probe = acquire_single_instance(cfg)
        if probe is None:
            print(f"[!] {cfg.get('bot_name', 'Бот')} сейчас работает. "
                  f"Сначала остановите его, потом сбрасывайте журнал.")
            return
        for f in (cfg["journal"], cfg["logfile"]):
            if os.path.exists(f):
                os.remove(f)
        print("[+] Журнал сброшен.")
        return

    journal = load_journal(cfg)

    if mode == "status":
        print_stats(journal, cfg)
        return

    ex = get_exchange()

    if mode == "symbols":
        syms, warn = active_symbols(ex, cfg)
        print(f"Зафиксированный список: {len(SYMBOLS)}, "
              f"из них торгуются сейчас: {len(syms)}")
        if warn:
            print(f"[!] {warn}")
        for x in syms:
            print("  ", x)
        return

    if mode == "refresh-symbols":
        # Подбирает список заново по текущим оборотам и печатает
        # его для ручной вставки. Сам ничего не меняет: подмена
        # состава на ходу обнулила бы чистоту форвард-статистики.
        fresh, err = discover_symbols(ex, cfg)
        if err:
            print(f"[!] {err}")
            return
        added = [x for x in fresh if x not in SYMBOLS]
        dropped = [x for x in SYMBOLS if x not in fresh]
        print(f"По текущим оборотам подошло бы {len(fresh)} инструментов.")
        print(f"  новых: {len(added)}   выпало: {len(dropped)}")
        if added:
            print("  добавились бы:", ", ".join(added))
        if dropped:
            print("  выпали бы:   ", ", ".join(dropped))
        print('\n' + "SYMBOLS = (")
        for x in fresh:
            print(f'    "{x}",')
        print(")")
        print('\n' + "Вставлять в код только осознанно: каждое обновление")
        print("обнуляет чистоту накопленной форвард-статистики.")
        return

    lock = acquire_single_instance(cfg)
    if lock is None:
        msg = (f"{cfg.get('bot_name', 'Бот')} уже запущен — второй экземпляр "
               f"не стартует, иначе два процесса испортят общий журнал.")
        print(f"[!] {msg}")
        log(f"[!] {msg}", cfg, show=False)
        return
    # Старый флаг остановки, оставшийся от прошлого запуска, не должен
    # погасить бота сразу после старта
    if os.path.exists(stop_path(cfg)):
        os.remove(stop_path(cfg))
    keep_awake(True)

    log("=" * 56, cfg, show=False)
    log(f"СТАРТ  депозит=${cfg['deposit']}  риск={cfg['risk_pct']:.0%}  "
        f"макс.позиций={cfg['max_open']}  плечо<={cfg['max_leverage']}x", cfg)
    log(f"Правила: {describe_rules(cfg)}", cfg)

    # Паспорт бота в журнале. Панель читает его отсюда, а не импортирует
    # код бота: иначе ей пришлось бы грузить pandas и ccxt (~110 МБ)
    # только ради того, чтобы узнать правила и размер риска.
    journal["meta"] = {
        "bot_id": cfg.get("bot_id"), "bot_name": cfg.get("bot_name"),
        "strategy": cfg.get("strategy"), "rules": describe_rules(cfg),
        "risk_pct": cfg["risk_pct"], "max_open": cfg["max_open"],
        "deposit": cfg["deposit"], "timeframe": cfg["timeframe"],
        "expect": cfg.get("expect"),
        "symbols": len(SYMBOLS),
    }
    save_journal(journal, cfg)

    symbols, warn = active_symbols(ex, cfg)
    if not symbols:
        log("[!] Не удалось получить список инструментов", cfg)
        return
    if warn:
        log(f"[!] {warn}", cfg)
    log(f"Список зафиксирован: {len(symbols)} инструментов — "
        + ", ".join(x.split("/")[0] for x in symbols[:12])
        + (" ..." if len(symbols) > 12 else ""), cfg)

    fails = 0
    while True:
        try:
            log(f"--- Цикл #{journal.get('cycles', 0) + 1} ---", cfg)
            opened = run_cycle(ex, journal, cfg, symbols)
            save_journal(journal, cfg)
            fails = 0

            t = journal["trades"]
            wr = (sum(1 for x in t if x["result"] == "WIN") / len(t) * 100) if t else 0
            avg_r = (sum(x["r_multiple"] for x in t) / len(t)) if t else 0
            log(f"Баланс=${journal['balance']:.2f}  сделок={len(t)}  "
                f"WR={wr:.1f}%  среднее={avg_r:+.3f}R  "
                f"открыто={len(journal['open'])}  новых={opened}", cfg)
            log(f"Следующий цикл через {cfg['scan_interval_min']} мин", cfg, show=False)
            if sleep_or_stop(cfg, cfg["scan_interval_min"] * 60):
                log("Остановлен из панели", cfg)
                break

        except KeyboardInterrupt:
            log("Остановлен пользователем", cfg)
            print_stats(journal, cfg)
            break
        except Exception as e:
            fails += 1
            log(f"ОШИБКА: {type(e).__name__}: {e} — повтор через 5 мин", cfg)
            # Закрытия, уже сделанные в этом цикле до ошибки, не теряем
            try:
                save_journal(journal, cfg)
            except Exception:
                pass
            # После сна ноутбука подключение к бирже может остаться
            # полумёртвым. Три ошибки подряд — открываем его заново.
            if fails >= 3:
                log("Три ошибки подряд — пересоздаю подключение к бирже", cfg)
                ex = get_exchange()
                fails = 0
            if sleep_or_stop(cfg, 300):
                log("Остановлен из панели", cfg)
                break

    keep_awake(False)


if __name__ == "__main__":
    main()
