"""
BotDemo/demo_trading.py
=======================
Демо-счёт Bitget: обе стратегии, настоящие заявки, вымышленные деньги.

Зачем, если бумажные боты уже работают. Бумажный бот симулирует
исполнение сам. Демо проверяет то, что симуляция знать не может:
реальные заполнения и проскальзывание, срабатывание стопов на бирже,
funding (здесь он не оценивается, а приходит от биржи в каждой
закрытой позиции) и сам код постановки заявок.

Сигналы — те же, что у бумажных ботов: по БОЕВЫМ свечам, тем же кодом.
Поэтому демо и бумага на общих монетах торгуют одинаковые сделки, и
расхождение между ними — это ровно цена реального исполнения.

── Какое это демо ─────────────────────────────────────────────
У Bitget два демо. Старое — контракты SBTC/SETH/SXRP, всего три монеты.
Новое — обычные символы (BTC/USDT:USDT), отдельные демо-ключи и
заголовок PAPTRADING: 1 в каждом запросе. В новом 45 контрактов, из
наших 35 монет доступны 15. Используется новое.

── ПРЕДОХРАНИТЕЛЬ ─────────────────────────────────────────────
Символы в новом демо ТЕ ЖЕ, что на реальном счёте. Демо от реальных
денег отделяют только ключ и заголовок. Поэтому:

1. Клиент биржи — подкласс DemoBitget, который добавляет заголовок
   PAPTRADING к КАЖДОМУ запросу на уровне подписи. Отправить запрос
   без него из этого файла невозможно конструктивно, а не «маловероятно».

2. При каждом старте ключ проверяется на бирже ДВУМЯ запросами баланса
   (только чтение):
     без заголовка — обязан получить ОТКАЗ. Если запрос прошёл, ключ
       работает на реальном счёте, и бот отказывается стартовать;
     с заголовком — обязан пройти.
   Сетевая ошибка на первом шаге — это «проверить не удалось», и бот
   тоже не стартует. Правило одно: сомнение трактуется в сторону отказа.

3. Каждая заявка проходит через place_order, который ещё раз проверяет
   тип клиента, флаг песочницы и то, что монета из разрешённого списка.

── Одна позиция на монету ─────────────────────────────────────
На фьючерсах Bitget в одностороннем режиме по монете может быть только
одна позиция. Если Дончиан держит лонг, а Supertrend откроет шорт по той
же монете, биржа не откроет вторую позицию, а закроет первую; два лонга
сольются в одну со средней ценой и перепутанными стопами. Поэтому монету
держит одна стратегия, пока позиция не закроется. Если обе дали сигнал
на одной свече, приоритет у Дончиана — за весь период у него вдвое
больше край на сделку.

Ключи — только из .env в корне проекта (см. .env.example). Нужны права
чтение + торговля. Вывод средств боту не нужен ни для чего.

Запуск:
  python demo_trading.py check    — проверить ключ и счёт, без заявок
  python demo_trading.py          — торговля
  python demo_trading.py status   — позиции и история
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta

import ccxt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Bot5_Donchian"))
sys.path.insert(0, os.path.join(ROOT, "Bot6_Supertrend"))

try:
    if sys.stdout is not None and sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import paper_trading_donchian as core          # noqa: E402
import paper_trading_supertrend as st_bot      # noqa: E402

# Правила стратегий берутся у бумажных ботов — одни и те же
STRATEGIES = {"donchian": core.CONFIG, "supertrend": st_bot.CONFIG}
PRIORITY = ("donchian", "supertrend")
STRATEGY_RU = {"donchian": "Дончиан", "supertrend": "Supertrend"}

CONFIG = {
    "bot_id":          "demo",
    "bot_name":        "Демо Bitget — обе стратегии",
    "allocation":      {"donchian": 0.5, "supertrend": 0.5},
    "risk_pct":        0.05,     # от доли счёта стратегии — как у бумажных ботов
    "max_open":        5,        # на каждую стратегию
    "max_notional":    3.0,      # объём позиции не больше 3x доли стратегии
    "leverage":        5,        # биржевое плечо: стоп 6.5%, ликвидация ~20% — стоп раньше
    "margin_mode":     "isolated",
    "margin_buffer":   1.15,     # свободной маржи нужно на 15% больше расчётной
    "timeframe":       "4h",
    "candles":         400,
    "scan_interval_min": 20,
    "max_signal_age_min": 30,
    "max_hold_bars":   200,
    "product_type":    "USDT-FUTURES",
    "journal":         os.path.join(HERE, "demo_journal.json"),
    "logfile":         os.path.join(HERE, "demo_log.txt"),
}


# ─────────────────────────────────────────────────────────────
#  Ключи
# ─────────────────────────────────────────────────────────────
ENV_FILE = os.path.join(ROOT, ".env")
ENV_NAMES = ("BITGET_DEMO_API_KEY", "BITGET_DEMO_API_SECRET", "BITGET_DEMO_API_PASSPHRASE")


def read_env():
    """Значения из .env в корне проекта; окружение процесса имеет приоритет."""
    vals = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip().strip('"').strip("'")
    return {n: os.environ.get(n) or vals.get(n, "") for n in ENV_NAMES}


def keys_configured():
    return all(read_env().values())


# ─────────────────────────────────────────────────────────────
#  Предохранитель
# ─────────────────────────────────────────────────────────────
class NotDemoError(RuntimeError):
    """Есть сомнение, что работа идёт с демо-счётом. Торговля запрещена."""


class DemoBitget(ccxt.bitget):
    """
    Клиент Bitget, который физически не умеет ходить на реальный счёт:
    заголовок PAPTRADING добавляется при подписи КАЖДОГО запроса.

    ccxt сам не добавляет его к запросу времени и к старым S-контрактам;
    первое нам не нужно для торговли, вторые мы не используем.
    """

    def sign(self, path, api="public", method="GET", params={}, headers=None, body=None):
        req = super().sign(path, api, method, params, headers, body)
        if path not in ("v2/public/time",):
            h = dict(req.get("headers") or {})
            h["PAPTRADING"] = "1"
            req["headers"] = h
        return req


def make_demo_client(keys=None):
    keys = keys or read_env()
    ex = DemoBitget({
        "apiKey": keys["BITGET_DEMO_API_KEY"],
        "secret": keys["BITGET_DEMO_API_SECRET"],
        "password": keys["BITGET_DEMO_API_PASSPHRASE"],
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    ex.set_sandbox_mode(True)
    return ex


def verify_demo_key(log=print):
    """
    Доказать на бирже, что ключ демонстрационный. Возвращает клиент и
    баланс или бросает NotDemoError. Только запросы баланса — ни одной
    заявки.
    """
    keys = read_env()
    missing = [n for n, v in keys.items() if not v]
    if missing:
        raise NotDemoError("не заданы ключи в .env: " + ", ".join(missing))

    # 1. БЕЗ заголовка — демо-ключ реальный счёт обязан отвергнуть
    plain = ccxt.bitget({
        "apiKey": keys["BITGET_DEMO_API_KEY"],
        "secret": keys["BITGET_DEMO_API_SECRET"],
        "password": keys["BITGET_DEMO_API_PASSPHRASE"],
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    try:
        plain.fetch_balance({"productType": CONFIG["product_type"]})
    except ccxt.NetworkError as e:
        raise NotDemoError(
            f"проверить ключ не удалось — сетевая ошибка ({type(e).__name__}). "
            f"Без проверки торговля не начинается, повторите позже.")
    except ccxt.BaseError as e:
        log(f"✔ Реальный счёт ключ не принял — так и должно быть "
            f"({type(e).__name__}: {str(e)[:120]})")
    else:
        raise NotDemoError(
            "КЛЮЧ РАБОТАЕТ НА РЕАЛЬНОМ СЧЁТЕ. Это не демо-ключ. Торговля "
            "не начата, ни одной заявки не отправлено. Удалите этот ключ из "
            ".env и создайте ключ в режиме «Демо-торговля».")

    # 2. С заголовком — демо обязано принять
    demo = make_demo_client(keys)
    try:
        bal = demo.fetch_balance({"productType": CONFIG["product_type"]})
    except ccxt.AuthenticationError as e:
        raise NotDemoError(f"демо-счёт отверг ключ: {str(e)[:200]}. "
                           f"Проверьте, что ключ создан в режиме демо и пароль (passphrase) верный.")
    except ccxt.BaseError as e:
        raise NotDemoError(f"демо-счёт недоступен: {type(e).__name__}: {str(e)[:200]}")
    usdt = bal.get("USDT") or {}
    total = float(usdt.get("total") or 0)
    free = float(usdt.get("free") or 0)
    log(f"✔ Демо-счёт принял ключ: всего {total:.2f} USDT, свободно {free:.2f}")
    return demo, total, free


def place_order(ex, symbol, side, amount, params, allowed):
    """Единственный путь к заявке. Любое сомнение — отказ."""
    if not isinstance(ex, DemoBitget) or not ex.options.get("sandboxMode"):
        raise NotDemoError("заявка через клиент без режима демо — запрещено")
    if symbol not in allowed:
        raise NotDemoError(f"{symbol} не в списке разрешённых демо-монет")
    return ex.create_order(symbol, "market", side, amount, None, params)


# ─────────────────────────────────────────────────────────────
#  Журнал и лог (общие помощники ядра)
# ─────────────────────────────────────────────────────────────
def log(msg, cfg=CONFIG, show=True):
    core.log(msg, cfg, show)


def load_journal(cfg=CONFIG):
    if os.path.exists(cfg["journal"]):
        with open(cfg["journal"], "r", encoding="utf-8") as f:
            return json.load(f)
    return {"created": datetime.now().isoformat(), "start_equity": None,
            "equity": None, "available": None, "open": [], "trades": [],
            "acted": {}, "cycles": 0, "foreign": [], "last_error": None,
            "demo_symbols": []}


# ─────────────────────────────────────────────────────────────
#  Цикл
# ─────────────────────────────────────────────────────────────
def demo_symbols(ex):
    """Зафиксированный список ∩ то, что есть в демо. Только убираем, не добавляем."""
    markets = ex.load_markets()
    return [s for s in core.SYMBOLS
            if (markets.get(s) or {}).get("active") and markets[s].get("swap")]


def fetch_live_positions(ex):
    out = {}
    for p in ex.fetch_positions(None, {"productType": CONFIG["product_type"]}):
        if float(p.get("contracts") or 0) > 0:
            out[p["symbol"]] = p
    return out


MAX_SETTLE_TRIES = 4


def settle_closed(ex, journal, pos, cfg):
    """
    Позиция пропала с биржи — берём её итог из истории позиций.

    Итог берём только у биржи и никогда не додумываем. Запись в
    истории появляется с задержкой в несколько секунд, а иногда
    запрос просто не проходит; если в этот момент записать сделку,
    в журнал попадёт нулевой PnL с меткой «убыток» — и статистика
    будет врать, не подавая виду.

    Поэтому: нет данных — сделку не закрываем, пробуем в следующем
    цикле. После MAX_SETTLE_TRIES попыток закрываем с честной
    пометкой «итог неизвестен», чтобы позиция не висела вечно.

    Возвращает True, если сделка записана, и False, если надо
    попробовать ещё раз позже.
    """
    opened_ms = int(datetime.fromisoformat(pos["opened"]).timestamp() * 1000)
    rec = None
    try:
        hist = ex.fetch_positions_history([pos["symbol"]], opened_ms - 60_000, 20,
                                          {"productType": cfg["product_type"]})
        side = "long" if pos["dir"] == 1 else "short"
        cands = [h for h in hist if (h.get("info") or {}).get("holdSide") == side
                 and int((h.get("info") or {}).get("utime") or 0) >= opened_ms]
        rec = max(cands, key=lambda h: int(h["info"]["utime"])) if cands else None
    except Exception as e:
        log(f"[!] История позиций {pos['symbol']}: {type(e).__name__}: {str(e)[:120]}", cfg)

    info = (rec or {}).get("info") or {}
    net = float(info["netProfit"]) if info.get("netProfit") not in (None, "") else None
    funding = float(info["totalFunding"]) if info.get("totalFunding") not in (None, "") else None
    close_px = float(info["closeAvgPrice"]) if info.get("closeAvgPrice") not in (None, "") else None

    reason = "другое"
    if close_px:
        d_stop = abs(close_px - pos["stop"]) / pos["entry"]
        d_take = abs(close_px - pos["take"]) / pos["entry"]
        reason = "stop" if d_stop < d_take else "take"
        if min(d_stop, d_take) > 0.01:
            reason = "time"
    r = (net / pos["risk_usd"]) if (net is not None and pos.get("risk_usd")) else None

    if net is None:
        pos["settle_tries"] = pos.get("settle_tries", 0) + 1
        if pos["settle_tries"] < MAX_SETTLE_TRIES:
            # На бирже позиции уже нет: показывать по ней «в плюсе
            # столько-то» было бы враньём
            pos["closing"] = True
            pos["unrealized"] = None
            log(f"[!] {pos['symbol']}: биржа ещё не отдала итог закрытой "
                f"позиции, попытка {pos['settle_tries']} из "
                f"{MAX_SETTLE_TRIES} — ждём следующего цикла", cfg)
            return False
        log(f"[!] {pos['symbol']}: итог закрытой позиции получить не "
            f"удалось за {MAX_SETTLE_TRIES} попытки. Записываю сделку "
            f"с пометкой «итог неизвестен» — в статистику она не идёт.", cfg)

    journal["trades"].append({
        **pos, "exit": close_px, "pnl": net, "funding": funding,
        "r_multiple": round(r, 3) if r is not None else None,
        "result": ("WIN" if net > 0 else "LOSS") if net is not None else "UNKNOWN",
        "exit_reason": reason if net is not None else "неизвестно",
        "closed": datetime.now().isoformat(),
    })
    em = "🟢 WIN " if (net or 0) > 0 else "🔴 LOSS"
    log(f"{em} [{STRATEGY_RU[pos['strategy']]}] {pos['symbol']} "
        f"{'ЛОНГ' if pos['dir'] == 1 else 'ШОРТ'} [{reason}] "
        f"итог {net if net is not None else '?'} USDT"
        + (f" ({r:+.2f}R)" if r is not None else "")
        + (f", funding {funding:+.4f}" if funding else ""), cfg)
    return True


def run_cycle(ex, journal, cfg, symbols):
    journal["cycles"] = journal.get("cycles", 0) + 1
    core.prune_acted(journal)

    # ── 1. Сверка с биржей: она источник истины ────────────────
    live = fetch_live_positions(ex)
    still = []
    for pos in journal["open"]:
        if pos["symbol"] in live:
            lp = live[pos["symbol"]]
            pos["mark"] = lp.get("markPrice")
            pos["unrealized"] = lp.get("unrealizedPnl")
            still.append(pos)
        elif not settle_closed(ex, journal, pos, cfg):
            still.append(pos)          # итога ещё нет, вернёмся в след. цикле
    journal["open"] = still

    ours = {p["symbol"] for p in journal["open"]}
    foreign = sorted(set(live) - ours)
    if foreign != journal.get("foreign"):
        if foreign:
            log(f"[!] На демо-счёте есть чужие позиции (открыты не ботом): "
                f"{', '.join(foreign)} — эти монеты бот не трогает", cfg)
        journal["foreign"] = foreign

    # ── 2. Выход по времени ────────────────────────────────────
    max_age = timedelta(hours=4 * cfg["max_hold_bars"])
    for pos in list(journal["open"]):
        if datetime.now() - datetime.fromisoformat(pos["opened"]) < max_age:
            continue
        side = "sell" if pos["dir"] == 1 else "buy"
        try:
            place_order(ex, pos["symbol"], side, pos["qty"],
                        {"reduceOnly": True, "marginMode": cfg["margin_mode"]}, symbols)
            log(f"⏱️  [{STRATEGY_RU[pos['strategy']]}] {pos['symbol']} закрыта по времени "
                f"({cfg['max_hold_bars']} свечей)", cfg)
        except NotDemoError:
            raise
        except Exception as e:
            log(f"[!] Не удалось закрыть {pos['symbol']} по времени: {str(e)[:150]}", cfg)

    # ── 3. Счёт ────────────────────────────────────────────────
    bal = ex.fetch_balance({"productType": cfg["product_type"]})
    usdt = bal.get("USDT") or {}
    equity = float(usdt.get("total") or 0)
    available = float(usdt.get("free") or 0)
    journal["equity"], journal["available"] = equity, available
    if journal.get("start_equity") is None:
        journal["start_equity"] = equity

    # ── 4. Сигналы ─────────────────────────────────────────────
    occupied = ours | set(foreign)
    opened = 0
    for strat in PRIORITY:
        scfg = dict(STRATEGIES[strat], timeframe=cfg["timeframe"], candles=cfg["candles"])
        mine = [p for p in journal["open"] if p["strategy"] == strat]
        if len(mine) >= cfg["max_open"]:
            continue
        params = core.strategy_params(scfg)
        fn = core.STRATEGY_FUNCS[strat]
        for sym in symbols:
            if len(mine) >= cfg["max_open"]:
                break
            if sym in occupied:
                continue
            # Свечи — БОЕВЫЕ, публичным клиентом: те же сигналы, что у бумаги
            df = core.fetch_candles(PUBLIC, sym, cfg["timeframe"], cfg["candles"])
            time.sleep(0.1)
            if df is None or len(df) < scfg["ema"] + 30:
                continue
            prepared = fn(df, params)
            i = len(prepared) - 1
            sigs = core.strategies.signal_fn(prepared, i, scfg, None, None, None)
            if not sigs:
                continue
            sig = sigs[0]
            signal_ts = int(prepared.timestamp.iloc[i])
            age = (PUBLIC.milliseconds() - signal_ts - core.TF_MS[cfg["timeframe"]]) / 60000
            if age > cfg["max_signal_age_min"]:
                continue
            key = f"{strat}|{sym}|{signal_ts}"
            if key in journal["acted"]:
                continue

            atr_val = float(prepared["atr"].iloc[i])
            price = ex.fetch_ticker(sym)["last"]
            d = sig["dir"]
            risk = atr_val * scfg["atr_mult"]
            stop = price - risk if d == 1 else price + risk
            take = price + risk * scfg["rr"] if d == 1 else price - risk * scfg["rr"]

            share = equity * cfg["allocation"][strat]
            risk_usd = share * cfg["risk_pct"]
            qty = risk_usd / risk
            notional = qty * price
            cap = share * cfg["max_notional"]
            note = ""
            if notional > cap:
                qty, notional, note = cap / price, cap, "урезано по объёму"
            qty = float(ex.amount_to_precision(sym, qty))
            notional = qty * price
            min_cost = ((ex.market(sym).get("limits") or {}).get("cost") or {}).get("min") or 5
            if qty <= 0 or notional < min_cost:
                log(f"⏭️  [{STRATEGY_RU[strat]}] {sym}: объём {notional:.2f} меньше минимума", cfg)
                journal["acted"][key] = datetime.now().isoformat()
                continue
            need_margin = notional / cfg["leverage"] * cfg["margin_buffer"]
            if available < need_margin:
                log(f"⏭️  [{STRATEGY_RU[strat]}] {sym}: не хватает маржи "
                    f"({available:.2f} < {need_margin:.2f})", cfg)
                continue

            params_order = {
                "marginMode": cfg["margin_mode"],
                "stopLoss": {"triggerPrice": float(ex.price_to_precision(sym, stop))},
                "takeProfit": {"triggerPrice": float(ex.price_to_precision(sym, take))},
            }
            try:
                order = place_order(ex, sym, "buy" if d == 1 else "sell", qty, params_order, symbols)
            except NotDemoError:
                raise
            except Exception as e:
                log(f"❌ [{STRATEGY_RU[strat]}] {sym}: заявка отклонена — "
                    f"{type(e).__name__}: {str(e)[:160]}", cfg)
                journal["acted"][key] = datetime.now().isoformat()
                continue

            fill = order.get("average") or price
            pos = {
                "strategy": strat, "symbol": sym, "dir": d, "type": sig["type"],
                "entry": fill, "stop": float(ex.price_to_precision(sym, stop)),
                "take": float(ex.price_to_precision(sym, take)), "qty": qty,
                "notional": round(notional, 2), "risk_usd": round(qty * risk, 4),
                "stop_pct": round(risk / price, 5), "atr": atr_val,
                "opened": datetime.now().isoformat(), "signal_ts": signal_ts,
                "signal_age_min": round(age, 1), "order_id": order.get("id"),
            }
            journal["open"].append(pos)
            journal["acted"][key] = datetime.now().isoformat()
            occupied.add(sym)
            mine.append(pos)
            available -= notional / cfg["leverage"]
            opened += 1
            log(f"✅ ОТКРЫТА [{STRATEGY_RU[strat]}] {sym} {'ЛОНГ' if d == 1 else 'ШОРТ'} "
                f"≈{fill} стоп={pos['stop']} ({risk / price:.2%}) тейк={pos['take']} "
                f"объём {notional:.2f} USDT риск {risk_usd:.2f} USDT"
                + (f" [{note}]" if note else ""), cfg)
    return opened


PUBLIC = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})


# ─────────────────────────────────────────────────────────────
#  Точка входа
# ─────────────────────────────────────────────────────────────
def prepare_account(ex, symbols, cfg):
    """Односторонний режим позиций, изолированная маржа, плечо — на каждой монете."""
    try:
        ex.set_position_mode(False, None, {"productType": cfg["product_type"]})
    except Exception as e:
        log(f"режим позиций: {str(e)[:120]}", cfg, show=False)
    for sym in symbols:
        try:
            ex.set_margin_mode(cfg["margin_mode"], sym, {"productType": cfg["product_type"]})
        except Exception as e:
            log(f"маржа {sym}: {str(e)[:100]}", cfg, show=False)
        try:
            ex.set_leverage(cfg["leverage"], sym, {"productType": cfg["product_type"],
                                                   "marginMode": cfg["margin_mode"]})
        except Exception as e:
            log(f"плечо {sym}: {str(e)[:100]}", cfg, show=False)


def cmd_check(cfg):
    print("ПРОВЕРКА ДЕМО-СЧЁТА BITGET (без единой заявки)")
    print("=" * 56)
    if not keys_configured():
        print("✘ Ключи не заданы. Создайте файл .env в корне проекта по образцу .env.example")
        return 2
    try:
        ex, total, free = verify_demo_key(log=print)
    except NotDemoError as e:
        print(f"✘ {e}")
        return 1
    syms = demo_symbols(ex)
    print(f"✔ Монет из нашего списка доступно в демо: {len(syms)} — "
          + ", ".join(s.split("/")[0] for s in syms))
    live = fetch_live_positions(ex)
    print(f"✔ Открытых позиций на демо-счёте: {len(live)}")
    share = total * cfg["allocation"]["donchian"]
    print(f"✔ На стратегию: {share:.2f} USDT, риск на сделку {share * cfg['risk_pct']:.2f} USDT")
    print("=" * 56)
    print("Всё готово. Демо-бота можно запускать из панели.")
    return 0


def main(cfg=None):
    cfg = dict(cfg or CONFIG)
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"

    if mode == "check":
        sys.exit(cmd_check(cfg))

    journal = load_journal(cfg)
    if mode == "status":
        print(json.dumps({k: journal.get(k) for k in ("equity", "available", "start_equity", "cycles")},
                         ensure_ascii=False))
        for p in journal["open"]:
            print(f"  [{STRATEGY_RU[p['strategy']]}] {p['symbol']} {'ЛОНГ' if p['dir'] == 1 else 'ШОРТ'} "
                  f"вход {p['entry']} стоп {p['stop']} тейк {p['take']}")
        return

    lock = core.acquire_single_instance(cfg)
    if lock is None:
        print(f"[!] {cfg['bot_name']} уже запущен")
        return
    if os.path.exists(core.stop_path(cfg)):
        os.remove(core.stop_path(cfg))

    log("=" * 56, cfg, show=False)
    log(f"СТАРТ {cfg['bot_name']}  риск {cfg['risk_pct']:.0%} от доли стратегии  "
        f"плечо {cfg['leverage']}x  до {cfg['max_open']} позиций на стратегию", cfg)
    try:
        ex, total, free = verify_demo_key(log=lambda m: log(m, cfg))
    except NotDemoError as e:
        journal["last_error"] = str(e)
        core.save_journal(journal, cfg)
        log(f"🛑 ОСТАНОВКА: {e}", cfg)
        return
    journal["last_error"] = None

    symbols = demo_symbols(ex)
    journal["demo_symbols"] = symbols
    journal["meta"] = {
        "bot_id": cfg["bot_id"], "bot_name": cfg["bot_name"],
        "rules": {s: core.describe_rules(dict(STRATEGIES[s], timeframe=cfg["timeframe"]))
                  for s in PRIORITY},
        "risk_pct": cfg["risk_pct"], "max_open": cfg["max_open"],
        "allocation": cfg["allocation"], "leverage": cfg["leverage"],
        "symbols": len(symbols),
    }
    core.save_journal(journal, cfg)
    log(f"Монет в демо: {len(symbols)} — {', '.join(s.split('/')[0] for s in symbols)}", cfg)
    prepare_account(ex, symbols, cfg)
    core.keep_awake(True)

    while True:
        try:
            log(f"--- Цикл #{journal.get('cycles', 0) + 1} ---", cfg)
            opened = run_cycle(ex, journal, cfg, symbols)
            journal["last_error"] = None
            core.save_journal(journal, cfg)
            eq, st = journal.get("equity"), journal.get("start_equity")
            log(f"Капитал={eq:.2f} USDT ({(eq - st):+.2f})  открыто={len(journal['open'])}  "
                f"сделок={len(journal['trades'])}  новых={opened}", cfg)
            if core.sleep_or_stop(cfg, cfg["scan_interval_min"] * 60):
                log("Остановлен из панели", cfg)
                break
        except NotDemoError as e:
            journal["last_error"] = str(e)
            core.save_journal(journal, cfg)
            log(f"🛑 ОСТАНОВКА: {e}", cfg)
            break
        except KeyboardInterrupt:
            log("Остановлен пользователем", cfg)
            break
        except Exception as e:
            journal["last_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            core.save_journal(journal, cfg)
            log(f"ОШИБКА: {type(e).__name__}: {e} — повтор через 5 мин", cfg)
            if core.sleep_or_stop(cfg, 300):
                log("Остановлен из панели", cfg)
                break
    core.keep_awake(False)


if __name__ == "__main__":
    main()
