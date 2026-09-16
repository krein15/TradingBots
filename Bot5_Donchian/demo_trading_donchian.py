"""
demo_trading_donchian.py
========================
Бот #5 на ДЕМО-счёте Bitget. Те же правила, настоящие заявки,
вымышленные деньги.

Зачем отдельно от paper_trading_donchian.py. Они отвечают на разные
вопросы и должны работать параллельно:

  paper_trading_donchian.py — 35 инструментов, исполнение
    симулируется локально. Меряет СТАТИСТИКУ стратегии: диверсификация
    по независимым пробоям — это то, чем стратегия живёт.

  demo_trading_donchian.py — 3 инструмента, заявки уходят на биржу.
    Меряет МЕХАНИКУ: правильно ли считается объём, проходят ли
    ордера, срабатывают ли стопы, сколько списывает funding.
    Статистику здесь мерить нельзя: у SBTC, SETH и SXRP средняя
    парная корреляция 0.78, они ходят почти как один актив.

Демо у Bitget — это отдельный тип контрактов SUSDT-FUTURES с
расчётом в вымышленной валюте SUSDT. Настоящих денег там нет по
устройству: SUSDT нельзя вывести и он ничего не стоит.

ПРЕДОХРАНИТЕЛЬ. Код постановки заявок на демо и на реальном счёте
у Bitget один и тот же, разница только в символе. Поэтому здесь
перед КАЖДОЙ заявкой проверяется, что инструмент рассчитывается в
SUSDT. Боевой символ в этот код не пройдёт физически — не потому
что маловероятно, а потому что проверка стоит на пути исполнения.
Реального торгового пути здесь нет и не будет.

Ключи API берутся только из переменных окружения или файла .env,
который в git не попадает. В коде их нет и быть не должно.

Нужны права: чтение + торговля фьючерсами. Право на ВЫВОД СРЕДСТВ
включать не нужно — боту оно не требуется ни для чего.

Запуск:
  python demo_trading_donchian.py check     — проверить связь и счёт
  python demo_trading_donchian.py           — торговля
  python demo_trading_donchian.py status    — позиции и история
  python demo_trading_donchian.py close-all — закрыть все позиции
"""

import json
import os
import sys
import time
from datetime import datetime

import ccxt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from Backtest import strategies
import paper_trading_donchian as paper

HERE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
#  Инструменты
# ─────────────────────────────────────────────────────────────
# Демо Bitget даёт ровно три контракта. Сигналы считаем по БОЕВЫМ
# свечам: у демо всего ~540 свечей истории, а EMA200 на 4ч требует
# заметно больше для прогрева. Цены при этом совпадают с боевыми с
# точностью 0.01% при стопе 6.5%, так что подмена безобидна.
SYMBOL_MAP = {
    "SBTC/SUSDT:SUSDT": "BTC/USDT:USDT",
    "SETH/SUSDT:SUSDT": "ETH/USDT:USDT",
    "SXRP/SUSDT:SUSDT": "XRP/USDT:USDT",
}

# Валюта расчётов демо-контрактов. Всё, что рассчитывается не в ней,
# этот бот трогать не имеет права.
DEMO_SETTLE = "SUSDT"

CONFIG = dict(paper.CONFIG)
CONFIG.update({
    "journal":  os.path.join(HERE, "demo_journal.json"),
    "logfile":  os.path.join(HERE, "demo_log.txt"),
    "scan_interval_min": 20,
    "leverage":  3,
    "margin_mode": "isolated",
})


# ─────────────────────────────────────────────────────────────
#  Ключи
# ─────────────────────────────────────────────────────────────
def load_env():
    """
    Читаем .env рядом со скриптом или в корне проекта.

    Без сторонних зависимостей и без записи куда-либо: значения
    попадают только в окружение текущего процесса.
    """
    for path in (os.path.join(HERE, ".env"),
                 os.path.join(os.path.dirname(HERE), ".env")):
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_credentials():
    load_env()
    key = os.environ.get("BITGET_API_KEY", "")
    secret = os.environ.get("BITGET_API_SECRET", "")
    password = os.environ.get("BITGET_API_PASSWORD", "")
    missing = [n for n, v in (("BITGET_API_KEY", key),
                              ("BITGET_API_SECRET", secret),
                              ("BITGET_API_PASSWORD", password)) if not v]
    return key, secret, password, missing


def get_exchange():
    key, secret, password, missing = get_credentials()
    if missing:
        raise RuntimeError(
            "Не заданы ключи: " + ", ".join(missing) + ".\n"
            "    Создай файл Bot5_Donchian/.env по образцу .env.example\n"
            "    и впиши туда ключи демо-счёта Bitget. В git он не попадёт.")
    return ccxt.bitget({
        "apiKey": key,
        "secret": secret,
        "password": password,
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })


# ─────────────────────────────────────────────────────────────
#  Предохранитель
# ─────────────────────────────────────────────────────────────
class NotDemoError(RuntimeError):
    """Попытка тронуть инструмент, который не является демо."""


def assert_demo(exchange, symbol):
    """
    Пропускает дальше ТОЛЬКО демо-контракты.

    Вызывается перед каждой заявкой. Это не проверка «на всякий
    случай», а единственный путь к исполнению: боевой символ сюда
    не пройдёт, потому что рассчитывается в USDT, а не в SUSDT.
    """
    if symbol not in SYMBOL_MAP:
        raise NotDemoError(f"{symbol} нет в списке демо-контрактов")
    market = exchange.market(symbol)
    settle = market.get("settle")
    if settle != DEMO_SETTLE:
        raise NotDemoError(
            f"{symbol} рассчитывается в {settle}, а не в {DEMO_SETTLE}. "
            f"Это боевой контракт — заявка не будет отправлена.")
    return market


def assert_demo_account(exchange, cfg):
    """
    Проверяем, что на счёте демо-валюта.

    Если бы ключи оказались от боевого счёта, здесь не окажется
    баланса в SUSDT — и бот откажется работать, не отправив ни
    одной заявки.
    """
    bal = exchange.fetch_balance({"productType": "SUSDT-FUTURES"})
    total = bal.get("total", {}) or {}
    demo_amount = total.get(DEMO_SETTLE)
    if demo_amount is None:
        raise NotDemoError(
            f"На счёте нет баланса в {DEMO_SETTLE}. Либо демо-торговля "
            f"не включена в кабинете Bitget, либо ключи от боевого "
            f"счёта. Торговля не начата.")
    return float(demo_amount)


# ─────────────────────────────────────────────────────────────
#  Данные и сигналы
# ─────────────────────────────────────────────────────────────
def signal_for(exchange, demo_symbol, cfg):
    """
    Сигнал по боевым свечам того же актива.
    Возвращает (сигнал, ATR, метка свечи) или (None, None, None).

    Сигнал со свечи, закрывшейся дольше max_signal_age_min назад,
    отбрасывается: бэктест входит сразу после закрытия, и вход
    часами позже — это уже другая сделка по другой цене.
    """
    real = SYMBOL_MAP[demo_symbol]
    df = paper.fetch_candles(exchange, real, cfg["timeframe"], cfg["candles"])
    if df is None or len(df) < cfg["ema"] + 30:
        return None, None, None
    prepared = strategies.donchian(df, paper.strategy_params(cfg))
    i = len(prepared) - 1
    sigs = strategies.signal_fn(prepared, i, cfg, None, None, None)
    if not sigs:
        return None, None, None
    signal_ts = int(prepared.timestamp.iloc[i])
    closed_ms = signal_ts + paper.TF_MS.get(cfg["timeframe"], 14_400_000)
    if (exchange.milliseconds() - closed_ms) / 60000 > cfg["max_signal_age_min"]:
        return None, None, None
    return sigs[0], float(prepared["atr"].iloc[i]), signal_ts


# ─────────────────────────────────────────────────────────────
#  Заявки
# ─────────────────────────────────────────────────────────────
def open_position(exchange, journal, demo_symbol, sig, atr_val, cfg, log):
    market = assert_demo(exchange, demo_symbol)      # ← предохранитель

    ticker = exchange.fetch_ticker(demo_symbol)
    price = ticker.get("last")
    if not price:
        log(f"⏭️  {demo_symbol}: нет цены")
        return False

    d = sig["dir"]
    risk = atr_val * cfg["atr_mult"]
    stop = price - risk if d == 1 else price + risk
    take = price + risk * cfg["rr"] if d == 1 else price - risk * cfg["rr"]

    qty, notional, note = paper.position_size(journal["balance"], price, stop, cfg)
    if qty <= 0:
        log(f"⏭️  {demo_symbol}: {note}")
        return False
    qty = float(exchange.amount_to_precision(demo_symbol, qty))
    if qty <= 0:
        log(f"⏭️  {demo_symbol}: объём округлился в ноль")
        return False

    side = "buy" if d == 1 else "sell"
    params = {
        "marginMode": cfg["margin_mode"],
        "hedged": False,
        # Стоп и тейк уходят ВМЕСТЕ с заявкой и живут на бирже.
        # Если бот упадёт, позиция останется защищённой — держать
        # стоп только в памяти процесса нельзя.
        "stopLoss": {"triggerPrice": float(exchange.price_to_precision(demo_symbol, stop))},
        "takeProfit": {"triggerPrice": float(exchange.price_to_precision(demo_symbol, take))},
    }

    try:
        order = exchange.create_order(demo_symbol, "market", side, qty, None, params)
    except Exception as e:
        log(f"❌ {demo_symbol}: заявка отклонена — {type(e).__name__}: {str(e)[:160]}")
        return False

    pos = {
        "symbol": demo_symbol, "real_symbol": SYMBOL_MAP[demo_symbol],
        "dir": d, "type": sig["type"],
        "entry": price, "stop": round(stop, 8), "take": round(take, 8),
        "qty": qty, "notional": round(notional, 4),
        "atr": round(atr_val, 8), "stop_pct": round(risk / price, 5),
        "order_id": order.get("id"),
        "opened": datetime.now().isoformat(),
    }
    journal.setdefault("open", []).append(pos)
    log(f"✅ ОТКРЫТА {demo_symbol} {'ЛОНГ' if d == 1 else 'ШОРТ'} "
        f"цена≈{price} стоп={pos['stop']} ({risk/price:.2%}) тейк={pos['take']} "
        f"объём={qty} (${notional:.2f}) id={order.get('id')}"
        + (f" [{note}]" if note else ""))
    return True


def sync_positions(exchange, journal, cfg, log):
    """
    Сверяем журнал с тем, что реально открыто на бирже.

    Источник истины — биржа, а не журнал: стоп мог сработать, пока
    бот не работал. Позиции, пропавшие с биржи, считаем закрытыми.
    """
    try:
        live = exchange.fetch_positions(list(SYMBOL_MAP))
    except Exception as e:
        log(f"[!] Не удалось получить позиции: {type(e).__name__}: {str(e)[:120]}")
        return
    open_now = {p["symbol"] for p in live
                if p.get("contracts") and float(p["contracts"]) > 0}

    still = []
    for pos in journal.get("open", []):
        if pos["symbol"] in open_now:
            still.append(pos)
            continue
        # Позиции нет — значит закрылась по стопу или тейку.
        # Реальный PnL берём из истории сделок биржи.
        pnl = None
        try:
            closed = exchange.fetch_my_trades(pos["symbol"], limit=20)
            rel = [t for t in closed
                   if t.get("timestamp", 0) >= _ts(pos["opened"])]
            if rel:
                pnl = sum(float(t.get("info", {}).get("profit") or 0) for t in rel)
        except Exception:
            pass
        journal.setdefault("trades", []).append({
            **pos,
            "closed": datetime.now().isoformat(),
            "pnl_reported": pnl,
        })
        log(f"🔚 ЗАКРЫТА {pos['symbol']} {'ЛОНГ' if pos['dir'] == 1 else 'ШОРТ'}"
            + (f"  PnL по данным биржи: {pnl:+.4f} {DEMO_SETTLE}" if pnl is not None
               else "  (PnL не удалось получить)"))
    journal["open"] = still


def _ts(iso):
    try:
        return int(datetime.fromisoformat(iso).timestamp() * 1000)
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────
#  Циклы и режимы
# ─────────────────────────────────────────────────────────────
def run_cycle(exchange, journal, cfg, log):
    journal["cycles"] = journal.get("cycles", 0) + 1
    sync_positions(exchange, journal, cfg, log)

    try:
        journal["balance"] = assert_demo_account(exchange, cfg)
    except NotDemoError as e:
        log(f"🛑 {e}")
        raise

    busy = {(p["symbol"], p["dir"]) for p in journal.get("open", [])}
    if len(journal.get("open", [])) >= cfg["max_open"]:
        log(f"Лимит позиций занят ({len(journal['open'])}/{cfg['max_open']})")
        return 0

    opened = 0
    acted = journal.setdefault("acted", {})
    for demo_symbol in SYMBOL_MAP:
        if len(journal.get("open", [])) >= cfg["max_open"]:
            break
        sig, atr_val, signal_ts = signal_for(exchange, demo_symbol, cfg)
        if not sig:
            continue
        # Один сигнал — одна заявка, даже после перезапуска
        key = f"{demo_symbol}|{signal_ts}"
        if key in acted or (demo_symbol, sig["dir"]) in busy:
            continue
        if open_position(exchange, journal, demo_symbol, sig, atr_val, cfg, log):
            acted[key] = datetime.now().isoformat()
            busy.add((demo_symbol, sig["dir"]))
            opened += 1
    return opened


def cmd_check(cfg):
    """Связь, счёт и предохранители — без единой заявки."""
    print("=" * 62)
    print("  ПРОВЕРКА ДЕМО-ПОДКЛЮЧЕНИЯ BITGET")
    print("=" * 62)

    key, _, _, missing = get_credentials()
    if missing:
        print(f"  [!] Не заданы ключи: {', '.join(missing)}")
        print("      Создай Bot5_Donchian/.env по образцу .env.example")
        return 1
    print(f"  Ключ найден: {key[:6]}…{key[-4:]} (длина {len(key)})")

    ex = get_exchange()
    ex.load_markets()
    print("  Связь с биржей: есть")

    print("\n  Предохранитель — боевые контракты не должны проходить:")
    for sym in ("BTC/USDT:USDT", "ETH/USDT:USDT"):
        try:
            assert_demo(ex, sym)
            print(f"    [!!] {sym}: ПРОШЁЛ — это ошибка, торговать нельзя")
            return 1
        except NotDemoError as e:
            print(f"    ✔ {sym}: отклонён — {str(e)[:70]}")
    print("  Демо-контракты:")
    for sym in SYMBOL_MAP:
        m = assert_demo(ex, sym)
        print(f"    ✔ {sym}: допущен (расчёт в {m['settle']})")

    print("\n  Счёт:")
    try:
        amount = assert_demo_account(ex, cfg)
        print(f"    Баланс демо: {amount:.2f} {DEMO_SETTLE}")
    except NotDemoError as e:
        print(f"    [!] {e}")
        return 1
    except Exception as e:
        print(f"    [!] Не удалось прочитать баланс: {type(e).__name__}: {str(e)[:140]}")
        return 1

    print("\n  Сигналы прямо сейчас:")
    for demo_symbol in SYMBOL_MAP:
        sig, atr_val, _ = signal_for(ex, demo_symbol, cfg)
        if sig:
            print(f"    {demo_symbol}: {sig['type']}")
        else:
            print(f"    {demo_symbol}: сигнала нет")

    print("\n  Всё готово. Запуск: python demo_trading_donchian.py")
    print("=" * 62)
    return 0


def cmd_close_all(cfg):
    ex = get_exchange()
    ex.load_markets()
    positions = ex.fetch_positions(list(SYMBOL_MAP))
    n = 0
    for p in positions:
        contracts = float(p.get("contracts") or 0)
        if contracts <= 0:
            continue
        assert_demo(ex, p["symbol"])          # ← предохранитель
        side = "sell" if p["side"] == "long" else "buy"
        ex.create_order(p["symbol"], "market", side, contracts, None,
                        {"reduceOnly": True, "marginMode": cfg["margin_mode"]})
        print(f"  закрыта {p['symbol']} {p['side']} {contracts}")
        n += 1
    print(f"Закрыто позиций: {n}")


def main():
    cfg = dict(CONFIG)
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"

    def log(msg, show=True):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        if show:
            print(line, flush=True)
        with open(cfg["logfile"], "a", encoding="utf-8", errors="replace") as f:
            f.write(line + "\n")

    if mode == "check":
        sys.exit(cmd_check(cfg))

    if mode == "close-all":
        cmd_close_all(cfg)
        return

    journal = paper.load_journal(cfg)

    if mode == "status":
        print(f"Баланс в журнале: {journal.get('balance')} {DEMO_SETTLE}")
        print(f"Открыто: {len(journal.get('open', []))}   "
              f"закрыто: {len(journal.get('trades', []))}   "
              f"циклов: {journal.get('cycles', 0)}")
        for p in journal.get("open", []):
            print(f"  {p['symbol']} {'ЛОНГ' if p['dir'] == 1 else 'ШОРТ'} "
                  f"вход={p['entry']} стоп={p['stop']} тейк={p['take']}")
        for t in journal.get("trades", [])[-10:]:
            print(f"  закрыта {t['symbol']} "
                  f"{'ЛОНГ' if t['dir'] == 1 else 'ШОРТ'} "
                  f"PnL={t.get('pnl_reported')}")
        return

    ex = get_exchange()
    ex.load_markets()

    log("=" * 56, show=False)
    log(f"СТАРТ ДЕМО  риск={cfg['risk_pct']:.0%}  "
        f"макс.позиций={cfg['max_open']}  плечо={cfg['leverage']}x")
    log(f"Инструменты: {', '.join(SYMBOL_MAP)}")

    # Ни одной заявки, пока не подтверждено, что счёт демонстрационный
    balance = assert_demo_account(ex, cfg)
    journal["balance"] = balance
    journal.setdefault("deposit", balance)
    log(f"Счёт подтверждён как демонстрационный: {balance:.2f} {DEMO_SETTLE}")

    for sym in SYMBOL_MAP:
        assert_demo(ex, sym)
        try:
            ex.set_margin_mode(cfg["margin_mode"], sym)
        except Exception:
            pass
        try:
            ex.set_leverage(cfg["leverage"], sym)
        except Exception as e:
            log(f"[!] Плечо для {sym}: {type(e).__name__}: {str(e)[:90]}", show=False)

    while True:
        try:
            log(f"--- Цикл #{journal.get('cycles', 0) + 1} ---")
            opened = run_cycle(ex, journal, cfg, log)
            paper.save_journal(journal, cfg)
            log(f"Баланс={journal['balance']:.2f} {DEMO_SETTLE}  "
                f"открыто={len(journal.get('open', []))}  новых={opened}")
            time.sleep(cfg["scan_interval_min"] * 60)
        except KeyboardInterrupt:
            log("Остановлен пользователем")
            break
        except NotDemoError as e:
            log(f"🛑 ОСТАНОВКА: {e}")
            break
        except Exception as e:
            log(f"ОШИБКА: {type(e).__name__}: {e} — повтор через 5 мин")
            time.sleep(300)


if __name__ == "__main__":
    main()
