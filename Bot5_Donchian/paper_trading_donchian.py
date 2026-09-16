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

from Backtest import strategies

HERE = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
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
    # 5% давали +556% за год, но с просадкой 68% и серией из 11
    # убытков подряд. 1-2% — то, что можно досидеть.
    "risk_pct":       0.02,
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
    with open(cfg["journal"], "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False, indent=2)


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


def get_symbols(exchange, cfg, allowed=None):
    """Ликвидные крипто-перпетуалы USDT, отсортированные по обороту."""
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


def fetch_candles(exchange, symbol, timeframe, limit):
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
    except Exception:
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


def check_exit(pos, df):
    """
    Задет ли стоп или тейк свечами, закрывшимися ПОСЛЕ входа.

    Смотрим high/low свечей, а не текущую цену: прежние боты
    опрашивали цену раз в N минут и пропускали касания между
    опросами. Здесь проверка совпадает с бэктестом свеча в свечу.

    Оба уровня задеты одной свечой — считаем стопом: порядок
    движения цены внутри свечи неизвестен.
    """
    seg = df[df.timestamp > pos["entry_ts"]]
    if seg.empty:
        return None, None, 0
    d, stop, take = pos["dir"], pos["stop"], pos["take"]
    for _, bar in seg.iterrows():
        if d == 1:
            hit_stop, hit_take = bar.low <= stop, bar.high >= take
        else:
            hit_stop, hit_take = bar.high >= stop, bar.low <= take
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
    })
    if pnl <= 0:
        unblock = datetime.now() + timedelta(hours=8)
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
    return {
        "channel": cfg["channel"], "atr_period": cfg["atr_period"],
        "atr_mult": cfg["atr_mult"], "rr": cfg["rr"], "ema": cfg["ema"],
        "allow_short": cfg["allow_short"], "max_hold_bars": cfg["max_hold_bars"],
    }


def run_cycle(exchange, journal, cfg, symbols):
    journal["cycles"] = journal.get("cycles", 0) + 1
    p = strategy_params(cfg)

    # ── 1. Открытые позиции ───────────────────────────────────
    still_open = []
    for pos in journal["open"]:
        df = fetch_candles(exchange, pos["symbol"], cfg["timeframe"], cfg["candles"])
        time.sleep(0.1)
        if df is None:
            still_open.append(pos)
            continue

        reason, ts, bars = check_exit(pos, df)
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

    opened = 0
    for sym in symbols:
        if len(journal["open"]) >= cfg["max_open"]:
            break
        if sym in on_cooldown:
            continue

        df = fetch_candles(exchange, sym, cfg["timeframe"], cfg["candles"])
        time.sleep(0.1)
        if df is None or len(df) < cfg["ema"] + 30:
            continue

        # Тот же код сигналов, что и в бэктесте
        prepared = strategies.donchian(df, p)
        i = len(prepared) - 1            # последняя ЗАКРЫТАЯ свеча
        sigs = strategies.signal_fn(prepared, i, cfg, None, None, None)
        if not sigs:
            continue
        sig = sigs[0]

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
            "entry_ts": int(prepared.timestamp.iloc[i]),
            "bars_held": 0,
        }
        journal["open"].append(pos)
        busy.add((sym, sig["dir"]))
        opened += 1
        log(f"✅ ОТКРЫТА {sym} {'ЛОНГ' if sig['dir'] == 1 else 'ШОРТ'} "
            f"вход={round(entry, 6)} стоп={round(stop, 6)} ({risk/entry:.2%}) "
            f"тейк={round(take, 6)} объём=${notional:.2f} "
            f"риск=${journal['balance']*cfg['risk_pct']:.2f}"
            + (f" [{note}]" if note else ""), cfg)

    return opened


# ─────────────────────────────────────────────────────────────
#  Статистика
# ─────────────────────────────────────────────────────────────
def print_stats(journal, cfg):
    t = journal["trades"]
    bal, dep = journal["balance"], journal["deposit"]
    print("=" * 62)
    print("  БОТ #5 — Дончиан 4ч, Bitget перпетуалы")
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
        print(f"  Ожидание по бэктесту: WR ~33%, +0.19R на сделку")
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
def main():
    cfg = CONFIG.copy()
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"

    if mode == "reset":
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
        syms, err = get_symbols(ex, cfg)
        if err:
            print(f"[!] {err}")
            return
        print(f"Инструментов: {len(syms)}")
        for s in syms:
            print("  ", s)
        return

    log("=" * 56, cfg, show=False)
    log(f"СТАРТ  депозит=${cfg['deposit']}  риск={cfg['risk_pct']:.0%}  "
        f"макс.позиций={cfg['max_open']}  плечо<={cfg['max_leverage']}x", cfg)
    log(f"Правила: Дончиан {cfg['channel']} свечей, стоп {cfg['atr_mult']} ATR, "
        f"тейк {cfg['rr']}R, фильтр EMA{cfg['ema']}, ТФ {cfg['timeframe']}", cfg)

    allowed = crypto_bases()
    log(f"Криптоактивов в белом списке (спот Binance): {len(allowed)}", cfg)

    symbols, err = get_symbols(ex, cfg, allowed)
    if err:
        log(f"[!] {err}", cfg)
        return
    log(f"Инструментов в работе: {len(symbols)} — {', '.join(s.split('/')[0] for s in symbols[:12])}"
        + (" ..." if len(symbols) > 12 else ""), cfg)

    last_symbol_refresh = time.time()

    while True:
        try:
            # Набор инструментов обновляем раз в сутки
            if time.time() - last_symbol_refresh > 86400:
                fresh, err2 = get_symbols(ex, cfg, allowed)
                if not err2 and fresh:
                    symbols = fresh
                    log(f"Список инструментов обновлён: {len(symbols)}", cfg)
                last_symbol_refresh = time.time()

            log(f"--- Цикл #{journal.get('cycles', 0) + 1} ---", cfg)
            opened = run_cycle(ex, journal, cfg, symbols)
            save_journal(journal, cfg)

            t = journal["trades"]
            wr = (sum(1 for x in t if x["result"] == "WIN") / len(t) * 100) if t else 0
            avg_r = (sum(x["r_multiple"] for x in t) / len(t)) if t else 0
            log(f"Баланс=${journal['balance']:.2f}  сделок={len(t)}  "
                f"WR={wr:.1f}%  среднее={avg_r:+.3f}R  "
                f"открыто={len(journal['open'])}  новых={opened}", cfg)
            log(f"Следующий цикл через {cfg['scan_interval_min']} мин", cfg, show=False)
            time.sleep(cfg["scan_interval_min"] * 60)

        except KeyboardInterrupt:
            log("Остановлен пользователем", cfg)
            print_stats(journal, cfg)
            break
        except Exception as e:
            log(f"ОШИБКА: {type(e).__name__}: {e} — повтор через 5 мин", cfg)
            time.sleep(300)


if __name__ == "__main__":
    main()
