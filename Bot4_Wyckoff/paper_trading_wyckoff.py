"""
paper_trading_wyckoff.py
========================
Бот #4 — Wyckoff + VSA (Volume Spread Analysis)
Биржа: Bitget
Таймфреймы: 1ч, 4ч

Стратегия:
  Ищем фазу накопления Wyckoff:
  1. Selling Climax (SC)  — резкое падение объём >5x
  2. Automatic Rally (AR) — отскок после SC
  3. Secondary Test (ST)  — повторный тест низов объём <50% SC
  4. Spring               — ложный пробой минимума SC
  5. ВХОД в лонг за Spring с коротким стопом

  Для шорта — зеркально (Buying Climax → Distribution)

Цель:
  Редкие сделки (5-15 в месяц)
  RR 1:7 — 1:10
  Короткий стоп за Spring (0.5-2%)

Журнал: wyckoff_journal.json
Лог:    wyckoff_log.txt

Запуск:
  python paper_trading_wyckoff.py          — старт
  python paper_trading_wyckoff.py status   — статистика
  python paper_trading_wyckoff.py reset    — сброс
  python paper_trading_wyckoff.py combined — все 4 бота
"""

import sys, time, os, json
import ccxt
try:
    from divergence_module import calc_rsi, find_divergence, calc_volume_profile
    HAS_DIV = True
except ImportError:
    HAS_DIV = False
import pandas as pd
import numpy as np
from datetime import datetime

CONFIG = {
    "timeframes":        ["1h", "4h"],        # возвращаем старшие TF — меньше шума
    "candles":           500,
    "vol_avg_period":    20,
    "min_usdt_vol":      5_000_000,  # только ликвидные монеты
    # Wyckoff параметры
    "sc_vol_mult":       5.0,        # только настоящая паника — 5x объём
    "ar_min_bounce":     0.03,       # Automatic Rally — отскок минимум 3%
    "st_vol_ratio":      0.6,        # Secondary Test — объём <60% от SC
    "st_price_pct":      0.02,       # ST не уходит дальше 2% от SC лоя
    "spring_pct":        0.005,      # Spring пробивает SC лой на 0.5-3%
    "spring_max_pct":    0.03,
    "spring_close_back": 0.003,      # Spring закрывается обратно выше SC лоя
    # Торговые параметры
    "rr_ratio":          7.0,        # RR 1:7
    "commission":        0.001,
    "stop_buffer":       0.002,
    "max_wait_bars":     5,          # ждём исполнения 5 свечей
    "initial_deposit":   50.0,
    "risk_pct":          0.05,
    "max_open_trades":   3,
    "session_filter":    True,        # торговать только в активные сессии
    "session_hours":     [(7,18)],    # UTC 7-18 (Лондон + Нью-Йорк)
    "cooldown_hours":    4,          # пауза после лосса 4ч
    "scan_interval":     30,         # каждые 30 минут для 1h/4h
    "journal_file":      "wyckoff_journal.json",
    "log_file":          "wyckoff_log.txt",
}

TIMEFRAME_LABELS = {"1h":"1 час","4h":"4 часа","1d":"1 день"}


# ─────────────────────────────────────────────
#  ФИЛЬТР ТОРГОВОЙ СЕССИИ
# ─────────────────────────────────────────────
def is_active_session(cfg):
    """
    Торгуем только в активные сессии (UTC):
    Лондон:   07:00 - 12:00
    Нью-Йорк: 13:00 - 18:00
    Пересечение: 13:00 - 17:00 — самое активное время
    Ночь Азии: пропускаем (низкий объём, много ложных сигналов)
    """
    if not cfg.get("session_filter", True):
        return True, "все сессии"
    hour = datetime.now().hour
    sessions = cfg.get("session_hours", [(7, 18)])
    for start, end in sessions:
        if start <= hour < end:
            if 7 <= hour < 12:
                return True, "🇬🇧 Лондон"
            elif 13 <= hour < 18:
                return True, "🇺🇸 Нью-Йорк"
            else:
                return True, "активная"
    return False, f"🌙 ночь/Азия ({hour}:00 UTC)"


# ─────────────────────────────────────────────
#  ЛОГ
# ─────────────────────────────────────────────
def log(msg, cfg, also_print=True):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_print:
        print(line)
    with open(cfg["log_file"], "a", encoding="utf-8", errors="replace") as f:
        f.write(line + "\n")


# ─────────────────────────────────────────────
#  ЖУРНАЛ
# ─────────────────────────────────────────────
def load_journal(cfg):
    fname = cfg["journal_file"]
    if os.path.exists(fname):
        with open(fname, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "created":         datetime.now().isoformat(),
        "initial_deposit": cfg["initial_deposit"],
        "balance":         cfg["initial_deposit"],
        "trades":          [],
        "pending":         [],
        "open":            [],
        "scan_count":      0,
    }


def save_journal(journal, cfg):
    with open(cfg["journal_file"], "w", encoding="utf-8") as f:
        json.dump(journal, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
#  БИРЖА — BITGET
# ─────────────────────────────────────────────
def get_exchange():
    return ccxt.bitget({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })


def get_symbols(exchange, min_vol):
    try:
        tickers = exchange.fetch_tickers()
        symbols = [(s, t.get("quoteVolume") or 0)
                   for s, t in tickers.items()
                   if s.endswith("/USDT")
                   and s != "BTC/USDT"
                   and (t.get("quoteVolume") or 0) >= min_vol]
        symbols.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in symbols]
    except Exception as e:
        print(f"[!] Ошибка Bitget: {e}")
        return []


def fetch_candles(exchange, symbol, timeframe, limit, cfg):
    try:
        ms_map = {"1h":3600000,"4h":14400000,"1d":86400000}
        ms    = ms_map.get(timeframe, 3600000)
        since = exchange.milliseconds() - limit * ms - ms * 10
        raw   = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not raw or len(raw) < 50:
            return None
        df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        p = cfg["vol_avg_period"]
        df["vol_avg"]   = df["volume"].rolling(p).mean()
        df["vol_ratio"] = df["volume"] / df["vol_avg"].replace(0, 1e-9)
        return df.reset_index(drop=True)
    except Exception:
        return None


def get_current_price(exchange, symbol):
    try:
        return exchange.fetch_ticker(symbol)["last"]
    except Exception:
        return None


# ─────────────────────────────────────────────
#  WYCKOFF — ПОИСК ФАЗ
# ─────────────────────────────────────────────
def find_wyckoff_accumulation(df, cfg):
    """
    Ищем фазу накопления Wyckoff:

    SC → AR → ST → Spring → ВХОД

    Визуально:
                AR
               /  \\
    SC        /    ST        Spring
    |\\       /      \\       /|
    | \\     /        \\     / |
    |  \\___/          \\___/  ← ВХОД ЗДЕСЬ
    |________________________ SC лой (уровень)
                              ↑ Spring пробивает и закрывается обратно
    """
    signals   = []
    sc_mult   = cfg["sc_vol_mult"]
    ar_min    = cfg["ar_min_bounce"]
    st_vol    = cfg["st_vol_ratio"]
    st_pct    = cfg["st_price_pct"]
    sp_min    = cfg["spring_pct"]
    sp_max    = cfg["spring_max_pct"]
    sp_back   = cfg["spring_close_back"]
    buf       = cfg["stop_buffer"]
    rr        = cfg["rr_ratio"]

    # Ищем SC в окне последних 100 свечей
    lookback = min(100, len(df) - 10)
    start    = len(df) - lookback

    for sc_i in range(start, len(df) - 15):
        sc_row = df.iloc[sc_i]
        sc_vol = sc_row["vol_ratio"]
        sc_low = sc_row["low"]
        sc_close = sc_row["close"]

        # ── ШАГ 1: Selling Climax ─────────────────
        # Медвежья свеча с огромным объёмом
        if sc_vol < sc_mult:
            continue
        if sc_close >= sc_row["open"]:
            continue  # должна быть медвежья свеча

        sc_drop = (sc_row["open"] - sc_close) / sc_row["open"]
        if sc_drop < 0.02:
            continue  # падение минимум 2%

        # ── ШАГ 2: Automatic Rally ────────────────
        # После SC — отскок минимум ar_min%
        ar_high = df.iloc[sc_i+1:sc_i+8]["high"].max()
        ar_bounce = (ar_high - sc_low) / sc_low
        if ar_bounce < ar_min:
            continue

        # ── ШАГ 3: Secondary Test ─────────────────
        # Повторный тест SC лоя на малом объёме
        st_found = False
        st_i     = None
        for j in range(sc_i + 3, min(sc_i + 25, len(df) - 5)):
            st_row = df.iloc[j]
            st_low = st_row["low"]
            st_vol_ratio = st_row["vol_ratio"]

            # ST близко к SC лою но не пробивает его
            dist = abs(st_low - sc_low) / sc_low
            if dist > st_pct:
                continue
            if st_vol_ratio > sc_vol * st_vol:
                continue  # объём должен быть меньше SC
            if st_low < sc_low * (1 - 0.005):
                continue  # не должен пробивать SC лой

            st_found = True
            st_i     = j
            break

        if not st_found:
            continue

        # ── ШАГ 4: Spring ─────────────────────────
        # Ложный пробой SC лоя — выбивает стопы
        for sp_i in range(st_i + 1, min(st_i + 15, len(df) - 1)):
            sp_row   = df.iloc[sp_i]
            sp_low   = sp_row["low"]
            sp_close = sp_row["close"]
            sp_vol   = sp_row["vol_ratio"]

            # Spring пробивает SC лой
            pct_below = (sc_low - sp_low) / sc_low
            if pct_below < sp_min or pct_below > sp_max:
                continue

            # Но ЗАКРЫВАЕТСЯ обратно выше SC лоя
            if sp_close < sc_low * (1 + sp_back):
                continue

            # Spring на относительно малом объёме (не паника)
            if sp_vol > sc_vol * 0.8:
                continue

            # ── ВХОД ──────────────────────────────
            # Лимитка на уровне SC лоя (зона поддержки)
            entry = sc_low * (1 + buf)
            stop  = sp_low * (1 - buf)    # стоп за лой Spring
            risk  = entry - stop
            if risk <= 0:
                continue

            # Тейк = AR хай (первая цель) или RR 1:7
            take_rr   = entry + risk * rr
            take_ar   = ar_high * 0.95    # чуть ниже AR хая
            take      = max(take_rr, take_ar)

            # Проверяем что цена ещё около уровня входа
            close_now = df.iloc[-1]["close"]
            dist_now  = abs(close_now - entry) / entry
            if dist_now > 0.05:
                continue  # цена ушла далеко

            actual_rr = round((take - entry) / risk, 1)
            if actual_rr < 3:
                continue  # минимум RR 1:3

            signals.append({
                "type":        "ACCUMULATION",
                "dir":         1,
                "entry_limit": round(entry, 6),
                "stop":        round(stop, 6),
                "take":        round(take, 6),
                "sc_low":      round(sc_low, 6),
                "ar_high":     round(ar_high, 6),
                "spring_low":  round(sp_low, 6),
                "sc_vol":      round(sc_vol, 1),
                "spring_pct":  round(pct_below * 100, 2),
                "rr":          actual_rr,
                "sc_bar":      sc_i,
                "spring_bar":  sp_i,
            })
            break  # один сигнал на SC

    return signals


def find_wyckoff_distribution(df, cfg):
    """
    Зеркально — фаза распределения для шортов:
    Buying Climax → AR вниз → ST → Upthrust → ШОРТ
    """
    signals  = []
    bc_mult  = cfg["sc_vol_mult"]
    ar_min   = cfg["ar_min_bounce"]
    st_vol   = cfg["st_vol_ratio"]
    st_pct   = cfg["st_price_pct"]
    ut_min   = cfg["spring_pct"]
    ut_max   = cfg["spring_max_pct"]
    ut_back  = cfg["spring_close_back"]
    buf      = cfg["stop_buffer"]
    rr       = cfg["rr_ratio"]

    lookback = min(100, len(df) - 10)
    start    = len(df) - lookback

    for bc_i in range(start, len(df) - 15):
        bc_row   = df.iloc[bc_i]
        bc_vol   = bc_row["vol_ratio"]
        bc_high  = bc_row["high"]
        bc_close = bc_row["close"]

        # Buying Climax — бычья свеча с огромным объёмом
        if bc_vol < bc_mult:
            continue
        if bc_close <= bc_row["open"]:
            continue

        bc_rise = (bc_close - bc_row["open"]) / bc_row["open"]
        if bc_rise < 0.02:
            continue

        # Automatic Reaction — откат вниз
        ar_low    = df.iloc[bc_i+1:bc_i+8]["low"].min()
        ar_bounce = (bc_high - ar_low) / bc_high
        if ar_bounce < ar_min:
            continue

        # Secondary Test — повторный тест хая на малом объёме
        st_found = False
        st_i     = None
        for j in range(bc_i + 3, min(bc_i + 25, len(df) - 5)):
            st_row = df.iloc[j]
            st_high = st_row["high"]
            st_vol_ratio = st_row["vol_ratio"]
            dist = abs(st_high - bc_high) / bc_high
            if dist > st_pct:
                continue
            if st_vol_ratio > bc_vol * st_vol:
                continue
            if st_high > bc_high * (1 + 0.005):
                continue
            st_found = True
            st_i     = j
            break

        if not st_found:
            continue

        # Upthrust — ложный пробой хая вверх
        for ut_i in range(st_i + 1, min(st_i + 15, len(df) - 1)):
            ut_row   = df.iloc[ut_i]
            ut_high  = ut_row["high"]
            ut_close = ut_row["close"]

            pct_above = (ut_high - bc_high) / bc_high
            if pct_above < ut_min or pct_above > ut_max:
                continue
            if ut_close > bc_high * (1 - ut_back):
                continue
            if ut_row["vol_ratio"] > bc_vol * 0.8:
                continue

            entry = bc_high * (1 - buf)
            stop  = ut_high * (1 + buf)
            risk  = stop - entry
            if risk <= 0:
                continue

            take_rr = entry - risk * rr
            take_ar = ar_low * 1.05
            take    = min(take_rr, take_ar)
            if take <= 0:
                continue

            close_now = df.iloc[-1]["close"]
            dist_now  = abs(close_now - entry) / entry
            if dist_now > 0.05:
                continue

            actual_rr = round((entry - take) / risk, 1)
            if actual_rr < 3:
                continue

            signals.append({
                "type":        "DISTRIBUTION",
                "dir":         -1,
                "entry_limit": round(entry, 6),
                "stop":        round(stop, 6),
                "take":        round(take, 6),
                "bc_high":     round(bc_high, 6),
                "ar_low":      round(ar_low, 6),
                "ut_high":     round(ut_high, 6),
                "bc_vol":      round(bc_vol, 1),
                "ut_pct":      round(pct_above * 100, 2),
                "rr":          actual_rr,
                "bc_bar":      bc_i,
                "ut_bar":      ut_i,
            })
            break

    return signals


# ─────────────────────────────────────────────
#  ОДИН ЦИКЛ
# ─────────────────────────────────────────────
def run_cycle(exchange, journal, cfg):
    now     = datetime.now().isoformat()
    balance = journal["balance"]
    comm    = cfg["commission"]

    # Автопополнение — если баланс упал ниже минимума для входа
    min_trade = cfg["initial_deposit"] * cfg["risk_pct"]
    if balance < min_trade and balance > 0:
        old_bal = balance
        journal["balance"] = cfg["initial_deposit"]
        balance = cfg["initial_deposit"]
        journal.setdefault("refills", []).append({
            "date": now,
            "from": round(old_bal, 4),
            "to":   cfg["initial_deposit"],
        })
        log(f"💰 АВТОПОПОЛНЕНИЕ: ${old_bal:.2f} → ${cfg['initial_deposit']:.2f} "
            f"(пополнений всего: {len(journal['refills'])})", cfg)

    # Автопополнение если баланс упал ниже 10% от депозита
    min_balance = cfg["initial_deposit"] * 0.1
    if balance <= min_balance and len(journal["open"]) == 0:
        old_bal = balance
        journal["balance"] = cfg["initial_deposit"]
        balance = cfg["initial_deposit"]
        journal["refills"] = journal.get("refills", 0) + 1
        print(f"  💰 АВТОПОПОЛНЕНИЕ #{journal['refills']}: "
              f"${old_bal:.2f} → ${balance:.2f}")

    # ── 1. Обновляем pending ──────────────────
    updated_pending = []
    updated_open    = list(journal["open"])

    for p in journal["pending"]:
        sym   = p["symbol"]
        price = get_current_price(exchange, sym)
        time.sleep(0.1)
        if price is None:
            updated_pending.append(p)
            continue

        p["bars_waited"] = p.get("bars_waited", 0) + 1
        entry   = p["entry_limit"]
        d       = p["dir"]
        filled  = (d ==  1 and price <= entry * 1.005) or \
                  (d == -1 and price >= entry * 0.995)
        expired = p["bars_waited"] >= cfg["max_wait_bars"]

        if filled:
            risk_usd      = balance * cfg["risk_pct"]
            risk_per_unit = abs(entry - p["stop"])
            qty = risk_usd / risk_per_unit if risk_per_unit > 0 else 0
            if qty > 0:
                pos = {**p, "qty": round(qty, 6),
                       "risk_usd": round(risk_usd, 4),
                       "opened_at": now}
                updated_open.append(pos)
                log(f"✅ ОТКРЫТА  {sym} "
                    f"{'ЛОНГ' if d==1 else 'ШОРТ'} "
                    f"[{p['wyckoff_type']}] "
                    f"вход={entry} стоп={p['stop']} "
                    f"тейк={p['take']} RR=1:{p['rr']} "
                    f"риск=${round(risk_usd,2)}", cfg)
        elif expired:
            log(f"⏱️  ОТМЕНЕНА {sym} [{p['wyckoff_type']}] "
                f"— не исполнилась", cfg)
        else:
            updated_pending.append(p)

    # ── 2. Обновляем open ─────────────────────
    still_open = []
    for pos in updated_open:
        sym   = pos["symbol"]
        price = get_current_price(exchange, sym)
        time.sleep(0.1)
        if price is None:
            still_open.append(pos)
            continue

        d     = pos["dir"]
        entry = pos["entry_limit"]
        stop  = pos["stop"]
        take  = pos["take"]
        qty   = pos["qty"]
        result = exit_p = None

        if d == 1:
            if price <= stop:   result, exit_p = "LOSS", stop
            elif price >= take: result, exit_p = "WIN",  take
        else:
            if price >= stop:   result, exit_p = "LOSS", stop
            elif price <= take: result, exit_p = "WIN",  take

        if result:
            pnl = (exit_p-entry)*qty*d - (entry+exit_p)*qty*comm
            balance += pnl
            trade = {**pos,
                     "result":        result,
                     "exit_price":    exit_p,
                     "pnl_usd":       round(pnl, 4),
                     "pnl_pct":       round(pnl/cfg["initial_deposit"]*100, 2),
                     "closed_at":     now,
                     "balance_after": round(balance, 4)}
            journal["trades"].append(trade)
            emoji = "🟢 WIN" if result == "WIN" else "🔴 LOSS"
            log(f"{emoji}  {sym} "
                f"[{pos.get('wyckoff_type','?')}] "
                f"PnL=${round(pnl,2):+.2f}  "
                f"баланс=${round(balance,2)}", cfg)
        else:
            still_open.append(pos)

    journal["open"]    = still_open
    journal["pending"] = updated_pending
    # Автопополнение если баланс упал ниже минимума
    min_balance = cfg["initial_deposit"] * 0.1  # 10% от депозита
    if balance <= min_balance:
        old_bal  = balance
        balance  = cfg["initial_deposit"]
        refills  = journal.get("refill_count", 0) + 1
        journal["refill_count"] = refills
        log(f"💰 АВТОПОПОЛНЕНИЕ #{refills}  "
            f"${old_bal:.2f} → ${balance:.2f}  "
            f"(баланс упал ниже ${min_balance:.2f})", cfg)

    journal["balance"] = round(balance, 4)

    # ── 3. Новые сигналы ──────────────────────
    open_count = len(journal["open"]) + len(journal["pending"])
    new_sigs   = 0

    if open_count >= cfg["max_open_trades"]:
        return journal, 0

    # Проверяем торговую сессию
    active, session_name = is_active_session(cfg)
    if not active:
        log(f"⏸️  {session_name} — пропускаем сканирование", cfg)
        return journal, 0
    log(f"✅ Сессия: {session_name}", cfg)

    existing = set()
    for p in journal["pending"] + journal["open"]:
        existing.add(f"{p['symbol']}_{p['dir']}")

    on_cooldown = set()
    for t in journal["trades"]:
        if t["result"] == "LOSS":
            closed = datetime.fromisoformat(t["closed_at"])
            age_h  = (datetime.now()-closed).total_seconds()/3600
            if age_h < cfg["cooldown_hours"]:
                on_cooldown.add(t["symbol"])

    try:
        symbols = get_symbols(exchange, cfg["min_usdt_vol"])
    except Exception:
        symbols = []

    for tf in cfg["timeframes"]:
        if open_count >= cfg["max_open_trades"]:
            break
        label = TIMEFRAME_LABELS.get(tf, tf)

        for sym in symbols[:60]:
            if open_count >= cfg["max_open_trades"]:
                break
            if sym in on_cooldown:
                continue

            df = fetch_candles(exchange, sym, tf, cfg["candles"], cfg)
            time.sleep(0.15)
            if df is None:
                continue

            # Ищем Wyckoff паттерны
            acc_sigs  = find_wyckoff_accumulation(df, cfg)
            dist_sigs = find_wyckoff_distribution(df, cfg)
            all_sigs  = acc_sigs + dist_sigs

            if not all_sigs:
                continue

            # Берём сигнал с лучшим RR
            all_sigs.sort(key=lambda x: x["rr"], reverse=True)
            sig = all_sigs[0]
            key = f"{sym}_{sig['dir']}"

            if key in existing:
                continue

            d_ru = "ЛОНГ" if sig["dir"] == 1 else "ШОРТ"
            pending_sig = {
                "symbol":       sym,
                "tf":           tf,
                "dir":          sig["dir"],
                "entry_limit":  sig["entry_limit"],
                "stop":         sig["stop"],
                "take":         sig["take"],
                "wyckoff_type": sig["type"],
                "rr":           sig["rr"],
                "added_at":     now,
                "bars_waited":  0,
            }
            journal["pending"].append(pending_sig)
            existing.add(key)
            open_count += 1
            new_sigs   += 1
            log(f"➕ WYCKOFF  {sym} [{tf}] {d_ru} "
                f"[{sig['type']}] RR=1:{sig['rr']} "
                f"лимит={sig['entry_limit']} "
                f"стоп={sig['stop']} тейк={sig['take']}", cfg)

    journal["scan_count"] = journal.get("scan_count", 0) + 1
    return journal, new_sigs


# ─────────────────────────────────────────────
#  СТАТИСТИКА
# ─────────────────────────────────────────────
def print_stats(journal, cfg):
    trades  = journal["trades"]
    balance = journal["balance"]
    init    = journal["initial_deposit"]
    scans   = journal.get("scan_count", 0)
    pnl     = balance - init
    pnl_pct = round(pnl / init * 100, 2)

    print(f"\n{'='*65}")
    print(f"  🎯 БОТ #4 — WYCKOFF + VSA | Bitget")
    print(f"  Старт: {journal['created'][:10]}  "
          f"Сканирований: {scans}")
    print(f"{'='*65}")
    s = "+" if pnl >= 0 else ""
    print(f"  Депозит:  ${init:.2f}")
    print(f"  Баланс:   ${balance:.2f}  "
          f"({s}${pnl:.2f}  {s}{pnl_pct}%)")

    if trades:
        wins   = [t for t in trades if t["result"] == "WIN"]
        losses = [t for t in trades if t["result"] == "LOSS"]
        wr     = round(len(wins)/len(trades)*100, 1)
        avg_w  = round(np.mean([t["pnl_usd"] for t in wins]),   2) if wins   else 0
        avg_l  = round(np.mean([t["pnl_usd"] for t in losses]), 2) if losses else 0
        longs  = [t for t in trades if t["dir"] ==  1]
        shorts = [t for t in trades if t["dir"] == -1]
        eq     = [init] + [t["balance_after"] for t in trades]
        peak   = np.maximum.accumulate(eq)
        dd     = round(((np.array(eq)-peak)/peak*100).min(), 2)
        avg_rr = round(np.mean([t.get("rr", 0) for t in trades]), 1)

        # По типам Wyckoff
        for wtype in ["ACCUMULATION","DISTRIBUTION"]:
            wt = [t for t in trades if t.get("wyckoff_type")==wtype]
            if wt:
                wt_wr = round(len([t for t in wt
                                   if t["result"]=="WIN"])/len(wt)*100,1)
                print(f"  {wtype:<14}: {len(wt):>3} сделок  WR={wt_wr}%")

        print(f"\n  Сделок:  {len(trades)}  "
              f"(🟢 {len(wins)}  🔴 {len(losses)})")
        print(f"  WR:      {wr}%")
        print(f"  Ср. RR:  1:{avg_rr}")
        print(f"  Ср. профит:  +${avg_w}")
        print(f"  Ср. убыток:  ${avg_l}")
        print(f"  Лонгов:  {len(longs)}  Шортов: {len(shorts)}")
        print(f"  Макс. просадка: {dd}%")

        print(f"\n  📋 Последние сделки:")
        print(f"  {'МОНЕТА':<14} {'TF':>3} {'ТИП':<16} "
              f"{'RR':>4} {'RES':>5} {'PNL$':>8} {'БАЛАНС':>8}")
        print(f"  {'-'*62}")
        for t in trades[-7:]:
            em = "🟢" if t["result"] == "WIN" else "🔴"
            d  = "Л" if t["dir"] == 1 else "Ш"
            wt = t.get("wyckoff_type","?")[:12]
            rr = t.get("rr", "?")
            print(f"  {em} {t['symbol']:<12} {t['tf']:>3} "
                  f"{wt:<14} 1:{rr} "
                  f"{t['result']:>5} {t['pnl_usd']:>+8.2f}$ "
                  f"${t['balance_after']:>7.2f}")

        pd.DataFrame(trades).to_csv("wyckoff_trades.csv", index=False)
        print(f"\n  [+] wyckoff_trades.csv обновлён")
    else:
        print(f"\n  Закрытых сделок пока нет — стратегия редкая,")
        print(f"  ждём качественные Wyckoff паттерны...")

    pending = journal["pending"]
    open_p  = journal["open"]
    if pending:
        print(f"\n  ⏳ Ожидают исполнения ({len(pending)}):")
        for p in pending:
            d = "ЛОНГ" if p["dir"] == 1 else "ШОРТ"
            print(f"     {p['symbol']:<16} [{p['tf']}] {d} "
                  f"[{p.get('wyckoff_type','?')}] "
                  f"RR=1:{p.get('rr','?')} "
                  f"лимит={p['entry_limit']}")
    if open_p:
        print(f"\n  🔓 Открытые позиции ({len(open_p)}):")
        for p in open_p:
            d = "ЛОНГ" if p["dir"] == 1 else "ШОРТ"
            print(f"     {p['symbol']:<16} [{p['tf']}] {d} "
                  f"[{p.get('wyckoff_type','?')}] "
                  f"вход={p['entry_limit']} "
                  f"стоп={p['stop']} тейк={p['take']}")

    print(f"\n  🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}")


# ─────────────────────────────────────────────
#  ОБЪЕДИНЁННАЯ СТАТИСТИКА
# ─────────────────────────────────────────────
def print_combined_stats():
    files = [
        ("paper_journal.json",    "Бот #1 EMA        "),
        ("structure_journal.json","Бот #2 Структура  "),
        ("smc_journal.json",      "Бот #3 SMC        "),
        ("wyckoff_journal.json",  "Бот #4 Wyckoff    "),
    ]
    all_trades = []
    total_init = 0

    print(f"\n{'='*70}")
    print(f"  📊 ОБЪЕДИНЁННАЯ СТАТИСТИКА — ВСЕ 4 БОТА")
    print(f"{'='*70}")
    print(f"  {'БОТ':<22} {'СДЕЛОК':>7} {'WR%':>6} "
          f"{'P&L$':>8} {'P&L%':>7} {'DD%':>7}")
    print(f"  {'-'*60}")

    for fname, label in files:
        if not os.path.exists(fname):
            print(f"  {label} — не запущен")
            continue
        with open(fname, "r", encoding="utf-8") as f:
            j = json.load(f)
        trades     = j["trades"]
        init       = j["initial_deposit"]
        bal        = j["balance"]
        pnl        = bal - init
        total_init += init
        if trades:
            wins = [t for t in trades if t["result"] == "WIN"]
            wr   = round(len(wins)/len(trades)*100, 1)
            eq   = [init] + [t["balance_after"] for t in trades]
            peak = np.maximum.accumulate(eq)
            dd   = round(((np.array(eq)-peak)/peak*100).min(), 2)
        else:
            wr, dd = 0, 0
        s = "+" if pnl >= 0 else ""
        print(f"  {label} {len(trades):>7} {wr:>5}%  "
              f"{s}{round(pnl,2):>7}$  "
              f"{s}{round(pnl/init*100,1):>5}%  {dd:>6}%")
        all_trades.extend(trades)

    if all_trades:
        wins     = [t for t in all_trades if t["result"] == "WIN"]
        total_wr = round(len(wins)/len(all_trades)*100, 1)
        print(f"\n  {'ИТОГО':<22} {len(all_trades):>7} {total_wr:>5}%")
    print(f"{'='*70}")


# ─────────────────────────────────────────────
#  ГЛАВНЫЙ ЦИКЛ
# ─────────────────────────────────────────────
def main():
    cfg  = CONFIG.copy()
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"

    if mode == "refill":
        journal = load_journal(cfg)
        old_bal = journal["balance"]
        journal["balance"] = cfg["initial_deposit"]
        save_journal(journal, cfg)
        print(f"[+] Баланс пополнен: ${old_bal:.2f} → ${cfg['initial_deposit']:.2f}")
        print(f"[+] История сделок сохранена: {len(journal['trades'])} сделок")
        return

    if mode == "reset":
        for f in [cfg["journal_file"], cfg["log_file"],
                  "wyckoff_trades.csv"]:
            if os.path.exists(f):
                os.remove(f)
        print("[+] Журнал сброшен.")
        return

    journal = load_journal(cfg)

    if mode == "status":
        print_stats(journal, cfg)
        return

    if mode == "combined":
        print_combined_stats()
        return

    interval = cfg["scan_interval"] * 60

    print("=" * 65)
    print(f"  🎯 БОТ #4 — WYCKOFF + VSA | Bitget")
    print(f"  Таймфреймы: {', '.join(cfg['timeframes'])}  |  Интервал: {cfg['scan_interval']}мин  SC объём: >{cfg['sc_vol_mult']}x")
    print(f"  Депозит: ${cfg['initial_deposit']}  "
          f"Риск: {int(cfg['risk_pct']*100)}%  "
          f"RR цель: 1:{cfg['rr_ratio']}")
    print(f"  SC объём: >{cfg['sc_vol_mult']}x  "
          f"Spring: {cfg['spring_pct']*100}-"
          f"{cfg['spring_max_pct']*100}%")
    print(f"  Стратегия редкая — ждём качественные паттерны!")
    print(f"  Остановка: Ctrl+C")
    print("=" * 65)

    log("="*50, cfg)
    log(f"СТАРТ БОТ#4 WYCKOFF  Bitget  "
        f"TF=1h/4h  депозит=${cfg['initial_deposit']}  "
        f"риск={int(cfg['risk_pct']*100)}%  "
        f"RR=1:{cfg['rr_ratio']}", cfg)

    ex = get_exchange()

    while True:
        try:
            log(f"--- ЦИКЛ #{journal.get('scan_count',0)+1} ---", cfg)
            journal, new_sigs = run_cycle(ex, journal, cfg)
            save_journal(journal, cfg)

            trades = journal["trades"]
            wins   = len([t for t in trades if t["result"]=="WIN"])
            wr     = round(wins/len(trades)*100,1) if trades else 0
            s      = "+" if journal["balance"] >= cfg["initial_deposit"] else ""
            log(f"Баланс=${journal['balance']}  "
                f"Сделок={len(trades)}  WR={wr}%  "
                f"P&L={s}"
                f"{round(journal['balance']-cfg['initial_deposit'],2)}$  "
                f"Новых паттернов={new_sigs}", cfg)

            log(f"Следующий цикл через "
                f"{cfg['scan_interval']} минут...", cfg)
            time.sleep(interval)

        except KeyboardInterrupt:
            log("ОСТАНОВЛЕН (Ctrl+C)", cfg)
            print("\n[!] Финальная статистика:")
            print_stats(journal, cfg)
            break
        except Exception as e:
            log(f"ОШИБКА: {e} — повтор через 5 минут", cfg)
            time.sleep(300)


if __name__ == "__main__":
    main()
