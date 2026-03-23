"""
paper_trading_auto.py
=====================
Автоматический режим — запустил один раз и оставил работать.
Сканирует каждые 30 минут, журнал обновляется сам.

Запуск:
  python paper_trading_auto.py          — старт (каждые 30 минут)
  python paper_trading_auto.py status   — статистика прямо сейчас
  python paper_trading_auto.py reset    — сбросить журнал

Остановка: Ctrl+C или закрыть окно CMD
"""

import sys, time, os, json
import ccxt
import pickle
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[!] sklearn не найден — pip install scikit-learn")

# Загружаем ML модель если есть
ML_MODEL = None
ML_THRESHOLD = 0.35
if SKLEARN_OK:
    try:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "ml_model_ema.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                ML_MODEL = pickle.load(f)
            ML_THRESHOLD = ML_MODEL.get("threshold", 0.35)
            print(f"[+] ML модель загружена  "
                  f"AUC={ML_MODEL.get('auc','?')}  "
                  f"порог={ML_THRESHOLD}  "
                  f"WR при пороге={ML_MODEL.get('threshold_wr','?')}%")
        else:
            print("[!] ML модель не найдена — работаем без ML")
    except Exception as e:
        print(f"[!] Ошибка загрузки ML: {e}")
        ML_MODEL = None

try:
    from divergence_module import calc_rsi, find_divergence
    HAS_DIV = True
except ImportError:
    HAS_DIV = False
import pandas as pd
import numpy as np
from datetime import datetime

CONFIG = {
    "timeframes":       ["5m", "15m"],
    "trend_timeframe":  "15m",           # только 15м определяет тренд BTC
    "candles":          300,
    "impulse_candles":  3,
    "min_price_chg":    0.003,       # 0.3% — для вялого рынка — ловим слабые импульсы
    "min_vol_mult":     1.2,          # снижено до 1.2x
    "vol_avg_period":   20,
    "fibo_entry":       0.236,       # изменено с 38.2% на 23.6% — ближе к цене
    "max_entry_miss":   0.05,        # увеличено до 5% — брать сигнал даже если цена ушла
    "min_usdt_vol":     1_000_000,
    "rr_ratio":          4.0,
    "commission":       0.001,
    "stop_buffer":      0.001,
    "max_wait_bars":    10,
    "btc_ema_period":   50,
    "btc_neutral_zone": 0.01,
    "initial_deposit":  50.0,
    "risk_pct":         0.05,
    "max_open_trades":  5,           # увеличено с 3 до 5
    "require_confirmation": True,    # монета должна быть сильнее/слабее BTC
    "confirmation_period":  20,      # свечей для сравнения с BTC
    "cooldown_minutes": 120,         # блокировка монеты после LOSS на 2 часа
    "journal_file":     "paper_journal.json",
    "log_file":         "paper_log.txt",
    "scan_interval":    10,          # минут между сканированиями
}

TIMEFRAME_LABELS = {"1m":"1м","5m":"5м","15m":"15м","1h":"1ч"}


# ─────────────────────────────────────────────
#  ФИЛЬТР ТОРГОВОЙ СЕССИИ
# ─────────────────────────────────────────────
def is_active_session(cfg):
    """
    Режим 24/7 — торгуем во все часы для сбора статистики.
    session_filter=False отключает ограничение по сессиям.
    Лондон:   07:00 - 12:00 UTC
    Нью-Йорк: 13:00 - 18:00 UTC
    Азия/ночь: торгуем (статистика не выявила преимущества сессий)
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
#  ЛОГИРОВАНИЕ
# ─────────────────────────────────────────────
def log(msg, cfg, show=True):
    """Пишет в лог-файл и в консоль одновременно."""
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if show:
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
        "created":          datetime.now().isoformat(),
        "initial_deposit":  cfg["initial_deposit"],
        "balance":          cfg["initial_deposit"],
        "trades":           [],
        "pending":          [],
        "open":             [],
        "cooldown":         {},
        "scan_count":       0,
    }


def save_journal(journal, cfg):
    with open(cfg["journal_file"], "w", encoding="utf-8") as f:
        json.dump(journal, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
#  БИРЖА
# ─────────────────────────────────────────────
def get_exchange():
    return ccxt.binance({"enableRateLimit": True})


def get_symbols(exchange, min_vol):
    tickers = exchange.fetch_tickers()
    symbols = [(s, t.get("quoteVolume") or 0)
               for s, t in tickers.items()
               if s.endswith("/USDT")
               and s != "BTC/USDT"
               and (t.get("quoteVolume") or 0) >= min_vol]
    symbols.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in symbols]


def fetch_candles(exchange, symbol, timeframe, limit):
    try:
        ms_map = {"1m":60000,"5m":300000,"15m":900000,"1h":3600000}
        ms    = ms_map.get(timeframe, 60000)
        since = exchange.milliseconds() - limit * ms - ms * 20
        raw   = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not raw or len(raw) < 50:
            return None
        df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.reset_index(drop=True)
    except Exception:
        return None


def get_current_price(exchange, symbol):
    try:
        return exchange.fetch_ticker(symbol)["last"]
    except Exception:
        return None


def get_btc_trend(exchange, timeframe, cfg):
    try:
        df = fetch_candles(exchange, "BTC/USDT", timeframe,
                           cfg["btc_ema_period"] + 20)
        if df is None:
            return "neutral", None, None, 0.0
        close  = df["close"]
        ema50  = close.ewm(span=cfg["btc_ema_period"], adjust=False).mean()
        lc     = close.iloc[-1]
        le     = ema50.iloc[-1]
        diff   = (lc - le) / le
        # Нет нейтральной зоны — строго выше или ниже EMA
        trend  = "bull" if diff > 0 else "bear"
        n       = cfg["impulse_candles"]
        btc_chg = (close.iloc[-1] - close.iloc[-1-n]) / close.iloc[-1-n]
        return trend, round(lc, 2), round(le, 2), round(btc_chg, 6)
    except Exception:
        return "neutral", None, None, 0.0


# ─────────────────────────────────────────────
#  ИНДИКАТОРЫ И СИГНАЛЫ
# ─────────────────────────────────────────────
def add_indicators(df, cfg):
    c, n, p = df["close"], cfg["impulse_candles"], cfg["vol_avg_period"]
    df["vol_avg"]      = df["volume"].rolling(p).mean()
    df["vol_ratio"]    = df["volume"] / df["vol_avg"].replace(0, 1e-9)
    df["price_chg"]    = (c - c.shift(n)) / c.shift(n)
    df["impulse_high"] = df["high"].rolling(n).max()
    df["impulse_low"]  = df["low"].rolling(n).min()
    return df


def get_coin_trend(df_1h):
    """Проверяем тренд монеты на 1ч — не берём лонг если монета падает"""
    if df_1h is None or len(df_1h) < 20:
        return "unknown"
    close = df_1h["close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    return "bull" if close.iloc[-1] > ema20.iloc[-1] else "bear"


def find_signals(df, cfg, btc_trend="neutral", btc_chg=0.0):
    """
    btc_chg — изменение BTC за последние N свечей.
    Логика подтверждения:
      ЛОНГ: разрешён если монета растёт СИЛЬНЕЕ чем BTC (относительная сила)
      ШОРТ: разрешён если монета падает СИЛЬНЕЕ чем BTC (относительная слабость)
      В боковике: оба направления без ограничений
    """
    min_chg = cfg["min_price_chg"]
    min_vol = cfg["min_vol_mult"]
    fibo    = cfg["fibo_entry"]
    buf     = cfg["stop_buffer"]
    rr      = cfg["rr_ratio"]
    start   = cfg["impulse_candles"] + cfg["vol_avg_period"]
    signals = []
    # В боковике — оба направления
    # В тренде — только направление тренда + подтверждение силы монеты
    # Строго: только направление тренда, нет нейтральной зоны
    allow_long  = btc_trend == "bull"
    allow_short = btc_trend == "bear"

    for i in range(start, len(df)):
        row = df.iloc[i]
        chg, vol = row["price_chg"], row["vol_ratio"]
        if abs(chg) < min_chg or vol < min_vol:
            continue
        imp_high = row["impulse_high"]
        imp_low  = row["impulse_low"]
        spread   = imp_high - imp_low
        if spread <= 0:
            continue

        if chg > 0 and allow_long:
            entry = imp_low + spread * fibo
            stop  = imp_low * (1 - buf)
            risk  = entry - stop
            if risk <= 0 or entry >= imp_high:
                continue
            take = entry + risk * rr
            signals.append({"bar": i, "dir": 1,
                             "entry_limit": round(entry, 6),
                             "stop": round(stop, 6),
                             "take": round(take, 6),
                             "chg_pct": round(chg*100, 2),
                             "vol_ratio": round(vol, 1)})

        elif chg < 0 and allow_short:
            entry = imp_high - spread * fibo
            stop  = imp_high * (1 + buf)
            risk  = stop - entry
            if risk <= 0 or entry <= imp_low:
                continue
            take = entry - risk * rr
            if take <= 0:
                continue
            signals.append({"bar": i, "dir": -1,
                             "entry_limit": round(entry, 6),
                             "stop": round(stop, 6),
                             "take": round(take, 6),
                             "chg_pct": round(chg*100, 2),
                             "vol_ratio": round(vol, 1)})
    return signals


# ─────────────────────────────────────────────
#  ОДИН ЦИКЛ СКАНИРОВАНИЯ
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
        time.sleep(0.08)
        if price is None:
            updated_pending.append(p)
            continue

        p["bars_waited"] = p.get("bars_waited", 0) + 1
        entry   = p["entry_limit"]
        d       = p["dir"]
        filled  = (d == 1 and price <= entry * 1.005) or \
                  (d == -1 and price >= entry * 0.995)
        expired = p["bars_waited"] >= cfg["max_wait_bars"]

        if filled:
            risk_usd      = balance * cfg["risk_pct"]
            risk_per_unit = abs(entry - p["stop"])
            qty = risk_usd / risk_per_unit if risk_per_unit > 0 else 0
            if qty > 0:
                pos = {**p, "qty": round(qty, 6),
                       "risk_usd": round(risk_usd, 4),
                       "opened_at": now, "status": "open"}
                updated_open.append(pos)
                log(f"✅ ОТКРЫТА  {sym} {'ЛОНГ' if d==1 else 'ШОРТ'} "
                    f"вход={entry} стоп={p['stop']} тейк={p['take']} "
                    f"риск=${round(risk_usd,2)}", cfg)
        elif expired:
            log(f"⏱️  ОТМЕНЕНА {sym} — не исполнилась за "
                f"{p['bars_waited']} циклов", cfg)
        else:
            updated_pending.append(p)

    # ── 2. Обновляем open ─────────────────────
    still_open = []
    for pos in updated_open:
        sym   = pos["symbol"]
        price = get_current_price(exchange, sym)
        time.sleep(0.08)
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
            pnl = (exit_p - entry) * qty * d - (entry + exit_p) * qty * comm
            balance += pnl
            trade = {**pos,
                     "result":       result,
                     "exit_price":   exit_p,
                     "pnl_usd":      round(pnl, 4),
                     "pnl_pct":      round(pnl / cfg["initial_deposit"] * 100, 2),
                     "closed_at":    now,
                     "balance_after":round(balance, 4)}
            journal["trades"].append(trade)
            # После LOSS — блокируем монету на cooldown_minutes
            if result == "LOSS":
                from datetime import timedelta
                unblock = (datetime.now() + timedelta(minutes=cfg["cooldown_minutes"])).isoformat()
                journal.setdefault("cooldown", {})[sym] = unblock
                log(f"⛔ КУЛДАУН  {sym} заблокирована до {unblock[:16]}", cfg)
            emoji = "🟢 WIN" if result == "WIN" else "🔴 LOSS"
            log(f"{emoji}  {sym} {'ЛОНГ' if d==1 else 'ШОРТ'} "
                f"PnL=${round(pnl,2):+.2f}  баланс=${round(balance,2)}", cfg)
        else:
            cur_pnl = round((price - entry) * qty * d, 2)
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

    if open_count < cfg["max_open_trades"]:
        existing = set()
        for p in journal["pending"] + journal["open"]:
            existing.add(f"{p['symbol']}_{p['dir']}")

        # Cooldown: пропускаем монеты заблокированные после LOSS
        on_cooldown = set()
        for sym_cd, unblock_str in journal.get("cooldown", {}).items():
            try:
                if datetime.now() < datetime.fromisoformat(unblock_str):
                    on_cooldown.add(sym_cd)
            except Exception:
                pass

        try:
            symbols = get_symbols(exchange, cfg["min_usdt_vol"])
        except Exception:
            symbols = []

        for tf in cfg["timeframes"]:
            if open_count >= cfg["max_open_trades"]:
                break
            # Тренд ВСЕГДА определяем по 15м — старший ТФ главный
            trend_tf = cfg.get("trend_timeframe", "15m")
            btc_trend, btc_p, btc_ema, btc_chg = get_btc_trend(exchange, trend_tf, cfg)

            trend_ru = {"bull":"🟢 БЫЧИЙ→лонги", "bear":"🔴 МЕДВЕЖИЙ→шорты",
                        "neutral":"🟡 БОКОВИК→лонги+шорты"}.get(btc_trend, btc_trend)
            log(f"BTC [{tf}] ${btc_p}  EMA50=${btc_ema}  {trend_ru}  "
                f"allow_long={btc_trend == 'bull'}  "
                f"allow_short={btc_trend == 'bear'}", cfg, show=True)

            for sym in symbols[:80]:
                if open_count >= cfg["max_open_trades"]:
                    break
                df = fetch_candles(exchange, sym, tf, cfg["candles"])
                time.sleep(0.08)
                if df is None:
                    continue
                df   = add_indicators(df, cfg)

                # Проверяем тренд монеты на 1ч
                df_1h      = fetch_candles(exchange, sym, "1h", 30)
                coin_trend = get_coin_trend(df_1h)

                # Фильтр: не берём лонг если монета на 1ч падает
                #         не берём шорт если монета на 1ч растёт
                effective_trend = btc_trend
                if btc_trend == "bull" and coin_trend == "bear":
                    continue  # BTC растёт но монета падает — пропускаем
                if btc_trend == "bear" and coin_trend == "bull":
                    continue  # BTC падает но монета растёт — пропускаем

                sigs = find_signals(df, cfg, effective_trend, btc_chg)
                recent = [s for s in sigs if s["bar"] >= len(df) - 3]
                if not recent:
                    continue

                sig = recent[-1]
                key = f"{sym}_{sig['dir']}"
                if key in existing:
                    continue

                # Пропускаем если монета на cooldown после LOSS
                if sym in on_cooldown:
                    log(f"⏸️  COOLDOWN {sym} — заблокирована после LOSS ({cfg['cooldown_minutes']} мин)", cfg)
                    continue

                current = df.iloc[-1]["close"]
                entry   = sig["entry_limit"]
                d       = sig["dir"]
                diff    = ((current - entry) / entry if d == 1
                           else (entry - current) / entry)
                if diff > cfg["max_entry_miss"]:
                    continue

                # Фильтр объёма — берём сигнал только если текущий объём выше среднего
                last_vol = df.iloc[-2]["vol_ratio"] if "vol_ratio" in df.columns else 1.0
                if last_vol < 1.2:
                    continue  # объём слабый — пропускаем

                pending_sig = {
                    "symbol":      sym, "tf": tf, "dir": d,
                    "entry_limit": entry,
                    "stop":        sig["stop"],
                    "take":        sig["take"],
                    "chg_pct":     sig["chg_pct"],
                    "vol_ratio":   sig["vol_ratio"],
                    "btc_trend":   btc_trend,
                    "added_at":    now,
                    "bars_waited": 0,
                }
                journal["pending"].append(pending_sig)
                existing.add(key)
                open_count += 1
                new_sigs   += 1
                d_ru = "ЛОНГ" if d == 1 else "ШОРТ"
                log(f"➕ СИГНАЛ   {sym} [{tf}] {d_ru} "
                    f"лимит={entry} стоп={sig['stop']} тейк={sig['take']}", cfg)

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
    total_pnl = balance - init
    pnl_pct   = round(total_pnl / init * 100, 2)

    print(f"\n{'='*60}")
    print(f"  📊 БУМАЖНЫЙ ТРЕЙДИНГ — СТАТИСТИКА")
    print(f"  Старт: {journal['created'][:10]}  "
          f"Сканирований: {scans}")
    print(f"{'='*60}")
    s = "+" if total_pnl >= 0 else ""
    print(f"  💰 Начальный депозит:  ${init:.2f}")
    print(f"  💵 Текущий баланс:     ${balance:.2f}")
    print(f"  📈 Прибыль/убыток:     {s}${total_pnl:.2f}  ({s}{pnl_pct}%)")

    if trades:
        wins    = [t for t in trades if t["result"] == "WIN"]
        losses  = [t for t in trades if t["result"] == "LOSS"]
        wr      = round(len(wins) / len(trades) * 100, 1)
        avg_w   = round(np.mean([t["pnl_usd"] for t in wins]),   2) if wins   else 0
        avg_l   = round(np.mean([t["pnl_usd"] for t in losses]), 2) if losses else 0
        longs   = [t for t in trades if t["dir"] ==  1]
        shorts  = [t for t in trades if t["dir"] == -1]
        eq      = [init] + [t["balance_after"] for t in trades]
        peak    = np.maximum.accumulate(eq)
        dd      = round(((np.array(eq) - peak) / peak * 100).min(), 2)

        print(f"\n  Сделок:      {len(trades)}  "
              f"(🟢 {len(wins)} побед  🔴 {len(losses)} потерь)")
        print(f"  WR:          {wr}%")
        print(f"  Ср. профит:  +${avg_w}")
        print(f"  Ср. убыток:  ${avg_l}")
        print(f"  Лонгов:      {len(longs)}  Шортов: {len(shorts)}")
        print(f"  Макс. просадка: {dd}%")

        print(f"\n  📋 Последние 7 сделок:")
        print(f"  {'МОНЕТА':<16} {'TF':>3} {'ТИП':>5} "
              f"{'RES':>5} {'PNL$':>8} {'БАЛАНС':>8}")
        print(f"  {'-'*54}")
        for t in trades[-7:]:
            em = "🟢" if t["result"] == "WIN" else "🔴"
            d  = "ЛОНГ" if t["dir"] == 1 else "ШОРТ"
            print(f"  {em} {t['symbol']:<14} {t['tf']:>3} {d:>5} "
                  f"{t['result']:>5} {t['pnl_usd']:>+8.2f}$ "
                  f"${t['balance_after']:>7.2f}")
        pd.DataFrame(trades).to_csv("paper_trades.csv", index=False)
        print(f"\n  [+] paper_trades.csv обновлён")
    else:
        print(f"\n  Закрытых сделок пока нет — ждём...")

    # Текущие позиции
    pending = journal["pending"]
    open_p  = journal["open"]
    if pending:
        print(f"\n  ⏳ Ожидают лимитки ({len(pending)}):")
        for p in pending:
            d = "ЛОНГ" if p["dir"] == 1 else "ШОРТ"
            print(f"     {p['symbol']:<16} [{p['tf']}] {d}  "
                  f"лимит={p['entry_limit']}  "
                  f"ждём {p.get('bars_waited',0)} цикл.")
    if open_p:
        print(f"\n  🔓 Открытые позиции ({len(open_p)}):")
        for p in open_p:
            d = "ЛОНГ" if p["dir"] == 1 else "ШОРТ"
            print(f"     {p['symbol']:<16} [{p['tf']}] {d}  "
                  f"вход={p['entry_limit']}  "
                  f"стоп={p['stop']}  тейк={p['take']}")

    print(f"\n  🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


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
        for f in [cfg["journal_file"], cfg["log_file"], "paper_trades.csv"]:
            if os.path.exists(f):
                os.remove(f)
        print("[+] Журнал сброшен. Начинаем заново.")
        return

    journal = load_journal(cfg)

    if mode == "status":
        print_stats(journal, cfg)
        return

    # ── АВТОМАТИЧЕСКИЙ РЕЖИМ ──────────────────
    interval = cfg["scan_interval"] * 60  # в секундах

    print(f"\n  Депозит: ${cfg['initial_deposit']}  "
          f"Риск: {int(cfg['risk_pct']*100)}%/сделку  "
          f"RR 1:{cfg['rr_ratio']}")
    print(f"  Интервал сканирования: каждые {cfg['scan_interval']} минут")
    print(f"  Лог: {cfg['log_file']}")
    print(f"  Остановка: Ctrl+C\n")
    log("="*50, cfg)
    log(f"СТАРТ  депозит=${cfg['initial_deposit']}  "
        f"риск={int(cfg['risk_pct']*100)}%  "
        f"интервал={cfg['scan_interval']}мин", cfg)

    ex = get_exchange()

    while True:
        try:
            log(f"--- ЦИКЛ #{journal.get('scan_count',0)+1} ---", cfg)
            journal, new_sigs = run_cycle(ex, journal, cfg)
            save_journal(journal, cfg)

            trades = journal["trades"]
            wins   = len([t for t in trades if t["result"] == "WIN"])
            wr     = round(wins/len(trades)*100,1) if trades else 0
            s      = "+" if journal["balance"] >= cfg["initial_deposit"] else ""
            log(f"Баланс=${journal['balance']}  "
                f"Сделок={len(trades)}  WR={wr}%  "
                f"P&L={s}{round(journal['balance']-cfg['initial_deposit'],2)}$  "
                f"Новых сигналов={new_sigs}", cfg)

            log(f"Следующий цикл через {cfg['scan_interval']} минут...", cfg)
            print(f"\n  Следующий запуск: "
                  f"{datetime.now().strftime('%H:%M')} + "
                  f"{cfg['scan_interval']} мин\n")

            time.sleep(interval)

        except KeyboardInterrupt:
            log("ОСТАНОВЛЕН пользователем (Ctrl+C)", cfg)
            print("\n\n[!] Остановлен. Финальная статистика:")
            print_stats(journal, cfg)
            break
        except Exception as e:
            log(f"ОШИБКА: {e} — повтор через 5 минут", cfg)
            time.sleep(300)


if __name__ == "__main__":
    main()
