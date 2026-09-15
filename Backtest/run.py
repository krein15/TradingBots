"""
Backtest/run.py
===============
Запуск бэктеста Бота #1 на исторических данных.

Стратегия берётся НЕ пересказом, а импортом настоящих функций
add_indicators / find_signals из самого бота. Если правишь бота —
бэктест автоматически проверяет новую версию, разойтись они не могут.

Примеры:
  python Backtest/run.py --start 2026-06-01 --end 2026-09-01
  python Backtest/run.py --symbols 30 --fill touch
  python Backtest/run.py --compare-fills

Ключи:
  --start/--end     период (UTC)
  --symbols N       сколько самых ликвидных пар взять
  --fill            limit | touch | market
  --compare-fills   прогнать все три модели исполнения и сравнить
  --leverage        предел плеча (1.0 = спот без плеча)
  --risk            риск на сделку (0.05 = 5%)
"""

import argparse
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

from Backtest import data, context, metrics
from Backtest.engine import Engine, FILL_LIMIT, FILL_TOUCH, FILL_MARKET

# Настоящий код бота — не копия
sys.path.insert(0, str(data.ROOT / "Bot1_EMA"))
import paper_trading_v2_clean as bot1


def build_symbol_data(exchange_id, symbols, timeframe, start, end, cfg,
                      exchange=None, verbose=True):
    """Качаем свечи и считаем индикаторы теми же функциями, что и бот."""
    out = {}
    for n, sym in enumerate(symbols, 1):
        df = data.load(exchange_id, sym, timeframe, start, end,
                       exchange=exchange)
        if df is None or len(df) < 120:
            if verbose:
                print(f"  [{n}/{len(symbols)}] {sym:<16} — данных мало, пропуск")
            continue
        df = bot1.add_indicators(df.copy(), cfg)
        out[sym] = df
        if verbose:
            print(f"  [{n}/{len(symbols)}] {sym:<16} {len(df)} свечей")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--end", default="2026-09-01")
    ap.add_argument("--symbols", type=int, default=15)
    ap.add_argument("--exchange", default="bitget")
    ap.add_argument("--timeframe", default="5m")
    ap.add_argument("--fill", default=FILL_LIMIT,
                    choices=[FILL_LIMIT, FILL_TOUCH, FILL_MARKET])
    ap.add_argument("--compare-fills", action="store_true")
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--risk", type=float, default=None)
    ap.add_argument("--deposit", type=float, default=None)
    ap.add_argument("--slippage", type=float, default=0.0005)
    ap.add_argument("--out", default=None, help="куда сохранить сделки (CSV)")
    args = ap.parse_args()

    cfg = bot1.CONFIG.copy()
    deposit = args.deposit if args.deposit is not None else cfg["initial_deposit"]
    risk = args.risk if args.risk is not None else cfg["risk_pct"]

    print("=" * 64)
    print("  БЭКТЕСТ — Бот #1 (адаптивный)")
    print("=" * 64)
    print(f"  Период:     {args.start} .. {args.end}")
    print(f"  Биржа:      {args.exchange}   таймфрейм: {args.timeframe}")
    print(f"  Депозит:    ${deposit}   риск: {risk:.0%}   "
          f"плечо<= {args.leverage}x")
    print(f"  Комиссия:   {cfg['commission']:.3%}   "
          f"проскальзывание: {args.slippage:.3%}")
    print()

    ex = data.get_exchange(args.exchange)

    print("  Список инструментов...")
    symbols = data.top_symbols(args.exchange, limit=args.symbols,
                               min_quote_vol=cfg["min_usdt_vol"], exchange=ex)
    print(f"  Взято пар: {len(symbols)}")
    print("  ВНИМАНИЕ: список построен по СЕГОДНЯШНИМ объёмам — это")
    print("  survivorship bias, монеты-неудачники сюда не попали.\n")

    print("  Загрузка свечей (первый раз из сети, дальше из кэша)...")
    t0 = time.time()
    symbols_data = build_symbol_data(args.exchange, symbols, args.timeframe,
                                     args.start, args.end, cfg, exchange=ex)
    print(f"  Готово за {time.time() - t0:.0f}с, инструментов: {len(symbols_data)}\n")

    if not symbols_data:
        print("[!] Нет данных — нечего тестировать")
        return

    # Детектору режима нужно 100 часовых свечей разгона, иначе первые
    # четверо суток бэктеста идут с режимом "?" — а это другая ветка
    # стратегии. Поэтому BTC грузим с запасом ДО начала периода.
    print("  Рыночный контекст по BTC...")
    lead_1h = pd.Timestamp(args.start, tz="UTC") - pd.Timedelta(hours=context.REGIME_WINDOW + 10)
    btc_1h = data.load(args.exchange, "BTC/USDT", "1h",
                       lead_1h.isoformat(), args.end, exchange=ex)

    btc_trend_tf = cfg.get("trend_timeframe", "15m")
    lead_tf = pd.Timestamp(args.start, tz="UTC") - pd.Timedelta(
        milliseconds=data.TF_MS[btc_trend_tf] * (cfg["btc_ema_period"] + 20))
    btc_tf = data.load(args.exchange, "BTC/USDT", btc_trend_tf,
                       lead_tf.isoformat(), args.end, exchange=ex)

    regime_lookup = btc_lookup = None
    if btc_1h is not None:
        rs = context.regime_series(btc_1h)
        regime_lookup = context.ContextLookup(rs, "1h", ["regime", "confidence"])
        start_ms = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000)
        in_period = rs[(rs.regime != "?") & (rs.timestamp >= start_ms)]
        dist = in_period.regime.value_counts()
        print(f"  Режимы за период: "
              + "  ".join(f"{k}={v}" for k, v in dist.items()))
    if btc_tf is not None:
        ts_ = context.btc_trend_series(btc_tf, cfg["btc_ema_period"],
                                       cfg["impulse_candles"])
        btc_lookup = context.ContextLookup(ts_, btc_trend_tf, ["trend", "btc_chg"])
        bull = (ts_.trend == "bull").mean() * 100
        print(f"  BTC бычий {bull:.0f}% времени, медвежий {100 - bull:.0f}%")
    print()

    fills = ([FILL_LIMIT, FILL_TOUCH, FILL_MARKET] if args.compare_fills
             else [args.fill])
    summaries, engines = [], {}

    for fm in fills:
        titles = {
            FILL_LIMIT: "Лимитка — цена должна дойти до уровня (честно)",
            FILL_TOUCH: "Касание +-0.3% — модель бумажного бота",
            FILL_MARKET: "Рынок по открытию следующей свечи",
        }
        print(f"  Прогон: {titles[fm]}")
        t0 = time.time()
        eng = Engine(
            cfg=cfg, signal_fn=bot1.find_signals,
            deposit=deposit, risk_pct=risk,
            max_open=cfg["max_open_trades"],
            commission=cfg["commission"], slippage=args.slippage,
            fill_model=fm, touch_tolerance=0.003,
            max_wait_bars=cfg["max_wait_bars"],
            cooldown_minutes=cfg["cooldown_minutes"],
            bad_hours=cfg.get("bad_hours", ()),
            max_leverage=args.leverage,
        ).run(symbols_data, regime_lookup, btc_lookup)
        print(f"  ...{time.time() - t0:.0f}с\n")

        s = metrics.print_report(eng, titles[fm],
                                 show_breakdowns=not args.compare_fills)
        s["label"] = fm
        summaries.append(s)
        engines[fm] = eng
        print()

    if len(summaries) > 1:
        print("=" * 64)
        print("  СРАВНЕНИЕ МОДЕЛЕЙ ИСПОЛНЕНИЯ")
        print("=" * 64)
        print(metrics.compare(summaries).to_string(index=False))
        print()

    out_path = args.out or (data.ROOT / "Backtest" / "backtest_trades.csv")
    eng = engines[fills[0]]
    tdf = eng.trades_df()
    if len(tdf):
        tdf.to_csv(out_path, index=False)
        print(f"  Сделки сохранены: {out_path}")


if __name__ == "__main__":
    main()
