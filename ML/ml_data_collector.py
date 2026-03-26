"""
ml_data_collector.py
====================
Сборщик данных для обучения ML модели.

Поддерживает все 4 бота:
  Bot1 EMA       — paper_journal.json
  Bot2 MeanRev   — meanrev_journal.json
  Bot3 Funding   — funding_journal.json
  Bot4 Breakout  — breakout_journal.json

Запуск:
  python ml_data_collector.py

Результат:
  ML\\ml_dataset.csv
  ML\\ml_dataset_summary.txt
"""

import os, json
import pandas as pd
import numpy as np
from datetime import datetime

# ── Пути к журналам ───────────────────────────────────────────
JOURNALS = [
    {
        "name": "EMA",
        "path": "C:\\TradingBots\\Bot1_EMA\\paper_journal.json",
    },
    {
        "name": "MeanRev",
        "path": "C:\\TradingBots\\Bot2_MeanRev\\meanrev_journal.json",
    },
    {
        "name": "Funding",
        "path": "C:\\TradingBots\\Bot3_Funding\\funding_journal.json",
    },
    {
        "name": "Breakout",
        "path": "C:\\TradingBots\\Bot4_Breakout\\breakout_journal.json",
    },
]

OUTPUT_DIR          = "C:\\TradingBots\\ML"
REGIME_HISTORY_PATH = "C:\\TradingBots\\ML\\regime_history.jsonl"
SHARED_STATE_PATH   = "C:\\TradingBots\\shared_state.json"

INITIAL_DEPOSIT = 50.0  # для расчёта risk_pct когда stop не сохранён


def load_regime_history():
    """
    Загружаем историю режимов из regime_history.jsonl.
    Возвращаем список записей отсортированных по времени.
    """
    records = []
    try:
        with open(REGIME_HISTORY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        records.sort(key=lambda x: x["ts"])
    except Exception:
        pass
    return records


def get_regime_at(closed_at, regime_records):
    """
    Находим режим рынка в момент закрытия сделки.
    Берём последнюю запись режима ДО closed_at.
    """
    if not regime_records or not closed_at:
        return "?", 0
    best = None
    for r in regime_records:
        if r["ts"] <= closed_at:
            best = r
        else:
            break
    if best:
        return best["regime"], best["confidence"]
    return "?", 0


def load_journal(bot):
    p = bot.get("path")
    if p and os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def parse_time(closed_at):
    """Разбираем дату и возвращаем временные признаки."""
    try:
        dt = datetime.fromisoformat(closed_at)
        hour        = dt.hour
        day_of_week = dt.weekday()
        is_weekend  = 1 if day_of_week >= 5 else 0
        is_night    = 1 if (hour >= 22 or hour < 6) else 0
        is_london   = 1 if 7 <= hour < 12 else 0
        is_newyork  = 1 if 13 <= hour < 18 else 0
        is_asia     = 1 if hour < 7 else 0
        return hour, day_of_week, is_weekend, is_night, is_london, is_newyork, is_asia
    except Exception:
        return 0, 0, 0, 0, 0, 0, 0


def extract_features(trade, bot_name, deposit, regime_records=None):
    """
    Извлекаем признаки из сделки.
    Поддерживает оба формата журналов:
      - старый (Bot1 EMA): exit_price, closed_at, vol_ratio, stop, take
      - новый (Bot2/3/4):  exit, closed, vol, (без stop/take в trades)
    """
    # ── Время ──────────────────────────────────────────────────
    closed_at = trade.get("closed_at") or trade.get("closed", "")
    hour, dow, is_we, is_night, is_lon, is_ny, is_asia = parse_time(closed_at)

    # ── Цены ───────────────────────────────────────────────────
    entry  = float(trade.get("entry", trade.get("entry_limit", 0)))
    stop   = float(trade.get("stop",  0))
    take   = float(trade.get("take",  0))
    # exit_price — старый формат, exit — новый
    exit_p = float(trade.get("exit_price") or trade.get("exit", 0))
    pnl    = float(trade.get("pnl_usd") or trade.get("pnl", 0))
    result = trade.get("result", "LOSS")
    d      = int(trade.get("dir", 1))

    # ── Risk/Reward ─────────────────────────────────────────────
    if stop > 0 and entry > 0:
        # Старый формат — есть stop/take
        risk   = abs(entry - stop)
        reward = abs(take - entry)
    else:
        # Новый формат — считаем из реального PnL
        # risk_pct = |pnl| / deposit (примерно, т.к. риск фиксирован 5%)
        risk   = abs(pnl) / deposit * entry if deposit > 0 and entry > 0 else 0
        # rr берём из конфига бота по типу
        sig_type = trade.get("type", "")
        rr_map   = {"MR": 2.0, "FR": 3.0, "BO": 3.0, "EMA": 4.0}
        rr_guess = next((v for k, v in rr_map.items() if k in sig_type), 3.0)
        reward   = risk * rr_guess

    rr_plan  = round(reward / risk, 2) if risk > 0 else 0
    stop_pct = round(risk / entry * 100, 3) if entry > 0 else 0
    take_pct = round(reward / entry * 100, 3) if entry > 0 else 0

    # ── Прогресс до TP ──────────────────────────────────────────
    if take > 0 and entry > 0 and take != entry:
        if d == 1:
            exit_progress = round((exit_p - entry) / (take - entry) * 100, 1)
        else:
            exit_progress = round((entry - exit_p) / (entry - take) * 100, 1)
    else:
        exit_progress = 100.0 if result == "WIN" else -round(1 / rr_plan * 100, 1) if rr_plan > 0 else -33.3

    # ── Таймфрейм ───────────────────────────────────────────────
    tf = trade.get("tf", "15m")
    tf_minutes = {"1m":1,"5m":5,"15m":15,"30m":30,
                  "1h":60,"4h":240,"1d":1440}.get(tf, 15)

    # ── Объём ───────────────────────────────────────────────────
    vol_ratio = float(
        trade.get("vol_ratio") or   # Bot1 EMA
        trade.get("vol") or         # Bot2/3/4
        0
    )

    # ── Специфичные признаки ────────────────────────────────────
    rsi       = float(trade.get("rsi",       0))
    adx       = float(trade.get("adx",       0))
    bb_width  = float(trade.get("bb_width",  0))
    fr        = float(trade.get("fr",        0))   # Funding Rate
    atr_ratio = float(trade.get("atr_ratio", 0))   # Breakout

    # Wyckoff/SMC (если когда-нибудь вернутся)
    sc_drop    = float(trade.get("sc_drop_pct",   0))
    ar_bounce  = float(trade.get("ar_bounce_pct", 0))
    spring_wick= float(trade.get("spring_wick",   0))

    # ── Сигнал и направление ────────────────────────────────────
    sig_type  = trade.get("structure") or trade.get("type") or trade.get("signal_type", "UNKNOWN")
    direction = "LONG" if d == 1 else "SHORT"
    target    = 1 if result == "WIN" else 0

    # Режим рынка в момент сделки
    # Приоритет: поле из журнала → история режимов → неизвестно
    regime      = trade.get("regime", "?")
    regime_conf = int(trade.get("regime_conf", 0))
    if regime == "?" and regime_records:
        regime, regime_conf = get_regime_at(closed_at, regime_records)

    return {
        # Идентификаторы
        "bot":           bot_name,
        "symbol":        trade.get("symbol", ""),
        "tf":            tf,
        "tf_minutes":    tf_minutes,
        "direction":     direction,
        "signal_type":   sig_type,
        "closed_at":     closed_at,

        # Цены
        "entry":         round(entry, 6),
        "stop":          round(stop, 6),
        "take":          round(take, 6),
        "exit_price":    round(exit_p, 6),

        # Риск-менеджмент
        "risk_pct":      stop_pct,
        "reward_pct":    take_pct,
        "rr_planned":    rr_plan,
        "exit_progress": exit_progress,

        # Время
        "hour":          hour,
        "day_of_week":   dow,
        "is_weekend":    is_we,
        "is_night":      is_night,
        "is_london":     is_lon,
        "is_newyork":    is_ny,
        "is_asia":       is_asia,

        # Индикаторы
        "vol_ratio":     vol_ratio,
        "rsi":           rsi,
        "adx":           adx,
        "bb_width":      bb_width,
        "funding_rate":  fr,
        "atr_ratio":     atr_ratio,

        # Wyckoff/SMC (legacy)
        "sc_drop_pct":   sc_drop,
        "ar_bounce_pct": ar_bounce,
        "spring_wick":   spring_wick,

        # Режим рынка
        "regime":        regime,
        "regime_conf":   regime_conf,

        # Результат
        "pnl_usd":       round(pnl, 4),
        "result":        result,
        "target":        target,
    }


def build_dataset():
    all_rows = []
    summary  = []

    print("=" * 60)
    print(f"  СБОРЩИК ML ДАТАСЕТА")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Загружаем историю режимов рынка
    regime_records = load_regime_history()
    if regime_records:
        print(f"  Загружено записей режима: {len(regime_records)} "
              f"(с {regime_records[0]['ts'][:10]} по {regime_records[-1]['ts'][:10]})")
    else:
        print("  [!] regime_history.jsonl не найден — запусти market_regime.py")

    for bot in JOURNALS:
        journal = load_journal(bot)
        if not journal:
            print(f"\n  ? {bot['name']} — журнал не найден ({bot['path']})")
            continue

        trades = journal.get("trades", [])
        if not trades:
            print(f"\n  ! {bot['name']} — нет завершённых сделок")
            continue

        deposit = float(journal.get("deposit", INITIAL_DEPOSIT))
        rows = [extract_features(t, bot["name"], deposit, regime_records) for t in trades]
        all_rows.extend(rows)

        wins = sum(1 for r in rows if r["target"] == 1)
        wr   = round(wins / len(rows) * 100, 1)

        print(f"\n  OK {bot['name']}: {len(rows)} сделок  WR={wr}%")
        df_bot = pd.DataFrame(rows)
        print(f"     Лонгов: {(df_bot['direction']=='LONG').sum()}  "
              f"Шортов: {(df_bot['direction']=='SHORT').sum()}")
        print(f"     Ср. объём: {df_bot['vol_ratio'].mean():.2f}x  "
              f"Ср. стоп: {df_bot['risk_pct'].mean():.3f}%")

        summary.append({"bot": bot["name"], "trades": len(rows), "wins": wins, "wr": wr})

    if not all_rows:
        print("\n[!] Нет данных для датасета")
        input("Enter...")
        return

    df = pd.DataFrame(all_rows)
    total_wr = round(df["target"].mean() * 100, 1)

    print(f"\n{'='*60}")
    print(f"  ИТОГО: {len(df)} сделок  WR={total_wr}%")
    print(f"{'='*60}")

    # ── Топ признаков ─────────────────────────────────────────
    print(f"\n  ТОП ПРИЗНАКОВ (корреляция с WIN):")
    skip = {"target","entry","stop","take","exit_price","pnl_usd","exit_progress"}
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]
    corrs = []
    for col in numeric:
        if df[col].std() > 0:
            c = df[col].corr(df["target"])
            if not np.isnan(c):
                corrs.append((col, c))
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, c in corrs[:10]:
        bar  = "█" * int(abs(c) * 30)
        sign = "+" if c > 0 else "-"
        print(f"  {col:<22} {sign}{abs(c):.3f}  {bar}")

    # ── По времени ────────────────────────────────────────────
    print(f"\n  WR ПО ВРЕМЕНИ СУТОК:")
    for h in [0, 4, 8, 12, 16, 20]:
        grp = df[(df["hour"] >= h) & (df["hour"] < h+4)]
        if len(grp) > 0:
            wr_h = round(grp["target"].mean() * 100, 1)
            bar  = "█" * int(wr_h / 5)
            print(f"  {h:02d}-{h+4:02d}ч: {wr_h:>5}%  {bar}  ({len(grp)} сд)")

    # ── По боту ───────────────────────────────────────────────
    print(f"\n  WR ПО БОТУ:")
    for bot_name in df["bot"].unique():
        grp = df[df["bot"] == bot_name]
        wr_b = round(grp["target"].mean() * 100, 1)
        bar  = "█" * int(wr_b / 5)
        print(f"  {bot_name:<12} {wr_b:>5}%  {bar}  ({len(grp)} сд)")

    # ── По направлению ────────────────────────────────────────
    print(f"\n  WR ЛОНГ vs ШОРТ:")
    for d in ["LONG", "SHORT"]:
        grp = df[df["direction"] == d]
        if len(grp) > 0:
            wr_d = round(grp["target"].mean() * 100, 1)
            bar  = "█" * int(wr_d / 5)
            print(f"  {d:<6} {wr_d:>5}%  {bar}  ({len(grp)} сд)")

    # ── По объёму ─────────────────────────────────────────────
    print(f"\n  WR ПО ОБЪЁМУ:")
    df["vol_bin"] = pd.cut(df["vol_ratio"],
                           bins=[0,1.5,2.0,3.0,5.0,100],
                           labels=["<1.5x","1.5-2x","2-3x","3-5x",">5x"])
    for label in ["<1.5x","1.5-2x","2-3x","3-5x",">5x"]:
        grp = df[df["vol_bin"] == label]
        if len(grp) > 0:
            wr_v = round(grp["target"].mean() * 100, 1)
            bar  = "█" * int(wr_v / 5)
            print(f"  {label:<8} {wr_v:>5}%  {bar}  ({len(grp)} сд)")

    # ── Сохраняем ─────────────────────────────────────────────
    for save_dir in [OUTPUT_DIR, os.getcwd()]:
        try:
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, "ml_dataset.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"\n[+] Датасет: {csv_path}")
            break
        except Exception as e:
            print(f"  Ошибка сохранения в {save_dir}: {e}")

    # Текстовый отчёт
    report = [
        f"ML ДАТАСЕТ  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Всего сделок: {len(df)}  WR: {total_wr}%",
        "",
    ]
    for s in summary:
        report.append(f"{s['bot']:<12}: {s['trades']:>4} сделок  WR={s['wr']}%")
    report += ["", "ТОП ПРИЗНАКОВ:"]
    for col, c in corrs[:10]:
        report.append(f"  {col:<22} {c:>+.3f}")
    report += ["", "ПОЛНЫЕ ДАННЫЕ: ml_dataset.csv"]

    for save_dir in [OUTPUT_DIR, os.getcwd()]:
        try:
            txt_path = os.path.join(save_dir, "ml_dataset_summary.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(report))
            print(f"[+] Сводка:  {txt_path}")
            break
        except Exception:
            pass

    print(f"\n  Загрузи ml_dataset_summary.txt в чат для анализа!")
    print(f"  Минимум для ML: 200+ сделок. Сейчас: {len(df)} сделок.\n")
    input("Нажми Enter для выхода...")


if __name__ == "__main__":
    try:
        build_dataset()
    except Exception as e:
        import traceback
        print(f"\nОШИБКА: {e}")
        print(traceback.format_exc())
        input("\nНажми Enter...")
