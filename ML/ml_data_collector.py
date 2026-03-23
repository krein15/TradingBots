"""
ml_data_collector.py
====================
Сборщик данных для обучения ML модели.

Собирает все сделки всех ботов в единый датасет
с расширенными признаками для обучения.

Запуск:
  python ml_data_collector.py

Результат:
  ml_dataset.csv — полный датасет для ML
  ml_dataset_summary.txt — краткая сводка

Положи в C:\\TradingBots\\
"""

import os, json
import pandas as pd
import numpy as np
from datetime import datetime

# Пути к журналам всех ботов
JOURNALS = [
    {
        "name":    "EMA",
        "path":    "C:\\TradingBots\\Bot1_EMA\\paper_journal.json",
        "logpath": "C:\\TradingBots\\Bot1_EMA\\paper_log.txt",
    },
    {
        "name":    "Structure",
        "path":    "C:\\TradingBots\\Bot2_Structure\\structure2_journal.json",
        "path2":   "C:\\TradingBots\\Bot2_Structure\\paper_journal_structure.json",
        "logpath": "C:\\TradingBots\\Bot2_Structure\\structure2_log.txt",
    },
    {
        "name":    "SMC",
        "path":    "C:\\TradingBots\\Bot3_SMC\\smc_journal.json",
        "logpath": "C:\\TradingBots\\Bot3_SMC\\smc_log.txt",
    },
    {
        "name":    "Wyckoff",
        "path":    "C:\\TradingBots\\Bot4_Wyckoff\\wyckoff_journal.json",
        "logpath": "C:\\TradingBots\\Bot4_Wyckoff\\wyckoff_log.txt",
    },
]

OUTPUT_DIR = "C:\\TradingBots\\ML"


def load_journal(bot):
    for key in ["path", "path2", "path3"]:
        p = bot.get(key)
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def extract_features(trade, bot_name):
    """
    Извлекаем все доступные признаки из сделки.
    Это будут входные данные для ML модели.
    """
    # Базовые данные
    closed_at = trade.get("closed_at", trade.get("close", ""))
    try:
        dt = datetime.fromisoformat(closed_at)
        hour       = dt.hour
        day_of_week= dt.weekday()
        is_weekend = 1 if dt.weekday() >= 5 else 0
        is_night   = 1 if (hour >= 22 or hour < 6) else 0
        is_london  = 1 if 8 <= hour < 12 else 0
        is_newyork = 1 if 13 <= hour < 18 else 0
        is_asia    = 1 if (hour >= 0 and hour < 8) else 0
    except Exception:
        hour = day_of_week = is_weekend = is_night = 0
        is_london = is_newyork = is_asia = 0

    # Цены
    entry    = float(trade.get("entry",   trade.get("entry_limit", 0)))
    stop     = float(trade.get("stop",    0))
    take     = float(trade.get("take",    0))
    exit_p   = float(trade.get("exit",    trade.get("exit_price", 0)))
    pnl      = float(trade.get("pnl",     trade.get("pnl_usd", 0)))
    result   = trade.get("result", "LOSS")
    d        = trade.get("dir",    1)

    # Риск-менеджмент признаки
    risk     = abs(entry - stop)
    reward   = abs(take  - entry)
    rr_plan  = round(reward / risk, 2) if risk > 0 else 0
    stop_pct = round(risk / entry * 100, 3) if entry > 0 else 0
    take_pct = round(reward / entry * 100, 3) if entry > 0 else 0

    # Насколько цена дошла до тейка/стопа
    if d == 1:
        exit_progress = round((exit_p - entry) / (take - entry) * 100, 1) \
                        if take != entry else 0
    else:
        exit_progress = round((entry - exit_p) / (entry - take) * 100, 1) \
                        if take != entry else 0

    # Направление
    direction = "LONG" if d == 1 else "SHORT"

    # Тип сигнала
    sig_type = trade.get("structure",
               trade.get("type",
               trade.get("signal_type", "UNKNOWN")))

    # Таймфрейм в минутах
    tf = trade.get("tf", "15m")
    tf_minutes = {"1m":1,"5m":5,"15m":15,"30m":30,
                  "1h":60,"4h":240,"1d":1440}.get(tf, 15)

    # Объём
    vol_ratio = float(trade.get("vol_ratio",
                      trade.get("vol",
                      trade.get("sc_vol_ratio", 0))))

    # Специфичные признаки по типу стратегии
    sc_drop    = float(trade.get("sc_drop_pct",   0))
    ar_bounce  = float(trade.get("ar_bounce_pct", 0))
    spring_wick= float(trade.get("spring_wick",   0))
    fvg_size   = float(trade.get("fvg_size",      0))
    adx        = float(trade.get("adx",           0))

    # Целевая переменная
    target = 1 if result == "WIN" else 0

    return {
        # ── Идентификаторы ────────────────────
        "bot":          bot_name,
        "symbol":       trade.get("symbol", ""),
        "tf":           tf,
        "tf_minutes":   tf_minutes,
        "direction":    direction,
        "signal_type":  sig_type,
        "closed_at":    closed_at,

        # ── Цены ──────────────────────────────
        "entry":        entry,
        "stop":         stop,
        "take":         take,
        "exit_price":   exit_p,

        # ── Риск-менеджмент ───────────────────
        "risk_pct":     stop_pct,
        "reward_pct":   take_pct,
        "rr_planned":   rr_plan,
        "exit_progress":exit_progress,

        # ── Время ─────────────────────────────
        "hour":         hour,
        "day_of_week":  day_of_week,
        "is_weekend":   is_weekend,
        "is_night":     is_night,
        "is_london":    is_london,
        "is_newyork":   is_newyork,
        "is_asia":      is_asia,

        # ── Объём и индикаторы ────────────────
        "vol_ratio":    vol_ratio,
        "adx":          adx,

        # ── Wyckoff специфика ─────────────────
        "sc_drop_pct":  sc_drop,
        "ar_bounce_pct":ar_bounce,
        "spring_wick":  spring_wick,

        # ── SMC специфика ─────────────────────
        "fvg_size":     fvg_size,

        # ── Результат ─────────────────────────
        "pnl_usd":      pnl,
        "result":       result,
        "target":       target,  # 1=WIN, 0=LOSS — это предсказывает ML
    }


def build_dataset():
    all_rows = []
    summary  = []

    print("=" * 60)
    print(f"  🤖 СБОРЩИК ML ДАТАСЕТА")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Создаём папку ML если не существует
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"  Папка: {OUTPUT_DIR} — OK")
    except Exception as e:
        print(f"  ОШИБКА создания папки: {e}")
        print(f"  Файлы будут сохранены в текущую папку")

    for bot in JOURNALS:
        journal = load_journal(bot)
        if not journal:
            print(f"\n  ❓ {bot['name']} — журнал не найден")
            continue

        trades = journal.get("trades", [])
        if not trades:
            print(f"\n  ⚠️  {bot['name']} — нет сделок")
            continue

        rows = [extract_features(t, bot["name"]) for t in trades]
        all_rows.extend(rows)

        wins = sum(1 for r in rows if r["target"] == 1)
        wr   = round(wins / len(rows) * 100, 1) if rows else 0

        print(f"\n  ✅ {bot['name']}: {len(rows)} сделок  WR={wr}%")

        # Статистика по признакам
        df_bot = pd.DataFrame(rows)
        print(f"     Лонгов: {(df_bot['direction']=='LONG').sum()}  "
              f"Шортов: {(df_bot['direction']=='SHORT').sum()}")
        print(f"     Ср. объём: {df_bot['vol_ratio'].mean():.2f}x")
        print(f"     Ср. стоп: {df_bot['risk_pct'].mean():.3f}%")
        print(f"     Ср. RR: {df_bot['rr_planned'].mean():.2f}")

        # Корреляция признаков с результатом
        numeric_cols = ["vol_ratio","risk_pct","rr_planned",
                        "hour","tf_minutes","exit_progress"]
        print(f"     Корреляция с WIN:")
        for col in numeric_cols:
            if df_bot[col].std() > 0:
                corr = df_bot[col].corr(df_bot["target"])
                bar  = "▓" * int(abs(corr) * 20) if not (corr != corr) else ""
                sign = "+" if corr > 0 else "-"
                print(f"       {col:<20} {sign}{abs(corr):.3f}  {bar}")

        summary.append({
            "bot":    bot["name"],
            "trades": len(rows),
            "wins":   wins,
            "wr":     wr,
        })

    if not all_rows:
        print("\n[!] Нет данных для датасета")
        return

    df = pd.DataFrame(all_rows)

    # ── Итоговая статистика ───────────────────
    total_wins = df["target"].sum()
    total_wr   = round(total_wins / len(df) * 100, 1)

    print(f"\n{'='*60}")
    print(f"  ИТОГО: {len(df)} сделок  WR={total_wr}%")
    print(f"{'='*60}")

    # Топ признаков по корреляции с результатом
    print(f"\n  📊 ТОП ПРИЗНАКОВ (корреляция с WIN):")
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c not in
               ["target","entry","stop","take","exit_price","pnl_usd"]]
    corrs = []
    for col in numeric:
        if df[col].std() > 0:
            corr = df[col].corr(df["target"])
            if corr == corr:  # проверка на NaN
                corrs.append((col, corr))
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, corr in corrs[:10]:
        bar  = "▓" * int(abs(corr) * 30) if not (corr != corr) else ""
        sign = "+" if corr > 0 else "-"
        print(f"  {col:<22} {sign}{abs(corr):.3f}  {bar}")

    print(f"\n  💡 Интерпретация:")
    for col, corr in corrs[:3]:
        if corr > 0.05:
            print(f"     Чем выше {col} → тем выше вероятность WIN")
        elif corr < -0.05:
            print(f"     Чем выше {col} → тем выше вероятность LOSS")

    # ── Анализ по времени ─────────────────────
    print(f"\n  🕐 WIN RATE ПО ВРЕМЕНИ СУТОК:")
    for h in [0, 4, 8, 12, 16, 20]:
        mask = (df["hour"] >= h) & (df["hour"] < h+4)
        grp  = df[mask]
        if len(grp) > 0:
            wr_h = round(grp["target"].mean() * 100, 1)
            bar  = "▓" * int(wr_h / 5) if wr_h == wr_h else ""
            print(f"  {h:02d}-{h+4:02d}ч: {wr_h:>5}%  {bar}  ({len(grp)} сделок)")

    # ── Анализ по таймфрейму ──────────────────
    print(f"\n  📈 WIN RATE ПО ТАЙМФРЕЙМУ:")
    for tf in sorted(df["tf"].unique()):
        grp = df[df["tf"] == tf]
        wr_tf = round(grp["target"].mean() * 100, 1)
        bar   = "▓" * int(wr_tf / 5) if wr_tf == wr_tf else ""
        print(f"  {tf:<6}: {wr_tf:>5}%  {bar}  ({len(grp)} сделок)")

    # ── Анализ по направлению ─────────────────
    print(f"\n  📊 WIN RATE ЛОНГ vs ШОРТ:")
    for d in ["LONG", "SHORT"]:
        grp = df[df["direction"] == d]
        if len(grp) > 0:
            wr_d = round(grp["target"].mean() * 100, 1)
            bar  = "▓" * int(wr_d / 5) if wr_d == wr_d else ""
            print(f"  {d:<6}: {wr_d:>5}%  {bar}  ({len(grp)} сделок)")

    # ── Анализ по объёму ──────────────────────
    print(f"\n  📊 WIN RATE ПО ОБЪЁМУ:")
    bins = [0, 1.5, 2.0, 3.0, 5.0, 100]
    labels = ["<1.5x", "1.5-2x", "2-3x", "3-5x", ">5x"]
    df["vol_bin"] = pd.cut(df["vol_ratio"], bins=bins, labels=labels)
    for label in labels:
        grp = df[df["vol_bin"] == label]
        if len(grp) > 0:
            wr_v = round(grp["target"].mean() * 100, 1)
            bar  = "▓" * int(wr_v / 5) if wr_v == wr_v else ""
            print(f"  {label:<8}: {wr_v:>5}%  {bar}  ({len(grp)} сделок)")

    # ── Сохраняем файлы ───────────────────────
    # Сохраняем CSV
    for save_dir in [OUTPUT_DIR, os.getcwd(), "."]:
        try:
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, "ml_dataset.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"\n[+] ML датасет: {csv_path}")
            break
        except Exception as e:
            print(f"  Не удалось сохранить в {save_dir}: {e}")
            continue

    # Текстовый отчёт
    report = []
    report.append(f"ML ДАТАСЕТ  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Всего сделок: {len(df)}  WR: {total_wr}%")
    report.append("")
    for s in summary:
        report.append(f"{s['bot']:<12}: {s['trades']:>4} сделок  WR={s['wr']}%")
    report.append("")
    report.append("ТОП ПРИЗНАКОВ:")
    for col, corr in corrs[:10]:
        report.append(f"  {col:<22} {corr:>+.3f}")
    report.append("")
    report.append("ПОЛНЫЕ ДАННЫЕ: ml_dataset.csv")

    for save_dir in [OUTPUT_DIR, os.getcwd(), "."]:
        try:
            txt_path = os.path.join(save_dir, "ml_dataset_summary.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(report))
            print(f"[+] Сводка:    {txt_path}")
            break
        except Exception as e:
            print(f"  Не удалось сохранить сводку в {save_dir}: {e}")
            continue

    print(f"\n  📤 Загрузи ml_dataset_summary.txt в чат для анализа!")
    print(f"  📤 ml_dataset.csv — для обучения ML модели")
    print(f"\n  Нужно минимум 200-300 сделок для ML.")
    print(f"  Сейчас: {len(df)} сделок — продолжаем собирать!\n")

    input("Нажми Enter для выхода...")


if __name__ == "__main__":
    try:
        build_dataset()
    except Exception as e:
        import traceback
        print(f"\nОШИБКА: {e}")
        print(traceback.format_exc())
        input("\nНажми Enter для выхода...")
