"""
combined_stats.py
=================
Объединённая статистика всех 4 ботов.

Запуск:
  python combined_stats.py

Положи этот файл в C:/TradingBots/
"""

import os, json
import numpy as np
import pandas as pd
from datetime import datetime

BOTS = [
    {
        "name":    "Бот #1 — EMA",
        "journal": "C:\\TradingBots\\Bot1_EMA\\paper_journal.json",
        "trades":  "C:\\TradingBots\\Bot1_EMA\\paper_trades.csv",
    },
    {
        "name":    "Бот #2 — Структура HH/HL",
        "journal": "C:\\TradingBots\\Bot2_Structure\\structure2_journal.json",
        "trades":  "C:\\TradingBots\\Bot2_Structure\\structure_trades.csv",
    },
    {
        "name":    "Бот #3 — SMC OB+FVG",
        "journal": "C:\\TradingBots\\Bot3_SMC\\smc_journal.json",
        "trades":  "C:\\TradingBots\\Bot3_SMC\\smc_trades.csv",
    },
    {
        "name":    "Бот #4 — Wyckoff Spring",
        "journal": "C:\\TradingBots\\Bot4_Wyckoff\\wyckoff_journal.json",
        "trades":  "C:\\TradingBots\\Bot4_Wyckoff\\wyckoff_trades.csv",
    },
]


def load_journal(path, path2=None, path3=None):
    for p in [path, path2, path3]:
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def calc_stats(journal):
    if not journal:
        return None
    trades  = journal["trades"]
    balance = journal["balance"]
    init    = journal["initial_deposit"]
    scans   = journal.get("scan_count", 0)
    pnl     = balance - init

    if not trades:
        return {
            "n": 0, "wr": 0, "pnl": pnl,
            "pnl_pct": round(pnl/init*100, 2),
            "balance": balance, "init": init,
            "dd": 0, "sharpe": 0,
            "avg_win": 0, "avg_loss": 0,
            "longs": 0, "shorts": 0,
            "scans": scans,
            "pending": len(journal.get("pending", [])),
            "open":    len(journal.get("open", [])),
            "created": journal.get("created", "")[:10],
        }

    wins   = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    longs  = [t for t in trades if t.get("dir") == 1]
    shorts = [t for t in trades if t.get("dir") == -1]

    eq   = [init] + [t["balance_after"] for t in trades]
    peak = np.maximum.accumulate(eq)
    dd   = round(((np.array(eq) - peak) / peak * 100).min(), 2)

    ret  = [t["pnl_usd"] / init for t in trades]
    sh   = round((np.mean(ret) / np.std(ret) * np.sqrt(252))
                 if np.std(ret) > 0 else 0, 2)

    return {
        "n":       len(trades),
        "wr":      round(len(wins) / len(trades) * 100, 1),
        "pnl":     round(pnl, 2),
        "pnl_pct": round(pnl / init * 100, 2),
        "balance": balance,
        "init":    init,
        "dd":      dd,
        "sharpe":  sh,
        "avg_win":  round(np.mean([t["pnl_usd"] for t in wins]),   2) if wins   else 0,
        "avg_loss": round(np.mean([t["pnl_usd"] for t in losses]), 2) if losses else 0,
        "longs":   len(longs),
        "shorts":  len(shorts),
        "scans":   scans,
        "pending": len(journal.get("pending", [])),
        "open":    len(journal.get("open", [])),
        "created": journal.get("created", "")[:10],
        "trades":  trades,
    }


def print_combined():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*72}")
    print(f"  📊 ОБЩАЯ СТАТИСТИКА ВСЕХ БОТОВ  |  {now}")
    print(f"{'='*72}")

    all_trades  = []
    all_wins    = []
    total_bots  = 0
    active_bots = 0

    # ── Сводная таблица ───────────────────────
    print(f"\n  {'БОТ':<26} {'N':>5} {'WR%':>6} {'P&L$':>8} "
          f"{'P&L%':>7} {'DD%':>7} {'БАЛАНС':>8}")
    print(f"  {'-'*68}")

    bot_stats = []
    for bot in BOTS:
        journal = load_journal(bot["journal"], bot.get("journal2"), bot.get("journal3"))
        stats   = calc_stats(journal)
        total_bots += 1

        if stats is None:
            print(f"  ❓ {bot['name']:<24} — файл не найден")
            continue

        active_bots += 1
        s  = "+" if stats["pnl"] >= 0 else ""
        wr_emoji = "🟢" if stats["wr"] >= 35 else "🟡" if stats["wr"] >= 25 else "🔴"

        print(f"  {wr_emoji} {bot['name']:<24} {stats['n']:>5} "
              f"{stats['wr']:>5}%  {s}{stats['pnl']:>7}$  "
              f"{s}{stats['pnl_pct']:>5}%  {stats['dd']:>6}%  "
              f"${stats['balance']:>7.2f}")

        bot_stats.append((bot["name"], stats))
        if stats.get("trades"):
            all_trades.extend(stats["trades"])
            all_wins.extend([t for t in stats["trades"] if t["result"] == "WIN"])

    # ── Итоговая строка ───────────────────────
    if all_trades:
        total_wr  = round(len(all_wins) / len(all_trades) * 100, 1)
        total_pnl = sum(s["pnl"] for _, s in bot_stats)
        wr_emoji  = "🟢" if total_wr >= 35 else "🟡" if total_wr >= 25 else "🔴"
        s = "+" if total_pnl >= 0 else ""
        print(f"  {'-'*68}")
        print(f"  {wr_emoji} {'ИТОГО':<24} {len(all_trades):>5} "
              f"{total_wr:>5}%  {s}{round(total_pnl,2):>7}$")

    # ── Детали по каждому боту ────────────────
    print(f"\n{'='*72}")
    print(f"  ДЕТАЛИ ПО КАЖДОМУ БОТУ")
    print(f"{'='*72}")

    for bot_name, stats in bot_stats:
        print(f"\n  📌 {bot_name}")
        print(f"     Старт: {stats['created']}  "
              f"Сканирований: {stats['scans']}")
        print(f"     Сделок: {stats['n']}  WR: {stats['wr']}%  "
              f"Sharpe: {stats['sharpe']}")
        print(f"     Ср. профит: +${stats['avg_win']}  "
              f"Ср. убыток: ${stats['avg_loss']}")
        print(f"     Лонгов: {stats['longs']}  "
              f"Шортов: {stats['shorts']}")
        print(f"     Просадка: {stats['dd']}%")
        print(f"     В очереди: {stats['pending']}  "
              f"Открыто: {stats['open']}")

        # Последние 3 сделки
        if stats.get("trades"):
            print(f"     Последние сделки:")
            for t in stats["trades"][-3:]:
                em = "🟢" if t["result"] == "WIN" else "🔴"
                d  = "ЛОНГ" if t.get("dir") == 1 else "ШОРТ"
                st = t.get("structure", "?")
                print(f"       {em} {t['symbol']:<14} {d}  "
                      f"[{st}]  {t['result']}  "
                      f"{t['pnl_usd']:>+.2f}$")

    # ── Лучшие сделки за всё время ───────────
    if all_trades:
        wins_sorted = sorted(
            [t for t in all_trades if t["result"] == "WIN"],
            key=lambda x: x["pnl_usd"], reverse=True
        )
        if wins_sorted:
            print(f"\n{'='*72}")
            print(f"  🏆 ТОП-5 ЛУЧШИХ СДЕЛОК ВСЕХ БОТОВ")
            print(f"{'='*72}")
            for t in wins_sorted[:5]:
                print(f"  🟢 {t['symbol']:<14} "
                      f"[{t.get('structure','?')}]  "
                      f"+${t['pnl_usd']:.2f}  "
                      f"баланс после: ${t['balance_after']:.2f}")

    print(f"\n{'='*72}")
    print(f"  🕐 {now}")
    print(f"  Нажми любую клавишу для выхода...")
    input()


if __name__ == "__main__":
    try:
        # Сначала выводим на экран
        print_combined()

        # Потом сохраняем в файл
        import io, sys
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        print_combined()
        sys.stdout = old_stdout
        result = buf.getvalue()

        fname = f"stats_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        saved = False
        for save_dir in [
            os.path.join("C:\\TradingBots", "ML"),
            "C:\\TradingBots",
            os.getcwd(),
            "."
        ]:
            try:
                os.makedirs(save_dir, exist_ok=True)
                fpath = os.path.join(save_dir, fname)
                with open(fpath, "w", encoding="utf-8", errors="replace") as f:
                    f.write(result)
                print(f"\n[+] Файл сохранён: {fpath}")
                print(f"    Загрузи его в чат для анализа!")
                saved = True
                break
            except Exception as e:
                print(f"  Не удалось сохранить в {save_dir}: {e}")
        if not saved:
            print("  [!] Не удалось сохранить файл")
    except Exception as e:
        import traceback
        print(f"ОШИБКА: {e}")
        print(traceback.format_exc())
    input("\nНажми Enter для выхода...")
