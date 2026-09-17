"""
Tools/audit_journals.py
=======================
Проверка журналов ботов: пересчитывает всё заново и ищет расхождения.

Зачем. Ошибку в учёте глазами не увидишь: баланс выглядит правдоподобно,
пока не сложишь сделки вручную. Эта проверка складывает — и сверяет
каждое число с тем, что записано.

Что проверяется:
  баланс = депозит + сумма всех PnL;
  цепочка «баланс после сделки» не рвётся;
  R каждой сделки = PnL / (риск на входе);
  цена выхода соответствует причине: по тейку — ровно уровень, по
    стопу — не лучше уровня (проскальзывание против нас);
  комиссия сделки совпадает с расчётной;
  геометрия открытых позиций: стоп и тейк по нужную сторону от входа;
  нет двух позиций по одной монете и стороне, лимит позиций соблюдён;
  объём позиции в пределах плеча и не ниже минимального;
  сигналы не старше допустимого возраста;
  в журнале нет NaN и бесконечностей.

Запуск:
  python Tools/audit_journals.py
"""

import json
import math
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

EPS = 1e-6

BOTS = [
    {"name": "Бот #5 — Дончиан", "journal": os.path.join(ROOT, "Bot5_Donchian", "donchian_journal.json"),
     "commission": 0.0006, "slippage": 0.0005, "max_leverage": 3.0, "min_notional": 5.0,
     "max_open": 5, "max_signal_age_min": 30, "kind": "paper"},
    {"name": "Бот #6 — Supertrend", "journal": os.path.join(ROOT, "Bot6_Supertrend", "supertrend_journal.json"),
     "commission": 0.0006, "slippage": 0.0005, "max_leverage": 3.0, "min_notional": 5.0,
     "max_open": 5, "max_signal_age_min": 30, "kind": "paper"},
    {"name": "Демо Bitget", "journal": os.path.join(ROOT, "BotDemo", "demo_journal.json"),
     "max_open": 5, "kind": "demo"},
]


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def audit_paper(j, cfg, say):
    problems = []
    deposit = j.get("deposit")
    balance = j.get("balance")
    trades = j.get("trades", [])
    opened = j.get("open", [])

    # 1. Баланс = депозит + сумма PnL
    total = sum(t.get("pnl", 0) for t in trades)
    if not finite(deposit) or not finite(balance):
        problems.append("депозит или баланс не число")
    elif abs(deposit + total - balance) > 1e-4:
        problems.append(f"баланс {balance:.4f} ≠ депозит {deposit} + сумма сделок {total:.4f} "
                        f"(расхождение {deposit + total - balance:+.4f})")
    else:
        say(f"баланс сходится: {deposit} {total:+.4f} = {balance:.4f}")

    # 2. Цепочка балансов и содержимое каждой сделки
    run = deposit
    for i, t in enumerate(trades, 1):
        tag = f"сделка {i} {t.get('symbol')}"
        run += t.get("pnl", 0)
        if abs(run - t.get("balance", run)) > 1e-4:
            problems.append(f"{tag}: записан баланс {t.get('balance')}, пересчёт даёт {run:.4f}")

        entry, stop, take = t.get("entry"), t.get("stop"), t.get("take")
        qty, d = t.get("qty"), t.get("dir")
        exit_p, reason = t.get("exit"), t.get("exit_reason")
        if not all(finite(x) for x in (entry, stop, take, qty, exit_p)):
            problems.append(f"{tag}: нечисловые значения")
            continue

        risk = abs(entry - stop) * qty
        if risk > 0:
            r_calc = t.get("pnl", 0) / risk
            if abs(r_calc - (t.get("r_multiple") or 0)) > 0.01:
                problems.append(f"{tag}: R записан {t.get('r_multiple')}, пересчёт {r_calc:.3f}")

        # Цена выхода против причины
        if reason == "take" and abs(exit_p - take) > max(take * 1e-6, 1e-9):
            problems.append(f"{tag}: выход по тейку, но цена {exit_p} ≠ уровень {take}")
        if reason == "stop":
            worse = exit_p <= stop + EPS if d == 1 else exit_p >= stop - EPS
            if not worse:
                problems.append(f"{tag}: выход по стопу {exit_p} ЛУЧШЕ уровня {stop} — "
                                f"проскальзывание не учтено")

        # Выход на той же свече, внутри которой вошли, — только по
        # уточнённому размаху. Иначе стоп мог прилететь от движения,
        # случившегося до входа.
        step = 14_400_000
        if (t.get("exit_ts") and t.get("entry_ts")
                and t["exit_ts"] - t["entry_ts"] == step
                and reason in ("stop", "take")
                and not t.get("entry_bar_checked")):
            problems.append(f"{tag}: выход по {reason} на свече входа, а её размах "
                            f"не уточнён по минуткам — движение могло быть до входа")

        # Комиссия
        if "fees" in t and finite(t["fees"]):
            calc = (abs(entry * qty) + abs(exit_p * qty)) * cfg["commission"]
            if abs(calc - t["fees"]) > max(calc * 0.02, 1e-6):
                problems.append(f"{tag}: комиссия {t['fees']:.6f}, расчёт {calc:.6f}")

        # PnL целиком
        gross = (exit_p - entry) * qty * d
        pnl_calc = gross - t.get("fees", 0)
        if abs(pnl_calc - t.get("pnl", 0)) > 1e-4:
            problems.append(f"{tag}: PnL {t.get('pnl')}, пересчёт {pnl_calc:.4f}")

    if trades:
        say(f"пересчитано сделок: {len(trades)}")

    # 3. Открытые позиции
    seen = set()
    for p in opened:
        tag = f"позиция {p.get('symbol')}"
        entry, stop, take, d = p.get("entry"), p.get("stop"), p.get("take"), p.get("dir")
        if not all(finite(x) for x in (entry, stop, take)):
            problems.append(f"{tag}: нечисловые уровни")
            continue
        if d == 1 and not (stop < entry < take):
            problems.append(f"{tag}: лонг, но уровни не по порядку стоп<вход<тейк: {stop}/{entry}/{take}")
        if d == -1 and not (take < entry < stop):
            problems.append(f"{tag}: шорт, но уровни не по порядку тейк<вход<стоп: {take}/{entry}/{stop}")
        key = (p.get("symbol"), d)
        if key in seen:
            problems.append(f"{tag}: две позиции по одной монете и стороне")
        seen.add(key)

        notional = p.get("notional")
        if finite(notional) and finite(balance):
            if notional > balance * cfg["max_leverage"] + 0.01:
                problems.append(f"{tag}: объём {notional} выше лимита плеча "
                                f"{balance * cfg['max_leverage']:.2f}")
            if notional < cfg["min_notional"] - 0.01:
                problems.append(f"{tag}: объём {notional} ниже минимального {cfg['min_notional']}")
        age = p.get("signal_age_min")
        if age is not None and age > cfg["max_signal_age_min"]:
            problems.append(f"{tag}: вход по сигналу возрастом {age} мин "
                            f"(допустимо {cfg['max_signal_age_min']})")
    if len(opened) > cfg["max_open"]:
        problems.append(f"открыто {len(opened)} позиций при лимите {cfg['max_open']}")
    if opened:
        say(f"проверено открытых позиций: {len(opened)}")
    return problems


def audit_demo(j, cfg, say):
    problems = []
    trades = j.get("trades", [])
    opened = j.get("open", [])
    seen = set()
    for p in opened:
        sym = p.get("symbol")
        if sym in seen:
            problems.append(f"{sym}: две позиции по одной монете — на бирже так нельзя, "
                            f"вторая закроет первую")
        seen.add(sym)
        d, entry, stop, take = p.get("dir"), p.get("entry"), p.get("stop"), p.get("take")
        if all(finite(x) for x in (entry, stop, take)):
            if d == 1 and not (stop < entry < take):
                problems.append(f"{sym}: лонг с вывернутыми уровнями")
            if d == -1 and not (take < entry < stop):
                problems.append(f"{sym}: шорт с вывернутыми уровнями")
    per = {}
    for t in trades:
        per[t.get("strategy")] = per.get(t.get("strategy"), 0) + 1
        if t.get("r_multiple") is not None and t.get("risk_usd") and finite(t.get("pnl")):
            calc = t["pnl"] / t["risk_usd"]
            if abs(calc - t["r_multiple"]) > 0.01:
                problems.append(f"{t.get('symbol')}: R записан {t['r_multiple']}, пересчёт {calc:.3f}")
    unknown = [t for t in trades if t.get("result") == "UNKNOWN"]
    if unknown:
        problems.append(f"сделок с неизвестным итогом: {len(unknown)} "
                        f"({', '.join(t.get('symbol', '?') for t in unknown)}) — "
                        f"биржа не отдала историю, в статистику они не идут")
    if trades:
        say(f"сделок: {len(trades)} " + ", ".join(f"{k}: {v}" for k, v in per.items()))
    if opened:
        say(f"открытых позиций: {len(opened)}, монеты не повторяются")
    if j.get("last_error"):
        problems.append(f"в журнале записана ошибка: {j['last_error'][:160]}")
    return problems


def main():
    print("=" * 66)
    print("  ПРОВЕРКА ЖУРНАЛОВ: пересчёт всех чисел заново")
    print("=" * 66)
    total = 0
    for cfg in BOTS:
        print(f"\n{cfg['name']}")
        if not os.path.exists(cfg["journal"]):
            print("  журнала нет — бот ещё не запускался")
            continue
        try:
            with open(cfg["journal"], "r", encoding="utf-8") as f:
                j = json.load(f)
        except Exception as e:
            print(f"  ✘ журнал не читается: {e}")
            total += 1
            continue

        notes = []
        problems = (audit_demo if cfg["kind"] == "demo" else audit_paper)(j, cfg, notes.append)
        for n in notes:
            print(f"  ✔ {n}")
        for p in problems:
            print(f"  ✘ {p}")
        total += len(problems)

    print("\n" + "=" * 66)
    print("  Расхождений не найдено" if total == 0 else f"  НАЙДЕНО РАСХОЖДЕНИЙ: {total}")
    print("=" * 66)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
