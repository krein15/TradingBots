"""
paper_trading_supertrend.py
===========================
Бот #6 — разворот Supertrend. Bitget перпетуальные фьючерсы, 4ч.

Исполнение, журнал, стопы и список инструментов — общие с Ботом #5:
ядро живёт в Bot5_Donchian/paper_trading_donchian.py, здесь подменены
только правила входа. Благодаря этому разница в результатах двух ботов
— это разница стратегий, а не разница в том, как они исполняют сделки.

Зачем второй бот. В широком поиске (README, «Широкий поиск: 13
семейств») Supertrend и Дончиан оказались двумя лидерами, прошедшими
все барьеры: случайные наборы монет и данные другой биржи. И они
дополняют друг друга:

  Дончиан силён на росте рынка, Supertrend — на падении;
  корреляция полугодовых результатов -0.23 (Bitget) / +0.04 (Binance);
  корреляция сигналов 0.12.

За весь период 2023-2026 Дончиан вдвое лучше на сделку (+0.195R против
+0.109R), поэтому Supertrend — не замена, а второй независимый поток
сделок, который сглаживает кривую капитала.

Параметры — плато, а не точка: при mult 3.0 и фильтре EMA200 прибыльны
все проверенные стопы 2.0 / 2.5 / 3.0 ATR. Выбран 2.5 — та же
геометрия риска, что у Бота #5.

Запуск:
  python paper_trading_supertrend.py           — торговля
  python paper_trading_supertrend.py status    — статистика
  python paper_trading_supertrend.py reset     — сбросить журнал
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Bot5_Donchian"))

import paper_trading_donchian as core  # noqa: E402

CONFIG = dict(core.CONFIG)
CONFIG.update({
    "bot_id":        "bot6",
    "bot_name":      "Бот #6 — Supertrend",
    "strategy":      "supertrend",

    # ── Правила ───────────────────────────────────────────────
    "mult":          3.0,     # ширина полос Supertrend в ATR
    "atr_period":    14,
    "atr_mult":      2.5,     # стоп = 2.5 ATR, как у Бота #5
    "rr":            3.0,     # тейк 3R без трейлинга — трейлинг вредит
    "ema":           200,     # лонги выше EMA200, шорты ниже
    "allow_short":   True,
    "max_hold_bars": 200,
    # Нижняя — среднее за весь период, верхняя — медиана на проверке,
    # где Supertrend повезло с медвежьим рынком
    "expect":        {"wr": 33, "r_lo": 0.11, "r_hi": 0.28},

    # ── Свой журнал и лог ─────────────────────────────────────
    "journal":       os.path.join(HERE, "supertrend_journal.json"),
    "logfile":       os.path.join(HERE, "supertrend_log.txt"),
})
CONFIG.pop("channel", None)   # параметр Дончиана, здесь не нужен


if __name__ == "__main__":
    core.main(CONFIG)
