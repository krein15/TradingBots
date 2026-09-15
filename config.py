"""
config.py
=========
Единая точка для всех путей проекта.

Раньше пути были захардкожены (C:/TradingBots/...) в 23 местах —
проект работал только если лежал ровно в этой папке. Теперь всё
считается от расположения этого файла, и проект можно положить куда угодно.

Использование из корня:
    from config import SHARED_STATE

Использование из папки бота (Bot1_EMA и т.п.):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import SHARED_STATE
"""

from pathlib import Path

# Корень проекта — папка, где лежит этот файл
ROOT = Path(__file__).resolve().parent

# ── Каталоги ──────────────────────────────────────────────
BOT1_DIR = ROOT / "Bot1_EMA"
BOT2_DIR = ROOT / "Bot2_MeanRev"
BOT3_DIR = ROOT / "Bot3_Funding"
BOT4_DIR = ROOT / "Bot4_Breakout"
ML_DIR   = ROOT / "ML"

# ── Общее состояние (market_regime.py пишет, боты читают) ──
SHARED_STATE   = ROOT / "shared_state.json"
REGIME_HISTORY = ML_DIR / "regime_history.jsonl"

# Максимальный возраст shared_state.json в минутах.
# Старше — считаем данные протухшими и режим неизвестным.
SHARED_STATE_MAX_AGE_MIN = 60

# ── ML ────────────────────────────────────────────────────
ML_DATASET = ML_DIR / "ml_dataset.csv"
ML_SUMMARY = ML_DIR / "ml_dataset_summary.txt"

# ── Журналы сделок (в git не попадают) ────────────────────
JOURNALS = {
    "EMA":      BOT1_DIR / "paper_journal.json",
    "MeanRev":  BOT2_DIR / "meanrev_journal.json",
    "Funding":  BOT3_DIR / "funding_journal.json",
    "Breakout": BOT4_DIR / "breakout_journal.json",
}

# ── Обученные модели (в git не попадают) ──────────────────
MODELS = {
    "EMA":      BOT1_DIR / "ml_model_ema.pkl",
    "MeanRev":  BOT2_DIR / "ml_model_meanrev.pkl",
    "Funding":  BOT3_DIR / "ml_model_funding.pkl",
    "Breakout": BOT4_DIR / "ml_model_breakout.pkl",
}
