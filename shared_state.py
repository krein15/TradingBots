"""
shared_state.py
===============
Чтение текущего режима рынка из shared_state.json.

Раньше каждый потребитель читал файл сам, со своим путём и своим
порогом свежести: Bot1 считал данные протухшими через 90 минут,
market_regime.py — через 60, а Bot2 не проверял свежесть вообще
и мог торговать по режиму недельной давности. Теперь логика одна.

Использование из папки бота:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from shared_state import read_regime

    regime, conf = read_regime()
"""

import json
from datetime import datetime

from config import SHARED_STATE, SHARED_STATE_MAX_AGE_MIN

# Режим неизвестен: файла нет, он повреждён или данные протухли
UNKNOWN = ("?", 0)


def read_regime(max_age_min=SHARED_STATE_MAX_AGE_MIN):
    """
    Возвращает (regime, confidence).

    regime: TREND_UP / TREND_DOWN / SIDEWAYS / BREAKOUT / VOLATILE / "?"
    Протухшие данные возвращаются как "?" — бот не должен принимать
    решения по режиму, посчитанному часы назад.
    """
    try:
        with open(SHARED_STATE, encoding="utf-8") as f:
            state = json.load(f)
        updated = datetime.fromisoformat(state["updated_at"])
        age_min = (datetime.now() - updated).total_seconds() / 60
        if age_min > max_age_min:
            return UNKNOWN
        return state.get("regime", "?"), state.get("confidence", 0)
    except Exception:
        return UNKNOWN
