# TradingBots — Автоматический торговый бот

Проект бумажного трейдинга на Python. 4 бота под разные режимы рынка,
Market Regime детектор, ML фильтры, автоматический журнал сделок.

---

## Архитектура

```
market_regime.py (каждые 30 мин)
  ├── TREND_UP / TREND_DOWN → Бот #1 Адаптивный  (Bitget spot, 5m)
  ├── SIDEWAYS              → Бот #1 + Бот #2     (Bitget / Binance)
  ├── BREAKOUT              → Бот #4 Breakout     (Binance, 15m/1h)
  └── VOLATILE              → пауза

Бот #3 Funding Rate — работает независимо от режима (Bitget фьючерсы, 1h)

shared_state.json ← market_regime.py пишет каждые 30 мин
      ↓
все боты читают режим → выбирают стратегию входа
      ↓
ml_data_collector.py → единый датасет (regime, signal_type, exchange, ...)
```

---

## Структура проекта

```
TradingBots/
├── Bot1_EMA/
│   ├── paper_trading_v2_clean.py
│   ├── ml_model_ema.pkl         # НЕ загружать на GitHub
│   ├── ml_train_ema.py
│   └── start_bot1.bat
│
├── Bot2_MeanRev/
│   ├── paper_trading_meanrev.py
│   ├── ml_model_meanrev.pkl     # НЕ загружать на GitHub
│   ├── ml_train_meanrev.py
│   └── start_bot2.bat
│
├── Bot3_Funding/
│   ├── paper_trading_funding.py
│   └── start_bot3.bat
│
├── Bot4_Breakout/
│   ├── paper_trading_breakout.py
│   └── start_bot4.bat
│
├── ML/
│   ├── ml_data_collector.py
│   ├── ml_train_ema.py
│   ├── ml_train_meanrev.py
│   ├── ml_train_funding.py      # запускать после 60+ сделок
│   ├── ml_train_breakout.py     # запускать после 150+ сделок
│   ├── ml_dataset.csv           # НЕ загружать на GitHub
│   └── regime_history.jsonl     # НЕ загружать на GitHub
│
├── market_regime.py
├── shared_state.json
├── start_regime.bat
├── update.bat
└── README.md
```

---

## Боты

### Бот #1 — Адаптивный (Bitget spot, 5m)
**Режим:** TREND_UP / TREND_DOWN / SIDEWAYS

**Стратегия меняется по режиму рынка:**

```
TREND_UP / TREND_DOWN → Channel Breakout
  Пробой 15-свечного канала с объёмом > 1.5x
  Вход на закрытии пробойной свечи
  Стоп за противоположный край канала
  RR 1:3

SIDEWAYS → Mean Reversion
  RSI < 35 + цена у нижней BB (±1%) → лонг к средней BB
  RSI > 65 + цена у верхней BB (±1%) → шорт к средней BB
  BTC медвежий → MR_ЛОНГИ заблокированы (WR 36% без фильтра)
  Объём < 3x, стоп 0.2-0.4%
  RR 1:2

VOLATILE → пауза
```

**ML фильтр:** Random Forest, AUC 0.608, порог 0.40
Каждый сигнал проходит ML перед добавлением в pending.

**Параметры:**
```python
timeframes       = ["5m"]
channel_bars     = 15       # свечей для канала
bo_vol_mult      = 1.5      # объём на пробое
min_breakout_pct = 0.001    # минимальный пробой 0.1%
rsi_oversold     = 35
rsi_overbought   = 65
min_stop_pct     = 0.002    # стоп <0.2% = шум → пропускаем
max_stop_pct     = 0.004    # стоп >0.4% = WR 0% → пропускаем
min_rr_mr        = 1.5      # минимальный RR для MR сигналов
bad_hours        = [14, 21, 22]
exchange         = "bitget"
```

**Статистика:**
```
Всего (старая стратегия): 604 сд  WR=23.5%
Новая стратегия (с 01.04): 20 сд  WR=50%
  MR_ШОРТ: WR=67%  PnL=+$4.35  ✅
  MR_ЛОНГ: WR=36%  (до фильтра BTC тренда)
  Channel BO: 0 сд (рынок пока не даёт пробоев)
```

---

### Бот #2 — Mean Reversion (Binance, 15m/1h)
**Режим:** SIDEWAYS

**Стратегия:** RSI + BB + ML фильтр, только ЛОНГИ

```
ШОРТЫ ОТКЛЮЧЕНЫ — математически убыточны:
  RR реальный 0.54 (WIN +$1.15 / LOSS -$2.12)
  Точка безубытка WR=65%, факт WR=42% → разрыв -23%
  107 сделок: SHORT PnL = -$32.35

ЛОНГИ работают:
  RR реальный 1.58 (WIN +$2.44 / LOSS -$1.54)
  Точка безубытка WR=39%, с ML фильтром WR=45%
```

**ML модель:** Random Forest, AUC 0.723 (только лонги)
OOS AUC 0.686, при пороге 0.40 WR=45% на тесте.

**Параметры:**
```python
timeframes     = ["15m", "1h"]
adx_max        = 25
rsi_oversold   = 30
rsi_overbought = 70   # не используется (шорты отключены)
min_rr         = 1.5
rr             = 2.0
```

**Статистика:** 107 сд, WR=36.4%, последние 20 сд WR=35%

---

### Бот #3 — Funding Rate (Bitget фьючерсы, 1h)
**Режим:** любой (лучший в TREND_DOWN)

**Стратегия:** контрариан по Funding Rate

```
FR < -0.03% → все шортят → ЛОНГУЕМ
FR > +0.03% → все лонгуют → ШОРТИМ
Подтверждение: RSI + BB
Лимитный ордер ±0.3%, RR 1:3
```

**Статистика:** 35 сд, WR=34.3%, PnL=-$38

**Наблюдение:** бот берёт малоликвидные монеты с аномальным FR.
ML модель исправит это автоматически после 60+ сделок с текущим WR.

---

### Бот #4 — Breakout (Binance, 15m/1h)
**Режим:** BREAKOUT

**Стратегия:** ATR squeeze + пробой с объёмом > 1.5x, RR 1:3

**Статистика:** 13 сд, WR=23.1%, PnL=-$7
Мало данных — нужно 150+ сделок для ML.

---

## Market Regime

| Режим | Условие | Бот |
|-------|---------|-----|
| TREND_UP | ADX>30, цена>EMA50 | Bot1 → Channel Breakout |
| TREND_DOWN | ADX>30, цена<EMA50 | Bot1 → Channel Breakout + Bot3 |
| SIDEWAYS | ADX<28, ATR норм. | Bot1 → MR + Bot2 MeanRev |
| BREAKOUT | BB squeeze + ATR↑ | Bot4 |
| VOLATILE | ATR>1.8x | пауза |

---

## ML система

| Бот | Модель | AUC | Сделок | Направление | Статус |
|-----|--------|-----|--------|-------------|--------|
| Bot1 | EMA | 0.608 | 604 | LONG+SHORT | ✅ Активна |
| Bot2 | MeanRev | 0.723 | 64 (LONG) | LONG only | ✅ Активна |
| Bot3 | Funding | — | 35 | — | ⏳ Нужно 60+ сд |
| Bot4 | Breakout | — | 13 | — | ⏳ Нужно 150+ сд |

### Переобучение
```bash
cd C:\TradingBots\Bot1_EMA     && python ml_train_ema.py
cd C:\TradingBots\Bot2_MeanRev && python ml_train_meanrev.py
# Bot3 и Bot4 — после накопления данных
```

---

## 🚀 Дорожная карта выхода на реальные деньги

```
Критерии выхода на реал:
  ✓ WR >= 35% на 200+ сделках (Bot1/2)
    или WR >= 30% на 60+ сделках с ML (Bot3)
  ✓ ML AUC >= 0.70
  ✓ Стабильность 2+ недели без деградации
  ✓ Просадка <= 20% за 30 дней
```

| Дата | Событие |
|------|---------|
| **07 апр 2026** | Bot3 достигает 60 сделок → обучаем ML |
| **09 апр 2026** | Bot2 достигает 200 сделок → оцениваем готовность |
| **Апр–Май 2026** | Bot1 новая стратегия: наблюдаем 100+ сделок |
| **🟢 Май 2026** | **Bot2 → реальные деньги** (если WR 35%+ держится) |
| **🟢 Май 2026** | **Bot3 → реальные деньги** (если WR 30%+ с ML) |
| **🟢 Июнь 2026** | **Bot1 → реальные деньги** (если новая стратегия WR 40%+) |
| Авг–Дек 2026 | Bot4 накапливает 150+ сд → ML → оценка |

> Боты идут на реал последовательно — сначала самые стабильные.
> Начинаем с минимального депозита $100-200 на каждый бот.

---

## Исправленные баги

| Баг | Дата | Описание |
|-----|------|----------|
| filled Bot1 | 25.03 | `price <= entry*1.005` → `±0.3%` |
| numpy bool JSON | 26.03 | `bool(bb_squeeze)` в market_regime |
| filled Bot3 | 27.03 | `±1%` → `±0.3%` |
| max_stop_pct | 27.03 | 0.8% → 0.4% |
| фьючерсы Bitget | 31.03 | фильтр `":" in s` в get_symbols |
| min_stop_pct MR | 02.04 | фильтр отсутствовал в MR блоке Bot1 |
| sklearn warning | 02.04 | np.array → pd.DataFrame в ml_filter |

---

## Запуск

```bash
# 1. Режим рынка (первым):
cd C:\TradingBots && python market_regime.py

# 2. Боты:
cd C:\TradingBots\Bot1_EMA       && python paper_trading_v2_clean.py
cd C:\TradingBots\Bot2_MeanRev   && python paper_trading_meanrev.py
cd C:\TradingBots\Bot3_Funding   && python paper_trading_funding.py
cd C:\TradingBots\Bot4_Breakout  && python paper_trading_breakout.py

# ML датасет:
cd C:\TradingBots && python ml_data_collector.py

# GitHub:
update.bat
```

---

## Установка

```bash
pip install ccxt pandas numpy scikit-learn
```

---

## Текущий статус (02.04.2026)

| Бот | Биржа | Сделок | WR | PnL | ML | Статус |
|-----|-------|--------|----|-----|----|--------|
| #1 Адаптивный | Bitget spot | 604 (20 новых) | 23.5% (50% новая) | -$440 | ✅ AUC 0.608 | 🔄 Новая стратегия |
| #2 MeanRev | Binance | 107 | 36.4% | -$47 | ✅ AUC 0.723 | 🎯 Цель: реал май 2026 |
| #3 Funding | Bitget futures | 35 | 34.3% | -$38 | ⏳ | 🎯 Цель: реал май 2026 |
| #4 Breakout | Binance | 13 | 23.1% | -$7 | ⏳ | 🔄 Накопление данных |

---

## Как продолжить в новом чате

```
Продолжаем проект торговых ботов.
GitHub: https://github.com/krein15/TradingBots
Прочитай README.md для контекста.
Задача: [что нужно сделать]
```

---

## Важные заметки

- Все боты: бумажные деньги (paper trading), депозит $50, риск 5%
- Запускать `market_regime.py` первым
- Bot2: `ml_model_meanrev.pkl` → папка `Bot2_MeanRev/`
- Bot1: `ml_model_ema.pkl` → папка `Bot1_EMA/`
- `*_journal.json`, `ml_dataset.csv`, `regime_history.jsonl` — не в GitHub
- Шорты Bot2 отключены — математически убыточны (RR реальный 0.54)
