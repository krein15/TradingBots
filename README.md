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
│   ├── ml_train_funding.py      # запускать после 150+ сделок
│   ├── ml_train_breakout.py     # запускать после 150+ сделок
│   ├── ml_dataset.csv           # НЕ загружать на GitHub
│   └── regime_history.jsonl     # НЕ загружать на GitHub
│
├── market_regime.py
├── shared_state.json            # текущий режим — читают все боты
├── start_regime.bat
├── update.bat
└── README.md
```

---

## Боты

### Бот #1 — Адаптивный (Bitget spot, 5m)
**Режим:** TREND_UP / TREND_DOWN / SIDEWAYS

**Ключевая идея — стратегия меняется по режиму рынка:**

```
TREND_UP / TREND_DOWN → Channel Breakout
  Пробой 20-свечного канала с объёмом > 2x
  Вход на закрытии пробойной свечи (по рынку)
  Стоп за противоположный край канала
  RR 1:3

SIDEWAYS / ? → Mean Reversion
  RSI < 35 + цена у нижней BB (±1%) → лонг к средней BB
  RSI > 65 + цена у верхней BB (±1%) → шорт к средней BB
  Объём < 3x (высокий объём в боковике опасен)
  RR 1:2

VOLATILE → пауза (нет сигналов)
```

**Биржа: Bitget spot** — старые 567 сделок с Binance сохранены в датасете
с пометкой `exchange=binance`, новые — `exchange=bitget`.

**Параметры:**
```python
timeframes       = ["5m"]
# Channel Breakout
channel_bars     = 20       # свечей для канала
bo_vol_mult      = 2.0      # объём на пробое
min_breakout_pct = 0.002    # минимальный пробой 0.2%
rr_trend         = 3.0      # RR в тренде

# Mean Reversion
rsi_oversold     = 35
rsi_overbought   = 65
bb_period        = 20
max_vol_mr       = 3.0      # макс. объём для MR
rr_sideways      = 2.0      # RR в боковике

# Общие
min_stop_pct     = 0.002
max_stop_pct     = 0.004    # стоп >0.4% = WR 0%
bad_hours        = [14, 21, 22]
risk_pct         = 0.05
exchange         = "bitget"
```

**Статистика:** 567 сделок (старая стратегия), WR 21.9% — стратегия переписана

**ML модель:** Random Forest, AUC 0.608, порог 0.40
Признаки: `risk_pct`, `vol_ratio`, `hour`, `regime_conf`, `body_ratio`,
`impulse_strength`, `bb_width`, `rsi`, `spread_pct`, `rel_strength`

---

### Бот #2 — Mean Reversion (Binance, 15m/1h)
**Режим:** SIDEWAYS (боковик)

**Стратегия:** RSI + Bollinger Bands + ML фильтр + BTC тренд фильтр

- ADX < 28 — только в боковике
- RSI < 35 + цена у нижней BB (±1%) → лонг
- RSI > 65 + цена у верхней BB (±1%) → шорт
- BTC медвежий → шорты заблокированы
- min_rr >= 1.5 — пропускаем сигналы с плохим RR
- ML фильтр (AUC 0.768): отсекает плохие шорты

**Параметры:**
```python
timeframes     = ["15m", "1h"]
adx_max        = 28
rsi_oversold   = 35
rsi_overbought = 65
min_rr         = 1.5
rr             = 2.0
```

**Статистика:** 82 сделки, WR 36.6%, тренд улучшается (последние 20 сд WR 40%)

**ML модель:** Random Forest, **AUC 0.768** — лучшая из всех моделей
```
vol_ratio  28% важности — высокий объём опасен для MR
adx        23% — сила тренда
bb_width   17% — ширина BB

Шорты одобренные ML: WR 81%
Шорты отклонённые ML: WR 12%
```

---

### Бот #3 — Funding Rate (Bitget фьючерсы, 1h)
**Режим:** любой (лучший в TREND_DOWN)

**Стратегия:** торгуем против толпы по Funding Rate

- FR > 0.03% → все лонгуют → ШОРТИМ
- FR < -0.03% → все шортят → ЛОНГУЕМ
- Подтверждение: RSI + BB
- Лимитный ордер ±0.3%
- RR 1:3, риск 5%

**Параметры:**
```python
fr_long_threshold  = 0.0003
fr_short_threshold = -0.0003
rsi_upper          = 60
rsi_lower          = 40
rr                 = 3.0
```

**Статистика:** 27 сделок, WR 29.6%, PnL -$25

**Проблема:** деградация WR с 50% → 10% в последних сделках.
Причина — бот берёт малоликвидные монеты (объём 0.1x) с аномальным FR.
ML модель нужна после накопления 150+ сделок.

---

### Бот #4 — Breakout (Binance, 15m/1h)
**Режим:** BREAKOUT

**Стратегия:** ATR squeeze + пробой с объёмом

- ATR < 100% среднего → консолидация (6 свечей)
- Пробой уровня с объёмом > 1.5x
- Свеча закрывается за уровнем (мин. 0.3%)
- Пробой свежий — не старше 3 свечей
- RR 1:3, риск 5%

**Статистика:** 11 сделок, WR 27.3%, PnL -$2

**Наблюдение:** первые 3 сделки (все LONG) WR 100%. Потом 8 LOSS —
преимущественно SHORT в медвежьем рынке. LONG WR 40%, SHORT WR 17%.
ML модель нужна после 150+ сделок.

---

## Market Regime

```bash
python market_regime.py   # или start_regime.bat
```

Каждые 30 минут записывает:
- `shared_state.json` — текущий режим (все боты читают)
- `ML/regime_history.jsonl` — история для ML анализа

| Режим | Условие | Бот |
|-------|---------|-----|
| TREND_UP | ADX>30, цена>EMA50, +DI>-DI | Bot1 → Channel Breakout |
| TREND_DOWN | ADX>30, цена<EMA50, -DI>+DI | Bot1 → Channel Breakout + Bot3 |
| SIDEWAYS | ADX<28, ATR нормальный | Bot1 → Mean Reversion + Bot2 |
| BREAKOUT | BB squeeze + ATR растёт | Bot4 |
| VOLATILE | ATR>1.8x среднего | пауза |

---

## ML система

### Статус моделей

| Бот | AUC | Сделок | Статус |
|-----|-----|--------|--------|
| Bot1 EMA | 0.608 | 567 | ✅ Активна |
| Bot2 MeanRev | **0.768** | 82 | ✅ Активна |
| Bot3 Funding | — | 27 | ⏳ Нужно 150+ сд |
| Bot4 Breakout | — | 11 | ⏳ Нужно 150+ сд |

### Признаки в датасете (Bot1)
```python
# Рыночный контекст
regime, regime_conf, exchange

# Качество сигнала
signal_type      # BO_ЛОНГ / BO_ШОРТ / MR_ЛОНГ / MR_ШОРТ
body_ratio       # чистота свечи
impulse_strength # сила в единицах ATR
bb_width, rsi    # для MR сигналов
spread_pct       # ликвидность монеты
rel_strength     # сила монеты vs BTC
btc_momentum     # скорость BTC
```

### Переобучение
```bash
# Bot1 (еженедельно):
cd C:\TradingBots\Bot1_EMA && python ml_train_ema.py

# Bot2 (еженедельно):
cd C:\TradingBots\Bot2_MeanRev && python ml_train_meanrev.py

# Bot3/4 (после 150+ сделок):
cd C:\TradingBots\Bot3_Funding  && python ml_train_funding.py
cd C:\TradingBots\Bot4_Breakout && python ml_train_breakout.py
```

---

## Исправленные баги

| Баг | Дата | Описание |
|-----|------|----------|
| filled Bot1 | 25.03 | `price <= entry*1.005` → `±0.3%` |
| numpy bool в JSON | 26.03 | `bool(bb_squeeze)` в market_regime |
| filled Bot3 | 27.03 | `±1%` → `±0.3%` в Funding |
| max_stop_pct Bot1 | 27.03 | 0.8% → 0.4% (стоп >0.4% WR=0%) |
| фьючерсы Bitget | 31.03 | фильтр `":" in s` в get_symbols |

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

# Утилиты Bot1:
python paper_trading_v2_clean.py refill  # пополнить баланс
python paper_trading_v2_clean.py reset   # сброс
python paper_trading_v2_clean.py status  # статистика
```

---

## Установка

```bash
pip install ccxt pandas numpy matplotlib scikit-learn
```

---

## Текущий статус (2026-03-31)

| Бот | Биржа | Стратегия | Сделок | WR | PnL | Статус |
|-----|-------|-----------|--------|----|-----|--------|
| #1 Адаптивный | Bitget spot | BO + MR по режиму | 567 | 21.9% | -$408 | 🔄 Новая стратегия |
| #2 MeanRev | Binance | RSI+BB + ML | 82 | 36.6% | -$44 | ✅ ML активна |
| #3 Funding | Bitget futures | Funding Rate | 27 | 29.6% | -$25 | ⚠️ Деградация WR |
| #4 Breakout | Binance | ATR + пробой | 11 | 27.3% | -$2 | 🔄 Мало данных |

**Точка безубыточности:** ~25% при RR 1:3, ~33% при RR 1:2

---

## Дорожная карта

- [x] 4 бота под разные режимы рынка
- [x] Market Regime детектор + shared_state.json + regime_history.jsonl
- [x] ML фильтр Bot1 (AUC 0.608)
- [x] ML фильтр Bot2 (AUC 0.768) — лучшая модель
- [x] Bot1: адаптивная стратегия (Channel Breakout в тренде / MR в боковике)
- [x] Bot1: переход на Bitget spot
- [x] Bot2: BTC тренд фильтр шортов + min_rr фильтр
- [x] Bot2: ML фильтр интегрирован в бота
- [x] Расширенные ML признаки (body_ratio, spread_pct, rel_strength, ...)
- [x] exchange как признак в датасете
- [x] Исправлен баг лимитных ордеров (Bot1, Bot3)
- [ ] Накопить 150+ сделок по Bot3/4 → запустить ML
- [ ] Исправить деградацию Bot3 (малоликвидные монеты)
- [ ] Переход всех ботов на Bitget
- [ ] Telegram уведомления
- [ ] Переход на реальные деньги (цель: WR 35%+ на 200+ сделках)

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

- Все боты: бумажные деньги (paper trading)
- Депозит $50, риск 5%, автопополнение при обнулении
- Запускать `market_regime.py` первым — боты читают `shared_state.json`
- Bot2: положить `ml_model_meanrev.pkl` в папку `Bot2_MeanRev/`
- Bot1: положить `ml_model_ema.pkl` в папку `Bot1_EMA/`
- `*_journal.json`, `ml_dataset.csv`, `regime_history.jsonl` — не в GitHub
