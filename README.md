# TradingBots — Автоматический торговый бот

Проект бумажного трейдинга на Python. 4 бота под разные режимы рынка,
ML мета-модель для определения режима, автоматический журнал сделок.

---

## Архитектура

```
Режим рынка (market_regime.py)
  ├── TREND_UP / TREND_DOWN → Бот #1 EMA
  ├── SIDEWAYS              → Бот #2 MeanReversion
  ├── BREAKOUT              → Бот #4 Breakout
  └── VOLATILE              → пауза

Бот #3 Funding Rate — работает независимо от режима
```

---

## Структура проекта

```
TradingBots/
├── Bot1_EMA/
│   ├── paper_trading_v2_clean.py
│   ├── divergence_module.py
│   ├── ml_model_ema.pkl         # НЕ загружать на GitHub
│   ├── ml_train_ema.py
│   └── start_bot1.bat
│
├── Bot2_MeanRev/
│   ├── paper_trading_meanrev.py
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
│   └── ml_dataset.csv           # НЕ загружать на GitHub
│
├── market_regime.py             # определяет режим рынка
├── start_regime.bat             # мониторинг режима
├── update.bat                   # git push на GitHub
└── README.md
```

---

## Боты

### Бот #1 — EMA (Binance, 5m)
**Режим:** TREND_UP / TREND_DOWN

**Стратегия:** Импульс + BTC EMA50 + ML фильтр

- BTC выше EMA50 → только лонги
- BTC ниже EMA50 → только шорты
- Лимитный вход по Фибо 23.6%
- ML фильтр: Random Forest, порог 0.34, WR при пороге 50%
- Дивергенция RSI как дополнительный фильтр
- RR 1:4, риск 5%

**Исправления:**
```
Баг лимитных ордеров исправлен — сделка открывается только
когда цена реально достигает уровня входа (±0.3%)
Старая логика: price <= entry → почти всегда true!
```

**Параметры:**
```python
timeframes    = ["5m"]
min_price_chg = 0.003
min_vol_mult  = 1.2
fibo_entry    = 0.236
rr_ratio      = 4.0
risk_pct      = 0.05
```

**Статистика:** 353 сделки, WR 23.8%

---

### Бот #2 — Mean Reversion (Binance, 15m/1h)
**Режим:** SIDEWAYS (боковик)

**Стратегия:** RSI + Bollinger Bands в флете

- ADX < 25 — только в боковике
- RSI < 30 + цена у нижней BB → лонг
- RSI > 70 + цена у верхней BB → шорт
- Объём < 3x — исключаем новостные движения
- Тейк: средняя линия BB (возврат к среднему)
- RR 1:2, риск 5%

**Параметры:**
```python
timeframes     = ["15m", "1h"]
adx_max        = 25
rsi_oversold   = 30
rsi_overbought = 70
max_vol_mult   = 3.0
rr             = 2.0
interval_min   = 10
```

**Статистика:** новый бот — ждём данных

---

### Бот #3 — Funding Rate (Bitget фьючерсы, 1h)
**Режим:** любой (работает всегда)

**Стратегия:** торгуем против толпы по Funding Rate

- FR > 0.03% → все лонгуют → ШОРТИМ
- FR < -0.03% → все шортят → ЛОНГУЕМ
- Подтверждение: RSI + Bollinger Bands
- Объём < 3x — исключаем новостные движения
- RR 1:3, риск 5%

**Почему работает:**
```
Когда FR экстремальный → толпа перегрета в одну сторону
Лонгисты/шортисты начинают закрывать позиции чтобы
не платить funding → рынок разворачивается
```

**Параметры:**
```python
timeframes           = ["1h"]
fr_long_threshold    = 0.0003   # > 0.03% → шорт
fr_short_threshold   = -0.0003  # < -0.03% → лонг
rsi_upper            = 60
rsi_lower            = 40
rr                   = 3.0
interval_min         = 15
```

**Статистика:** новый бот — ждём данных

---

### Бот #4 — Breakout (Binance, 15m/1h)
**Режим:** BREAKOUT (пробой консолидации)

**Стратегия:** ATR squeeze + пробой с объёмом

- ATR сжат (< 70% среднего) → консолидация
- Пробой уровня с объёмом > 2x
- Свеча закрывается за уровнем
- BTC тренд фильтр
- RR 1:3, риск 5%

**Параметры:**
```python
timeframes       = ["15m", "1h"]
atr_squeeze      = 0.7
breakout_vol     = 2.0
min_breakout_pct = 0.005
rr               = 3.0
interval_min     = 15
```

**Статистика:** новый бот — ждём данных

---

## ML модель (Бот #1)

### Текущая модель
- Алгоритм: Random Forest + калибровка
- Обучена на: 353 EMA сделках
- ROC-AUC: 0.584
- Порог: 0.34
- WR при пороге: 50%

### Топ признаков
```
risk_pct    -0.183  (меньше стоп → лучше результат)
reward_pct  -0.170
hour        +0.107  (время входа важно)
vol_ratio   +0.105  (объём важен)
```

### Ключевые выводы из данных (353 EMA сделки)
```
Стоп 0.2-0.4%  → WR 30%  (лучшая зона)
Стоп >2%       → WR 0%   (никогда не выигрывает)
Объём 2.5-4x   → WR 31%  (оптимальный диапазон)
Лондон 07-12   → WR 22.7% vs 17.1% остальное
```

### Переобучение
```bash
cd C:\TradingBots\Bot1_EMA
python ml_train_ema.py
```

---

## Market Regime (определитель режима рынка)

```bash
# Запуск мониторинга:
start_regime.bat

# Или напрямую:
python market_regime.py
```

Показывает каждые 30 минут:
```
📈 РЕЖИМ РЫНКА: TREND_UP  (уверенность: 60%)
   Рекомендуемый бот: Bot1_EMA — лонги по тренду
   ADX: 30.3  ATR ratio: 0.75  BTC: $71063 > EMA50: $70297
```

**Режимы:**
```
TREND_UP   → ADX > 30 + цена > EMA50 + +DI > -DI
TREND_DOWN → ADX > 30 + цена < EMA50 + -DI > +DI
SIDEWAYS   → ADX < 25 + ATR нормальный
BREAKOUT   → BB squeeze + ATR растёт
VOLATILE   → ATR > 1.8x среднего
```

---

## Исправленные баги

### Критический баг лимитных ордеров (исправлен 2026-03-25)
```python
# БЫЛО (неверно): сделка открывалась когда цена НИЖЕ входа
filled = (d == 1 and price <= entry * 1.005)
# Это почти всегда true при падении рынка!

# СТАЛО (верно): только когда цена В ЗОНЕ ±0.3% от входа
filled = (entry * 0.997 <= price <= entry * 1.003)
```
Исправлено во всех 6 ботах.

---

## Запуск

```bash
# Боты:
cd C:\TradingBots\Bot1_EMA       && python paper_trading_v2_clean.py
cd C:\TradingBots\Bot2_MeanRev   && python paper_trading_meanrev.py
cd C:\TradingBots\Bot3_Funding   && python paper_trading_funding.py
cd C:\TradingBots\Bot4_Breakout  && python paper_trading_breakout.py

# Режим рынка:
cd C:\TradingBots && python market_regime.py

# Статистика + ML датасет:
cd C:\TradingBots && python ml_data_collector.py

# Обновить GitHub:
update.bat

# Пополнить баланс (история сохраняется):
python paper_trading_v2_clean.py refill

# Полный сброс:
python paper_trading_v2_clean.py reset
```

---

## Установка

```bash
pip install ccxt pandas numpy matplotlib scikit-learn
```

---

## Текущий статус (2026-03-25)

| Бот | Стратегия | Биржа | Сделок | WR | Статус |
|-----|-----------|-------|--------|-----|--------|
| #1 EMA | Momentum + ML | Binance | 353 | 23.8% | ✅ Активен |
| #2 MeanRev | RSI + BB в боковике | Binance | 0 | — | 🆕 Новый |
| #3 Funding | Funding Rate | Bitget | 0 | — | 🆕 Новый |
| #4 Breakout | ATR + пробой | Binance | 0 | — | 🆕 Новый |

**Общий WR:** 23.8% (353 сделки EMA)
**Точка безубыточности:** ~20% при RR 1:4

---

## Дорожная карта

- [x] 4 бота под разные режимы рынка
- [x] ML фильтр для бота #1
- [x] Market Regime детектор
- [x] Исправлен баг лимитных ордеров
- [x] RSI дивергенция
- [ ] Накопить 100+ сделок по ботам #2, #3, #4
- [ ] ML для ботов #2, #3, #4
- [ ] Интеграция market_regime в ботов (автовыбор стратегии)
- [ ] Переход на реальные деньги (цель: WR 35%+)

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
- Журналы `*_journal.json` — не загружать на GitHub
- ML датасет `ml_dataset.csv` — не загружать на GitHub
- `divergence_module.py` — положить в папку каждого бота
