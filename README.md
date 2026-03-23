# TradingBots — Автоматический торговый бот

Проект бумажного трейдинга на Python. 4 бота с разными стратегиями,
ML фильтр, автоматический журнал сделок.

---

## Структура проекта

```
TradingBots/
├── Bot1_EMA/                    # Бот #1 — EMA + ML фильтр
│   ├── paper_trading_v2_clean.py
│   ├── divergence_module.py
│   ├── ml_model_ema.pkl         # НЕ загружать на GitHub
│   ├── ml_train_ema.py
│   └── start_bot1.bat
│
├── Bot2_Structure/              # Бот #2 — Структура рынка BOS
│   ├── paper_trading_structure_v2.py
│   ├── divergence_module.py
│   └── start_bot2.bat
│
├── Bot3_SMC/                    # Бот #3 — SMC MSS+OB+Дивергенция
│   ├── paper_trading_smc_v2.py
│   ├── divergence_module.py
│   └── start_bot3.bat
│
├── Bot4_Wyckoff/                # Бот #4 — Wyckoff Spring
│   ├── paper_trading_wyckoff.py
│   ├── divergence_module.py
│   └── start_bot4.bat
│
├── ML/                          # ML датасет и статистика
│   ├── ml_data_collector.py
│   ├── ml_train_ema.py
│   └── ml_dataset.csv           # НЕ загружать на GitHub
│
├── Tools/
│   └── debug_scanner.py
│
├── combined_stats.py            # Общая статистика всех ботов
├── show_stats.bat               # Запуск статистики
├── collect_ml_data.bat          # Сбор ML данных
└── README.md
```

---

## Боты

### Бот #1 — EMA (Binance, 5m/15m)
**Стратегия:** Импульс + фильтр BTC EMA50 + ML фильтр

- Сканирует 150+ монет Binance
- BTC выше EMA50 → только лонги
- BTC ниже EMA50 → только шорты
- Лимитный вход по Фибо 23.6%
- Стоп: под лоу импульсной свечи
- ML фильтр: Random Forest, порог 0.34, WR при пороге 50%
- Дивергенция RSI как дополнительный фильтр
- Фильтр сессии: Лондон + Нью-Йорк (07-18 UTC)
- RR 1:4, риск 5% на сделку

**Параметры:**
```python
timeframes = ["5m", "15m"]
min_price_chg = 0.02      # 2% импульс
min_vol_mult = 1.8        # объём 1.8x
fibo_entry = 0.236        # Фибо 23.6%
rr_ratio = 4.0
risk_pct = 0.05
session_hours = [(7, 18)] # UTC
```

**Статистика:** 217 сделок, WR 25.3%

---

### Бот #2 — Структура рынка (Binance, 5m/15m)
**Стратегия:** BOS (Break of Structure) + ретест

- Свинги: 5 свечей с каждой стороны
- Структура HH+HL → бычья → лонги
- Структура LH+LL → медвежья → шорты
- BOS = пробой последнего свинга
- Вход на ретесте пробитого уровня
- Стоп за свинг-хай/лой
- Дивергенция RSI как фильтр
- ADX > 25 (фильтр флета)
- RR 1:4, риск 5%

**Параметры:**
```python
timeframes = ["5m", "15m"]
swing_n = 5
min_swing_pct = 0.015
rr = 4.0
```

**Статистика:** 42 сделки, WR 11.9%

---

### Бот #3 — SMC v2 (Bitget, 1h/4h)
**Стратегия:** MSS (Market Structure Shift) + Order Block + Дивергенция

- MSS = слом структуры с объёмом 2x+
- Order Block = последняя свеча перед MSS
- Стоп за тело OB (короткий стоп)
- Дивергенция RSI обязательна
- HTF тренд по EMA50
- BTC тренд фильтр
- Только Лондон + Нью-Йорк (07-18 UTC)
- RR 1:4, риск 5%

**Параметры:**
```python
timeframes = ["1h", "4h"]
min_vol = 2.0
mss_min_pct = 0.005
require_div = True
rr = 4.0
interval_min = 30
```

**Статистика:** 123 сделки, WR 8.1% (переработан — ждём новых данных)

---

### Бот #4 — Wyckoff Spring (Bitget, 1h/4h)
**Стратегия:** Wyckoff фаза накопления + Volume Profile + Дивергенция

- SC (Selling Climax): объём 5x+, падение 3%+
- AR (Automatic Rally): отскок 2%+
- ST (Secondary Test): объём < 40% от SC
- Spring: ложный пробой SC лоя
- Spring закрывается в верхней половине свечи
- Volume Profile: Spring ниже POC
- Бычья дивергенция RSI обязательна
- RR 1:7, риск 5%

**Параметры:**
```python
timeframes = ["1h", "4h"]
sc_vol_mult = 5.0
ar_min_pct = 0.02
st_vol_ratio = 0.4
rr_ratio = 7.0
interval_min = 30
```

**Статистика:** 5 сделок, WR 0% (редкая стратегия — ждём паттернов)

---

## ML модель (Бот #1)

### Текущая модель
- Алгоритм: Random Forest + калибровка вероятностей
- Обучена на: 217 EMA сделках
- ROC-AUC: 0.584
- Порог: 0.34
- WR при пороге: 50%

### Признаки
```
tf_minutes     # таймфрейм в минутах
risk_pct       # размер стопа %
reward_pct     # размер тейка %
rr_planned     # плановый RR
vol_ratio      # объём относительно среднего
hour           # час входа (UTC)
day_of_week    # день недели
is_london      # Лондонская сессия (07-12 UTC)
is_newyork     # Нью-Йорк сессия (13-18 UTC)
is_night       # ночь (22-06 UTC)
is_asia        # Азия (00-07 UTC)
dir_enc        # направление (лонг/шорт)
```

### Ключевые выводы из данных
```
Стоп <0.3%   → WR 30%  (лучший результат!)
Стоп >2%     → WR 0%   (никогда не выигрывает)
Лондон       → WR 22.7% vs 17.1% в остальное время
Объём 3-5x   → WR 25%  (оптимальный диапазон)
```

### Переобучение
```bash
cd C:\TradingBots\Bot1_EMA
python ml_train_ema.py
```
Запускать раз в неделю когда накопятся новые сделки.

---

## Запуск

### Запуск ботов
```bash
# Двойной клик на bat файле или через CMD:
cd C:\TradingBots\Bot1_EMA && python paper_trading_v2_clean.py
cd C:\TradingBots\Bot2_Structure && python paper_trading_structure_v2.py
cd C:\TradingBots\Bot3_SMC && python paper_trading_smc_v2.py
cd C:\TradingBots\Bot4_Wyckoff && python paper_trading_wyckoff.py
```

### Статистика
```bash
# Общая статистика всех ботов:
cd C:\TradingBots && python combined_stats.py

# Статус отдельного бота:
python paper_trading_v2_clean.py status

# Пополнить баланс без сброса истории:
python paper_trading_v2_clean.py refill

# Сброс журнала:
python paper_trading_v2_clean.py reset
```

### ML данные
```bash
cd C:\TradingBots && python ml_data_collector.py
```

---

## Установка зависимостей

```bash
pip install ccxt pandas numpy matplotlib scikit-learn
```

---

## Текущий статус (2026-03-23)

| Бот | Сделок | WR | Статус |
|-----|--------|-----|--------|
| #1 EMA | 217 | 25.3% | ✅ Работает + ML |
| #2 Структура | 42 | 11.9% | ✅ Работает |
| #3 SMC v2 | 0 | — | 🆕 Новая версия |
| #4 Wyckoff | 5 | 0% | ⏳ Ждёт паттернов |

**Общий WR:** 18.1% (387 сделок)
**Точка безубыточности при RR 1:4:** ~20% WR

---

## Дорожная карта

- [x] Momentum сканер
- [x] Бумажный трейдинг
- [x] 4 разные стратегии
- [x] ML фильтр для бота #1
- [x] Дивергенция RSI
- [x] Фильтр торговых сессий
- [ ] ML для ботов #2, #3, #4 (нужно 100+ сделок каждому)
- [ ] Fear & Greed фильтр
- [ ] Переход на реальные деньги (после стабильного WR 35%+)
- [ ] Самообучение (адаптация параметров)

---

## Как продолжить работу в новом чате

Напиши Claude:
```
Продолжаем проект торговых ботов.
GitHub: https://github.com/[твой_ник]/TradingBots
Прочитай README.md для контекста.
Текущая задача: [опиши что нужно сделать]
```

---

## Важные заметки

- Все боты используют **бумажные деньги** (paper trading)
- Депозит $50, риск 5% на сделку
- Автопополнение баланса при обнулении
- Cooldown 2-3 часа после LOSS по монете
- Журналы сделок: `*_journal.json` — не загружать на GitHub
- ML датасет: `ml_dataset.csv` — не загружать на GitHub
