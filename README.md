# TradingBots — Автоматический торговый бот

Проект бумажного трейдинга на Python. 4 бота под разные режимы рынка,
Market Regime детектор, ML фильтр, автоматический журнал сделок.

---

## Архитектура

```
market_regime.py (каждые 30 мин)
  ├── TREND_UP / TREND_DOWN → Бот #1 EMA        (Binance, 5m)
  ├── SIDEWAYS              → Бот #2 MeanRev     (Binance, 15m/1h)
  ├── BREAKOUT              → Бот #4 Breakout    (Binance, 15m/1h)
  └── VOLATILE              → пауза

Бот #3 Funding Rate — работает независимо от режима (Bitget, 1h)

shared_state.json ← market_regime.py пишет каждые 30 мин
      ↓
все боты читают текущий режим и записывают его в каждую сделку
      ↓
ml_data_collector.py собирает в единый датасет с признаком regime
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
│   ├── ml_dataset.csv           # НЕ загружать на GitHub
│   └── regime_history.jsonl     # НЕ загружать на GitHub
│
├── market_regime.py             # определяет режим рынка
├── shared_state.json            # текущий режим — читают все боты
├── start_regime.bat
├── update.bat                   # git push
└── README.md
```

---

## Боты

### Бот #1 — EMA (Binance, 5m)
**Режим:** TREND_UP / TREND_DOWN

**Стратегия:** Импульс + BTC EMA50 + ML фильтр

- BTC выше EMA50 → только лонги, ниже → только шорты
- Лимитный вход по Фибо 23.6%
- Стоп под лоу импульсной свечи
- Фильтр стопа: только 0.2–0.5% (зона WR 30–36%)
- Блокировка плохих часов: 14, 21, 22 UTC (WR 7–9%)
- ML фильтр: Random Forest, порог 0.34
- Записывает `regime` из shared_state.json в каждую сделку
- RR 1:4, риск 5%

**Параметры:**
```python
timeframes    = ["5m"]
min_price_chg = 0.003
min_vol_mult  = 1.2
fibo_entry    = 0.236
min_stop_pct  = 0.002        # стоп <0.2% — выбивается шумом
max_stop_pct  = 0.005        # стоп >0.5% — WR падает
bad_hours     = [14, 21, 22]
rr_ratio      = 4.0
risk_pct      = 0.05
```

**Статистика:** 391 сделка, WR 23.8%

**Ключевые инсайты (391 сделка):**
```
Стоп 0.2-0.3%               → WR 31%
Стоп 0.2-0.4%               → WR 30%  (лучшая зона)
Стоп 0.2-0.5% + хор. часы  → WR 45-49%
Стоп >0.5%                  → WR ≤17% (фильтруется)
Объём 3-5x                  → WR 36%
Часы 10, 11, 15, 16, 19 UTC → WR 33-44% (хорошие)
Часы 14, 21, 22 UTC         → WR 7-9%  (блокируются)
```

---

### Бот #2 — Mean Reversion (Binance, 15m/1h)
**Режим:** SIDEWAYS (боковик)

**Стратегия:** RSI + Bollinger Bands в флете

- ADX < 25 — только в боковике
- RSI < 30 + цена у нижней BB → лонг
- RSI > 70 + цена у верхней BB → шорт
- Тейк: средняя линия BB (возврат к среднему)
- Записывает `regime` из shared_state.json в каждую сделку
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

**Статистика:** 23 сделки, WR 30.4%

**Ключевые инсайты:**
```
Средний стоп:  0.34%  среднее движение WIN: 0.78%
RR 1:3 убьёт 5 из 7 WIN — стратегии нужен RR 1:2
Открытые сделки с rr=0.03 и rr=0.54 — нужен фильтр min_rr >= 1.5
```

---

### Бот #3 — Funding Rate (Bitget фьючерсы, 1h)
**Режим:** любой (работает всегда)

**Стратегия:** торгуем против толпы по Funding Rate

- FR > 0.03% → все лонгуют → ШОРТИМ
- FR < -0.03% → все шортят → ЛОНГУЕМ
- Подтверждение: RSI + BB
- Записывает `regime` из shared_state.json в каждую сделку
- RR 1:3, риск 5%

**Параметры:**
```python
timeframes           = ["1h"]
fr_long_threshold    = 0.0003
fr_short_threshold   = -0.0003
rsi_upper            = 60
rsi_lower            = 40
rr                   = 3.0
interval_min         = 15
```

**Статистика:** 8 сделок, WR 50%

**Известная проблема:**
```
DOT/USDT входил 4 раза подряд с одинаковыми параметрами —
дедупликация по символу не работает. Нужен фикс.
```

---

### Бот #4 — Breakout (Binance, 15m/1h)
**Режим:** BREAKOUT (пробой консолидации)

**Стратегия:** ATR squeeze + пробой с объёмом

- ATR < 90% среднего → консолидация
- Пробой уровня с объёмом > 1.5x
- Свеча закрывается за уровнем (минимум 0.3%)
- BTC тренд фильтр
- Записывает `regime` из shared_state.json в каждую сделку
- RR 1:3, риск 5%

**Параметры:**
```python
timeframes       = ["15m", "1h"]
atr_squeeze      = 0.9
breakout_vol     = 1.5
min_breakout_pct = 0.003
rr               = 3.0
interval_min     = 15
```

**Статистика:** 1 сделка, WR 0% — ждём данных

---

## Market Regime

```bash
python market_regime.py   # или start_regime.bat
```

Каждые 30 минут определяет режим и пишет два файла:

- **`shared_state.json`** — текущий режим, все боты читают перед сканированием
- **`ML/regime_history.jsonl`** — история всех измерений для ML

```
📈 РЕЖИМ РЫНКА: TREND_UP  (уверенность: 60%)
   ADX: 30.3  ATR ratio: 0.75  BTC: $71063 > EMA50: $70297
[+] Сохранено в shared_state.json и regime_history.jsonl
```

| Режим | Условие | Бот |
|-------|---------|-----|
| TREND_UP | ADX > 30, цена > EMA50, +DI > -DI | Bot1 EMA — лонги |
| TREND_DOWN | ADX > 30, цена < EMA50, -DI > +DI | Bot1 EMA — шорты |
| SIDEWAYS | ADX < 25, ATR нормальный | Bot2 MeanRev |
| BREAKOUT | BB squeeze + ATR растёт | Bot4 Breakout |
| VOLATILE | ATR > 1.8x среднего | пауза |

---

## ML система

### Модель Bot1 EMA
- Алгоритм: Random Forest + калибровка
- Обучена на: 391 EMA сделке
- ROC-AUC: 0.584, порог: 0.34, WR при пороге: 50%

### ml_data_collector.py
Собирает сделки всех 4 ботов в единый датасет. Новые признаки:

```python
# Из каждой сделки:
regime       # TREND_UP / SIDEWAYS / BREAKOUT / VOLATILE
regime_conf  # уверенность 0-100%

# Если поля нет в журнале — берётся из regime_history.jsonl
# по ближайшей записи до времени закрытия сделки
```

### Переобучение Bot1
```bash
cd C:\TradingBots\Bot1_EMA && python ml_train_ema.py
```
Запускать раз в неделю. ML для Bot2/3/4 — после 100+ сделок каждому.

---

## Исправленные баги

### Баг лимитных ордеров (исправлен 2026-03-25)
```python
# БЫЛО — срабатывало почти всегда:
filled = (d == 1 and price <= entry * 1.005)

# СТАЛО — только когда цена реально в зоне входа ±0.3%:
filled = (entry * 0.997 <= price <= entry * 1.003)
```
Исправлено во всех 4 ботах.

### Баг numpy bool в JSON (исправлен 2026-03-26)
```python
# market_regime.py — bb_squeeze возвращал numpy bool_, не сериализуемый в JSON
# БЫЛО:  "bb_squeeze": bb_squeeze
# СТАЛО: "bb_squeeze": bool(bb_squeeze)
```

---

## Запуск

```bash
# 1. Сначала — режим рынка (пишет shared_state.json):
cd C:\TradingBots && python market_regime.py

# 2. Боты:
cd C:\TradingBots\Bot1_EMA       && python paper_trading_v2_clean.py
cd C:\TradingBots\Bot2_MeanRev   && python paper_trading_meanrev.py
cd C:\TradingBots\Bot3_Funding   && python paper_trading_funding.py
cd C:\TradingBots\Bot4_Breakout  && python paper_trading_breakout.py

# 3. ML датасет:
cd C:\TradingBots && python ml_data_collector.py

# GitHub:
update.bat

# Пополнить баланс:
python paper_trading_v2_clean.py refill

# Сброс:
python paper_trading_v2_clean.py reset
```

---

## Установка

```bash
pip install ccxt pandas numpy matplotlib scikit-learn
```

---

## Текущий статус (2026-03-26)

| Бот | Стратегия | Биржа | Сделок | WR | Статус |
|-----|-----------|-------|--------|----|--------|
| #1 EMA | Momentum + ML | Binance | 391 | 23.8% | ✅ Активен |
| #2 MeanRev | RSI + BB боковик | Binance | 23 | 30.4% | 🔄 Копим данные |
| #3 Funding | Funding Rate | Bitget | 8 | 50% | 🔄 Копим данные |
| #4 Breakout | ATR + пробой | Binance | 1 | 0% | 🔄 Копим данные |

**Общая точка безубыточности:** ~20% при RR 1:4, ~33% при RR 1:2

---

## Дорожная карта

- [x] 4 бота под разные режимы рынка
- [x] ML фильтр для бота #1
- [x] Market Regime детектор
- [x] Исправлен баг лимитных ордеров (все боты)
- [x] RSI дивергенция
- [x] Запись режима рынка в каждую сделку (shared_state.json)
- [x] regime_history.jsonl — история режимов для ML
- [x] stop/take/rr в журналах Bot2/3/4
- [ ] Фикс дедупликации Bot3 Funding (DOT входил 4 раза)
- [ ] Фильтр min_rr >= 1.5 в Bot2 MeanRev
- [ ] Накопить 100+ сделок по ботам #2, #3, #4
- [ ] ML для ботов #2, #3, #4
- [ ] Интеграция market_regime: боты торгуют только в свой режим
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
- Запускать market_regime.py первым — боты читают shared_state.json
- `*_journal.json` — не загружать на GitHub
- `ml_dataset.csv`, `regime_history.jsonl` — не загружать на GitHub
- `divergence_module.py` — положить в папку Bot1_EMA
