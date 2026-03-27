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
все боты читают режим → адаптируют параметры входа
      ↓
ml_data_collector.py → единый датасет с признаками regime + fibo_used
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
├── market_regime.py
├── shared_state.json            # текущий режим — читают все боты
├── start_regime.bat
├── update.bat
└── README.md
```

---

## Боты

### Бот #1 — EMA адаптивный (Binance, 5m)
**Режим:** TREND_UP / TREND_DOWN / SIDEWAYS

**Ключевая идея — адаптивная точка входа по Фибо:**
```
SIDEWAYS / ?  → fibo_entry 23.6% — ждём откат (классика)
TREND_*       → fibo_trend  5%   — входим почти сразу на импульсе

Почему: в тренде (ADX=44) импульс сильный (1-3%).
Откат 23.6% долгий — цена уходит пока ждём.
При 5% вход ближе к пробою → меньше ожидания → лучше исполнение.
```

**Стратегия:**
- BTC выше EMA50 → только лонги, ниже → только шорты
- Импульсная свеча + объём 1.2x
- Стоп под лоу / над хаем импульсной свечи
- ML фильтр: Random Forest, порог 0.34
- Читает `shared_state.json` → логирует режим + Фибо
- RR 1:4, риск 5%

**Параметры:**
```python
timeframes    = ["5m"]
fibo_entry    = 0.236        # SIDEWAYS — откат 23.6%
fibo_trend    = 0.05         # TREND — вход почти сразу (5%)
min_stop_pct  = 0.002        # стоп <0.2% — выбивается шумом
max_stop_pct  = 0.004        # стоп >0.4% — WR=0% на данных
bad_hours     = [14, 21, 22]
rr_ratio      = 4.0
risk_pct      = 0.05
```

**Статистика:** 432 сделки, WR 23.4%

**Ключевые инсайты:**
```
SIDEWAYS → WR 33% (бот лучше работает в боковике)
TREND_DOWN → WR 15% (до патча адаптивного Фибо)
EMA шорты в SIDEWAYS → WR 67% (лучший результат)
Стоп 0.2-0.4% → WR 29-30%
Стоп >0.4%    → WR 0% (фильтруется)
```

---

### Бот #2 — Mean Reversion (Binance, 15m/1h)
**Режим:** SIDEWAYS (боковик)

**Стратегия:** RSI + Bollinger Bands в флете + BTC тренд фильтр

- ADX < 28 — только в боковике
- RSI < 35 + цена у нижней BB (±1%) → лонг
- RSI > 65 + цена у верхней BB (±1%) → шорт
- **BTC тренд фильтр шортов:** шорты заблокированы если BTC медвежий
- **min_rr = 1.5** — пропускаем сигналы с RR < 1.5 (когда BB слишком узкие)
- Тейк: средняя линия BB, RR 1:2, риск 5%

**Параметры:**
```python
timeframes     = ["15m", "1h"]
adx_max        = 28
rsi_oversold   = 35
rsi_overbought = 65
min_rr         = 1.5         # минимальный RR — отсекаем мусорные сигналы
rr             = 2.0
```

**Статистика:** 41 сделка, WR 36.6%

**Ключевые инсайты:**
```
LONG:  RR реальный 2.06 — работает (+$1.39)
SHORT: RR реальный 0.47 — убыточны в TREND_DOWN (-$32)
Решение: BTC тренд фильтр блокирует шорты в медвежьем рынке
```

---

### Бот #3 — Funding Rate (Bitget фьючерсы, 1h)
**Режим:** любой (лучший в TREND_DOWN)

**Стратегия:** торгуем против толпы по Funding Rate

- FR > 0.03% → все лонгуют → ШОРТИМ
- FR < -0.03% → все шортят → ЛОНГУЕМ
- Подтверждение: RSI + BB
- Лимитный ордер ±0.3% от входа
- RR 1:3, риск 5%

**Параметры:**
```python
timeframes           = ["1h"]
fr_long_threshold    = 0.0003
fr_short_threshold   = -0.0003
rsi_upper            = 60
rsi_lower            = 40
rr                   = 3.0
filled_zone          = ±0.3%   # исправлен баг ±1%
```

**Статистика:** 15 сделок, WR 46.7%, PnL +$7

**Корреляция с режимом:**
```
TREND_DOWN → WR 60% (+$13)  лучший режим
SIDEWAYS   → WR 0%  (-$9)   FR не работает в боковике
```

---

### Бот #4 — Breakout (Binance, 15m/1h)
**Режим:** BREAKOUT

**Стратегия:** ATR squeeze + пробой с объёмом

- ATR < 100% среднего → консолидация (6 свечей)
- Пробой с объёмом > 1.5x, свеча закрывается за уровнем
- Пробой свежий — не старше 3 свечей
- RR 1:3, риск 5%

**Параметры:**
```python
timeframes       = ["15m", "1h"]
consol_bars      = 6     # было 10 — стоп слишком большой
atr_squeeze      = 1.0   # было 0.9 — слишком строго
breakout_vol     = 1.5
min_breakout_pct = 0.003
```

**Статистика:** 3 сделки, WR 66.7%, PnL +$13

---

## Market Regime

```bash
python market_regime.py   # или start_regime.bat
```

Каждые 30 минут определяет режим и пишет:
- `shared_state.json` — текущий режим (боты читают перед сканированием)
- `ML/regime_history.jsonl` — история для ML анализа

**Статистика за первые сутки (49 измерений):**
```
TREND_DOWN : 71% времени  ADX ср. 44.1
SIDEWAYS   : 29% времени  ADX ср. 25.9
```

**Корреляция режима с результатами:**
```
SIDEWAYS   → WR 40%  PnL +$2   (все боты)
TREND_DOWN → WR 28%  PnL -$8   (все боты)

Лучшие сочетания:
  EMA шорты в SIDEWAYS   → WR 67%
  Funding в TREND_DOWN   → WR 60% (+$13)
  MeanRev в SIDEWAYS     → WR 60%
```

| Режим | Условие | Рекомендуемый бот |
|-------|---------|-------------------|
| TREND_UP | ADX>30, цена>EMA50, +DI>-DI | Bot1 EMA |
| TREND_DOWN | ADX>30, цена<EMA50, -DI>+DI | Bot1 EMA + Bot3 Funding |
| SIDEWAYS | ADX<28, ATR нормальный | Bot2 MeanRev |
| BREAKOUT | BB squeeze + ATR растёт | Bot4 Breakout |
| VOLATILE | ATR>1.8x среднего | пауза |

---

## ML система

### Модель Bot1 EMA
- Random Forest + калибровка, 432 сделки
- ROC-AUC: 0.584, порог: 0.34, WR при пороге: 50%

### Признаки в датасете
```python
regime        # TREND_UP / TREND_DOWN / SIDEWAYS / BREAKOUT
regime_conf   # уверенность 0-100%
fibo_used     # 0.05 (тренд) или 0.236 (боковик) — Bot1 EMA
```

### Переобучение
```bash
cd C:\TradingBots\Bot1_EMA && python ml_train_ema.py
```
ML для Bot2/3/4 — после 100+ сделок каждому.

---

## Исправленные баги

| Баг | Дата | Описание |
|-----|------|----------|
| Лимитные ордера Bot1 | 25.03 | `price <= entry*1.005` → `±0.3%` |
| numpy bool в JSON | 26.03 | `bool(bb_squeeze)` в market_regime |
| None в regime_history | 26.03 | try/catch в save_regime |
| Лимитные ордера Bot3 | 27.03 | `±1%` → `±0.3%` в Funding |

---

## Запуск

```bash
# 1. Сначала — режим рынка:
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

## Текущий статус (2026-03-27)

| Бот | Стратегия | Сделок | WR | PnL | Статус |
|-----|-----------|--------|----|-----|--------|
| #1 EMA | Адаптивный Фибо + ML | 432 | 23.4% | -$281 | 🔄 Адаптивный Фибо |
| #2 MeanRev | RSI+BB + BTC фильтр | 41 | 36.6% | -$31 | 🔄 BTC фильтр шортов |
| #3 Funding | Funding Rate | 15 | 46.7% | +$7 | ✅ Лучший |
| #4 Breakout | ATR + пробой | 3 | 66.7% | +$13 | 🔄 Мало данных |

**Точка безубыточности:** ~20% при RR 1:4, ~33% при RR 1:2

---

## Дорожная карта

- [x] 4 бота под разные режимы рынка
- [x] ML фильтр для бота #1
- [x] Market Regime детектор + shared_state.json
- [x] regime_history.jsonl — история режимов для ML
- [x] Адаптивный Фибо Bot1 (5% тренд / 23.6% боковик)
- [x] BTC тренд фильтр шортов Bot2 MeanRev
- [x] min_rr фильтр Bot2 (отсекаем RR < 1.5)
- [x] Исправлен баг лимитных ордеров (Bot1, Bot3)
- [x] stop/take/rr в журналах всех ботов
- [x] Корреляция режима с результатами
- [ ] Фикс дедупликации Bot3 Funding (одна монета входит несколько раз)
- [ ] Накопить 100+ сделок по Bot2/3/4
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
- Запускать `market_regime.py` первым — боты читают `shared_state.json`
- `*_journal.json`, `ml_dataset.csv`, `regime_history.jsonl` — не в GitHub
- `divergence_module.py` — только в папке Bot1_EMA
