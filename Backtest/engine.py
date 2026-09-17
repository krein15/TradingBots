"""
Backtest/engine.py
==================
Движок бэктеста с честной моделью исполнения.

Чем отличается от бумажной торговли, которая крутилась раньше:

1. Внутрисвечные касания.
   Бумажный бот опрашивал цену раз в 10 минут через fetch_ticker и
   сравнивал со стопом. На 5-минутном таймфрейме со стопом 0.6% между
   опросами помещалось несколько касаний — часть стопов и тейков
   просто не замечалась. Здесь проверяются high/low каждой свечи.

2. Лимитная заявка исполняется только при достижении цены.
   Бумажный бот считал заявку исполненной, если цена оказалась
   РЯДОМ (±0.3%). Настоящая лимитка на покупку исполняется, только
   если цена дошла до неё. Модель переключается параметром
   fill_model — можно посчитать оба варианта и увидеть, сколько
   результата давала именно эта поблажка.

3. Проскальзывание. Стоп — рыночная заявка, исполняется хуже уровня.

4. Ограничение плеча.
   При риске 5% и стопе 0.6% размер позиции выходит 8x от депозита.
   Для спота это невозможно: нельзя купить на $400, имея $50.
   Бумажный бот этого не проверял, здесь объём режется под депозит.

Когда в одной свече задеты и стоп, и тейк, порядок движения цены
внутри свечи неизвестен — засчитываем стоп. Пессимистично и не
завышает результат.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ── Модели исполнения входа ───────────────────────────────────
FILL_LIMIT = "limit"     # честная лимитка: цена должна дойти до уровня
FILL_TOUCH = "touch"     # как в бумажном боте: цена подошла на +-tolerance
FILL_MARKET = "market"   # рыночный вход по открытию следующей свечи


@dataclass
class Order:
    symbol: str
    direction: int          # 1 = лонг, -1 = шорт
    entry: float
    stop: float
    take: float
    created_ts: int
    signal_type: str = "?"
    regime: str = "?"
    regime_conf: int = 0
    meta: dict = field(default_factory=dict)
    bars_waited: int = 0


@dataclass
class Position:
    order: Order
    qty: float
    entry_price: float
    entry_ts: int
    notional: float
    capped: bool = False
    bars_held: int = 0
    best_price: float = 0.0   # экстремум в нашу сторону, для трейлинга
    # Стоп на момент входа. Трейлинг двигает order.stop, а риск сделки —
    # это расстояние до ИСХОДНОГО стопа. Без этого поля R считался от
    # подтянутого стопа, и при стопе у цены входа раздувался до +7R.
    initial_stop: float = 0.0
    exit_kind: str = "stop"   # станет "trail", когда стоп подтянут


@dataclass
class Trade:
    symbol: str
    direction: int
    signal_type: str
    regime: str
    regime_conf: int
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    stop: float
    take: float
    qty: float
    notional: float
    result: str
    pnl: float
    fees: float
    balance_after: float
    bars_held: int
    capped: bool
    exit_reason: str = "stop"
    initial_stop: float = 0.0   # для расчёта R; stop — финальный уровень


class Engine:
    """
    Прогон стратегии по подготовленным данным.

    symbols_data: {symbol: DataFrame с индикаторами и колонками
                   timestamp/open/high/low/close/volume}
    signal_fn:    (df_slice, cfg, btc_trend, btc_chg, regime) -> [сигналы]
    """

    def __init__(self, cfg, signal_fn,
                 deposit=50.0, risk_pct=0.05, max_open=5,
                 commission=0.001, slippage=0.0005,
                 fill_model=FILL_LIMIT, touch_tolerance=0.003,
                 max_wait_bars=10, cooldown_minutes=120,
                 bad_hours=(), max_leverage=1.0,
                 signal_max_age_bars=3,
                 trail_atr=None, max_hold_bars=None):
        self.cfg = cfg
        self.signal_fn = signal_fn
        self.deposit = deposit
        self.risk_pct = risk_pct
        self.max_open = max_open
        self.commission = commission
        self.slippage = slippage
        self.fill_model = fill_model
        self.touch_tolerance = touch_tolerance
        self.max_wait_bars = max_wait_bars
        self.cooldown_ms = cooldown_minutes * 60_000
        self.bad_hours = set(bad_hours)
        self.max_leverage = max_leverage
        self.signal_max_age_bars = signal_max_age_bars
        # Трейлинг-стоп в единицах ATR: без него трендовую
        # стратегию не проверить — она живёт тем, что даёт
        # прибыли тянуться, а не упирается в фиксированный тейк.
        self.trail_atr = trail_atr
        # Принудительный выход по времени: позиция не должна
        # занимать лимит неделями, как было в первом прогоне
        # (максимум 2780 свечей в одной сделке).
        self.max_hold_bars = max_hold_bars

        self.balance = deposit
        self.trades = []
        self.equity = []          # (timestamp, баланс)
        self.pending = []
        self.positions = []
        self.cooldown = {}        # symbol -> до какого времени заблокирована
        self.counters = {
            "signals": 0, "orders": 0, "filled": 0, "expired": 0,
            "skip_cooldown": 0, "skip_bad_hour": 0, "skip_max_open": 0,
            "skip_duplicate": 0, "skip_no_cash": 0, "capped": 0,
            "broke_bars": 0,
        }
        self.broke_at = None

    # ── Вспомогательное ───────────────────────────────────────
    def _fee(self, notional):
        return notional * self.commission

    def _is_broke(self):
        return self.balance < self.deposit * self.risk_pct

    def _size(self, entry, stop):
        """Объём по риску, урезанный лимитом плеча -> (qty, notional, capped)."""
        risk_per_unit = abs(entry - stop)
        if risk_per_unit <= 0 or entry <= 0:
            return 0.0, 0.0, False
        qty = (self.balance * self.risk_pct) / risk_per_unit
        notional = qty * entry
        cap = self.balance * self.max_leverage
        if notional > cap:
            return cap / entry, cap, True
        return qty, notional, False

    # ── Выход ─────────────────────────────────────────────────
    def _exit_price(self, direction, level, is_stop):
        """Стоп исполняется рыночно и хуже уровня; тейк — лимиткой."""
        if not is_stop:
            return level
        slip = level * self.slippage
        return level - slip if direction == 1 else level + slip

    def _close(self, pos, level, reason, ts):
        """
        reason: stop | take | trail | time.
        Тейк — лимитная заявка, исполняется по уровню. Всё остальное
        рыночное и проскальзывает против нас.
        Результат считаем по знаку PnL, а не по причине выхода: при
        трейлинге и выходе по времени сделка может закрыться в плюс,
        и записывать её как LOSS было бы неверно.
        """
        d = pos.order.direction
        exit_price = level if reason == "take" else self._exit_price(d, level, True)
        gross = (exit_price - pos.entry_price) * pos.qty * d
        fees = self._fee(pos.notional) + self._fee(exit_price * pos.qty)
        pnl = gross - fees
        self.balance += pnl

        self.trades.append(Trade(
            symbol=pos.order.symbol, direction=d,
            signal_type=pos.order.signal_type,
            regime=pos.order.regime, regime_conf=pos.order.regime_conf,
            entry_ts=pos.entry_ts, exit_ts=ts,
            entry_price=pos.entry_price, exit_price=exit_price,
            stop=pos.order.stop, take=pos.order.take,
            initial_stop=pos.initial_stop or pos.order.stop,
            qty=pos.qty, notional=pos.notional,
            result="WIN" if pnl > 0 else "LOSS",
            pnl=pnl, fees=fees, balance_after=self.balance,
            bars_held=pos.bars_held, capped=pos.capped,
            exit_reason=reason,
        ))
        if pnl <= 0:
            self.cooldown[pos.order.symbol] = ts + self.cooldown_ms

    def _check_exit(self, pos, high, low, ts):
        """
        Задет ли стоп или тейк этой свечой.
        Оба задеты -> считаем стопом: порядок движения цены внутри
        свечи неизвестен, оптимистичное допущение завысило бы WR.
        """
        d = pos.order.direction
        if d == 1:
            hit_stop, hit_take = low <= pos.order.stop, high >= pos.order.take
        else:
            hit_stop, hit_take = high >= pos.order.stop, low <= pos.order.take

        if hit_stop:
            self._close(pos, pos.order.stop, pos.exit_kind, ts)
            return True
        if hit_take:
            self._close(pos, pos.order.take, "take", ts)
            return True
        return False

    def _update_trail(self, pos, high, low, atr):
        """
        Подтягиваем стоп за ценой на trail_atr * ATR от достигнутого
        экстремума. Стоп двигается только в нашу сторону — назад он
        не откатывается никогда.
        """
        if not self.trail_atr or not atr or atr <= 0:
            return
        d = pos.order.direction
        if d == 1:
            pos.best_price = max(pos.best_price, high)
            new_stop = pos.best_price - self.trail_atr * atr
            if new_stop > pos.order.stop:
                pos.order.stop = new_stop
                pos.exit_kind = "trail"
        else:
            pos.best_price = min(pos.best_price, low)
            new_stop = pos.best_price + self.trail_atr * atr
            if new_stop < pos.order.stop:
                pos.order.stop = new_stop
                pos.exit_kind = "trail"

    # ── Вход ──────────────────────────────────────────────────
    def _try_fill(self, order, o, h, l):
        """Цена исполнения на этой свече или None."""
        entry = order.entry
        if self.fill_model == FILL_MARKET:
            return o * (1 + self.slippage * order.direction)
        if self.fill_model == FILL_TOUCH:
            lo, hi = entry * (1 - self.touch_tolerance), entry * (1 + self.touch_tolerance)
            return entry if (l <= hi and h >= lo) else None
        # FILL_LIMIT: цена должна реально дойти до уровня
        if order.direction == 1:
            return entry if l <= entry else None
        return entry if h >= entry else None

    # ── Главный цикл ──────────────────────────────────────────
    def run(self, symbols_data, regime_lookup=None, btc_lookup=None,
            progress=None, trade_from=None):
        """
        symbols_data: {symbol: DataFrame}. Все символы проходятся
        синхронно по общей временной шкале — депозит и лимит
        одновременных позиций у них общие, как и в бою.

        trade_from — метка времени (мс), раньше которой новые позиции не
        открываются. Нужна, чтобы индикаторы прогревались на истории ДО
        оцениваемого периода. Иначе, если нарезать данные ровно по началу
        периода, первый месяц EMA200 считается по неполному окну, и
        сигналы в нём не те, что увидел бы бот в бою.
        """
        frames = {}
        for sym, df in symbols_data.items():
            if df is None or len(df) < 60:
                continue
            frames[sym] = {
                "df": df,
                "ts": df["timestamp"].to_numpy(dtype="int64"),
                "o": df["open"].to_numpy(dtype="float64"),
                "h": df["high"].to_numpy(dtype="float64"),
                "l": df["low"].to_numpy(dtype="float64"),
                "c": df["close"].to_numpy(dtype="float64"),
                "atr": (df["atr"].to_numpy(dtype="float64")
                        if "atr" in df.columns else None),
            }
        if not frames:
            return self

        timeline = np.unique(np.concatenate([f["ts"] for f in frames.values()]))
        # Указатель на текущую свечу каждого символа
        ptr = {sym: 0 for sym in frames}
        warmup = 60   # свечей на прогрев индикаторов

        for step, ts in enumerate(timeline):
            if progress and step % progress == 0:
                done = step / len(timeline) * 100
                print(f"    {done:5.1f}%  {pd.to_datetime(ts, unit='ms', utc=True)}  "
                      f"баланс=${self.balance:.2f}  сделок={len(self.trades)}",
                      flush=True)

            bars = {}
            for sym, f in frames.items():
                i = ptr[sym]
                if i < len(f["ts"]) and f["ts"][i] == ts:
                    bars[sym] = i
                    ptr[sym] = i + 1
            if not bars:
                continue

            # 1. Открытые позиции — внутрисвечные касания
            still_open = []
            for pos in self.positions:
                i = bars.get(pos.order.symbol)
                if i is None:
                    still_open.append(pos)
                    continue
                f = frames[pos.order.symbol]
                pos.bars_held += 1

                # Трейлинг подтягиваем ДО проверки касаний: стоп,
                # выставленный по этой же свече, не может быть ею
                # же и исполнен — иначе получилось бы подглядывание.
                closed = self._check_exit(pos, f["h"][i], f["l"][i], int(ts))
                if closed:
                    continue
                if self.trail_atr and f["atr"] is not None:
                    self._update_trail(pos, f["h"][i], f["l"][i], f["atr"][i])
                if self.max_hold_bars and pos.bars_held >= self.max_hold_bars:
                    self._close(pos, f["c"][i], "time", int(ts))
                    continue
                still_open.append(pos)
            self.positions = still_open

            # 2. Заявки — исполнение или истечение срока
            still_pending = []
            for order in self.pending:
                i = bars.get(order.symbol)
                if i is None:
                    still_pending.append(order)
                    continue
                f = frames[order.symbol]
                order.bars_waited += 1

                price = self._try_fill(order, f["o"][i], f["h"][i], f["l"][i])
                if price is None:
                    if order.bars_waited >= self.max_wait_bars:
                        self.counters["expired"] += 1
                    else:
                        still_pending.append(order)
                    continue

                # Место занято — заявка не пропадает, а ждёт дальше,
                # пока не истечёт её срок. Выбрасывать её здесь
                # значило бы терять сигналы тем чаще, чем лучше идёт
                # торговля, и незаметно искажать выборку.
                if len(self.positions) >= self.max_open:
                    self.counters["skip_max_open"] += 1
                    if order.bars_waited < self.max_wait_bars:
                        still_pending.append(order)
                    else:
                        self.counters["expired"] += 1
                    continue

                qty, notional, capped = self._size(price, order.stop)
                if qty <= 0:
                    self.counters["skip_no_cash"] += 1
                    continue
                if capped:
                    self.counters["capped"] += 1

                pos = Position(order=order, qty=qty, entry_price=price,
                               entry_ts=int(ts), notional=notional,
                               capped=capped, best_price=price,
                               initial_stop=order.stop)
                self.counters["filled"] += 1

                # Позиция могла быть выбита той же свечой, на которой
                # вошли: цена дошла до лимита и пошла дальше к стопу.
                # Игнорировать это — значит завышать результат.
                pos.bars_held = 1
                if not self._check_exit(pos, f["h"][i], f["l"][i], int(ts)):
                    self.positions.append(pos)
            self.pending = still_pending

            # 3. Новые сигналы
            if trade_from is not None and ts < trade_from:
                continue                 # прогрев: считаем, но не торгуем
            self.equity.append((int(ts), self.balance))
            if self._is_broke():
                self.counters["broke_bars"] += 1
                if self.broke_at is None:
                    self.broke_at = int(ts)
                continue

            hour = pd.Timestamp(ts, unit="ms", tz="UTC").hour
            if hour in self.bad_hours:
                self.counters["skip_bad_hour"] += 1
                continue
            if len(self.positions) + len(self.pending) >= self.max_open:
                self.counters["skip_max_open"] += 1
                continue

            regime, regime_conf = ("?", 0)
            if regime_lookup is not None:
                got = regime_lookup.at(int(ts))
                if got:
                    regime, regime_conf = got[0], int(got[1])

            btc_trend, btc_chg = ("neutral", 0.0)
            if btc_lookup is not None:
                got = btc_lookup.at(int(ts))
                if got:
                    btc_trend, btc_chg = got[0], float(got[1])

            busy = {(p.order.symbol, p.order.direction) for p in self.positions}
            busy |= {(o.symbol, o.direction) for o in self.pending}

            for sym, i in bars.items():
                if len(self.positions) + len(self.pending) >= self.max_open:
                    break
                if i < warmup:
                    continue
                if self.cooldown.get(sym, 0) > ts:
                    self.counters["skip_cooldown"] += 1
                    continue

                f = frames[sym]
                sigs = self.signal_fn(f["df"], i, self.cfg,
                                      btc_trend, btc_chg, regime)
                if not sigs:
                    continue
                self.counters["signals"] += len(sigs)

                sig = sigs[-1]
                if i - sig["bar"] > self.signal_max_age_bars:
                    continue
                key = (sym, sig["dir"])
                if key in busy:
                    self.counters["skip_duplicate"] += 1
                    continue

                self.pending.append(Order(
                    symbol=sym, direction=sig["dir"],
                    entry=sig["entry_limit"], stop=sig["stop"],
                    take=sig["take"], created_ts=int(ts),
                    signal_type=sig.get("type", "?"),
                    regime=regime, regime_conf=regime_conf,
                    meta={k: sig.get(k) for k in
                          ("vol_ratio", "rsi", "body_ratio", "atr_pct")},
                ))
                busy.add(key)
                self.counters["orders"] += 1

        return self

    # ── Результаты ────────────────────────────────────────────
    def trades_df(self):
        if not self.trades:
            return pd.DataFrame()
        df = pd.DataFrame([t.__dict__ for t in self.trades])
        df["entry_dt"] = pd.to_datetime(df["entry_ts"], unit="ms", utc=True)
        df["exit_dt"] = pd.to_datetime(df["exit_ts"], unit="ms", utc=True)
        return df

    def equity_df(self):
        if not self.equity:
            return pd.DataFrame()
        df = pd.DataFrame(self.equity, columns=["timestamp", "balance"])
        df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df
