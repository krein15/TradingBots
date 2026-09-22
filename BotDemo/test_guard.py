"""
BotDemo/test_guard.py
=====================
Тесты предохранителя демо-бота.

Зачем. Ключ Bitget один на оба счёта: демо и реальный. Значит, «мы
работаем с демо-ключом» ничего не гарантирует — гарантировать может
только инструмент. В схеме susdt заявка возможна лишь по контракту с
расчётом в SUSDT: эта валюта не существует вне демо и не выводится.

Здесь проверяется, что боевой контракт в place_order не проходит, и
что до create_order дело не доходит вовсе. Сеть не нужна: рынки и
заявка подменяются.

Запуск:
  python BotDemo/test_guard.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import demo_trading as demo  # noqa: E402

SBTC = "SBTC/SUSDT:SUSDT"
BTC = "BTC/USDT:USDT"
MARKETS = {
    SBTC: {"symbol": SBTC, "settle": "SUSDT", "swap": True, "active": True, "base": "SBTC"},
    BTC: {"symbol": BTC, "settle": "USDT", "swap": True, "active": True, "base": "BTC"},
}


class FakeClient:
    """Клиент, который громко жалуется, если заявка всё-таки ушла."""

    def __init__(self, scheme, sandbox=False):
        self.options = {"demoScheme": scheme}
        if sandbox:
            self.options["sandboxMode"] = True
        self.sent = []

    def market(self, symbol):
        return MARKETS.get(symbol)

    def create_order(self, symbol, *a, **k):
        self.sent.append(symbol)
        return {"id": "1", "average": 100.0}


class FakeDemoBitget(FakeClient, demo.DemoBitget):
    """Для схемы paptrading важен сам класс клиента."""

    def __init__(self, sandbox=True):
        FakeClient.__init__(self, "paptrading", sandbox)


def expect_refusal(fn, must_contain):
    try:
        fn()
    except demo.NotDemoError as e:
        assert must_contain in str(e), f"отказ не по той причине: {e}"
        return str(e)
    raise AssertionError("заявка НЕ была отклонена")


def test_real_contract_refused():
    """Боевой контракт не проходит, и заявка не уходит."""
    ex = FakeClient("susdt")
    msg = expect_refusal(
        lambda: demo.place_order(ex, BTC, "buy", 1.0, {}, {BTC: BTC}),
        "боевой контракт")
    assert ex.sent == [], "заявка по боевому контракту УШЛА на биржу"
    print(f"  ✔  боевой контракт отклонён: {msg[:70]}…")


def test_demo_contract_passes():
    """Демо-контракт проходит — предохранитель не глухой."""
    ex = FakeClient("susdt")
    demo.place_order(ex, SBTC, "buy", 1.0, {}, {SBTC: BTC})
    assert ex.sent == [SBTC], "заявка по демо-контракту не ушла"
    print("  ✔  демо-контракт (расчёт в SUSDT) проходит")


def test_symbol_not_allowed():
    """Инструмент вне списка не проходит, даже если он демо."""
    ex = FakeClient("susdt")
    expect_refusal(lambda: demo.place_order(ex, SBTC, "buy", 1.0, {}, {}),
                   "не в списке")
    assert ex.sent == []
    print("  ✔  инструмент вне разрешённого списка отклонён")


def test_client_without_scheme():
    """Клиент в обход demo_client не может отправить заявку."""
    ex = FakeClient("susdt")
    ex.options = {}
    expect_refusal(lambda: demo.place_order(ex, SBTC, "buy", 1.0, {}, {SBTC: BTC}),
                   "в обход demo_client")
    assert ex.sent == []
    print("  ✔  клиент без схемы отклонён")


def test_paptrading_needs_sandbox():
    """В схеме paptrading заявка без режима песочницы запрещена."""
    ex = FakeDemoBitget(sandbox=False)
    expect_refusal(lambda: demo.place_order(ex, BTC, "buy", 1.0, {}, {BTC: BTC}),
                   "без режима демо")
    assert ex.sent == []
    ok = FakeDemoBitget(sandbox=True)
    demo.place_order(ok, BTC, "buy", 1.0, {}, {BTC: BTC})
    assert ok.sent == [BTC]
    print("  ✔  paptrading: без режима песочницы отказ, с ним проходит")


def test_scheme_table():
    """Схемы описаны непротиворечиво."""
    assert demo.SCHEMES["susdt"]["currency"] == "SUSDT"
    assert demo.SCHEMES["susdt"]["header"] is False
    assert demo.SCHEMES["paptrading"]["header"] is True
    print("  ✔  таблица схем на месте")


if __name__ == "__main__":
    print("=" * 62)
    print("  ТЕСТЫ ПРЕДОХРАНИТЕЛЯ ДЕМО-БОТА")
    print("=" * 62)
    test_real_contract_refused()
    test_demo_contract_passes()
    test_symbol_not_allowed()
    test_client_without_scheme()
    test_paptrading_needs_sandbox()
    test_scheme_table()
    print("=" * 62)
