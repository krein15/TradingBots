"""
Panel/server.py
===============
Панель управления ботами — вместо чёрного окна cmd.

Запуск двойным кликом по «Запустить панель.vbs» в корне проекта.
Консоль не появляется: сервер работает через pythonw, а интерфейс
открывается окном-приложением Chrome (или Edge) без адресной строки.

── Только стандартная библиотека ──────────────────────────────
Первая версия импортировала код ботов ради их настроек и ccxt ради
цен — и занимала 213–309 МБ, больше любого бота, хотя только читает
файлы. pandas и ccxt вместе весят ~110 МБ, плюс ccxt держит в памяти
описания всех 790 рынков. Теперь:
  настройки бот сам пишет в журнал при старте («meta»), панель их читает;
  цены — один HTTP-запрос к публичному API Bitget раз в 20 секунд;
  в итоге сервер весит ~25 МБ.

── Как устроено ───────────────────────────────────────────────
  сервер слушает ТОЛЬКО 127.0.0.1;
  боты — отдельные процессы: закрытие панели их не останавливает;
  «работает ли бот» — по блокировке журнала, которую держит сам бот;
  остановка — файл-флаг, бот выходит между циклами; не вышел за
    20 секунд — принудительно;
  включённые в панели боты, которые упали, поднимаются через 30 секунд.

Команды принимаются только с секретным токеном, который есть лишь у
страницы самой панели, и с правильным Host (защита от DNS rebinding).
Ключи демо-счёта сервер не читает и не отдаёт — только проверяет, что
в .env заданы все три имени.
"""

import csv
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STATIC = HERE / "static"
STATE_FILE = HERE / "state.json"
LOG_FILE = HERE / "panel.log"

HOST, PORT = "127.0.0.1", 8765
TOKEN = secrets.token_urlsafe(24)
# Меняется при каждом старте сервера. Окно, открытое до перезапуска,
# видит смену и перезагружается само — иначе у него остаётся токен
# прежнего сервера, и кнопки молча не работают.
INSTANCE = secrets.token_hex(6)

# Под pythonw у процесса нет консоли: sys.stdout и sys.stderr равны None
if sys.stdout is None:
    sys.stdout = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
if sys.stderr is None:
    sys.stderr = sys.stdout


def plog(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")


PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

# Реестр ботов. «fallback» — то, что показать, пока бот ни разу не
# запускался и не записал свой паспорт в журнал. Как только запустится,
# источником истины становится его meta.
BOTS = {
    "bot5": {
        "kind": "paper", "short": "Дончиан",
        "script": ROOT / "Bot5_Donchian" / "paper_trading_donchian.py",
        "journal": ROOT / "Bot5_Donchian" / "donchian_journal.json",
        "log": ROOT / "Bot5_Donchian" / "donchian_log.txt",
        "fallback": {"bot_name": "Бот #5 — Дончиан", "risk_pct": 0.05, "max_open": 5, "deposit": 50.0,
                     "rules": "Дончиан 20 свечей, стоп 2.5 ATR, тейк 3.0R, фильтр EMA200, ТФ 4h"},
        # Диапазон, а не точка: нижняя граница — медиана по случайным
        # наборам монет на проверке, верхняя — среднее за весь период
        "expect": {"wr": 35, "r_lo": 0.10, "r_hi": 0.20},
    },
    "bot6": {
        "kind": "paper", "short": "Supertrend",
        "script": ROOT / "Bot6_Supertrend" / "paper_trading_supertrend.py",
        "journal": ROOT / "Bot6_Supertrend" / "supertrend_journal.json",
        "log": ROOT / "Bot6_Supertrend" / "supertrend_log.txt",
        "fallback": {"bot_name": "Бот #6 — Supertrend", "risk_pct": 0.05, "max_open": 5, "deposit": 50.0,
                     "rules": "Supertrend ×3.0, стоп 2.5 ATR, тейк 3.0R, фильтр EMA200, ТФ 4h"},
        # Нижняя — среднее за весь период, верхняя — медиана на проверке,
        # где Supertrend повезло с медвежьим рынком
        "expect": {"wr": 33, "r_lo": 0.11, "r_hi": 0.28},
    },
    "demo": {
        "kind": "demo", "short": "Демо",
        "script": ROOT / "BotDemo" / "demo_trading.py",
        "journal": ROOT / "BotDemo" / "demo_journal.json",
        "log": ROOT / "BotDemo" / "demo_log.txt",
        "fallback": {"bot_name": "Демо Bitget — обе стратегии", "risk_pct": 0.05, "max_open": 5},
    },
}

ENV_FILE = ROOT / ".env"
DEMO_KEY_NAMES = ("BITGET_DEMO_API_KEY", "BITGET_DEMO_API_SECRET", "BITGET_DEMO_API_PASSPHRASE")


def demo_keys_configured():
    """Заданы ли все три ключа. Сами значения не читаются наружу."""
    if not ENV_FILE.exists():
        return False
    found = set()
    try:
        for line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() in DEMO_KEY_NAMES and v.strip().strip('"').strip("'"):
                    found.add(k.strip())
    except Exception:
        return False
    return found == set(DEMO_KEY_NAMES)


# ─────────────────────────────────────────────────────────────
#  Процессы
# ─────────────────────────────────────────────────────────────
def lock_path(spec):
    return str(spec["journal"]) + ".lock"


def pid_path(spec):
    return str(spec["journal"]) + ".pid"


def stop_path(spec):
    return str(spec["journal"]) + ".stop"


def load_desired():
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_desired(d):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, STATE_FILE)


def is_running(spec):
    """Пробуем взять блокировку бота: взяли — значит никто её не держит."""
    path = lock_path(spec)
    if not os.path.exists(path):
        return False
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return False
    try:
        if os.name == "nt":
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError:
                return True
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            return False
        import fcntl
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def read_pid(spec):
    try:
        return int(Path(pid_path(spec)).read_text(encoding="utf-8").strip())
    except Exception:
        return None


_lock = threading.Lock()
_restart_after = {}
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def start_bot(bot_id):
    spec = BOTS[bot_id]
    with _lock:
        d = load_desired()
        d[bot_id] = True
        save_desired(d)
        if is_running(spec):
            return "уже работает"
        err = open(Path(spec["journal"]).with_suffix(".stderr.txt"), "a", encoding="utf-8")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        flags = NO_WINDOW | (0x00000200 if os.name == "nt" else 0)   # + CREATE_NEW_PROCESS_GROUP
        subprocess.Popen([str(PYTHON), "-u", str(spec["script"])],
                         cwd=str(spec["script"].parent), env=env,
                         stdout=subprocess.DEVNULL, stderr=err,
                         stdin=subprocess.DEVNULL, creationflags=flags)
        plog(f"запуск {bot_id}")
        return "запущен"


def stop_bot(bot_id):
    spec = BOTS[bot_id]
    with _lock:
        d = load_desired()
        d[bot_id] = False
        save_desired(d)
    if not is_running(spec):
        return "не работал"
    Path(stop_path(spec)).write_text("stop", encoding="utf-8")
    for _ in range(20):
        time.sleep(1)
        if not is_running(spec):
            plog(f"остановлен {bot_id} (штатно)")
            return "остановлен"
    pid = read_pid(spec)
    if pid:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               capture_output=True, creationflags=NO_WINDOW)
            else:
                os.kill(pid, 9)
        except Exception as e:
            plog(f"не удалось завершить {bot_id}: {e}")
    try:
        os.remove(stop_path(spec))
    except OSError:
        pass
    plog(f"остановлен {bot_id} (принудительно)")
    return "остановлен принудительно"


def demo_check():
    """Проверка демо-ключа отдельным процессом: ключи трогает только он."""
    spec = BOTS["demo"]
    try:
        r = subprocess.run([str(PYTHON), "-u", str(spec["script"]), "check"],
                           cwd=str(spec["script"].parent), capture_output=True,
                           timeout=120, creationflags=NO_WINDOW,
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        text = (r.stdout or b"").decode("utf-8", "replace") + (r.stderr or b"").decode("utf-8", "replace")
        return {"ok": r.returncode == 0, "code": r.returncode, "output": text[-4000:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "code": -1, "output": "Проверка не уложилась в 2 минуты — нет связи с Bitget?"}


def watchdog():
    while True:
        try:
            d = load_desired()
            for bot_id, spec in BOTS.items():
                if not d.get(bot_id) or is_running(spec):
                    _restart_after.pop(bot_id, None)
                    continue
                # Демо без ключей поднимать бессмысленно — только шум в логе
                if spec["kind"] == "demo" and not demo_keys_configured():
                    continue
                due = _restart_after.get(bot_id)
                if due is None:
                    _restart_after[bot_id] = time.time() + 30
                elif time.time() >= due:
                    plog(f"{bot_id} не работает, хотя включён — перезапуск")
                    start_bot(bot_id)
                    _restart_after.pop(bot_id, None)
        except Exception as e:
            plog(f"watchdog: {e}")
        time.sleep(5)


# ─────────────────────────────────────────────────────────────
#  Цены — один запрос к публичному API, без ccxt
# ─────────────────────────────────────────────────────────────
_prices = {"ts": 0, "data": {}}


def price_loop():
    url = "https://api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES"
    while True:
        try:
            wanted = set()
            for spec in BOTS.values():
                j = read_journal(spec) or {}
                wanted |= {p["symbol"] for p in j.get("open", [])}
            if wanted:
                req = urllib.request.Request(url, headers={"User-Agent": "tradingbots-panel"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    rows = json.loads(r.read()).get("data") or []
                by_id = {row.get("symbol"): row for row in rows}
                data = {}
                for sym in wanted:
                    base, rest = sym.split("/", 1)
                    row = by_id.get(base + rest.split(":")[0])     # BTC/USDT:USDT -> BTCUSDT
                    if row and row.get("lastPr"):
                        data[sym] = float(row["lastPr"])
                _prices["data"], _prices["ts"] = data, time.time()
        except Exception as e:
            plog(f"цены: {type(e).__name__}: {str(e)[:120]}")
        time.sleep(20)


# ─────────────────────────────────────────────────────────────
#  Данные для интерфейса
# ─────────────────────────────────────────────────────────────
_journal_cache = {}


def read_journal(spec):
    """Журнал может подменяться прямо сейчас — тогда отдаём прошлую версию."""
    path = str(spec["journal"])
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    cached = _journal_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]                      # не перечитываем без изменений
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        _journal_cache[path] = (mtime, j)
        return j
    except Exception:
        return cached[1] if cached else None


def tail(path, n):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 60_000))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
        return lines[-n:]
    except Exception:
        return []


def _ts(iso):
    try:
        return int(datetime.fromisoformat(iso).timestamp() * 1000)
    except Exception:
        return None


def last_cycle_time(spec):
    for line in reversed(tail(spec["log"], 40)):
        if line.startswith("[") and ("Баланс=" in line or "Капитал=" in line):
            return line[1:20]
    return None


def paper_state(bot_id, spec):
    j = read_journal(spec) or {}
    meta = {**spec["fallback"], **(j.get("meta") or {})}
    deposit = float(j.get("deposit", meta.get("deposit", 50.0)))
    balance = float(j.get("balance", deposit))
    trades = j.get("trades", [])
    wins = sum(1 for t in trades if t.get("result") == "WIN")
    rs = [t.get("r_multiple", 0) for t in trades]

    equity = [[_ts(j.get("created", "")) or int(time.time() * 1000), deposit]]
    for t in trades:
        ts = _ts(t.get("closed", ""))
        if ts:
            equity.append([ts, float(t.get("balance", balance))])
    peak, max_dd = deposit, 0.0
    for _, b in equity:
        peak = max(peak, b)
        max_dd = min(max_dd, (b - peak) / peak * 100 if peak else 0)

    positions, unreal = [], 0.0
    for p in j.get("open", []):
        price = _prices["data"].get(p["symbol"])
        risk = abs(p["entry"] - p["stop"]) * p["qty"]
        pnl = (price - p["entry"]) * p["qty"] * p["dir"] if price else None
        unreal += pnl or 0
        positions.append({
            "symbol": p["symbol"].split("/")[0], "dir": p["dir"],
            "entry": p["entry"], "stop": p["stop"], "take": p["take"],
            "price": price, "pnl": pnl,
            "r": (pnl / risk) if (pnl is not None and risk > 0) else None,
            "notional": p.get("notional"), "opened": p.get("opened"),
        })

    return {
        "id": bot_id, "kind": "paper", "short": spec["short"],
        "name": meta["bot_name"], "rules": meta["rules"],
        "risk_pct": meta["risk_pct"], "max_open": meta["max_open"],
        "running": is_running(spec), "enabled": bool(load_desired().get(bot_id)),
        "pid": read_pid(spec), "cycles": j.get("cycles", 0),
        "last_cycle": last_cycle_time(spec),
        "deposit": deposit, "balance": balance, "unrealized": unreal,
        "trades_count": len(trades), "wins": wins,
        "wr": (wins / len(trades) * 100) if trades else None,
        "avg_r": (sum(rs) / len(rs)) if rs else None,
        "max_dd": max_dd, "equity": equity, "positions": positions,
        "trades": [{
            "symbol": t["symbol"].split("/")[0], "dir": t["dir"],
            "entry": t.get("entry"), "exit": t.get("exit"),
            "reason": t.get("exit_reason"), "r": t.get("r_multiple"),
            "pnl": t.get("pnl"), "closed": t.get("closed"), "balance": t.get("balance"),
        } for t in trades[-100:]][::-1],
        # Источник истины — паспорт бота в журнале; список ниже нужен
        # только пока бот ни разу не запускался
        "expect": (meta.get("expect") or spec["expect"]),
    }


def demo_state(spec):
    j = read_journal(spec) or {}
    meta = {**spec["fallback"], **(j.get("meta") or {})}
    trades = j.get("trades", [])
    per = {}
    for strat in ("donchian", "supertrend"):
        ts = [t for t in trades if t.get("strategy") == strat]
        rs = [t["r_multiple"] for t in ts if t.get("r_multiple") is not None]
        per[strat] = {
            "trades": len(ts), "wins": sum(1 for t in ts if t.get("result") == "WIN"),
            "avg_r": (sum(rs) / len(rs)) if rs else None,
            "pnl": sum(t.get("pnl") or 0 for t in ts),
            "funding": sum(t.get("funding") or 0 for t in ts),
            "open": sum(1 for p in j.get("open", []) if p.get("strategy") == strat),
        }
    return {
        "id": "demo", "kind": "demo", "short": spec["short"], "name": meta["bot_name"],
        "rules": meta.get("rules"), "risk_pct": meta["risk_pct"], "max_open": meta["max_open"],
        "leverage": meta.get("leverage"), "symbols": meta.get("symbols"),
        "running": is_running(spec), "enabled": bool(load_desired().get("demo")),
        "pid": read_pid(spec), "cycles": j.get("cycles", 0),
        "last_cycle": last_cycle_time(spec),
        "keys": demo_keys_configured(), "last_error": j.get("last_error"),
        "equity": j.get("equity"), "start_equity": j.get("start_equity"),
        "available": j.get("available"), "foreign": j.get("foreign", []),
        "per_strategy": per,
        "positions": [{
            "strategy": p["strategy"], "symbol": p["symbol"].split("/")[0], "dir": p["dir"],
            "entry": p["entry"], "stop": p["stop"], "take": p["take"],
            "price": p.get("mark") or _prices["data"].get(p["symbol"]),
            "pnl": p.get("unrealized"),
            "r": (p["unrealized"] / p["risk_usd"]) if (p.get("unrealized") is not None and p.get("risk_usd")) else None,
            "notional": p.get("notional"), "opened": p.get("opened"),
        } for p in j.get("open", [])],
        "trades": [{
            "strategy": t.get("strategy"), "symbol": t["symbol"].split("/")[0], "dir": t["dir"],
            "entry": t.get("entry"), "exit": t.get("exit"), "reason": t.get("exit_reason"),
            "r": t.get("r_multiple"), "pnl": t.get("pnl"), "funding": t.get("funding"),
            "closed": t.get("closed"),
        } for t in trades[-100:]][::-1],
    }


def research():
    out = {}
    try:
        out["projection"] = json.loads((ROOT / "Backtest" / "projection_100.json").read_text(encoding="utf-8"))
    except Exception:
        out["projection"] = None
    try:
        with open(ROOT / "Backtest" / "research_summary_4h.csv", encoding="utf-8") as f:
            out["families"] = list(csv.DictReader(f))
    except Exception:
        out["families"] = []
    return out


def state():
    return {
        "instance": INSTANCE,
        "now": int(time.time() * 1000),
        "prices_age": (time.time() - _prices["ts"]) if _prices["ts"] else None,
        "bots": [paper_state(b, BOTS[b]) for b in ("bot5", "bot6")],
        "demo": demo_state(BOTS["demo"]),
    }


# ─────────────────────────────────────────────────────────────
#  HTTP
# ─────────────────────────────────────────────────────────────
MIME = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".js": "application/javascript; charset=utf-8", ".svg": "image/svg+xml"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _host_ok(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost")

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str))

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, "forbidden", "text/plain")
        u = urlparse(self.path)
        if u.path == "/api/ping":
            return self._json({"ok": True, "app": "tradingbots-panel"})
        if u.path == "/api/state":
            return self._json(state())
        if u.path == "/api/research":
            return self._json(research())
        if u.path == "/api/log":
            bot = parse_qs(u.query).get("bot", ["bot5"])[0]
            if bot not in BOTS:
                return self._json({"error": "нет такого бота"}, 404)
            return self._json({"lines": tail(BOTS[bot]["log"], 400)})

        name = "index.html" if u.path in ("/", "/index.html") else u.path.lstrip("/")
        path = (STATIC / name).resolve()
        if STATIC not in path.parents or not path.is_file():
            return self._send(404, "not found", "text/plain")
        body = path.read_bytes()
        if name == "index.html":
            body = body.replace(b"__PANEL_TOKEN__", TOKEN.encode())
        self._send(200, body, MIME.get(path.suffix, "application/octet-stream"))

    def do_POST(self):
        if not self._host_ok() or self.headers.get("X-Panel-Token") != TOKEN:
            return self._json({"error": "запрещено"}, 403)
        u = urlparse(self.path)
        bot = parse_qs(u.query).get("bot", [None])[0]
        if u.path in ("/api/start", "/api/stop"):
            if bot not in BOTS:
                return self._json({"error": "нет такого бота"}, 404)
            if u.path == "/api/start" and BOTS[bot]["kind"] == "demo" and not demo_keys_configured():
                return self._json({"error": "не заданы ключи демо-счёта в .env"}, 400)
            result = start_bot(bot) if u.path == "/api/start" else stop_bot(bot)
            return self._json({"ok": True, "result": result})
        if u.path == "/api/demo/check":
            return self._json(demo_check())
        if u.path == "/api/quit":
            self._json({"ok": True})
            threading.Thread(target=lambda: (time.sleep(0.5), os._exit(0)), daemon=True).start()
            return
        self._json({"error": "нет такой команды"}, 404)


# ─────────────────────────────────────────────────────────────
#  Запуск
# ─────────────────────────────────────────────────────────────
def find_app_browser():
    env = os.environ
    for c in (
        Path(env.get("ProgramFiles", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
        Path(env.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Google/Chrome/Application/chrome.exe",
        Path(env.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
        Path(env.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe",
        Path(env.get("ProgramFiles", r"C:\Program Files")) / "Microsoft/Edge/Application/msedge.exe",
    ):
        if c.is_file():
            return c
    return None


def open_window(url):
    browser = find_app_browser()
    if browser:
        try:
            subprocess.Popen([str(browser), f"--app={url}", "--window-size=1480,940"],
                             creationflags=NO_WINDOW)
            return
        except Exception as e:
            plog(f"окно приложения: {e}")
    webbrowser.open(url)


def already_running():
    try:
        with urllib.request.urlopen(f"http://{HOST}:{PORT}/api/ping", timeout=1.5) as r:
            return json.loads(r.read()).get("app") == "tradingbots-panel"
    except Exception:
        return False


def main():
    url = f"http://{HOST}:{PORT}/"
    no_window = "--no-window" in sys.argv
    if already_running():
        if not no_window:
            open_window(url)
        return
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    threading.Thread(target=watchdog, daemon=True).start()
    threading.Thread(target=price_loop, daemon=True).start()
    plog(f"панель запущена на {url}")
    if not no_window:
        threading.Timer(0.6, open_window, args=(url,)).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
