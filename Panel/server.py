"""
Panel/server.py
===============
Панель управления ботами — вместо чёрного окна cmd.

Запуск двойным кликом по «Запустить панель.vbs» в корне проекта.
Консоль не появляется: сервер работает через pythonw, а интерфейс
открывается окном-приложением Chrome (или Edge) без адресной строки.

Как устроено:
  сервер слушает ТОЛЬКО 127.0.0.1 — снаружи к нему не подключиться;
  боты — отдельные процессы. Закрытие панели их не останавливает, а
    повторный запуск панели находит уже работающих ботов;
  «работает ли бот» панель узнаёт по блокировке его журнала, которую
    держит сам процесс бота (см. acquire_single_instance в ядре). Это
    надёжнее, чем помнить PID: блокировку снимает ОС, когда процесс
    умирает любым способом;
  остановка — файл-флаг: бот выходит между циклами, а не посреди
    записи журнала. Если за 20 секунд не вышел — принудительно;
  боты, включённые в панели, перезапускаются, если упали.

Защита команд. Любой сайт, открытый в браузере, технически может
отправить запрос на localhost. Поэтому команды принимаются только с
секретным токеном в заголовке, который есть лишь у страницы самой
панели, и только с правильным заголовком Host (защита от DNS rebinding).
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

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Bot5_Donchian"))
sys.path.insert(0, str(ROOT / "Bot6_Supertrend"))

# Под pythonw у процесса нет консоли: sys.stdout и sys.stderr равны None,
# и любой вывод в них падает. Всё служебное — в файл.
if sys.stdout is None:
    sys.stdout = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
if sys.stderr is None:
    sys.stderr = sys.stdout


def plog(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")


import paper_trading_donchian as core        # noqa: E402
import paper_trading_supertrend as st_bot    # noqa: E402

PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

BOTS = {
    "bot5": {
        "cfg": core.CONFIG,
        "short": "Дончиан",
        "script": ROOT / "Bot5_Donchian" / "paper_trading_donchian.py",
        # Диапазон, а не точка: нижняя граница — медиана по случайным
        # наборам монет на проверке, верхняя — среднее за весь период
        "expect": {"wr": 35, "r_lo": 0.10, "r_hi": 0.20},
    },
    "bot6": {
        "cfg": st_bot.CONFIG,
        "short": "Supertrend",
        "script": ROOT / "Bot6_Supertrend" / "paper_trading_supertrend.py",
        # Нижняя граница — среднее за весь период, верхняя — медиана на
        # проверке, где Supertrend повезло с медвежьим рынком
        "expect": {"wr": 33, "r_lo": 0.11, "r_hi": 0.28},
    },
}


# ─────────────────────────────────────────────────────────────
#  Состояние процессов
# ─────────────────────────────────────────────────────────────
def load_desired():
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_desired(d):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, STATE_FILE)


def is_running(cfg):
    """
    Работает ли бот — пробуем взять ЕГО блокировку. Взяли — значит
    никто её не держит: сразу отпускаем и говорим «не работает».
    """
    path = core.lock_path(cfg)
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


def read_pid(cfg):
    try:
        return int(Path(core.pid_path(cfg)).read_text(encoding="utf-8").strip())
    except Exception:
        return None


_lock = threading.Lock()
_restart_after = {}      # bot_id -> не раньше какого времени перезапускать


def start_bot(bot_id):
    spec = BOTS[bot_id]
    cfg = spec["cfg"]
    with _lock:
        d = load_desired()
        d[bot_id] = True
        save_desired(d)
        if is_running(cfg):
            return "уже работает"
        err = open(Path(cfg["journal"]).with_suffix(".stderr.txt"), "a", encoding="utf-8")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        flags = 0
        if os.name == "nt":
            flags = 0x08000000 | 0x00000200   # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
        subprocess.Popen([str(PYTHON), "-u", str(spec["script"])],
                         cwd=str(spec["script"].parent), env=env,
                         stdout=subprocess.DEVNULL, stderr=err,
                         stdin=subprocess.DEVNULL, creationflags=flags)
        plog(f"запуск {bot_id}")
        return "запущен"


def stop_bot(bot_id):
    cfg = BOTS[bot_id]["cfg"]
    with _lock:
        d = load_desired()
        d[bot_id] = False
        save_desired(d)
    if not is_running(cfg):
        return "не работал"
    Path(core.stop_path(cfg)).write_text("stop", encoding="utf-8")
    for _ in range(20):
        time.sleep(1)
        if not is_running(cfg):
            plog(f"остановлен {bot_id} (штатно)")
            return "остановлен"
    # Не вышел за 20 секунд — скорее всего завис на сетевом запросе
    pid = read_pid(cfg)
    if pid:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               capture_output=True, creationflags=0x08000000)
            else:
                os.kill(pid, 9)
        except Exception as e:
            plog(f"не удалось завершить {bot_id}: {e}")
    try:
        os.remove(core.stop_path(cfg))
    except OSError:
        pass
    plog(f"остановлен {bot_id} (принудительно)")
    return "остановлен принудительно"


def watchdog():
    """Включённые в панели боты, которые упали, поднимаются через 30 с."""
    while True:
        try:
            d = load_desired()
            for bot_id, spec in BOTS.items():
                if not d.get(bot_id):
                    _restart_after.pop(bot_id, None)
                    continue
                if is_running(spec["cfg"]):
                    _restart_after.pop(bot_id, None)
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
#  Цены открытых позиций
# ─────────────────────────────────────────────────────────────
_prices = {"ts": 0, "data": {}}


def price_loop():
    """Раз в 20 с — текущие цены по открытым позициям обоих ботов."""
    import ccxt
    ex = None
    while True:
        try:
            symbols = set()
            for spec in BOTS.values():
                j = read_journal(spec["cfg"])
                for p in (j or {}).get("open", []):
                    symbols.add(p["symbol"])
            if symbols:
                if ex is None:
                    ex = ccxt.bitget({"enableRateLimit": True,
                                      "options": {"defaultType": "swap"}})
                t = ex.fetch_tickers(sorted(symbols))
                _prices["data"] = {s: v.get("last") for s, v in t.items() if v.get("last")}
                _prices["ts"] = time.time()
        except Exception as e:
            plog(f"цены: {type(e).__name__}: {str(e)[:120]}")
        time.sleep(20)


# ─────────────────────────────────────────────────────────────
#  Данные для интерфейса
# ─────────────────────────────────────────────────────────────
_journal_cache = {}


def read_journal(cfg):
    """
    Журнал может быть в процессе подмены — тогда отдаём прошлую версию,
    а не падаем и не показываем пустоту.
    """
    path = cfg["journal"]
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        _journal_cache[path] = j
        return j
    except FileNotFoundError:
        return None
    except Exception:
        return _journal_cache.get(path)


def tail(path, n):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 200_000))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
        return lines[-n:]
    except Exception:
        return []


def _ts(iso):
    try:
        return int(datetime.fromisoformat(iso).timestamp() * 1000)
    except Exception:
        return None


def bot_state(bot_id):
    spec = BOTS[bot_id]
    cfg = spec["cfg"]
    j = read_journal(cfg) or {}
    deposit = float(j.get("deposit", cfg["deposit"]))
    balance = float(j.get("balance", deposit))
    trades = j.get("trades", [])
    opened = j.get("open", [])

    wins = [t for t in trades if t.get("result") == "WIN"]
    rs = [t.get("r_multiple", 0) for t in trades]

    # Кривая капитала по закрытым сделкам
    equity = []
    start_ts = _ts(j.get("created", "")) or int(time.time() * 1000)
    equity.append([start_ts, deposit])
    for t in trades:
        ts = _ts(t.get("closed", ""))
        if ts:
            equity.append([ts, float(t.get("balance", balance))])
    peak, max_dd = deposit, 0.0
    for _, b in equity:
        peak = max(peak, b)
        max_dd = min(max_dd, (b - peak) / peak * 100 if peak else 0)

    positions, unreal = [], 0.0
    for p in opened:
        price = _prices["data"].get(p["symbol"])
        risk = abs(p["entry"] - p["stop"]) * p["qty"]
        pnl = (price - p["entry"]) * p["qty"] * p["dir"] if price else None
        if pnl is not None:
            unreal += pnl
        positions.append({
            "symbol": p["symbol"].split("/")[0], "dir": p["dir"],
            "entry": p["entry"], "stop": p["stop"], "take": p["take"],
            "price": price, "pnl": pnl,
            "r": (pnl / risk) if (pnl is not None and risk > 0) else None,
            "notional": p.get("notional"), "opened": p.get("opened"),
            "stop_pct": p.get("stop_pct"),
        })

    logs = tail(cfg["logfile"], 250)
    last_cycle = None
    for line in reversed(logs):
        if "Баланс=" in line and line.startswith("["):
            last_cycle = line[1:20]
            break

    return {
        "id": bot_id,
        "name": cfg.get("bot_name", bot_id),
        "short": spec["short"],
        "rules": core.describe_rules(cfg),
        "risk_pct": cfg["risk_pct"],
        "max_open": cfg["max_open"],
        "running": is_running(cfg),
        "enabled": bool(load_desired().get(bot_id)),
        "pid": read_pid(cfg),
        "created": j.get("created"),
        "cycles": j.get("cycles", 0),
        "last_cycle": last_cycle,
        "deposit": deposit,
        "balance": balance,
        "unrealized": unreal,
        "trades_count": len(trades),
        "wins": len(wins),
        "wr": (len(wins) / len(trades) * 100) if trades else None,
        "avg_r": (sum(rs) / len(rs)) if rs else None,
        "max_dd": max_dd,
        "equity": equity,
        "positions": positions,
        "trades": [{
            "symbol": t["symbol"].split("/")[0], "dir": t["dir"],
            "entry": t.get("entry"), "exit": t.get("exit"),
            "reason": t.get("exit_reason"), "r": t.get("r_multiple"),
            "pnl": t.get("pnl"), "closed": t.get("closed"),
            "balance": t.get("balance"),
        } for t in trades[-100:]][::-1],
        "expect": spec["expect"],
    }


def research():
    """Результаты исследований — то, с чем сравнивать живую торговлю."""
    out = {}
    try:
        out["projection"] = json.loads(
            (ROOT / "Backtest" / "projection_100.json").read_text(encoding="utf-8"))
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
        "now": int(time.time() * 1000),
        "prices_age": (time.time() - _prices["ts"]) if _prices["ts"] else None,
        "bots": [bot_state(b) for b in BOTS],
    }


# ─────────────────────────────────────────────────────────────
#  HTTP
# ─────────────────────────────────────────────────────────────
MIME = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".js": "application/javascript; charset=utf-8", ".svg": "image/svg+xml"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass                                   # под pythonw писать некуда

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
            return self._json({"lines": tail(BOTS[bot]["cfg"]["logfile"], 400)})

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
        q = parse_qs(u.query)
        bot = q.get("bot", [None])[0]
        if u.path in ("/api/start", "/api/stop"):
            if bot not in BOTS:
                return self._json({"error": "нет такого бота"}, 404)
            result = start_bot(bot) if u.path == "/api/start" else stop_bot(bot)
            return self._json({"ok": True, "result": result})
        if u.path == "/api/quit":
            self._json({"ok": True})
            threading.Thread(target=lambda: (time.sleep(0.5), os._exit(0)),
                             daemon=True).start()
            return
        self._json({"error": "нет такой команды"}, 404)


# ─────────────────────────────────────────────────────────────
#  Запуск
# ─────────────────────────────────────────────────────────────
def find_app_browser():
    """Chrome в приоритете, затем Edge — оба умеют режим приложения."""
    env = os.environ
    candidates = [
        Path(env.get("ProgramFiles", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
        Path(env.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Google/Chrome/Application/chrome.exe",
        Path(env.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
        Path(env.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe",
        Path(env.get("ProgramFiles", r"C:\Program Files")) / "Microsoft/Edge/Application/msedge.exe",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def open_window(url):
    browser = find_app_browser()
    if browser:
        try:
            subprocess.Popen([str(browser), f"--app={url}", "--window-size=1480,940"],
                             creationflags=0x08000000 if os.name == "nt" else 0)
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
        # Панель уже работает в фоне — просто показать окно ещё раз
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
