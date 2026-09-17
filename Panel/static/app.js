/* TradingBots — панель управления.
   Данные из журналов и логов — недоверенный текст: в DOM только через
   textContent, никакого innerHTML со значениями. */
"use strict";

const TOKEN = document.querySelector('meta[name="panel-token"]').content;
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const SVGNS = "http://www.w3.org/2000/svg";
const BOT_COLOR = { bot5: "var(--bot5)", bot6: "var(--bot6)" };

let S = null;              // последнее состояние с сервера
let research = null;
let eqView = "chart";
let tab = "positions";
let logBot = "bot5";
const busy = new Set();    // боты, по которым идёт команда
let demoCheck = null;      // результат последней проверки демо-ключа
let lastSig = "";          // отпечаток данных — перерисовываем только при изменении
let lastRenderAt = 0;

const STRAT_RU = { donchian: "Дончиан", supertrend: "Supertrend" };
BOT_COLOR["demo-donchian"] = "var(--bot5)";
BOT_COLOR["demo-supertrend"] = "var(--bot6)";
const demoBot = strat => ({ id: "demo-" + strat, short: "Демо · " + (STRAT_RU[strat] || strat) });

// ── Утилиты ───────────────────────────────────────────────
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style") el.style.cssText = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
// replaceChildren(null) печатает на странице слово "null" — браузер
// приводит его к тексту. h() такие значения отбрасывает, а прямые
// вызовы нет, поэтому для них есть fill().
function fill(el, ...kids) {
  el.replaceChildren(...kids.flat().filter(k => k != null && k !== false));
  return el;
}
function s(tag, attrs = {}) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) el.setAttribute(k, v);
  return el;
}
// Округление до центов ДО показа. Без него слагаемые на экране не
// сходятся с суммой: −$5.13 и +$3.30 дают −$1.83, а посчитанная
// отдельно разница показывается как −$1.82. Каждое число по
// отдельности верное, а вместе — бессмыслица.
const cents = v => Math.round((v || 0) * 100) / 100;
const money = (v, sign = false) => {
  if (v == null || !isFinite(v)) return "—";
  const a = Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (!sign) return (v < 0 ? "−$" : "$") + a;
  return (v > 0 ? "+$" : v < 0 ? "−$" : "$") + a;
};
const pct = (v, d = 1, sign = true) => v == null || !isFinite(v) ? "—"
  : (sign && v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(d) + "%";
const rr = v => v == null || !isFinite(v) ? "—" : (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(2) + "R";
const tone = v => v == null || Math.abs(v) < 1e-9 ? "flat" : v > 0 ? "up" : "down";
const arrow = v => v == null || Math.abs(v) < 1e-9 ? "" : v > 0 ? "▲ " : "▼ ";
const price = v => {
  if (v == null || !isFinite(v)) return "—";
  const d = v >= 100 ? 2 : v >= 1 ? 4 : 6;
  return v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
};
function ago(ms) {
  if (!ms) return "—";
  const m = Math.round((Date.now() - ms) / 60000);
  if (m < 1) return "только что";
  if (m < 60) return `${m} мин назад`;
  const hh = Math.floor(m / 60);
  if (hh < 48) return `${hh} ч ${m % 60} мин назад`;
  return `${Math.floor(hh / 24)} дн назад`;
}
const parseLogTs = t => t ? new Date(t.replace(" ", "T")).getTime() : null;
const dt = ms => new Date(ms).toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
const dshort = ms => new Date(ms).toLocaleDateString("ru-RU", { day: "2-digit", month: "2-digit" });

function toast(msg) {
  const t = h("div", { class: "toast", role: "status" }, msg);
  document.body.append(t);
  setTimeout(() => { t.style.opacity = "0"; setTimeout(() => t.remove(), 300); }, 2600);
}

async function api(path, method = "GET") {
  const r = await fetch(path, { method, headers: method === "POST" ? { "X-Panel-Token": TOKEN } : {} });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
}

// ── Тема ──────────────────────────────────────────────────
function applyTheme(t) {
  if (t) document.documentElement.dataset.theme = t;
  else delete document.documentElement.dataset.theme;
}
try { applyTheme(localStorage.getItem("tb-theme")); } catch (e) {}
$("#theme").addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem("tb-theme", next); } catch (e) {}
  render();
});

// ── Герой ─────────────────────────────────────────────────
function renderHero() {
  const bots = S.bots;
  // Крупное число — деньги, которые уже на счетах. Раньше здесь была
  // сумма баланса и плавающей прибыли по открытым позициям, и человек
  // видел «капитал $98.18» при убытке −$5.13: цифры не сходились,
  // потому что в одном числе смешаны зафиксированное и ещё нет.
  const dep = cents(bots.reduce((a, b) => a + b.deposit, 0));
  const bal = cents(bots.reduce((a, b) => a + b.balance, 0));
  const unr = cents(bots.reduce((a, b) => a + (b.unrealized || 0), 0));
  const realized = cents(bal - dep);
  const eq = cents(bal + unr);
  $("#hero-value").textContent = money(bal);

  const delta = $("#hero-delta");
  delta.replaceChildren(
    h("span", { class: tone(realized) },
      arrow(realized) + money(realized, true) + " закрытыми сделками (" + pct(dep ? realized / dep * 100 : 0) + ")"),
    h("span", { class: "hero-note" },
      unr ? `открытые позиции ${money(unr, true)} — если закрыть их сейчас, будет ${money(eq)}`
          : "открытых позиций нет")
  );

  const trades = bots.reduce((a, b) => a + b.trades_count, 0);
  const wins = bots.reduce((a, b) => a + b.wins, 0);
  const open = bots.reduce((a, b) => a + b.positions.length, 0);
  const running = bots.filter(b => b.running).length;
  const chip = (label, value, cls) => h("div", { class: "chip" },
    h("div", { class: "chip-label" }, label), h("div", { class: "chip-value " + (cls || "") }, value));
  $("#hero-side").replaceChildren(
    chip("Боты работают", `${running} из ${bots.length}`),
    chip("Открыто позиций", String(open)),
    chip("Сделок закрыто", String(trades)),
    chip("Доля прибыльных", trades ? pct(wins / trades * 100, 0, false) : "—"),
  );
}

// ── Карточки ботов ────────────────────────────────────────
function renderBots() {
  const wrap = $("#bots");
  wrap.replaceChildren(...S.bots.map(botCard));
}

function botCard(b) {
  // Те же правила, что в шапке: округляем до центов заранее, чтобы
  // числа на карточке сходились между собой, и нигде не смешиваем
  // зафиксированное с плавающим
  const bal = cents(b.balance);
  const dep = cents(b.deposit);
  const unreal = cents(b.unrealized);
  const realized = cents(bal - dep);      // то, что уже зафиксировано
  const eq = cents(bal + unreal);         // сколько будет, если закрыть всё сейчас
  const lastTs = parseLogTs(b.last_cycle);
  const stateCls = b.running ? "on" : (b.enabled ? "restarting" : "");
  const stateTxt = b.running ? "Работает" : (b.enabled ? "Перезапуск…" : "Остановлен");

  const isBusy = busy.has(b.id);
  const btn = h("button", {
    class: "btn" + (b.running ? "" : " primary"), disabled: isBusy || null,
    onclick: () => toggleBot(b),
  }, isBusy ? "Секунду…" : b.running ? "Остановить" : "Запустить");

  const tile = (label, value, sub, cls) => h("div", { class: "tile" },
    h("div", { class: "tile-label" }, label),
    h("div", { class: "tile-value " + (cls || "") }, value),
    sub ? h("div", { class: "tile-sub" }, sub) : null);

  const need = 100;
  const progress = Math.min(100, b.trades_count / need * 100);

  const posChips = b.positions.length
    ? b.positions.map(p => h("span", { class: "pos-chip" },
        h("b", {}, p.symbol),
        h("span", { class: "muted" }, p.dir === 1 ? "лонг" : "шорт"),
        h("span", { class: tone(p.r) }, p.r == null ? "…" : arrow(p.r) + rr(p.r))))
    : [h("span", { class: "muted" }, "Открытых позиций нет — бот ждёт пробоя")];

  return h("article", { class: "card bot", "aria-label": b.name },
    h("div", { class: "bot-head" },
      h("span", { class: "bot-key", style: `background:${BOT_COLOR[b.id]}` }),
      h("div", { style: "flex:1;min-width:0" },
        h("div", { class: "bot-title", style: "display:flex;gap:10px;align-items:baseline;flex-wrap:wrap" },
          b.name,
          h("span", { class: "muted", style: "font-size:13px;font-weight:500" },
            `на счёте ${money(bal)}`)),
        h("div", { class: "bot-rules" }, `${b.rules} · риск ${Math.round(b.risk_pct * 100)}% · до ${b.max_open} позиций`)),
      h("span", { class: "status " + stateCls }, h("span", { class: "status-dot" }), stateTxt),
      btn),

    // Закрытый результат и незакрытые позиции — РАЗНЫЕ плитки.
    // Пока они были сложены в одну «Прибыль», прибыль по открытым
    // позициям перекрывала убыток по закрытым, и казалось, что
    // проигранная сделка не отразилась на балансе.
    h("div", { class: "tiles" },
      tile("Закрытые сделки", arrow(realized) + money(realized, true),
        `баланс ${money(bal)} · ${pct(dep ? realized / dep * 100 : 0)}`, tone(realized)),
      tile("Открытые позиции", b.positions.length ? arrow(unreal) + money(unreal, true) : "—",
        b.positions.length ? `${b.positions.length} шт · закрыть сейчас — на счёте ${money(eq)}` : "нет открытых",
        tone(unreal)),
      tile("Сделок", String(b.trades_count), b.wr == null ? "WR —" : `WR ${b.wr.toFixed(0)}%`),
      tile("Средний результат", b.avg_r == null ? "—" : rr(b.avg_r), `просадка ${pct(b.max_dd, 1, false)}`, tone(b.avg_r))),

    h("div", { class: "expect" },
      h("div", { class: "expect-row" },
        h("span", {}, `Ожидание по бэктесту: WR ~${b.expect.wr}%, ${rr(b.expect.r_lo)} … ${rr(b.expect.r_hi)} на сделку`),
        h("span", { class: "muted" }, `${b.trades_count} из ${need} сделок до первых выводов`)),
      h("div", { class: "meter", role: "progressbar", "aria-valuenow": Math.round(progress), "aria-valuemin": 0, "aria-valuemax": 100 },
        h("span", { style: `width:${progress}%` }))),

    h("div", { class: "mini-pos" }, posChips),

    h("div", { class: "bot-foot" },
      h("span", {}, lastTs ? `Последний цикл ${ago(lastTs)}` : "Циклов ещё не было"),
      h("span", {}, `цикл №${b.cycles}` + (b.running && b.pid ? ` · процесс ${b.pid}` : "")))
  );
}

async function toggleBot(b) {
  busy.add(b.id); renderBots();
  try {
    const r = await api(`/api/${b.running ? "stop" : "start"}?bot=${b.id}`, "POST");
    toast(`${b.short}: ${r.result}`);
  } catch (e) {
    toast(`Не получилось: ${e.message}`);
  } finally {
    // Запуск занимает несколько секунд: процесс поднимается и берёт блокировку
    setTimeout(async () => { busy.delete(b.id); await refresh(); }, b.running ? 300 : 2500);
  }
}

// ── Кривая баланса ────────────────────────────────────────
// ── Демо-счёт ─────────────────────────────────────────────
function renderDemo() {
  const d = S.demo;
  const box = $("#demo");
  if (!d) { box.replaceChildren(); return; }

  const err = d.last_error;
  const stateCls = d.running ? "on" : err ? "error" : (d.enabled ? "restarting" : "");
  const stateTxt = d.running ? "Работает" : err ? "Остановлен с ошибкой" : (d.enabled && d.keys ? "Перезапуск…" : "Остановлен");
  const isBusy = busy.has("demo");

  const startBtn = h("button", {
    class: "btn" + (d.running ? "" : " primary"),
    disabled: isBusy || (!d.running && !d.keys) || null,
    title: !d.keys ? "Сначала нужны демо-ключи в .env" : null,
    onclick: () => toggleBot(d),
  }, isBusy ? "Секунду…" : d.running ? "Остановить" : "Запустить");

  const checkBtn = h("button", {
    class: "btn", disabled: !d.keys || demoCheck === "running" || null,
    onclick: runDemoCheck,
  }, demoCheck === "running" ? "Проверяю…" : "Проверить подключение");

  const kids = [
    h("div", { class: "bot-head" },
      h("div", { style: "flex:1;min-width:0" },
        h("div", { class: "bot-title", style: "display:flex;gap:10px;align-items:center" },
          d.name, h("span", { class: "demo-badge" }, "демо-счёт")),
        h("div", { class: "bot-rules" },
          "Настоящие заявки, вымышленные деньги · обе стратегии на одном счёте · " +
          (d.symbols ? `${d.symbols} монет · ` : "") +
          (d.leverage ? `плечо ${d.leverage}x · ` : "") +
          `риск ${Math.round(d.risk_pct * 100)}% доли стратегии · до ${d.max_open} позиций на стратегию`)),
      h("span", { class: "status " + stateCls }, h("span", { class: "status-dot" }), stateTxt),
      checkBtn, startBtn),
  ];

  if (!d.keys) {
    kids.push(h("div", { class: "callout" },
      h("b", {}, "Нужны ключи демо-счёта Bitget. "), "Три шага:",
      h("ol", {},
        h("li", {}, "На bitget.com переключитесь в режим «Демо-торговля» и там создайте API-ключ. Права: чтение и торговля. Вывод средств не включайте."),
        h("li", {}, "В корне проекта скопируйте файл .env.example в .env и впишите ключ, секрет и пароль (passphrase)."),
        h("li", {}, "Нажмите «Проверить подключение» — ни одной заявки отправлено не будет."))));
  }
  if (err) {
    kids.push(h("div", { class: "callout bad" }, h("b", {}, "Бот остановился: "), err));
  }
  if (demoCheck && demoCheck !== "running") {
    kids.push(h("div", { class: "callout" + (demoCheck.ok ? "" : " bad") },
      h("b", {}, demoCheck.ok ? "Проверка пройдена" : "Проверка не пройдена"),
      h("div", { class: "check-out", style: "margin-top:8px" }, demoCheck.output.trim())));
  }

  if (d.equity != null) {
    const res = d.start_equity != null ? d.equity - d.start_equity : null;
    const funding = Object.values(d.per_strategy).reduce((a, x) => a + (x.funding || 0), 0);
    const tile = (label, value, sub, cls) => h("div", { class: "tile" },
      h("div", { class: "tile-label" }, label), h("div", { class: "tile-value " + (cls || "") }, value),
      sub ? h("div", { class: "tile-sub" }, sub) : null);
    const usdt = v => v == null ? "—" : v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + " USDT";
    kids.push(h("div", { class: "tiles" },
      tile("Капитал демо-счёта", usdt(d.equity),
        "по данным биржи, вместе с открытыми позициями"
        + (d.start_equity != null ? ` · старт ${usdt(d.start_equity)}` : "")),
      tile("Результат", res == null ? "—" : arrow(res) + (res > 0 ? "+" : "") + usdt(res),
        "с учётом открытых позиций", tone(res)),
      tile("Свободная маржа", usdt(d.available), null),
      tile("Funding", usdt(funding), "по закрытым позициям, от биржи", tone(funding))));

    kids.push(h("div", { class: "strat-rows" }, ["donchian", "supertrend"].map(s => {
      const x = d.per_strategy[s];
      const cell = (label, value, cls) => h("div", { class: "strat-cell" },
        h("div", { class: "tile-label" }, label), h("b", { class: cls || "" }, value));
      return h("div", { class: "strat-row" },
        h("span", { class: "bot-key", style: `background:${s === "donchian" ? "var(--bot5)" : "var(--bot6)"}` }),
        h("div", {}, h("b", {}, STRAT_RU[s])),
        cell("Открыто", String(x.open)),
        cell("Сделок", String(x.trades)),
        cell("WR", x.trades ? pct(x.wins / x.trades * 100, 0, false) : "—"),
        cell("Средний R", rr(x.avg_r), tone(x.avg_r)),
        cell("Итог", x.trades ? (x.pnl > 0 ? "+" : "") + x.pnl.toFixed(2) : "—", tone(x.pnl)));
    })));
  }

  if (d.positions.length) {
    kids.push(h("div", { class: "mini-pos" }, d.positions.map(p => h("span", { class: "pos-chip" },
      h("span", { class: "botdot", style: `background:${BOT_COLOR["demo-" + p.strategy]};margin-right:0` }),
      h("b", {}, p.symbol),
      h("span", { class: "muted" }, p.dir === 1 ? "лонг" : "шорт"),
      h("span", { class: tone(p.r) }, p.r == null ? "…" : arrow(p.r) + rr(p.r))))));
  }
  if (d.foreign && d.foreign.length) {
    kids.push(h("div", { class: "card-sub" },
      `На демо-счёте есть позиции, открытые не ботом: ${d.foreign.map(s => s.split("/")[0]).join(", ")} — эти монеты бот не трогает.`));
  }

  const lastTs = parseLogTs(d.last_cycle);
  kids.push(h("div", { class: "bot-foot" },
    h("span", {}, lastTs ? `Последний цикл ${ago(lastTs)}` : "Циклов ещё не было"),
    h("span", {}, `цикл №${d.cycles}` + (d.running && d.pid ? ` · процесс ${d.pid}` : ""))));

  box.replaceChildren(h("article", { class: "card demo", "aria-label": d.name }, kids));
}

async function runDemoCheck() {
  demoCheck = "running";
  renderDemo();
  try {
    demoCheck = await api("/api/demo/check", "POST");
  } catch (e) {
    demoCheck = { ok: false, output: "Не удалось выполнить проверку: " + e.message };
  }
  renderDemo();
}

function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1;
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(st => span / st <= n) || 10 * mag;
  const start = Math.ceil(lo / step) * step;
  const out = [];
  for (let v = start; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

function stepValueAt(points, t) {
  let v = points[0][1];
  for (const [ts, b] of points) { if (ts <= t) v = b; else break; }
  return v;
}

function renderEquity() {
  const now = S.now;
  const series = S.bots.map(b => ({
    id: b.id, name: b.short, color: BOT_COLOR[b.id],
    pts: [...b.equity, [now, b.balance]],
  }));

  $("#eq-legend").replaceChildren(...series.map(sr =>
    h("span", {}, h("i", { style: `background:${sr.color}` }), sr.name)));

  const box = $("#eq-chart");
  const W = Math.max(320, box.clientWidth || 900), H = 300;
  const m = { l: 56, r: 150, t: 12, b: 30 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;

  const allT = series.flatMap(sr => sr.pts.map(p => p[0]));
  const allV = series.flatMap(sr => sr.pts.map(p => p[1]));
  let t0 = Math.min(...allT), t1 = Math.max(...allT);
  if (t1 - t0 < 3600e3) t0 = t1 - 3600e3;
  let v0 = Math.min(...allV), v1 = Math.max(...allV);
  const pad = Math.max((v1 - v0) * 0.15, Math.max(v1, 1) * 0.04);
  v0 -= pad; v1 += pad;

  const X = t => m.l + (t - t0) / (t1 - t0) * iw;
  const Y = v => m.t + (1 - (v - v0) / (v1 - v0)) * ih;

  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Кривая баланса ботов по закрытым сделкам" });

  for (const v of niceTicks(v0, v1, 5)) {
    svg.append(s("line", { x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v), stroke: "var(--grid)", "stroke-width": 1 }));
    const tx = s("text", { x: m.l - 10, y: Y(v) + 4, "text-anchor": "end", class: "tick" });
    tx.textContent = "$" + v.toLocaleString("en-US", { maximumFractionDigits: v < 100 ? 1 : 0 });
    svg.append(tx);
  }
  const days = (t1 - t0) / 86400e3;
  const nx = Math.min(6, Math.max(2, Math.floor(iw / 120)));
  for (let i = 0; i <= nx; i++) {
    const t = t0 + (t1 - t0) * i / nx;
    const tx = s("text", { x: X(t), y: H - 8, "text-anchor": i === 0 ? "start" : i === nx ? "end" : "middle", class: "tick" });
    tx.textContent = days < 2 ? new Date(t).toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" }) : dshort(t);
    svg.append(tx);
  }
  svg.append(s("line", { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih, stroke: "var(--axis)", "stroke-width": 1 }));

  // Ступенчатая линия: баланс меняется только в момент закрытия сделки
  const ends = [];
  for (const sr of series) {
    let d = "";
    sr.pts.forEach(([t, v], i) => {
      if (i === 0) d += `M${X(t)},${Y(v)}`;
      else d += `H${X(t)}V${Y(v)}`;
    });
    svg.append(s("path", { d, fill: "none", stroke: sr.color, "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round" }));
    const [lt, lv] = sr.pts[sr.pts.length - 1];
    ends.push({ sr, x: X(lt), y: Y(lv), v: lv });
  }
  // Подписи на концах; если сходятся — разводим с выносками
  ends.sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) {
    ends[i].ly = Math.max(ends[i].y, (ends[i - 1].ly ?? ends[i - 1].y) + 18);
  }
  for (const e of ends) {
    const ly = e.ly ?? e.y;
    if (Math.abs(ly - e.y) > 1) {
      svg.append(s("line", { x1: e.x + 6, y1: e.y, x2: e.x + 14, y2: ly, stroke: "var(--axis)", "stroke-width": 1 }));
    }
    svg.append(s("circle", { cx: e.x, cy: e.y, r: 4, fill: e.sr.color, stroke: "var(--surface)", "stroke-width": 2 }));
    const tx = s("text", { x: e.x + 16, y: ly + 4, class: "endlabel" });
    tx.textContent = `${e.sr.name} ${money(e.v)}`;
    svg.append(tx);
  }

  // Перекрестье и подсказка
  const cross = s("line", { y1: m.t, y2: m.t + ih, stroke: "var(--ink-3)", "stroke-width": 1, visibility: "hidden" });
  svg.append(cross);
  const dots = series.map(sr => {
    const c = s("circle", { r: 4, fill: sr.color, stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden" });
    svg.append(c); return c;
  });
  const hit = s("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" });
  svg.append(hit);

  const tip = h("div", { class: "tip", hidden: true });
  const snapTimes = [...new Set(allT)].sort((a, b) => a - b);
  const move = ev => {
    const r = svg.getBoundingClientRect();
    const px = (ev.clientX - r.left) * (W / r.width);
    const tRaw = t0 + (px - m.l) / iw * (t1 - t0);
    let t = snapTimes.reduce((best, x) => Math.abs(x - tRaw) < Math.abs(best - tRaw) ? x : best, snapTimes[0]);
    t = Math.min(Math.max(t, t0), t1);
    const cx = X(t);
    cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.setAttribute("visibility", "visible");
    series.forEach((sr, i) => {
      const v = stepValueAt(sr.pts, t);
      dots[i].setAttribute("cx", cx); dots[i].setAttribute("cy", Y(v)); dots[i].setAttribute("visibility", "visible");
    });
    tip.replaceChildren(
      h("div", { class: "tip-time" }, dt(t)),
      ...series.map(sr => {
        const v = stepValueAt(sr.pts, t);
        return h("div", { class: "tip-row" },
          h("span", { class: "tip-val" }, money(v)),
          h("span", { class: "tip-key" }, h("i", { style: `background:${sr.color}` }), sr.name));
      }));
    tip.hidden = false;
    tip.style.left = (cx / W * r.width) + "px";
    tip.style.top = ((ev.clientY - r.top)) + "px";
  };
  const leave = () => {
    cross.setAttribute("visibility", "hidden");
    dots.forEach(d => d.setAttribute("visibility", "hidden"));
    tip.hidden = true;
  };
  hit.addEventListener("pointermove", move);
  hit.addEventListener("pointerleave", leave);

  const noTrades = S.bots.every(b => b.trades_count === 0);
  fill(box, svg, tip,
    noTrades && h("div", { class: "card-sub", style: "text-align:center;margin-top:6px" },
      "Закрытых сделок пока нет — кривая оживёт, когда закроется первая позиция"));

  // Табличный двойник графика
  const rows = [];
  for (const b of S.bots) for (const [t, v] of b.equity) rows.push({ b, t, v });
  rows.sort((a, c) => c.t - a.t);
  $("#eq-table").replaceChildren(h("table", {},
    h("thead", {}, h("tr", {}, h("th", {}, "Время"), h("th", {}, "Бот"), h("th", { class: "num" }, "Баланс"))),
    h("tbody", {}, rows.map(r => h("tr", {},
      h("td", {}, dt(r.t)),
      h("td", {}, h("span", { class: "botdot", style: `background:${BOT_COLOR[r.b.id]}` }), r.b.short),
      h("td", { class: "num" }, money(r.v)))))));
}

// ── Позиции и сделки ──────────────────────────────────────
function sideCell(dir) {
  return h("span", { class: "side" }, dir === 1 ? "▲ Лонг" : "▼ Шорт");
}

function renderPositions() {
  const rows = [
    ...S.bots.flatMap(b => b.positions.map(p => ({ b, p }))),
    ...(S.demo ? S.demo.positions.map(p => ({ b: demoBot(p.strategy), p })) : []),
  ];
  const el = $('[data-panel="positions"]');
  if (!rows.length) {
    el.replaceChildren(h("div", { class: "empty" },
      "Открытых позиций нет. Стратегии на 4ч входят редко — в среднем около одной сделки в сутки на бота."));
    return;
  }
  const age = S.prices_age;
  el.replaceChildren(
    h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {},
        h("th", {}, "Бот"), h("th", {}, "Монета"), h("th", {}, "Направление"),
        h("th", { class: "num" }, "Вход"), h("th", { class: "num" }, "Цена сейчас"),
        h("th", { class: "num" }, "Стоп"), h("th", { class: "num" }, "Тейк"),
        h("th", { class: "num" }, "Объём"), h("th", { class: "num" }, "Результат"),
        h("th", { class: "num" }, "Открыта"))),
      h("tbody", {}, rows.map(({ b, p }) => h("tr", {},
        h("td", {}, h("span", { class: "botdot", style: `background:${BOT_COLOR[b.id]}` }), b.short),
        h("td", {}, h("b", {}, p.symbol)),
        h("td", {}, sideCell(p.dir)),
        h("td", { class: "num" }, price(p.entry)),
        h("td", { class: "num" }, price(p.price)),
        h("td", { class: "num" }, price(p.stop)),
        h("td", { class: "num" }, price(p.take)),
        h("td", { class: "num" }, money(p.notional)),
        h("td", { class: "num " + tone(p.r) }, p.r == null ? "—" : `${arrow(p.r)}${rr(p.r)} · ${money(p.pnl, true)}`),
        h("td", { class: "num muted" }, ago(new Date(p.opened).getTime()))))))),
    h("div", { class: "card-sub", style: "margin-top:10px" },
      age == null ? "Цены ещё загружаются…" : `Цены обновлены ${Math.round(age)} с назад · результат без учёта комиссии выхода`));
}

function renderTrades() {
  const rows = [
    ...S.bots.flatMap(b => b.trades.map(t => ({ b, t }))),
    ...(S.demo ? S.demo.trades.map(t => ({ b: demoBot(t.strategy), t })) : []),
  ].sort((a, c) => new Date(c.t.closed) - new Date(a.t.closed));
  const el = $('[data-panel="trades"]');
  if (!rows.length) {
    el.replaceChildren(h("div", { class: "empty" }, "Закрытых сделок пока нет."));
    return;
  }
  const reason = { stop: "стоп", take: "тейк", time: "по времени", trail: "трейлинг" };
  el.replaceChildren(h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {},
      h("th", {}, "Закрыта"), h("th", {}, "Бот"), h("th", {}, "Монета"), h("th", {}, "Направление"),
      h("th", { class: "num" }, "Вход"), h("th", { class: "num" }, "Выход"), h("th", {}, "Причина"),
      h("th", { class: "num" }, "Результат"), h("th", { class: "num" }, "Баланс после"))),
    h("tbody", {}, rows.map(({ b, t }) => h("tr", {},
      h("td", { class: "muted" }, dt(new Date(t.closed).getTime())),
      h("td", {}, h("span", { class: "botdot", style: `background:${BOT_COLOR[b.id]}` }), b.short),
      h("td", {}, h("b", {}, t.symbol)),
      h("td", {}, sideCell(t.dir)),
      h("td", { class: "num" }, price(t.entry)),
      h("td", { class: "num" }, price(t.exit)),
      h("td", {}, reason[t.reason] || t.reason || "—"),
      h("td", { class: "num " + tone(t.r) }, `${arrow(t.r)}${rr(t.r)} · ${money(t.pnl, true)}`),
      h("td", { class: "num" }, money(t.balance))))))));
}

// ── Журнал ────────────────────────────────────────────────
async function renderLog() {
  const box = $("#log");
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  let lines = [];
  try { lines = (await api(`/api/log?bot=${logBot}`)).lines; } catch (e) { return; }
  box.replaceChildren(...lines.map(line => {
    const m = line.match(/^\[([^\]]+)\]\s?(.*)$/);
    const text = m ? m[2] : line;
    let cls = "";
    if (/WIN/.test(text)) cls = "l-win";
    else if (/LOSS|ОШИБКА|\[!\]|СЛИТ/.test(text)) cls = "l-loss";
    else if (/ОТКРЫТА/.test(text)) cls = "l-open";
    return h("div", {},
      m ? h("span", { class: "l-ts" }, m[1] + "  ") : null,
      h("span", { class: cls }, text));
  }));
  if (atBottom) box.scrollTop = box.scrollHeight;
}

// ── Стратегии ─────────────────────────────────────────────
const FAMILY_RU = {
  donchian: "Пробой Дончиана", supertrend: "Supertrend", ma_cross: "Пересечение скользящих",
  tsmom: "Моментум (TSMOM)", keltner: "Канал Кельтнера", macd: "MACD", rsi_momo: "RSI-моментум",
  pullback: "Откат к скользящей", bb_squeeze: "Сжатие Боллинджера", nr7: "Узкий диапазон (NR7)",
  meanrev: "Возврат к среднему (RSI+BB)", zscore: "Возврат к среднему (z-score)",
  xsmom: "Кросс-секционный моментум",
};

function projectionChart(year) {
  const rows = [
    { key: "both", name: "Оба бота по $50", color: "var(--bot-both)" },
    { key: "donchian", name: "Только Дончиан $100", color: "var(--bot5)" },
    { key: "supertrend", name: "Только Supertrend $100", color: "var(--bot6)" },
  ];
  const W = 640, rowH = 46, m = { l: 180, r: 24, t: 10, b: 30 };
  const H = m.t + rows.length * rowH + m.b;
  const iw = W - m.l - m.r;
  const maxV = Math.max(...rows.map(r => year.subsets[r.key].p90), 120) * 1.08;
  const X = v => m.l + v / maxV * iw;

  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Диапазон капитала через год по случайным наборам монет" });
  for (const v of niceTicks(0, maxV, 5)) {
    svg.append(s("line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: "var(--grid)", "stroke-width": 1 }));
    const tx = s("text", { x: X(v), y: H - 10, "text-anchor": "middle", class: "tick" });
    tx.textContent = "$" + v;
    svg.append(tx);
  }
  // Стартовые $100
  svg.append(s("line", { x1: X(100), x2: X(100), y1: m.t - 4, y2: H - m.b, stroke: "var(--ink-3)", "stroke-width": 1.5 }));
  const st = s("text", { x: X(100) + 5, y: m.t + 6, class: "tick" });
  st.textContent = "старт";
  svg.append(st);

  rows.forEach((r, i) => {
    const d = year.subsets[r.key];
    const cy = m.t + i * rowH + rowH / 2;
    const lbl = s("text", { x: m.l - 14, y: cy - 3, "text-anchor": "end", class: "endlabel" });
    lbl.textContent = r.name;
    svg.append(lbl);
    const sub = s("text", { x: m.l - 14, y: cy + 13, "text-anchor": "end", class: "tick" });
    sub.textContent = `в минусе ${d.loss_share}% наборов`;
    svg.append(sub);
    // 10–90 перцентиль — полоса, медиана — точка, худший — засечка
    svg.append(s("rect", { x: X(d.p10), y: cy - 5, width: Math.max(2, X(d.p90) - X(d.p10)), height: 10, rx: 5,
      fill: r.color, "fill-opacity": 0.28 }));
    svg.append(s("line", { x1: X(d.worst), x2: X(d.worst), y1: cy - 9, y2: cy + 9, stroke: r.color, "stroke-width": 2, "stroke-linecap": "round" }));
    svg.append(s("circle", { cx: X(d.median), cy, r: 6, fill: r.color, stroke: "var(--surface)", "stroke-width": 2 }));
    const mv = s("text", { x: X(d.median), y: cy - 12, "text-anchor": "middle", class: "endlabel" });
    mv.textContent = money(d.median).replace(".00", "");
    svg.append(mv);
  });
  return svg;
}

function renderResearch() {
  const el = $('[data-panel="research"]');
  if (!research) { el.replaceChildren(h("div", { class: "empty" }, "Загрузка…")); return; }
  const proj = research.projection;
  const kids = [];

  if (proj && proj.years && proj.years.length) {
    const oos = proj.years.find(y => y.tag === "ПРОВЕРКА") || proj.years[proj.years.length - 1];
    const b = oos.subsets.both;
    const fact = (label, value, sub, cls) => h("div", { class: "tile", style: "border:1px solid var(--ring);border-radius:11px" },
      h("div", { class: "tile-label" }, label), h("div", { class: "tile-value " + (cls || "") }, value),
      sub ? h("div", { class: "tile-sub" }, sub) : null);

    kids.push(h("div", { class: "grid-2" },
      h("div", {},
        h("div", { class: "card-title" }, "$100 через год при риске 5%"),
        h("div", { class: "card-sub", style: "margin-bottom:8px" },
          `Период ${oos.from.slice(0, 7)} … ${oos.to.slice(0, 7)} — отбор правил его не видел. ` +
          "20 случайных наборов по 25 монет: полоса — от 10-го до 90-го перцентиля, точка — медиана, засечка — худший набор."),
        h("div", { class: "chart" }, projectionChart(oos))),
      h("div", { style: "display:flex;flex-direction:column;gap:10px" },
        h("div", { class: "facts" },
          fact("Медиана, оба бота", money(b.median), pct((b.median - 100)), tone(b.median - 100)),
          fact("Типичный диапазон", `${money(b.p10).replace(".00", "")} … ${money(b.p90).replace(".00", "")}`, "10-й … 90-й перцентиль"),
          fact("Худший набор монет", money(b.worst), pct(b.worst - 100), tone(b.worst - 100)),
          fact("Просадка по пути", pct(b.dd_median, 0, false), "медиана — это нормально для 5%", "down")),
        h("div", { class: "callout" },
          h("b", {}, "Два бота вместе надёжнее каждого по отдельности. "),
          `На проверке год закончили в минусе ${b.loss_share}% наборов — против ${oos.subsets.donchian.loss_share}% у одного Дончиана. ` +
          "Но просадка около −65% по пути — цена риска 5%: чтобы из неё выйти, нужен рост почти втрое.")),
    ));

    // Остальные годы — для понимания разброса
    kids.push(h("div", { class: "card-sub", style: "margin:18px 0 6px" },
      "Другие годы — на них правила отбирались, поэтому оценка оптимистичнее:"));
    kids.push(h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "Год"), h("th", {}, "Сценарий"),
        h("th", { class: "num" }, "Медиана"), h("th", { class: "num" }, "10% … 90%"),
        h("th", { class: "num" }, "Худший"), h("th", { class: "num" }, "В минусе"), h("th", { class: "num" }, "Просадка"))),
      h("tbody", {}, proj.years.flatMap(y => [["both", "Оба бота"], ["donchian", "Дончиан"], ["supertrend", "Supertrend"]].map(([k, name]) => {
        const d = y.subsets[k];
        return h("tr", {},
          h("td", { class: "muted" }, `${y.from.slice(0, 7)} … ${y.to.slice(0, 7)}` + (y.tag === "ПРОВЕРКА" ? " · проверка" : "")),
          h("td", {}, name),
          h("td", { class: "num " + tone(d.median - 100) }, money(d.median)),
          h("td", { class: "num" }, `${money(d.p10)} … ${money(d.p90)}`),
          h("td", { class: "num" }, money(d.worst)),
          h("td", { class: "num" }, d.loss_share + "%"),
          h("td", { class: "num" }, pct(d.dd_median, 0, false)));
      }))))));
  }

  if (research.families && research.families.length) {
    const fam = [...research.families].sort((a, c) => (+c["плюсовых"]) - (+a["плюсовых"]));
    kids.push(h("div", { style: "margin-top:26px" },
      h("div", { class: "card-title" }, "13 семейств стратегий"),
      h("div", { class: "card-sub", style: "margin-bottom:8px" },
        "Доля настроек, прибыльных на проверочном периоде. Высокая доля значит, что край не зависит от точного выбора параметров.")));
    kids.push(h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "Стратегия"), h("th", {}, "Тип"),
        h("th", { class: "num" }, "Настроек"), h("th", {}, "Прибыльных на проверке"), h("th", { class: "num" }, "Медиана"))),
      h("tbody", {}, fam.map(f => {
        const share = +f["плюсовых"];
        const live = f["семейство"] === "donchian" || f["семейство"] === "supertrend";
        const color = f["семейство"] === "donchian" ? "var(--bot5)" : f["семейство"] === "supertrend" ? "var(--bot6)" : "var(--ink-3)";
        const med = +f["медиана_проверка"];
        return h("tr", {},
          h("td", {}, live ? h("b", {}, FAMILY_RU[f["семейство"]] || f["семейство"]) : (FAMILY_RU[f["семейство"]] || f["семейство"]),
            live ? h("span", { class: "muted" }, "  · в работе") : null),
          h("td", { class: "muted" }, f["тип"]),
          h("td", { class: "num" }, f["настроек"]),
          h("td", {}, h("div", { class: "bar-cell" },
            h("div", { class: "bar-track" }, h("div", { class: "bar-fill", style: `width:${share}%;background:${color}` })),
            h("span", { style: "min-width:36px;text-align:right;font-variant-numeric:tabular-nums" }, share + "%"))),
          h("td", { class: "num " + tone(med) }, f["единица"] === "%/год" ? pct(med) + "/год" : rr(med)));
      })))));
    kids.push(h("div", { class: "callout", style: "margin-top:14px" },
      h("b", {}, "Что из этого следует. "),
      "Возврат к среднему не работает ни в одной настройке. Следование за трендом работает как класс — " +
      "но только с фиксированным тейком 3R: трейлинг-стоп на 3 ATR выбивает сделки раньше, чем они доходят до цели."));
  }
  el.replaceChildren(...kids);
}

// ── Цикл обновления ───────────────────────────────────────
function render() {
  if (!S) return;
  renderHero();
  renderBots();
  renderDemo();
  renderEquity();
  if (tab === "positions") renderPositions();
  if (tab === "trades") renderTrades();
  if (tab === "research") renderResearch();
}

async function refresh() {
  document.body.classList.remove("stale");
  try {
    const next = await api("/api/state");
    // Сервер перезапустился — у страницы устаревший токен и, возможно,
    // устаревший код. Перезагружаемся, чтобы кнопки снова работали.
    if (S && S.instance && next.instance && S.instance !== next.instance) {
      location.reload();
      return;
    }
    const all = [...next.bots, ...(next.demo ? [next.demo] : [])];
    const running = all.filter(b => b.running).length;
    $("#meta").textContent = `${running} из ${all.length} ботов работают · обновлено ${new Date().toLocaleTimeString("ru-RU")}`;
    // Перерисовка только если данные изменились — или раз в минуту,
    // чтобы обновились «N мин назад». Иначе Chrome пересобирал всю
    // страницу каждые 5 секунд впустую.
    const sig = JSON.stringify({ ...next, now: 0, prices_age: 0 });
    S = next;
    if (sig !== lastSig || Date.now() - lastRenderAt > 60000) {
      lastSig = sig;
      lastRenderAt = Date.now();
      render();
    }
    if (tab === "log") renderLog();
  } catch (e) {
    // Панель выключена или связь пропала — держим последний кадр приглушённым
    $("#meta").textContent = "нет связи с панелью";
    $$(".app > section").forEach(x => x.classList.add("stale"));
    return;
  }
  $$(".app > section").forEach(x => x.classList.remove("stale"));
}

// ── Переключатели ─────────────────────────────────────────
$$("[data-eqview]").forEach(b => b.addEventListener("click", () => {
  eqView = b.dataset.eqview;
  $$("[data-eqview]").forEach(x => x.setAttribute("aria-selected", x === b));
  $("#eq-chart").hidden = eqView !== "chart";
  $("#eq-table").hidden = eqView !== "table";
}));
$$("[data-tab]").forEach(b => b.addEventListener("click", async () => {
  tab = b.dataset.tab;
  $$("[data-tab]").forEach(x => x.setAttribute("aria-selected", x === b));
  $$("[data-panel]").forEach(p => p.hidden = p.dataset.panel !== tab);
  $("#log-bot").hidden = tab !== "log";
  if (tab === "research" && !research) {
    try { research = await api("/api/research"); } catch (e) { research = { projection: null, families: [] }; }
  }
  if (tab === "log") { await renderLog(); const box = $("#log"); box.scrollTop = box.scrollHeight; }
  render();
}));
$$("[data-logbot]").forEach(b => b.addEventListener("click", async () => {
  logBot = b.dataset.logbot;
  $$("[data-logbot]").forEach(x => x.setAttribute("aria-selected", x === b));
  await renderLog();
  const box = $("#log"); box.scrollTop = box.scrollHeight;
}));

const dlg = $("#quit-dialog");
$("#quit").addEventListener("click", () => dlg.showModal());
$("#quit-cancel").addEventListener("click", () => dlg.close());
$("#quit-ok").addEventListener("click", async () => {
  try { await api("/api/quit", "POST"); } catch (e) {}
  dlg.close();
  document.body.replaceChildren(h("div", { class: "empty", style: "padding:120px 20px;font-size:15px" },
    "Панель закрыта. Боты продолжают работать в фоне — это окно можно закрыть."));
});

let resizeT;
addEventListener("resize", () => { clearTimeout(resizeT); resizeT = setTimeout(render, 120); });

// Свёрнутое окно ничего не опрашивает: данные нужны только тому, кто смотрит
refresh();
setInterval(() => { if (!document.hidden) refresh(); }, 5000);
document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
