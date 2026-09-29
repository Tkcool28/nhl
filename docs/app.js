/* NHL SOG Edge v1.2 — static frontend. No dependencies. */
const $ = s => document.querySelector(s);
const LS_LOG = "sog_log_v1", LS_SET = "sog_settings_v1";
let SLATE = null, SEL = {pid: null, line: 2.5};

const settings = () => Object.assign(
  {owner: "Tkcool28", repo: "nhl", pat: ""},
  JSON.parse(localStorage.getItem(LS_SET) || "{}"));
const saveSettings = s => localStorage.setItem(LS_SET, JSON.stringify(s));
const getLog = () => JSON.parse(localStorage.getItem(LS_LOG) || "[]");
const saveLog = l => localStorage.setItem(LS_LOG, JSON.stringify(l));

/* ---------- math ---------- */
const imp = o => { o = parseFloat(o); if (!isFinite(o) || o === 0) return NaN;
  return o > 0 ? 100 / (o + 100) : -o / (-o + 100); };
const fmtOdds = o => (o > 0 ? "+" : "") + Math.round(o);
function nbPmf(mu, r, kmax) {
  const p = r / (r + mu), out = [];
  // log-space for stability
  let lg = 0;
  for (let k = 0; k <= kmax; k++) {
    if (k === 0) lg = r * Math.log(1 - p);
    else lg += Math.log(k + r - 1) - Math.log(k) + Math.log(p);
    out.push(Math.exp(lg));
  }
  return out;
}
function edgeCalc(modelP, overOdds, underOdds) {
  const io = imp(overOdds), iu = imp(underOdds);
  if (!isFinite(io) || !isFinite(iu)) return null;
  const nvO = io / (io + iu), nvU = iu / (io + iu);
  const eO = (modelP - nvO) * 100, eU = ((1 - modelP) - nvU) * 100;
  const side = eO >= eU ? "over" : "under";
  const price = side === "over" ? parseFloat(overOdds) : parseFloat(underOdds);
  const prob = side === "over" ? modelP : 1 - modelP;
  const win = price > 0 ? price : 10000 / -price; // profit per 100 staked
  const ev = prob * win - (1 - prob) * 100;
  return {nvO, nvU, eO, eU, side, ev,
    edge: side === "over" ? eO : eU, price};
}

/* ---------- views ---------- */
function nav() {
  document.querySelectorAll(".tabs button").forEach(b =>
    b.onclick = () => {
      document.querySelectorAll(".tabs button").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      ({slate: vSlate, log: vLog, settings: vSettings})[b.dataset.view]();
    });
}
function vSlate() {
  const el = $("#view");
  if (!SLATE) { el.innerHTML = `<div class="card">Loading slate…</div>`; return; }
  if (!SLATE.players.length) {
    el.innerHTML = `<div class="card"><div class="big">No games today</div>
      <div class="dim">${SLATE.note || ""}<br>Slate date: ${SLATE.slate_date}</div></div>`;
    return;
  }
  const byGame = {};
  SLATE.players.forEach(p => { (byGame[p.game_id] = byGame[p.game_id] || []).push(p); });
  let h = `<div class="dim" style="margin-bottom:6px">${SLATE.slate_date} · ${SLATE.players.length} eligible skaters · model v1.2 · μ = model's expected shots</div>`;
  Object.entries(byGame).forEach(([gid, ps]) => {
    ps.sort((a, b) => b.mu_gbm - a.mu_gbm);
    h += `<div class="game-head">${ps[0].matchup || ""}</div>`;
    ps.forEach(p => {
      h += `<button class="prow" data-pid="${p.player_id}">
        <span><span class="nm">${p.player_name}</span><br><span class="tm">${p.team} · ${p.position} · TOI ${p.toi_l10.toFixed(1)}</span></span>
        <span style="text-align:right"><span class="mu">μ ${p.mu_gbm.toFixed(2)}</span><br>
        <span class="tm">o2.5 ${fmtOdds(p.fair_odds_over_2_5)} · ${(p.p_over_2_5 * 100).toFixed(0)}%</span></span>
      </button>`;
    });
  });
  el.innerHTML = h;
  el.querySelectorAll(".prow").forEach(b =>
    b.onclick = () => openPlayer(parseInt(b.dataset.pid)));
}

function playerById(pid) { return SLATE.players.find(p => p.player_id === pid); }

function openPlayer(pid) {
  SEL = {pid, line: 2.5};
  renderSheet();
  $("#sheet").classList.remove("hidden");
  $("#sheet").onclick = e => { if (e.target.id === "sheet") $("#sheet").classList.add("hidden"); };
}

function renderSheet() {
  const p = playerById(SEL.pid), L = SEL.line;
  const key = String(L).replace(".", "_");
  const mp = p["p_over_" + key], fairO = p["fair_odds_over_" + key], fairU = p["fair_odds_under_" + key];
  const alpha = (p.alpha || {})[String(L)] || 2.5;
  const flags = (p.reliability_flags || []).filter(f => f !== "ineligible");
  const b = $("#sheet-body");
  b.innerHTML = `
    <div class="row"><div><div class="big">${p.player_name}</div>
      <div class="dim">${p.team} vs ${p.opp} · ${p.home_away} · ${p.position} · μ ${p.mu_gbm.toFixed(2)} (EB ${p.mu_eb.toFixed(2)})</div></div>
      <button class="btn-ghost btn" style="width:auto;margin:0;padding:8px 12px" id="close">✕</button></div>
    <div class="dim" style="margin:6px 0">P(over ${L}) <span class="glow" style="font-size:20px">${(mp * 100).toFixed(1)}%</span>
      · fair ${fmtOdds(fairO)} / ${fmtOdds(fairU)}</div>
    <canvas class="chart" id="dist" width="520" height="190"></canvas>
    <div class="linebtns">${[1.5, 2.5, 3.5, 4.5].map(x =>
      `<button data-l="${x}" class="${x === L ? "active" : ""}">${x}</button>`).join("")}</div>
    <div class="card"><div class="dim">MODEL CONFIDENCE</div>
      <div class="kv"><span>P(over ${L})</span><b>${(mp * 100).toFixed(1)}%</b></div>
      <div class="kv"><span>Fair odds over / under</span><b>${fmtOdds(fairO)} / ${fmtOdds(fairU)}</b></div>
      <div class="kv"><span>History depth</span><b>${p.history_depth} games</b></div>
      <div class="kv"><span>Trailing TOI / shots</span><b>${p.toi_l10.toFixed(1)} min · ${p.shots_l10.toFixed(2)}/g</b></div>
      ${flags.length ? `<div style="margin-top:6px">${flags.map(f => `<span class="flag">${f}</span>`).join("")}</div>` : ""}
      <div class="note" style="margin-top:6px">v1.2 runs slightly hot at low P(over) and cool in the 0.55–0.80 band (2025-26 holdout). Treat mid-range edges as measured, not gospel.</div>
    </div>
    <div class="card"><div class="dim">QUICK CHECK — DK STYLE</div>
      <div class="dim" style="margin:2px 0 6px">Pick the "N+" market, type the book price, compare to the model.</div>
      <div class="linebtns" id="qlines">${[["2+", 1.5], ["3+", 2.5], ["4+", 3.5], ["5+", 4.5]].map(([lbl, x]) =>
        `<button data-l="${x}" class="${x === 2.5 ? "active" : ""}">${lbl}</button>`).join("")}</div>
      <label>Book price on <span id="qlbl">3+</span> (e.g. -120)</label>
      <input type="number" id="qodds" placeholder="-120">
      <div id="qout"><div class="note">Type the price to see the model's number for it.</div></div>
      <button class="btn btn-log" id="qlog">📝 Log this line</button>
      <div class="note" style="margin-top:6px">Single-price check: edge is vs the raw book price. Books bake vig into it, so the true no-vig edge runs a touch higher than shown.</div>
    </div>
    <div class="card"><div class="dim">BOOK ODDS → CHECK THE SPOT (both sides, de-vigged)</div>
      <div class="row" style="gap:8px">
        <div style="flex:1"><label>Over ${L}</label><input type="number" id="bo" placeholder="-110"></div>
        <div style="flex:1"><label>Under ${L}</label><input type="number" id="bu" placeholder="-110"></div>
      </div>
      <div id="evout"></div>
    </div>
    <button class="btn btn-log" id="logline">📝 Log this line</button>
    <div class="note" style="margin-top:8px">v1 is a measurement instrument — log everything, bet on nothing until the sample speaks. 8pp+ gaps usually mean missing info: check news first.</div>`;
  $("#close").onclick = () => $("#sheet").classList.add("hidden");
  b.querySelectorAll(".linebtns button").forEach(x =>
    x.onclick = () => { SEL.line = parseFloat(x.dataset.l); renderSheet(); });
  drawDist(p, L, alpha);
  const upd = () => {
    const r = edgeCalc(mp, $("#bo").value, $("#bu").value);
    const o = $("#evout");
    if (!r) { o.innerHTML = `<div class="note">Type both sides to de-vig and check the spot.</div>`; return; }
    const good = r.ev > 0;
    const w = Math.min(100, Math.abs(r.ev) / 2);
    o.innerHTML = `
      <div class="kv"><span>No-vig book P(${r.side})</span><b>${((r.side === "over" ? r.nvO : r.nvU) * 100).toFixed(1)}%</b></div>
      <div class="kv"><span>Model − market edge</span><b style="color:${r.edge >= 0 ? "var(--good)" : "var(--bad)"}">${r.edge >= 0 ? "+" : ""}${r.edge.toFixed(1)}pp ${r.side === "over" ? "▲ over" : "▼ under"}</b></div>
      <div class="evbar"><div style="width:${w}%;background:${good ? "var(--good)" : "var(--bad)"}"></div></div>
      <div class="verdict ${good ? "good" : "bad"}">${good ? "✓ VALUE SPOT" : "✕ NO EDGE"} · EV ${r.ev >= 0 ? "+" : ""}${r.ev.toFixed(1)} / $100</div>`;
    o.dataset.calc = JSON.stringify(r);
  };
  $("#bo").oninput = upd; $("#bu").oninput = upd; upd();
  $("#logline").onclick = () => logLine(p, L, mp, fairO);

  /* ---- DK-style quick check: single "N+" price vs model ---- */
  let qline = 2.5;
  const qkey = () => String(qline).replace(".", "_");
  const qlbl = () => ({1.5: "2+", 2.5: "3+", 3.5: "4+", 4.5: "5+"})[qline];
  const qupd = () => {
    const mp = p["p_over_" + qkey()], fairO = p["fair_odds_over_" + qkey()];
    $("#qlbl").textContent = qlbl();
    const o = $("#qout"), price = parseFloat($("#qodds").value), io = imp(price);
    if (!isFinite(io)) {
      o.innerHTML = `<div class="note">Type the price to see the model's number for it.</div>`;
      o.dataset.calc = ""; return;
    }
    const edge = (mp - io) * 100;
    const win = price > 0 ? price : 10000 / -price;
    const ev = mp * win - (1 - mp) * 100;
    const good = edge > 0;
    o.innerHTML = `
      <div class="kv"><span>Model P(${qlbl()} shots)</span><b class="glow">${(mp * 100).toFixed(1)}%</b></div>
      <div class="kv"><span>Model fair price</span><b>${fmtOdds(fairO)}</b></div>
      <div class="kv"><span>Book ${fmtOdds(price)} implies</span><b>${(io * 100).toFixed(1)}%</b></div>
      <div class="kv"><span>Model − book edge</span><b style="color:${edge >= 0 ? "var(--good)" : "var(--bad)"}">${edge >= 0 ? "+" : ""}${edge.toFixed(1)}pp</b></div>
      <div class="verdict ${good ? "good" : "bad"}">${good ? "✓ VALUE" : "✕ NO EDGE"} · EV ${ev >= 0 ? "+" : ""}${ev.toFixed(1)} / $100</div>`;
    o.dataset.calc = JSON.stringify({mp, fairO, price, io, edge, ev});
  };
  b.querySelectorAll("#qlines button").forEach(x =>
    x.onclick = () => {
      b.querySelectorAll("#qlines button").forEach(y => y.classList.remove("active"));
      x.classList.add("active"); qline = parseFloat(x.dataset.l); qupd();
    });
  $("#qodds").oninput = qupd;
  $("#qlog").onclick = () => {
    const c = JSON.parse($("#qout").dataset.calc || "null");
    if (!c) { alert("Type the book price first."); return; }
    const obs = {
      observation_id: uid(),
      observed_at: new Date().toISOString(),
      run_id: SLATE.run_id || ("slate-" + SLATE.slate_date),
      slate_date: SLATE.slate_date,
      game_id: p.game_id, player_id: p.player_id, player_name: p.player_name,
      team: p.team, opp: p.opp, home_away: p.home_away, position: p.position,
      line: qline, market_label: qlbl(),
      over_price: c.price, under_price: null,
      imp_over: +c.io.toFixed(4), imp_under: null,
      devig_over: null, devig_under: null,
      model_p_over: +c.mp.toFixed(4), model_fair_over: c.fairO,
      edge_pp: +c.edge.toFixed(2),
      edge_direction: c.edge >= 0 ? "model_over" : "model_under",
      edge_basis: "raw_implied_single_side",
      ev_per_100: +c.ev.toFixed(1),
      baseline_version: "v1.2", eligible: true, synced: false,
    };
    const log = getLog(); log.push(obs); saveLog(log);
    syncLog();
    alert(`Logged: ${p.player_name} ${qlbl()} @ ${fmtOdds(c.price)} (edge ${c.edge >= 0 ? "+" : ""}${c.edge.toFixed(1)}pp vs raw price)`);
  };
}

function drawDist(p, L, alpha) {
  const cv = $("#dist"), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height, mu = p.mu_gbm;
  const pmf = nbPmf(mu, alpha, 14);
  const mx = Math.max(...pmf);
  ctx.clearRect(0, 0, W, H);
  const bw = W / 15;
  let pover = 0;
  for (let k = 0; k <= 14; k++) {
    const hgt = (pmf[k] / mx) * (H - 44);
    const x = k * bw + 4, y = H - 24 - hgt;
    const isOver = k > L;
    if (isOver) pover += pmf[k];
    ctx.fillStyle = isOver ? "rgba(79,195,247,.85)" : "rgba(139,148,179,.4)";
    ctx.fillRect(x, y, bw - 8, hgt);
    ctx.fillStyle = "#8b94b3"; ctx.font = "10px sans-serif"; ctx.textAlign = "center";
    ctx.fillText(k, x + (bw - 8) / 2, H - 8);
  }
  const lx = (L + 0.5) * bw + 4;
  ctx.strokeStyle = "#ffb84d"; ctx.setLineDash([5, 4]); ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(lx, 8); ctx.lineTo(lx, H - 24); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#ffb84d"; ctx.font = "bold 11px sans-serif"; ctx.textAlign = "left";
  ctx.fillText("line " + L, lx + 5, 20);
  ctx.fillStyle = "#aee2ff"; ctx.font = "bold 13px sans-serif";
  ctx.shadowColor = "#4fc3f7"; ctx.shadowBlur = 12;
  ctx.fillText("P(over) " + (pover * 100).toFixed(1) + "%", lx + 5, 38);
  ctx.shadowBlur = 0;
}

/* ---------- logging ---------- */
function uid() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 8); }
function logLine(p, L, mp, fairO) {
  const calc = JSON.parse($("#evout").dataset.calc || "null");
  if (!calc) { alert("Type both book odds first — the log needs the market snapshot."); return; }
  const obs = {
    observation_id: uid(),
    observed_at: new Date().toISOString(),
    run_id: SLATE.run_id || ("slate-" + SLATE.slate_date),
    slate_date: SLATE.slate_date,
    game_id: p.game_id, player_id: p.player_id, player_name: p.player_name,
    team: p.team, opp: p.opp, home_away: p.home_away, position: p.position,
    line: L, over_price: parseFloat($("#bo").value), under_price: parseFloat($("#bu").value),
    imp_over: +imp($("#bo").value).toFixed(4), imp_under: +imp($("#bu").value).toFixed(4),
    devig_over: +calc.nvO.toFixed(4), devig_under: +calc.nvU.toFixed(4),
    model_p_over: +mp.toFixed(4), model_fair_over: fairO,
    edge_pp: +calc.edge.toFixed(2),
    edge_direction: calc.side === "over" ? "model_over" : "model_under",
    ev_per_100: +calc.ev.toFixed(1),
    baseline_version: "v1.2", eligible: true, synced: false,
  };
  const log = getLog(); log.push(obs); saveLog(log);
  syncLog();
  alert(`Logged: ${p.player_name} o${L} @ ${$("#bo").value} (edge ${calc.edge >= 0 ? "+" : ""}${calc.edge.toFixed(1)}pp ${calc.side})`);
}

async function gh(path, method = "GET", body = null) {
  const s = settings();
  if (!s.pat) throw new Error("no-pat");
  const r = await fetch(`https://api.github.com/repos/${s.owner}/${s.repo}/contents/${path}`, {
    method, headers: {Authorization: "Bearer " + s.pat, Accept: "application/vnd.github+json",
      "Content-Type": "application/json"},
    body: body ? JSON.stringify(body) : null,
  });
  if (!r.ok) { const t = await r.text(); throw new Error(r.status + " " + t.slice(0, 120)); }
  return r.json();
}
const b64 = s => btoa(unescape(encodeURIComponent(s)));
const ub64 = s => decodeURIComponent(escape(atob(s)));

async function syncLog() {
  const s = settings(), log = getLog();
  const pending = log.filter(o => !o.synced);
  if (!pending.length) return;
  if (!s.pat) { vLog(); return; }
  const byDay = {};
  pending.forEach(o => { (byDay[o.slate_date] = byDay[o.slate_date] || []).push(o); });
  for (const [day, obs] of Object.entries(byDay)) {
    const path = `data/market_log/obs_${day}.jsonl`;
    let sha = null, cur = "";
    try { const f = await gh(path); sha = f.sha; cur = ub64(f.content); }
    catch (e) { if (!String(e.message).startsWith("404")) throw e; }
    const lines = obs.map(o => { const {synced, ...rest} = o; return JSON.stringify(rest); });
    const next = cur + lines.join("\n") + "\n";
    const put = await gh(path, "PUT", {message: `market log ${day} (${obs.length} obs)`, content: b64(next), ...(sha ? {sha} : {})});
    obs.forEach(o => o.synced = true);
    saveLog(log);
  }
}

function vLog() {
  const el = $("#view"), log = getLog().slice().reverse();
  const s = settings();
  const nPred = SLATE && SLATE.players ? SLATE.players.length : 0;
  el.innerHTML = `<div class="card"><div class="big">📝 Auto-log</div>
    <div class="dim">Every skater on today's dash is prediction-logged automatically — no taps needed.
    <b style="color:var(--ice)">${nPred} predictions</b> for ${SLATE ? SLATE.slate_date : "—"} are in the repo's immutable log
    (<span style="font-family:monospace">data/market_log/predictions_*.jsonl</span>).</div></div>
  <div class="card"><div class="row">
      <div><div class="big">${log.length} market snapshots</div>
      <div class="dim">${log.filter(o => o.synced).length} synced to GitHub${s.pat ? "" : " · <b>no PAT set</b> — logging locally only"}</div></div>
      <button class="btn-ghost btn" style="width:auto;margin:0" id="syncbtn">↻ Sync</button></div>
    <div style="margin-top:8px" class="row"><button class="btn-ghost btn" style="margin:0" id="exp">⬇ Export JSON</button></div>
    <div class="note" style="margin-top:8px">Book-odds snapshots still need one tap when you check a spot — only you see your book's prices. Sync writes append-only JSONL to <span style="font-family:monospace">data/market_log/</span>; the morning job auto-joins results.</div></div>
    ${log.map(o => `<div class="logrow"><span class="pill ${o.synced ? "sync" : "pend"}">${o.synced ? "synced" : "pending"}</span>
      <b>${o.player_name}</b> o${o.line} @ ${o.over_price > 0 ? "+" : ""}${o.over_price}
      <span class="dim">edge ${o.edge_pp >= 0 ? "+" : ""}${o.edge_pp}pp ${o.edge_direction === "model_over" ? "▲" : "▼"} · EV ${o.ev_per_100 >= 0 ? "+" : ""}${o.ev_per_100}/$100 · ${o.slate_date}</span></div>`).join("") || `<div class="dim">Nothing logged yet.</div>`}`;
  $("#syncbtn").onclick = () => syncLog().then(vLog).catch(e => alert("Sync failed: " + e.message));
  $("#exp").onclick = () => {
    const blob = new Blob([getLog().map(o => { const {synced, ...r } = o; return JSON.stringify(r); }).join("\n")], {type: "application/json"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "sog_market_log.jsonl"; a.click();
  };
}

function vSettings() {
  const s = settings(), el = $("#view");
  el.innerHTML = `<div class="card"><div class="big">Settings</div>
    <label>GitHub owner</label><input type="text" id="so" value="${s.owner}">
    <label>Repo</label><input type="text" id="sr" value="${s.repo}">
    <label>Fine-grained PAT <span class="dim">(repo → Contents: read+write, stored only in this browser)</span></label>
    <input type="password" id="sp" value="${s.pat}" placeholder="github_pat_…">
    <button class="btn btn-go" id="ssave">Save</button>
    <button class="btn-ghost btn" id="stest">Test connection</button>
    <div class="note" style="margin-top:10px">The PAT never leaves your browser except to talk to GitHub's API. Create one at github.com → Settings → Developer settings → Personal access tokens → Fine-grained → this repo → Contents: Read and write.</div></div>`;
  $("#ssave").onclick = () => { saveSettings({owner: $("#so").value.trim(), repo: $("#sr").value.trim(), pat: $("#sp").value.trim()}); alert("Saved."); };
  $("#stest").onclick = async () => {
    try { const me = await (await fetch("https://api.github.com/user", {headers: {Authorization: "Bearer " + $("#sp").value.trim()}})).json();
      alert(me.login ? "Connected as @" + me.login : "Failed: " + JSON.stringify(me).slice(0, 100)); }
    catch (e) { alert("Failed: " + e.message); }
  };
}

/* ---------- boot ---------- */
fetch("data/latest.json").then(r => r.json()).then(s => { SLATE = s; vSlate(); })
  .catch(() => { $("#view").innerHTML = `<div class="card">Couldn't load slate data.</div>`; });
nav();
