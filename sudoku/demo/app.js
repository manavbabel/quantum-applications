/* Sudoku annealing demo: pick -> solve -> anneal -> send to the quantum annealer.
   Data comes from data.js (window.DEMO), written by precompute.py.

   URL hash hook, for screenshots and staff checks:
     #size=9&puzzle=0&screen=solve              blurred grid before Start
     #size=9&puzzle=0&screen=solve&fill=12      started, 12 squares filled plus one clash
     #size=4&puzzle=0&screen=solve&fill=all     finished
     #size=9&puzzle=0&screen=anneal&t=0.5       thermometer at t (0 hot, 1 cold), pencilling skipped
     #size=6&puzzle=1&screen=send&stage=2       send, stopping on stage 2
*/
"use strict";

var DEMO = window.DEMO;
var FILE = location.protocol === "file:";
var STAGE_MIN = 2500;
// the thermometer knob runs from 4% (HOT) to 95% (COLD) of the tube
var KNOB_TOP = .04, KNOB_SPAN = .91;

function $(id) { return document.getElementById(id); }
function el(tag, cls, parent) {
  var e = document.createElement(tag);
  if (cls) e.className = cls;
  if (parent) parent.appendChild(e);
  return e;
}
function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
function smooth(a, b, x) { var u = clamp((x - a) / (b - a), 0, 1); return u * u * (3 - 2 * u); }
function sleep(ms) { return new Promise(function (res) { setTimeout(res, ms); }); }
function fmt(x) { return Math.round(x).toLocaleString("en-GB"); }

var state = { screen: null, size: "9", index: 0, P: null };

/* ---------- screens ---------- */

function show(name) {
  var prev = state.screen;
  if (prev === "solve") pauseTimer();
  if (prev === "anneal") leaveAnneal();
  if (prev === "send" && name !== "send") stopSend();
  state.screen = name;
  document.querySelectorAll(".screen").forEach(function (s) { s.classList.toggle("on", s.id === name); });
  if (name === "solve") resumeTimer();
  if (name === "anneal") enterAnneal();
}

function loadPuzzle(size, index) {
  state.size = String(size);
  state.index = index;
  state.P = DEMO.puzzles[state.size][index];
  buildSolve();
  an.P = null;
}

function newPuzzle() {
  var list = DEMO.puzzles[state.size];
  var i = Math.floor(Math.random() * list.length);
  if (list.length > 1 && i === state.index) i = (i + 1 + Math.floor(Math.random() * (list.length - 1))) % list.length;
  loadPuzzle(state.size, i);
  show("solve");
}

function back() {
  if (state.screen === "solve") show("pick");
  else if (state.screen === "anneal") show("solve");
  else if (state.screen === "send") show("anneal");
}

/* ---------- grids ---------- */

// Thin printed cell lines and thick box lines, drawn over the cells in grid units.
function linesSvg(n, br, bc) {
  function line(cls, x1, y1, x2, y2) {
    return '<line class="' + cls + '" x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '" vector-effect="non-scaling-stroke"/>';
  }
  var thin = "", thick = "";
  for (var i = 1; i < n; i++) {
    if (i % bc) thin += line("thin", i, 0, i, n); else thick += line("thick", i, 0, i, n);
    if (i % br) thin += line("thin", 0, i, n, i); else thick += line("thick", 0, i, n, i);
  }
  return '<svg class="lines" viewBox="0 0 ' + n + " " + n + '" preserveAspectRatio="none">' + thin + thick +
    '<rect class="edge" x="0" y="0" width="' + n + '" height="' + n + '" vector-effect="non-scaling-stroke"/></svg>';
}

function buildGrid(P, slot) {
  var n = P.n;
  slot.innerHTML = "";
  var root = el("div", "grid", slot);
  root.style.setProperty("--n", n);
  var cells = [];
  for (var r = 0; r < n; r++) {
    cells.push([]);
    for (var c = 0; c < n; c++) {
      var cell = el("div", "cell", root);
      cell.style.setProperty("--r", r);
      cell.style.setProperty("--c", c);
      cell.dataset.r = r;
      cell.dataset.c = c;
      el("div", "bg", cell);
      var val = el("div", "val", cell);
      if (P.grid[r][c]) { cell.classList.add("clue"); val.textContent = P.grid[r][c]; }
      el("div", "rip", cell);
      cells[r].push({ el: cell, val: val });
    }
  }
  root.insertAdjacentHTML("beforeend", linesSvg(n, P.box[0], P.box[1]));
  return { root: root, cells: cells };
}

function setSheetHead(sheet, P) {
  var head = sheet.querySelector(".sheet-head");
  if (head) head.innerHTML = "<span>Sudoku No. " + (state.index + 1) + "</span><span>" + P.n + " &times; " + P.n + "</span>";
}

function sameUnit(P, r1, c1, r2, c2) {
  return r1 === r2 || c1 === c2 ||
    (Math.floor(r1 / P.box[0]) === Math.floor(r2 / P.box[0]) && Math.floor(c1 / P.box[1]) === Math.floor(c2 / P.box[1]));
}

/* ---------- 1. pick ---------- */

function buildPick() {
  document.querySelectorAll(".card-grid").forEach(function (slot) {
    var n = +slot.dataset.n, P = DEMO.puzzles[n][0];
    var s = linesSvg(n, P.box[0], P.box[1]).replace("</svg>", "");
    for (var r = 0; r < n; r++) {
      for (var c = 0; c < n; c++) {
        if (P.grid[r][c]) {
          s += '<text x="' + (c + .5) + '" y="' + (r + .5) + '" font-size=".58" text-anchor="middle" dominant-baseline="central">' + P.grid[r][c] + "</text>";
        }
      }
    }
    slot.innerHTML = s + "</svg>";
  });
  document.querySelectorAll(".stars").forEach(function (s) {
    var lit = +s.className.match(/s(\d)/)[1];
    for (var i = 0; i < 3; i++) s.insertAdjacentHTML("beforeend", '<svg class="' + (i < lit ? "lit" : "") + '"><use href="#i-star"/></svg>');
  });
  document.querySelectorAll(".card").forEach(function (card) {
    card.addEventListener("click", function () {
      var list = DEMO.puzzles[card.dataset.size];
      loadPuzzle(card.dataset.size, Math.floor(Math.random() * list.length));
      show("solve");
    });
  });
}

/* ---------- 2. solve ---------- */

var sv = {};

function buildSolve() {
  var P = state.P;
  var sheet = $("solve-sheet");
  var g = buildGrid(P, sheet.querySelector(".grid-slot"));
  sv = {
    grid: g, entries: P.grid.map(function (row) { return row.slice(); }),
    sel: null, started: false, solved: false, running: false, acc: 0, t0: 0, raf: 0, shown: ""
  };
  g.root.addEventListener("pointerdown", function (e) {
    var cell = e.target.closest(".cell");
    if (cell && sv.started && !sv.solved) select(+cell.dataset.r, +cell.dataset.c);
  });
  setSheetHead(sheet, P);
  sheet.classList.remove("started");
  $("solve").classList.remove("started", "solved");
  var btn = $("anneal-it-btn");
  btn.classList.remove("primary", "attn");
  buildPad(P.n);
  setTimerText(0);
}

function buildPad(n) {
  var pad = $("pad");
  pad.innerHTML = "";
  pad.style.setProperty("--pc", n === 4 ? 2 : 3);
  for (var v = 1; v <= n; v++) {
    var k = el("button", "key digit", pad);
    k.textContent = v;
    k.dataset.v = v;
  }
  var e = el("button", "key erase", pad);
  e.dataset.v = 0;
  e.innerHTML = '<svg><use href="#i-erase"/></svg>Erase';
}

function startSolve() {
  if (sv.started) return;
  sv.started = true;
  $("solve-sheet").classList.add("started");
  $("solve").classList.add("started");
  var first = state.P.cells[0];
  select(first[0], first[1]);
  sv.t0 = performance.now();
  sv.running = true;
  tick();
}

function select(r, c) {
  var P = state.P;
  sv.sel = [r, c];
  for (var i = 0; i < P.n; i++) {
    for (var j = 0; j < P.n; j++) {
      var cl = sv.grid.cells[i][j].el.classList;
      cl.toggle("sel", i === r && j === c);
      cl.toggle("peer", sameUnit(P, r, c, i, j) && !(i === r && j === c));
    }
  }
}

function moveSel(dr, dc) {
  if (!sv.sel) return;
  var n = state.P.n;
  select((sv.sel[0] + dr + n) % n, (sv.sel[1] + dc + n) % n);
}

function setEntry(v) {
  if (!sv.started || sv.solved || !sv.sel) return;
  var r = sv.sel[0], c = sv.sel[1];
  if (state.P.grid[r][c]) return;
  sv.entries[r][c] = v;
  var cell = sv.grid.cells[r][c];
  cell.val.textContent = v || "";
  cell.el.classList.toggle("ink", v > 0);
  checkSolve();
}

function checkSolve() {
  var P = state.P, n = P.n, g = sv.entries, full = true, anyBad = false;
  for (var r = 0; r < n; r++) {
    for (var c = 0; c < n; c++) {
      var v = g[r][c], bad = false;
      if (!v) full = false;
      else {
        for (var i = 0; i < n && !bad; i++) {
          for (var j = 0; j < n && !bad; j++) {
            if ((i !== r || j !== c) && g[i][j] === v && sameUnit(P, r, c, i, j)) bad = true;
          }
        }
      }
      if (bad) anyBad = true;
      sv.grid.cells[r][c].el.classList.toggle("clash", bad);
    }
  }
  if (full && !anyBad) finishSolve();
}

function finishSolve() {
  pauseTimer();
  sv.solved = true;
  var secs = (Math.floor(sv.acc / 100) / 10).toFixed(1);
  setTimerText(sv.acc);
  $("done-time").textContent = secs + " s";
  $("solve").classList.add("solved");
  sv.grid.cells.forEach(function (row) { row.forEach(function (x) { x.el.classList.remove("sel", "peer"); }); });
  sv.grid.root.classList.add("ripple");
  var btn = $("anneal-it-btn");
  btn.classList.add("primary", "attn");
}

function elapsed() { return sv.acc + (sv.running ? performance.now() - sv.t0 : 0); }
function tick() {
  if (!sv.running) return;
  setTimerText(elapsed());
  sv.raf = requestAnimationFrame(tick);
}
function setTimerText(ms) {
  var s = (Math.floor(ms / 100) / 10).toFixed(1);
  if (s !== sv.shown) { sv.shown = s; $("timer").firstChild.nodeValue = s; }
}
function pauseTimer() {
  if (!sv.running) return;
  sv.acc = elapsed();
  sv.running = false;
  cancelAnimationFrame(sv.raf);
}
function resumeTimer() {
  if (sv.started && !sv.solved && !sv.running) {
    sv.t0 = performance.now();
    sv.running = true;
    tick();
  }
}

/* ---------- 3. anneal ---------- */

var an = { P: null, timers: [] };

function buildAnneal() {
  var P = state.P, n = P.n, br = P.box[0], bc = P.box[1];
  var sheet = $("anneal-sheet");
  var g = buildGrid(P, sheet.querySelector(".grid-slot"));
  setSheetHead(sheet, P);
  g.root.classList.add("pre");
  g.root.style.setProperty("--br", br);
  g.root.style.setProperty("--bc", bc);
  g.root.style.setProperty("--bmax", Math.max(br, bc));
  var index = {};
  P.variables.forEach(function (x, i) { index[x[0] + "," + x[1] + "," + x[2]] = i; });
  var marks = new Array(P.variables.length), heats = [];
  P.cells.forEach(function (rc, k) {
    var r = rc[0], c = rc[1], cell = g.cells[r][c];
    heats.push(cell.el.insertBefore(el("div", "heat"), cell.val));
    var box = el("div", "marks", cell.el);
    for (var v = 1; v <= n; v++) {
      var mk = el("span", "mk", box);
      mk.style.left = ((v - 1) % bc) * 100 / bc + "%";
      mk.style.top = Math.floor((v - 1) / bc) * 100 / br + "%";
      mk.style.setProperty("--d", k * 14 + "ms");
      var mi = el("span", "mi", mk);
      mi.textContent = v;
      var i = index[r + "," + c + "," + v];
      if (i === undefined) mk.classList.add("x");
      else {
        marks[i] = mi;
        mk.classList.add("live");
        if (Math.random() < .5) mk.classList.add("alt");
        mk.style.animationDelay = (-Math.random()).toFixed(3) + "s";
        mk.style.animationDuration = (.42 + Math.random() * .26).toFixed(3) + "s";
      }
    }
    var s = P.solution[r][c];
    var fin = el("div", "fin", cell.el);
    fin.innerHTML = "<span>" + s + "</span>";
    // grow out of the pencil mark's position in the cell
    fin.style.transformOrigin = (5 + 90 * ((s - 1) % bc + .5) / bc).toFixed(1) + "% " + (5 + 90 * (Math.floor((s - 1) / bc) + .5) / br).toFixed(1) + "%";
  });
  // the hottest cell anywhere in the recording shows as strong red
  var hmax = 0;
  P.frames.forEach(function (f) { f.heat.forEach(function (h) { hmax = Math.max(hmax, h); }); });
  an.heatGain = hmax > 0 ? .7 / hmax : 0;
  an.pace = pace(P.frames);
  an.P = P;
  an.grid = g;
  an.marks = marks;
  an.heats = heats;
  an.lastP = new Float32Array(marks.length).fill(-1);
  an.lastH = new Float32Array(heats.length).fill(-1);
  an.t = 0;
  an.cold = false;
  an.introDone = false;
  an.amp = -1;
  an.status = "";
  an.raf = 0;
  an.auto = 0;
  // jitter amplitude in px at full heat, about 5% of a cell
  an.ampMax = $("anneal-sheet").querySelector(".grid-slot").clientWidth / n * .055;
  $("anneal-stamp").classList.remove("on");
  $("anneal-btn").classList.remove("hidden");
  $("anneal-btn").classList.add("attn");
  $("send-btn").classList.add("hidden");
  setStep("pencil");
  renderAnneal();
}

// Thermometer position of each recorded frame: 35% evenly spaced, 65% by how much the
// marks and heat change between frames, so stretches where nothing happens take less travel.
function pace(frames) {
  var steps = [], total = 0, F = frames.length - 1;
  for (var k = 1; k <= F; k++) {
    var a = 0, p0 = frames[k - 1].p, p1 = frames[k].p, h0 = frames[k - 1].heat, h1 = frames[k].heat;
    for (var i = 0; i < p0.length; i++) a += Math.abs(p1[i] - p0[i]);
    for (i = 0; i < h0.length; i++) a += Math.abs(h1[i] - h0[i]);
    steps.push(a);
    total += a;
  }
  var s = [0], c = 0;
  for (k = 1; k <= F; k++) {
    c += steps[k - 1];
    s.push(.35 * k / F + .65 * (total > 0 ? c / total : k / F));
  }
  return s;
}

// Fractional frame index at thermometer position t.
function frameAt(t) {
  var s = an.pace, k = 0;
  while (k < s.length - 2 && s[k + 1] <= t) k++;
  return k + clamp((t - s[k]) / (s[k + 1] - s[k] || 1), 0, 1);
}

function setStep(which) {
  $("step-pencil").className = which === "pencil" ? "on" : "done";
  $("step-cool").className = which === "cool" ? "on" : "";
}

function enterAnneal(skipIntro) {
  if (an.P !== state.P) {
    buildAnneal();
    if (skipIntro) finishIntro(true);
    else runIntro();
  } else if (an.introDone) {
    an.grid.root.classList.add("jitter");
    if (an.cold) $("anneal-stamp").classList.add("on");
  } else {
    runIntro();
  }
}

function leaveAnneal() {
  stopAuto();
  clearIntro();
  if (an.grid) an.grid.root.classList.remove("jitter");
}

function clearIntro() {
  an.timers.forEach(clearTimeout);
  an.timers = [];
}

// The classical step: pencil in every number, then strike out the ones already used nearby.
function runIntro() {
  clearIntro();
  var g = an.grid.root, spread = an.P.cells.length * 14;
  g.classList.add("pre");
  g.classList.remove("strike", "jitter");
  setStatus("Pencilling in every number that could go in each square...");
  an.timers.push(setTimeout(function () { g.classList.remove("pre"); }, 60));
  // each cell strikes out 14 ms after the previous one, and its marks fade for 1.45 s
  an.timers.push(setTimeout(function () { g.classList.add("strike"); }, 500 + spread));
  an.timers.push(setTimeout(function () { finishIntro(false); }, 2000 + 2 * spread));
}

function finishIntro(instant) {
  clearIntro();
  var g = an.grid.root;
  g.classList.remove("pre");
  g.classList.add("strike", "jitter");
  if (!instant) {
    g.classList.add("settle");
    an.timers.push(setTimeout(function () { g.classList.remove("settle"); }, 800));
  }
  an.introDone = true;
  an.lastP.fill(-1);
  an.lastH.fill(-1);
  an.status = "";
  setStep("cool");
  renderAnneal();
}

function setT(t) {
  an.t = clamp(t, 0, 1);
  if (!an.raf) an.raf = requestAnimationFrame(renderAnneal);
}

function setStatus(text) {
  if (text !== an.status) { an.status = text; $("anneal-status").textContent = text; }
}

function renderAnneal() {
  an.raf = 0;
  var P = an.P, t = an.t;
  var F = P.frames.length, f = frameAt(t), i0 = Math.floor(f), i1 = Math.min(i0 + 1, F - 1), a = f - i0;
  if (an.introDone) {
    var p0 = P.frames[i0].p, p1 = P.frames[i1].p;
    for (var i = 0; i < an.marks.length; i++) {
      var p = p0[i] + a * (p1[i] - p0[i]);
      if (Math.abs(p - an.lastP[i]) > .004) {
        an.lastP[i] = p;
        an.marks[i].style.transform = "scale(" + (.4 + .8 * p).toFixed(3) + ")";
        an.marks[i].style.opacity = (.12 + .88 * p).toFixed(3);
      }
    }
    var h0 = P.frames[i0].heat, h1 = P.frames[i1].heat;
    for (var k = 0; k < an.heats.length; k++) {
      var o = Math.min(1, (h0[k] + a * (h1[k] - h0[k])) * an.heatGain);
      if (Math.abs(o - an.lastH[k]) > .005) {
        an.lastH[k] = o;
        an.heats[k].style.opacity = o.toFixed(3);
      }
    }
    // ten steps of amplitude: each change restyles every jittering mark
    var amp = an.ampMax * Math.ceil((1 - t) * 10) / 10;
    if (amp !== an.amp) {
      an.amp = amp;
      an.grid.root.style.setProperty("--amp", amp.toFixed(2));
    }
    if (t < .12) setStatus("Hot: the qubits jiggle and flip between numbers.");
    else if (t < .55) setStatus("Cooling: the jiggling calms and squares start to settle.");
    else if (t < .999) setStatus("Colder: the last squares settle on numbers that break no rules.");
    else setStatus("Cold: every square has settled on one number, and no rules are broken.");
  }
  // thermometer: liquid top and knob share the same offset, as a share of the tube
  var off = "translateY(" + ((KNOB_TOP + KNOB_SPAN * t) * 100).toFixed(2) + "%)";
  $("liquid").style.transform = off;
  $("knob-rail").style.transform = off;
  // red fades out before blue fades in, through the steel of the base layer
  var hot = (1 - smooth(.25, .6, t)).toFixed(3), cold = smooth(.4, .75, t).toFixed(3);
  $("liq-hot").style.opacity = hot;
  $("bulb-hot").style.opacity = hot;
  $("liq-cold").style.opacity = cold;
  $("bulb-cold").style.opacity = cold;
  $("thermo").setAttribute("aria-valuenow", Math.round((1 - t) * 100));
  if (an.introDone && t >= .999 && !an.cold) goCold();
  else if (an.cold && t < .995) unCold();
}

function goCold() {
  an.cold = true;
  var g = an.grid.root;
  g.classList.remove("ripple");
  g.classList.add("cold");
  void g.offsetWidth; // restart the ripple animation
  g.classList.add("ripple");
  an.timers.push(setTimeout(function () { if (an.cold) $("anneal-stamp").classList.add("on"); }, 2 * an.P.n * 55 + 300));
  $("anneal-btn").classList.add("hidden");
  $("send-btn").classList.remove("hidden");
  $("send-btn").classList.add("primary", "attn");
}

function unCold() {
  an.cold = false;
  an.grid.root.classList.remove("cold", "ripple");
  $("anneal-stamp").classList.remove("on");
  $("anneal-btn").classList.remove("hidden", "attn");
  $("send-btn").classList.remove("primary", "attn");
}

function autoAnneal() {
  if (!an.P || an.cold) return;
  if (!an.introDone) finishIntro(true);
  stopAuto();
  $("anneal-btn").classList.remove("attn");
  var t0 = an.t, start = performance.now(), dur = 6000 * (1 - t0);
  function step(now) {
    var u = clamp((now - start) / dur, 0, 1);
    var e = u < .5 ? 2 * u * u : 1 - Math.pow(2 - 2 * u, 2) / 2;
    setT(t0 + (1 - t0) * e);
    an.auto = u < 1 ? requestAnimationFrame(step) : 0;
  }
  an.auto = requestAnimationFrame(step);
}

function stopAuto() {
  if (an.auto) cancelAnimationFrame(an.auto);
  an.auto = 0;
}

function wireThermo() {
  var thermo = $("thermo"), tube = $("tube"), drag = null;
  function at(e) { setT(((e.clientY - drag.top) / drag.h - KNOB_TOP) / KNOB_SPAN); }
  thermo.addEventListener("pointerdown", function (e) {
    if (!an.P) return;
    stopAuto();
    if (!an.introDone) finishIntro(true);
    $("anneal-btn").classList.remove("attn");
    var r = tube.getBoundingClientRect();
    drag = { top: r.top, h: r.height };
    thermo.setPointerCapture(e.pointerId);
    at(e);
  });
  thermo.addEventListener("pointermove", function (e) { if (drag) at(e); });
  function end() { drag = null; }
  thermo.addEventListener("pointerup", end);
  thermo.addEventListener("pointercancel", end);
  thermo.addEventListener("lostpointercapture", end);
}

/* ---------- 4. send ---------- */

var sd = { token: 0 };
var api = { live: false };

function checkStatus() {
  if (FILE) return;
  fetch("/api/status", { cache: "no-store" })
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (j) { api.live = !!(j && j.live); })
    .catch(function () { api.live = false; });
}

function requestSample(P, index) {
  var fallback = { source: "simulation", energies: P.fallback.energies, best: P.fallback.best };
  if (FILE || !api.live) return Promise.resolve(fallback);
  var ctrl = new AbortController();
  var timer = setTimeout(function () { ctrl.abort(); }, 20000);
  return fetch("/api/sample", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ size: P.n, index: index }),
    signal: ctrl.signal
  })
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (j) {
      var ok = j && j.source === "dwave" && Array.isArray(j.energies) && j.energies.length > 0 &&
        Array.isArray(j.best) && j.best.length === P.variables.length;
      return ok ? j : fallback;
    })
    .catch(function () { return fallback; })
    .finally(function () { clearTimeout(timer); });
}

function stopSend() {
  sd.token++;
  document.querySelector(".pulse").classList.remove("go");
}

function startSend(hold) {
  var token = ++sd.token, P = state.P;
  document.querySelectorAll("#stages li").forEach(function (li) { li.className = ""; });
  document.querySelectorAll(".view").forEach(function (v) { v.classList.remove("on"); });
  $("again-btn").classList.add("hidden");
  setCaption("");
  var reply = requestSample(P, state.index);
  runStages(token, P, reply, hold || 0);
}

async function runStages(token, P, reply, hold) {
  var stages = [stage1, stage2, stage3, stage4, stage5];
  var items = document.querySelectorAll("#stages li");
  var result = null;
  for (var k = 0; k < stages.length; k++) {
    if (token !== sd.token) return;
    items[k].className = "on";
    document.querySelectorAll(".view").forEach(function (v, i) { v.classList.toggle("on", i === k); });
    var t0 = performance.now();
    var out = await stages[k](token, P, reply, result);
    if (k === 2) result = out;
    if (token !== sd.token || hold === k + 1) return;
    var wait = STAGE_MIN - (performance.now() - t0);
    if (wait > 0) await sleep(wait);
    if (token !== sd.token) return;
    items[k].className = "done";
  }
  $("again-btn").classList.remove("hidden");
  $("again-btn").classList.add("attn");
}

// Runs fn(u) for u from 0 to 1 over ms milliseconds; stops early if the send is cancelled.
function animate(token, ms, fn) {
  return new Promise(function (res) {
    var s = performance.now();
    function f(now) {
      if (token !== sd.token) return res();
      var u = clamp((now - s) / ms, 0, 1);
      fn(u);
      if (u < 1) requestAnimationFrame(f); else res();
    }
    requestAnimationFrame(f);
  });
}

function setHud(items) {
  var hud = $("hud");
  hud.innerHTML = "";
  return items.map(function (it) {
    var c = el("div", "ctr", hud);
    el("span", "ctr-l", c).textContent = it[0];
    var v = el("span", "ctr-v", c);
    v.textContent = it[1];
    return v;
  });
}

function setCaption(text, cls) {
  var c = $("caption");
  if (text) c.textContent = text; else c.innerHTML = "&nbsp;";
  c.className = "caption" + (cls ? " " + cls : "");
}

// Canvas drawn in CSS pixels; k is the backing-store scale (devicePixelRatio, or more for zooming).
function sizeCanvas(canvas, w, h, k) {
  k = k || window.devicePixelRatio || 1;
  canvas.width = Math.round(w * k);
  canvas.height = Math.round(h * k);
  var ctx = canvas.getContext("2d");
  ctx.setTransform(k, 0, 0, k, 0, 0);
  return ctx;
}

// Stage 1: each blank cell splits into one glowing dot per remaining candidate,
// and lines join qubits that share an "exactly one" rule.
async function stage1(token, P) {
  var hud = setHud([["Qubits", 0], ["Couplers", 0]]);
  setCaption("Every number that could still go in a square becomes a qubit.");
  var sq = $("v1-sq"), n = P.n;
  sq.querySelectorAll(".qdot, .clue, .lines").forEach(function (e) { e.remove(); });
  var size = sq.getBoundingClientRect().width, q = size / n;
  sq.style.setProperty("--q", q + "px");
  sq.insertAdjacentHTML("beforeend", linesSvg(n, P.box[0], P.box[1]));
  for (var r = 0; r < n; r++) {
    for (var c = 0; c < n; c++) {
      if (!P.grid[r][c]) continue;
      var d = el("span", "clue", sq);
      d.textContent = P.grid[r][c];
      d.style.left = (c + .5) * 100 / n + "%";
      d.style.top = (r + .5) * 100 / n + "%";
    }
  }
  // dot positions, in grid units
  var byCell = {};
  P.variables.forEach(function (x, i) { (byCell[x[0] + "," + x[1]] = byCell[x[0] + "," + x[1]] || []).push(i); });
  var pos = new Array(P.variables.length), dots = [];
  P.cells.forEach(function (rc, k) {
    var ids = byCell[rc[0] + "," + rc[1]] || [], m = ids.length;
    var dpx = Math.min(q * .2, q * .8 / Math.max(m, 1) * .72);
    ids.forEach(function (i, j) {
      var x = rc[1] + .5, y = rc[0] + .1 + .8 * (j + .5) / m;
      pos[i] = [x, y];
      var dot = el("span", "qdot", sq);
      dot.style.left = x * 100 / n + "%";
      dot.style.top = y * 100 / n + "%";
      dot.style.setProperty("--d", dpx.toFixed(1) + "px");
      // start at the cell centre, then spread into the column
      dot.style.setProperty("--dy", ((rc[0] + .5 - y) * q).toFixed(1) + "px");
      dot.style.setProperty("--del", k * 22 + "ms");
      dots.push(dot);
    });
  });
  var ctx = sizeCanvas($("v1-canvas"), size, size);
  ctx.clearRect(0, 0, size, size);
  sq.getBoundingClientRect(); // commit the start state so the dots transition in
  dots.forEach(function (d) { d.classList.add("on"); });
  var spread = P.cells.length * 22 + 500;
  await animate(token, spread, function (u) { hud[0].textContent = fmt(P.variables.length * u); });
  // couplers: every pair inside each group, drawn as gentle arcs
  var pairs = [];
  P.groups.forEach(function (g) {
    for (var a = 0; a < g.length; a++) for (var b = a + 1; b < g.length; b++) pairs.push([g[a], g[b]]);
  });
  ctx.globalCompositeOperation = "lighter";
  ctx.lineWidth = 1.3;
  ctx.strokeStyle = n === 9 ? "rgba(110, 200, 255, .3)" : "rgba(110, 200, 255, .45)";
  var drawn = 0;
  await animate(token, 1500, function (u) {
    var upto = Math.floor(pairs.length * u);
    ctx.beginPath();
    for (; drawn < upto; drawn++) {
      var A = pos[pairs[drawn][0]], B = pos[pairs[drawn][1]];
      var x1 = A[0] * q, y1 = A[1] * q, x2 = B[0] * q, y2 = B[1] * q;
      var mx = (x1 + x2) / 2, my = (y1 + y2) / 2, dx = x2 - x1, dy = y2 - y1;
      var bend = x1 === x2 ? .35 : .14;
      ctx.moveTo(x1, y1);
      ctx.quadraticCurveTo(mx - dy * bend, my + dx * bend, x2, y2);
    }
    ctx.stroke();
    hud[1].textContent = fmt(P.num_couplers * u);
  });
}

var chipCache = { key: "", coupled: null, index: null, xy: null };

// Lookups for the chip, and its qubit positions centred in the unit square.
function chipIndex() {
  if (!chipCache.index) {
    var chip = DEMO.chip, index = new Map(), coupled = new Set();
    chip.qubits.forEach(function (q, i) { index.set(q, i); });
    chip.couplers.forEach(function (e) { coupled.add(Math.min(e[0], e[1]) + "," + Math.max(e[0], e[1])); });
    var lo = [1, 1], hi = [0, 0];
    chip.xy.forEach(function (p) {
      lo = [Math.min(lo[0], p[0]), Math.min(lo[1], p[1])];
      hi = [Math.max(hi[0], p[0]), Math.max(hi[1], p[1])];
    });
    var s = Math.max(hi[0] - lo[0], hi[1] - lo[1], 1e-9);
    var ox = (1 - (hi[0] - lo[0]) / s) / 2, oy = (1 - (hi[1] - lo[1]) / s) / 2;
    chipCache.xy = chip.xy.map(function (p) { return [ox + (p[0] - lo[0]) / s, oy + (p[1] - lo[1]) / s]; });
    chipCache.index = index;
    chipCache.coupled = coupled;
  }
  return chipCache.index;
}

// The chip lattice is drawn once per size and zoom; only the highlight canvas changes per puzzle.
function drawChipBase(size, k) {
  var key = Math.round(size) + "x" + k.toFixed(2);
  if (chipCache.key === key) return;
  chipCache.key = key;
  var chip = DEMO.chip, idx = chipIndex(), xy = chipCache.xy, pad = size * .04, w = size - 2 * pad;
  var ctx = sizeCanvas($("v2-base"), size, size, k);
  ctx.clearRect(0, 0, size, size);
  ctx.strokeStyle = "rgba(120, 170, 215, .07)";
  ctx.lineWidth = .6;
  ctx.beginPath();
  chip.couplers.forEach(function (e) {
    var a = xy[idx.get(e[0])], b = xy[idx.get(e[1])];
    if (!a || !b) return;
    ctx.moveTo(pad + a[0] * w, pad + a[1] * w);
    ctx.lineTo(pad + b[0] * w, pad + b[1] * w);
  });
  ctx.stroke();
  ctx.fillStyle = "rgba(150, 190, 225, .5)";
  var s = Math.max(1.2, w / 400);
  xy.forEach(function (p) { ctx.fillRect(pad + p[0] * w - s / 2, pad + p[1] * w - s / 2, s, s); });
}

// Bounding box of the bulk of the points, ignoring the odd isolated qubit far away.
function coreBox(pts) {
  function med(a) { var s = a.slice().sort(function (x, y) { return x - y; }); return s[s.length >> 1]; }
  var xs = pts.map(function (p) { return p[0]; }), ys = pts.map(function (p) { return p[1]; });
  var mx = med(xs), my = med(ys);
  var dx = 4 * med(xs.map(function (x) { return Math.abs(x - mx); })) + .03;
  var dy = 4 * med(ys.map(function (y) { return Math.abs(y - my); })) + .03;
  var box = [1, 1, 0, 0];
  pts.forEach(function (p) {
    if (Math.abs(p[0] - mx) > dx || Math.abs(p[1] - my) > dy) return;
    box = [Math.min(box[0], p[0]), Math.min(box[1], p[1]), Math.max(box[2], p[0]), Math.max(box[3], p[1])];
  });
  return box;
}

// Stage 2: the chip's qubits drawn faintly, the puzzle's chains lit one by one, then a zoom
// into the corner of the chip they use.
async function stage2(token, P) {
  var hud = setHud([["Physical qubits", 0], ["Longest chain", 0]]);
  setCaption("Each qubit becomes a chain of real qubits on the chip.");
  var die = $("v2-die"), zoom = $("v2-zoom"), size = die.getBoundingClientRect().width;
  var chip = DEMO.chip, idx = chipIndex(), cxy = chipCache.xy, pad = size * .04, w = size - 2 * pad;
  var chains = P.qpu.chains, all = [];
  chains.forEach(function (ch) { ch.forEach(function (q) { var p = cxy[idx.get(q)]; if (p) all.push(p); }); });
  var box = all.length ? coreBox(all) : [0, 0, 1, 1];
  var span = Math.max(box[2] - box[0], box[3] - box[1], .02);
  var z = clamp(.62 / span, 1, 4);
  // draw at the zoomed resolution so the zoom stays sharp (capped to keep the canvas small)
  var k = Math.min(z * (window.devicePixelRatio || 1), 2600 / size);
  zoom.style.transition = "none";
  zoom.style.transform = "none";
  drawChipBase(size, k);
  var ctx = sizeCanvas($("v2-glow"), size, size, k);
  ctx.clearRect(0, 0, size, size);
  var done = 0, phys = 0, longest = 0;
  var rad = Math.max(1.4, w / 260 / Math.sqrt(z));
  function xy(q) { var p = cxy[idx.get(q)]; return p ? [pad + p[0] * w, pad + p[1] * w] : null; }
  await animate(token, 2000, function (u) {
    var upto = Math.floor(chains.length * u);
    for (; done < upto; done++) {
      var ch = chains[done], pts = ch.map(xy);
      ctx.globalCompositeOperation = "lighter";
      ctx.strokeStyle = "rgba(130, 215, 255, .9)";
      ctx.lineWidth = Math.max(1.5, rad * .8);
      ctx.beginPath();
      for (var a = 0; a < ch.length; a++) {
        for (var b = a + 1; b < ch.length; b++) {
          if (pts[a] && pts[b] && chipCache.coupled.has(Math.min(ch[a], ch[b]) + "," + Math.max(ch[a], ch[b]))) {
            ctx.moveTo(pts[a][0], pts[a][1]);
            ctx.lineTo(pts[b][0], pts[b][1]);
          }
        }
      }
      ctx.stroke();
      pts.forEach(function (p) {
        if (!p) return;
        ctx.fillStyle = "rgba(60, 170, 255, .22)";
        ctx.beginPath();
        ctx.arc(p[0], p[1], rad * 3, 0, 7);
        ctx.fill();
        ctx.fillStyle = "#e9f8ff";
        ctx.beginPath();
        ctx.arc(p[0], p[1], rad, 0, 7);
        ctx.fill();
      });
      phys += ch.length;
      longest = Math.max(longest, ch.length);
    }
    hud[0].textContent = fmt(phys);
    hud[1].textContent = fmt(longest);
  });
  if (token !== sd.token) return;
  setCaption("The puzzle uses " + fmt(phys) + " of the chip's " + fmt(chip.qubits.length) + " qubits.");
  if (z > 1.05) {
    // centre the chains, but keep the view on the chip
    var tx = clamp(size / 2 - z * (pad + (box[0] + box[2]) / 2 * w), size * (1 - z), 0);
    var ty = clamp(size / 2 - z * (pad + (box[1] + box[3]) / 2 * w), size * (1 - z), 0);
    zoom.style.transition = "transform 1.4s cubic-bezier(.45, 0, .2, 1)";
    zoom.style.transform = "translate(" + tx.toFixed(1) + "px, " + ty.toFixed(1) + "px) scale(" + z.toFixed(3) + ")";
    await sleep(1500);
  }
}

function buildFridge() {
  // [top y, half width, thickness, label]
  var plates = [[52, 170, 15, "300 K"], [128, 150, 12, "50 K"], [198, 130, 12, "4 K"], [262, 112, 11, "1 K"], [322, 96, 10, "0.1 K"], [378, 82, 10, "0.015 K"]];
  var cx = 200, s = "";
  for (var i = 0; i < plates.length - 1; i++) {
    var A = plates[i], B = plates[i + 1], y1 = A[0] + A[2], y2 = B[0];
    [-.78, -.3, .3, .78].forEach(function (f) {
      s += '<rect x="' + (cx + f * B[1] - 3.5) + '" y="' + y1 + '" width="7" height="' + (y2 - y1) + '" fill="url(#g-rod)"/>';
    });
  }
  // coaxial lines snaking down between the plates
  [-56, -30, 34, 60].forEach(function (dx, j) {
    var d = "M" + (cx + dx) + " 30";
    for (var y = 30; y < 378; y += 40) {
      var w = (j % 2 ? 7 : -7);
      d += " C" + (cx + dx + w) + " " + (y + 13) + " " + (cx + dx - w) + " " + (y + 27) + " " + (cx + dx * .85) + " " + (y + 40);
    }
    s += '<path d="' + d + '" fill="none" stroke="' + (j % 2 ? "#b87333" : "#9aa4ad") + '" stroke-width="2.2" opacity=".85"/>';
  });
  s += '<rect x="' + (cx - 4) + '" y="388" width="8" height="46" fill="url(#g-rod)"/>';
  plates.forEach(function (p) {
    var y = p[0], rx = p[1], th = p[2], ry = rx * .13;
    s += '<ellipse cx="' + cx + '" cy="' + (y + th) + '" rx="' + rx + '" ry="' + ry + '" fill="#5c4311"/>';
    s += '<rect x="' + (cx - rx) + '" y="' + y + '" width="' + 2 * rx + '" height="' + th + '" fill="url(#g-side)"/>';
    s += '<ellipse cx="' + cx + '" cy="' + y + '" rx="' + rx + '" ry="' + ry + '" fill="url(#g-top)"/>';
    s += '<text class="lbl" x="' + (cx + rx + 14) + '" y="' + (y + th / 2 + 5) + '">' + p[3] + "</text>";
  });
  document.getElementById("fridge-parts").innerHTML = s;
}

// Stage 3: a pulse runs down into the fridge until the reply arrives.
async function stage3(token, P, reply) {
  setHud([["Answers asked for", fmt(P.qpu.num_reads)], ["Chip temperature", "0.015 K"]]);
  setCaption("Sending the problem to the quantum computer...");
  var pulse = document.querySelector(".pulse");
  pulse.classList.add("go");
  var result = await reply;
  if (token !== sd.token) return result;
  var dot = $("staff-dot");
  dot.className = result.source === "dwave" ? "dwave" : "sim";
  dot.title = result.source === "dwave" ? "Last answer: D-Wave" : "Last answer: recorded simulation";
  return result;
}

// Stage 4: the reads drop into a histogram of energies.
async function stage4(token, P, reply, result) {
  document.querySelector(".pulse").classList.remove("go");
  var E = result.energies.map(function (e) { return Math.max(0, Math.round(e)); });
  var N = E.length, zeros = E.filter(function (e) { return e === 0; }).length;
  var hud = setHud([["Answers", 0], ["Broke no rules", 0]]);
  setCaption("Each ball is one answer, sorted by how many rules it broke.");
  // bin 0 holds exactly the perfect answers; the rest share bins of width w
  var maxE = Math.max.apply(null, E), w = Math.max(1, Math.ceil(maxE / 12));
  var nb = Math.max(6, 1 + Math.ceil(maxE / w));
  function binOf(e) { return e === 0 ? 0 : 1 + Math.floor((e - 1) / w); }
  var counts = new Array(nb).fill(0);
  E.forEach(function (e) { counts[binOf(e)]++; });
  var hist = $("hist");
  hist.innerHTML = "";
  hist.className = "hist";
  var rect = hist.getBoundingClientRect(), W = rect.width, H = rect.height;
  var top = 34, labelH = 34, Hc = H - top - labelH, gap = Math.max(8, W * .012);
  var cw = (W - gap * (nb + 1)) / nb, maxC = Math.max.apply(null, counts);
  var best = { d: 0, k: 1 };
  for (var k = 1; k <= 12; k++) {
    var d = Math.min((cw - 6) / k, (Hc - 6) / Math.ceil(maxC / k));
    if (d > best.d) best = { d: d, k: k };
  }
  var D = Math.min(best.d, 34), K = Math.max(1, Math.min(best.k, Math.floor((cw - 6) / D)));
  el("div", "axis", hist).textContent = "Rules broken";
  for (var b = 0; b < nb; b++) {
    var x = gap + b * (cw + gap);
    var bin = el("div", "bin" + (b === 0 ? " zero" : ""), hist);
    bin.style.cssText = "left:" + x + "px;top:" + top + "px;width:" + cw + "px;height:" + Hc + "px";
    var lab = el("div", "bin-label" + (b === 0 ? " zero" : ""), hist);
    var lo = b === 0 ? 0 : 1 + (b - 1) * w, hi = b === 0 ? 0 : b * w;
    lab.textContent = lo === hi ? lo : lo + "-" + hi;
    lab.style.cssText = "left:" + x + "px;width:" + cw + "px;top:" + (top + Hc + 10) + "px";
  }
  var filled = new Array(nb).fill(0), spread = 1900 / N;
  E.forEach(function (e, i) {
    var b = binOf(e), j = filled[b]++;
    var row = Math.floor(j / K), col = j % K, inRow = Math.min(K, counts[b] - row * K);
    var x = gap + b * (cw + gap) + (cw - inRow * D) / 2 + col * D;
    var y = top + Hc - 3 - (row + 1) * D;
    var ball = el("span", "ball" + (b === 0 ? " z" : ""), hist);
    var s = D * .9;
    ball.style.cssText = "left:" + (x + D * .05) + "px;top:" + (y + D * .05) + "px;width:" + s + "px;height:" + s + "px;" +
      "--fall:" + (-(y + D + 30)).toFixed(0) + "px;--del:" + Math.round(i * spread) + "ms";
  });
  hist.getBoundingClientRect();
  hist.classList.add("go");
  var landed = function (u) { return Math.min(N, Math.max(0, Math.floor((u * (1900 + 500) - 500) / spread) + 1)); };
  await animate(token, 2400, function (u) {
    var m = landed(u);
    hud[0].textContent = fmt(m);
    hud[1].textContent = fmt(E.slice(0, m).filter(function (e) { return e === 0; }).length);
  });
  hud[0].textContent = fmt(N);
  hud[1].textContent = fmt(zeros);
  hist.classList.add("lit");
  setCaption(zeros + " of " + N + " answers broke no rules", zeros ? "good" : "");
}

// Map a read to a grid: a cell gets v when exactly one of its variables is on, else 0.
function decodeRead(P, bits) {
  var g = P.grid.map(function (row) { return row.slice(); }), on = {};
  P.variables.forEach(function (x, i) {
    if (!bits[i]) return;
    var key = x[0] + "," + x[1];
    on[key] = (on[key] || 0) + 1;
    g[x[0]][x[1]] = x[2];
  });
  P.cells.forEach(function (rc) { if (on[rc[0] + "," + rc[1]] !== 1) g[rc[0]][rc[1]] = 0; });
  return g;
}

// Stage 5: the best read fills the grid cell by cell; wrong cells go red.
async function stage5(token, P, reply, result) {
  var g = decodeRead(P, result.best);
  var items = [["Squares filled", 0], ["Wrong", 0]];
  if (result.source === "dwave" && result.qpu_access_time) items.push(["Time on the chip", (result.qpu_access_time / 1e6).toFixed(3) + " s"]);
  var hud = setHud(items);
  setCaption("The best answer goes on the grid...");
  var sheet = $("result-sheet");
  var grid = buildGrid(P, sheet.querySelector(".grid-slot"));
  $("result-stamp").classList.remove("on");
  var wrong = [];
  P.cells.forEach(function (rc) {
    var r = rc[0], c = rc[1], cell = grid.cells[r][c];
    cell.el.classList.add("ink");
    cell.val.textContent = g[r][c] || "";
    if (g[r][c] !== P.solution[r][c]) wrong.push(cell);
  });
  await sleep(500);
  var step = Math.min(70, 2200 / P.cells.length), shown = 0;
  await animate(token, step * P.cells.length, function (u) {
    var upto = Math.min(P.cells.length, Math.ceil(u * P.cells.length));
    for (; shown < upto; shown++) {
      var rc = P.cells[shown];
      grid.cells[rc[0]][rc[1]].el.classList.add("in");
    }
    hud[0].textContent = fmt(shown);
  });
  if (token !== sd.token) return;
  await sleep(450);
  if (token !== sd.token) return;
  wrong.forEach(function (cell) { cell.el.classList.add("wrong"); });
  hud[1].textContent = fmt(wrong.length);
  if (wrong.length) {
    setCaption("Today's quantum computers still make mistakes on bigger problems.", "bad");
  } else {
    grid.root.classList.add("ripple");
    setTimeout(function () { if (token === sd.token) $("result-stamp").classList.add("on"); }, 2 * P.n * 55 + 200);
    setCaption("The quantum computer solved it!", "good");
  }
}

/* ---------- wiring ---------- */

function onKey(e) {
  if (state.screen === "solve") {
    var n = state.P.n;
    if (!sv.started) {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); startSolve(); }
      return;
    }
    if (/^[1-9]$/.test(e.key) && +e.key <= n) setEntry(+e.key);
    else if (e.key === "Backspace" || e.key === "Delete" || e.key === "0") setEntry(0);
    else if (e.key === "ArrowUp") moveSel(-1, 0);
    else if (e.key === "ArrowDown") moveSel(1, 0);
    else if (e.key === "ArrowLeft") moveSel(0, -1);
    else if (e.key === "ArrowRight") moveSel(0, 1);
    else return;
    e.preventDefault();
  } else if (state.screen === "anneal" && an.P) {
    var dt = { ArrowDown: .04, ArrowUp: -.04, PageDown: .2, PageUp: -.2, End: 1, Home: -1 }[e.key];
    if (dt === undefined) return;
    e.preventDefault();
    stopAuto();
    if (!an.introDone) finishIntro(true);
    setT(an.t + dt);
  }
}

function wire() {
  document.querySelectorAll("[data-act=back]").forEach(function (b) { b.addEventListener("click", back); });
  document.querySelectorAll("[data-act=new]").forEach(function (b) { b.addEventListener("click", newPuzzle); });
  $("start-btn").addEventListener("click", startSolve);
  $("pad").addEventListener("click", function (e) {
    var k = e.target.closest("button");
    if (k) setEntry(+k.dataset.v);
  });
  $("anneal-it-btn").addEventListener("click", function () { show("anneal"); });
  $("anneal-btn").addEventListener("click", autoAnneal);
  $("send-btn").addEventListener("click", function () { show("send"); startSend(); });
  $("again-btn").addEventListener("click", function () { show("pick"); });
  document.addEventListener("keydown", onKey);
  // no long-press menu on the touchscreen
  document.addEventListener("contextmenu", function (e) { e.preventDefault(); });
  wireThermo();
}

// See the comment at the top of the file.
function fromHash() {
  var h = new URLSearchParams(location.hash.slice(1));
  var screen = h.get("screen");
  if (!screen) return false;
  var size = h.get("size") || "9";
  if (!DEMO.puzzles[size]) return false;
  loadPuzzle(size, clamp(+(h.get("puzzle") || 0), 0, DEMO.puzzles[size].length - 1));
  if (screen === "solve") {
    show("solve");
    var fill = h.get("fill");
    if (fill) {
      startSolve();
      var P = state.P, cells = P.cells, m = fill === "all" ? cells.length : Math.min(+fill, cells.length);
      sv.acc = 37400 + m * 900;
      sv.t0 = performance.now();
      for (var i = 0; i < m; i++) {
        select(cells[i][0], cells[i][1]);
        setEntry(P.solution[cells[i][0]][cells[i][1]]);
      }
      if (fill !== "all" && m < cells.length) {
        // a deliberate clash: copy a clue from the same row into the next blank
        var rc = cells[m], row = P.grid[rc[0]], clue = row.find(function (v) { return v; });
        select(rc[0], rc[1]);
        setEntry(clue || P.solution[rc[0]][rc[1]] % P.n + 1);
      }
    }
  } else if (screen === "anneal") {
    state.screen = "anneal";
    document.querySelectorAll(".screen").forEach(function (s) { s.classList.toggle("on", s.id === "anneal"); });
    var t = h.get("t");
    enterAnneal(t !== null);
    if (t !== null) setT(+t);
  } else if (screen === "send") {
    state.screen = "send";
    document.querySelectorAll(".screen").forEach(function (s) { s.classList.toggle("on", s.id === "send"); });
    startSend(+(h.get("stage") || 0));
  } else {
    show("pick");
  }
  return true;
}

function init() {
  if (!DEMO) {
    document.body.innerHTML = '<p class="nodata">data.js is missing. Run precompute.py to make it.</p>';
    return;
  }
  buildPick();
  buildFridge();
  wire();
  checkStatus();
  setInterval(checkStatus, 30000);
  if (!fromHash()) show("pick");
}

init();
