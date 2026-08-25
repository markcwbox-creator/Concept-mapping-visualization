/*
 * Concept Collider front end.
 *
 * No build step, no dependencies, no CDN — open index.html through any static
 * server and it runs. Two views share one data set:
 *
 *   map view       Canvas 2-D scatter of the whole concept space at
 *                  precomputed UMAP/PCA coordinates. Layout is baked in
 *                  Python on purpose: a force simulation in the browser is
 *                  slow above a few thousand nodes and, worse, lands
 *                  somewhere different on every reload, which destroys the
 *                  spatial memory that makes a map worth having.
 *
 *   collider       Pick two concepts, and every bridge view is recomputed
 *                  live from the quantised vectors — no server, no
 *                  precomputed pair list to be limited by.
 */

'use strict';

const state = {
  graph: null,
  space: null,          // { vecs, n, dim, knnIdx, knnSim, k, typicalNN }
  byId: new Map(),
  slots: { a: null, b: null },
  hover: null,
  view: { x: 0, y: 0, scale: 1 },
  domainColour: new Map(),
  showEdges: true,
  showLabels: true,
};

// A colourblind-safe qualitative ramp, extended by rotating lightness so that
// maps with many domains stay distinguishable rather than collapsing to mush.
const PALETTE = [
  '#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#b07aa1', '#76b7b2',
  '#edc948', '#ff9da7', '#9c755f', '#bab0ac', '#86bcb6', '#d37295',
];

// ---------------------------------------------------------------- data load

async function load() {
  let graph;
  try {
    const res = await fetch('data/graph.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    graph = await res.json();
  } catch (err) {
    showBanner(
      'No graph found at <code>web/data/graph.json</code>. Build one with ' +
      '<code>python -m collider all --config configs/qwen3-1.7b-4bit.yaml</code>, ' +
      'then reload.', 'error');
    return;
  }
  state.graph = graph;
  graph.nodes.forEach((nd) => state.byId.set(nd.id, nd));

  graph.domains.forEach((d, i) => state.domainColour.set(d, PALETTE[i % PALETTE.length]));

  if (graph.vectors) {
    const buf = await (await fetch('data/' + graph.vectors.file)).arrayBuffer();
    const vecs = Collider.dequantise(buf, graph.vectors.n, graph.vectors.dim, graph.vectors.scale);
    const k = graph.knn.k;
    const knnIdx = Int32Array.from(graph.knn.idx.flat());
    const knnSim = Float32Array.from(graph.knn.sim.flat());
    // Median nearest-neighbour distance: the natural yardstick for "is this
    // region of the space empty or crowded".
    const nn = graph.knn.sim.map((row) => 1 - row[0]).sort((a, b) => a - b);
    state.space = {
      vecs, n: graph.vectors.n, dim: graph.vectors.dim,
      knnIdx, knnSim, k, typicalNN: nn[Math.floor(nn.length / 2)] || 0.1,
    };
  } else {
    showBanner('This build has no embedded vectors, so only the precomputed ' +
      'suggested pairs can be explored. Rebuild with <code>embed_vectors: true</code>.', 'warn');
  }

  const prov = graph.meta && graph.meta.provenance;
  if (prov && /smoke|random/i.test(prov)) {
    showBanner('<strong>Smoke-test data.</strong> ' + prov +
      ' The interface is real; the semantics are not. Run a real extraction ' +
      'before drawing any conclusion from what you see here.', 'error');
  }

  renderMeta();
  renderLegend();
  renderSuggested();
  renderAbout();
  resetView();
  draw();
}

// -------------------------------------------------------------------- chrome

function showBanner(html, kind) {
  const el = document.getElementById('banner');
  el.innerHTML = html;
  el.className = 'banner ' + (kind || '');
}

function renderMeta() {
  const m = state.graph.meta || {};
  const bits = [];
  if (m.model) bits.push(m.model);
  if (m.layer != null) bits.push('layer ' + m.layer + (m.n_layers ? '/' + m.n_layers : ''));
  if (m.pooling) bits.push(m.pooling + ' pooling');
  if (m.encoder) bits.push(m.encoder);
  bits.push(state.graph.nodes.length + ' concepts');
  bits.push(state.graph.layout.method + ' layout');
  document.getElementById('build-meta').textContent = bits.join(' · ');
}

function renderLegend() {
  const el = document.getElementById('legend');
  el.innerHTML = state.graph.domains.map((d) =>
    `<span class="chip"><i style="background:${state.domainColour.get(d)}"></i>${d}</span>`
  ).join('');
}

function renderAbout() {
  const m = state.graph.meta || {};
  const iso = m.isotropy_after || {};
  const isoBefore = m.isotropy_before || {};
  const fmt = (v) => (typeof v === 'number' ? v.toFixed(3) : '—');
  document.getElementById('about').innerHTML = `
    <h3>What you are looking at</h3>
    <p>Each point is a concept, positioned by a ${state.graph.layout.method.toUpperCase()}
    projection of its vector from <code>${m.model || 'the encoder'}</code>. Nearby points
    are conceptually similar <em>to the model</em>, which is not the same as similar to you —
    that gap is the interesting part.</p>

    <h3>Is this space actually usable?</h3>
    <p>Raw LLM activations are anisotropic: everything looks similar to everything.
    These numbers say whether the correction worked.</p>
    <table class="kv">
      <tr><th></th><th>before</th><th>after</th></tr>
      <tr><td>mean cosine of random pairs</td><td>${fmt(isoBefore.mean_cosine)}</td><td>${fmt(iso.mean_cosine)}</td></tr>
      <tr><td>spread (std)</td><td>${fmt(isoBefore.cosine_std)}</td><td>${fmt(iso.cosine_std)}</td></tr>
      <tr><td>variance in top direction</td><td>${fmt(isoBefore.top1_variance_ratio)}</td><td>${fmt(iso.top1_variance_ratio)}</td></tr>
    </table>
    <p class="dim">Mean cosine should fall toward zero and the spread should rise.
    If it did not, distances here mean very little.</p>

    <h3>Reading a collision</h3>
    <ul>
      <li><strong>Stepping stones</strong> — the cheapest chain through the
      neighbour graph. Nothing invented; every step is a real concept.</li>
      <li><strong>Balanced bridges</strong> — concepts genuinely between the two,
      scored so that a close neighbour of one side cannot win.</li>
      <li><strong>Nearest the blend</strong> — what sits closest to the midpoint
      vector. If <em>vacancy</em> is above 1, nothing in this map names that
      blend: a description without a word.</li>
      <li><strong>Orthogonal</strong> — far from both poles and unaligned with the
      tension between them. The sideways answer.</li>
    </ul>`;
}

// ----------------------------------------------------------------- map view

const canvas = document.getElementById('map');
const ctx = canvas.getContext('2d');

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = r.width * dpr;
  canvas.height = r.height * dpr;
  canvas.style.width = r.width + 'px';
  canvas.style.height = r.height + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function resetView() {
  const r = canvas.getBoundingClientRect();
  state.view = {
    scale: Math.min(r.width, r.height) * 0.44,
    x: r.width / 2,
    y: r.height / 2,
  };
  draw();
}

const toScreen = (nd) => ({
  x: nd.x * state.view.scale + state.view.x,
  y: -nd.y * state.view.scale + state.view.y,
});

function draw() {
  if (!state.graph) return;
  const r = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, r.width, r.height);
  const g = state.graph;
  const zoom = state.view.scale / (Math.min(r.width, r.height) * 0.44);

  const highlight = new Set();
  const pathSet = new Set();
  if (state.lastReport) {
    state.lastReport.path.forEach((i) => pathSet.add(i));
    ['balanced', 'midpoint', 'orthogonal'].forEach((key) =>
      state.lastReport[key].slice(0, 6).forEach((b) => highlight.add(b.idx)));
  }
  [state.slots.a, state.slots.b].forEach((s) => { if (s != null) highlight.add(s); });

  // Edges. Above ~40k they stop conveying anything at zoomed-out scales and
  // just cost frame time, so they fade out until you zoom in.
  if (state.showEdges && g.edges.length < 60000) {
    ctx.lineWidth = 0.6;
    ctx.strokeStyle = `rgba(140,150,170,${Math.min(0.35, 12 / Math.sqrt(g.edges.length))})`;
    ctx.beginPath();
    for (const e of g.edges) {
      const a = toScreen(g.nodes[e.s]), b = toScreen(g.nodes[e.t]);
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
    }
    ctx.stroke();
  }

  // The stepping-stone path drawn over the top, so a collision reads as a
  // route across the map rather than a list in a side panel.
  if (pathSet.size > 1 && state.lastReport) {
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    state.lastReport.path.forEach((idx, i) => {
      const p = toScreen(g.nodes[idx]);
      if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  const baseR = Math.max(2, 3 * Math.sqrt(zoom));
  for (let i = 0; i < g.nodes.length; i++) {
    const nd = g.nodes[i];
    const p = toScreen(nd);
    if (p.x < -20 || p.y < -20 || p.x > r.width + 20 || p.y > r.height + 20) continue;
    const isSlot = i === state.slots.a || i === state.slots.b;
    const isHot = highlight.has(i) || pathSet.has(i);
    const dim = state.lastReport && !isHot;
    ctx.globalAlpha = dim ? 0.25 : 1;
    ctx.beginPath();
    ctx.arc(p.x, p.y, isSlot ? baseR * 2.1 : (isHot ? baseR * 1.5 : baseR), 0, 6.284);
    ctx.fillStyle = state.domainColour.get(nd.domain) || '#888';
    ctx.fill();
    if (isSlot) {
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = '#fff';
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  if (state.showLabels) {
    ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
    ctx.textAlign = 'center';
    // Only label what the user can actually read: selected, highlighted,
    // hovered, and — once zoomed in — everything on screen.
    for (let i = 0; i < g.nodes.length; i++) {
      const isSlot = i === state.slots.a || i === state.slots.b;
      const show = isSlot || highlight.has(i) || pathSet.has(i) ||
        i === state.hover || zoom > 2.2;
      if (!show) continue;
      const p = toScreen(g.nodes[i]);
      if (p.x < 0 || p.y < 0 || p.x > r.width || p.y > r.height) continue;
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(14,17,23,0.9)';
      ctx.strokeText(g.nodes[i].label, p.x, p.y - baseR - 5);
      ctx.fillStyle = isSlot ? '#fff' : '#c9d1d9';
      ctx.fillText(g.nodes[i].label, p.x, p.y - baseR - 5);
    }
  }
}

function nodeAt(px, py) {
  const g = state.graph;
  let best = null, bestD = 14 * 14;
  for (let i = 0; i < g.nodes.length; i++) {
    const p = toScreen(g.nodes[i]);
    const d = (p.x - px) ** 2 + (p.y - py) ** 2;
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

// ------------------------------------------------------------- interactions

let dragging = null;

canvas.addEventListener('mousedown', (e) => {
  dragging = { x: e.offsetX, y: e.offsetY, moved: false };
});
window.addEventListener('mouseup', (e) => {
  if (dragging && !dragging.moved) {
    const i = nodeAt(e.offsetX, e.offsetY);
    if (i != null) selectNode(i);
  }
  dragging = null;
});
canvas.addEventListener('mousemove', (e) => {
  if (dragging) {
    const dx = e.offsetX - dragging.x, dy = e.offsetY - dragging.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) dragging.moved = true;
    state.view.x += dx;
    state.view.y += dy;
    dragging.x = e.offsetX;
    dragging.y = e.offsetY;
    draw();
    return;
  }
  const i = nodeAt(e.offsetX, e.offsetY);
  if (i !== state.hover) { state.hover = i; draw(); }
  const tip = document.getElementById('tooltip');
  if (i == null) { tip.className = 'tooltip hidden'; return; }
  const nd = state.graph.nodes[i];
  tip.innerHTML = `<strong>${nd.label}</strong><span class="dom">${nd.domain}</span>
                   <p>${nd.definition}</p>`;
  tip.className = 'tooltip';
  tip.style.left = Math.min(e.offsetX + 14, canvas.clientWidth - 280) + 'px';
  tip.style.top = (e.offsetY + 14) + 'px';
});
canvas.addEventListener('mouseleave', () => {
  document.getElementById('tooltip').className = 'tooltip hidden';
  state.hover = null; draw();
});
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const f = Math.exp(-e.deltaY * 0.0015);
  // Zoom about the cursor, so the thing you are pointing at stays put.
  state.view.x = e.offsetX - (e.offsetX - state.view.x) * f;
  state.view.y = e.offsetY - (e.offsetY - state.view.y) * f;
  state.view.scale *= f;
  draw();
}, { passive: false });

function selectNode(i) {
  if (state.slots.a === null || (state.slots.b !== null && state.slots.a !== null)) {
    if (state.slots.a !== null && state.slots.b !== null) {
      state.slots.a = i; state.slots.b = null; state.lastReport = null;
    } else {
      state.slots.a = i;
    }
  } else if (i !== state.slots.a) {
    state.slots.b = i;
  }
  renderSlots();
  if (state.slots.a !== null && state.slots.b !== null) runCollision();
  else draw();
}

function renderSlots() {
  for (const key of ['a', 'b']) {
    const el = document.getElementById('slot-' + key);
    const i = state.slots[key];
    el.querySelector('.name').textContent =
      i == null ? 'pick a concept' : state.graph.nodes[i].label;
    el.className = 'slot' + (i == null ? ' empty' : '');
  }
  document.getElementById('collide-btn').disabled =
    state.slots.a == null || state.slots.b == null;
}

// -------------------------------------------------------------- the collider

function runCollision() {
  const { a, b } = state.slots;
  if (a == null || b == null) return;
  if (!state.space) {
    showBanner('This build has no embedded vectors, so live collisions are unavailable.', 'warn');
    return;
  }
  const report = Collider.collide(state.space, a, b, { topN: 8 });
  state.lastReport = report;
  renderCollision(report);
  document.querySelector('.tab[data-tab="collision"]').click();
  draw();
}

function bridgeRow(bridge) {
  const nd = state.graph.nodes[bridge.idx];
  const pull = Math.round(100 * bridge.dA / (bridge.dA + bridge.dB));
  return `<li class="bridge" data-idx="${bridge.idx}">
    <div class="bridge-head">
      <span class="dot" style="background:${state.domainColour.get(nd.domain)}"></span>
      <span class="bridge-label">${nd.label}</span>
      <span class="bridge-num">${bridge.balance.toFixed(2)}</span>
    </div>
    <div class="lean" title="A ${bridge.dA.toFixed(2)} · B ${bridge.dB.toFixed(2)}">
      <span style="left:${pull}%"></span>
    </div>
    <p class="bridge-def">${nd.definition}</p>
  </li>`;
}

function renderCollision(rep) {
  const g = state.graph;
  const A = g.nodes[rep.aIdx], B = g.nodes[rep.bIdx];
  const pcts = (g.meta && g.meta.cosine_percentiles) || {};
  const pct = Collider.distancePercentile(pcts, rep.distance);
  const vac = rep.midpointVacancy;
  const vacLabel = vac > 1.5
    ? 'nothing in this map names that blend'
    : vac > 0.9 ? 'the blend is only loosely occupied'
      : 'the blend is already densely occupied';

  const section = (title, items, note) => `
    <h4>${title} <span class="note">${note}</span></h4>
    <ul class="bridges">${items.map(bridgeRow).join('')}</ul>`;

  document.getElementById('collision-empty').classList.add('hidden');
  const el = document.getElementById('collision');
  el.classList.remove('hidden');
  el.innerHTML = `
    <div class="pair-head">
      <div class="pole"><span class="tag">A</span>${A.label}<em>${A.domain}</em></div>
      <div class="gap">
        <div class="gap-num">${rep.distance.toFixed(3)}</div>
        <div class="gap-lab">cosine distance${pct != null ? ` · ~${Math.round(pct)}th pctile` : ''}</div>
      </div>
      <div class="pole"><span class="tag">B</span>${B.label}<em>${B.domain}</em></div>
    </div>

    <div class="vacancy">
      <strong>Blend vacancy ${vac.toFixed(2)}×</strong> — ${vacLabel}.
    </div>

    <h4>Stepping stones <span class="note">cheapest real chain between them</span></h4>
    ${rep.path.length
      ? `<ol class="path">${rep.path.map((i) =>
          `<li data-idx="${i}"><span class="dot" style="background:${state.domainColour.get(g.nodes[i].domain)}"></span>${g.nodes[i].label}</li>`
        ).join('')}</ol>
        <p class="dim small">${rep.path.length - 1} hops · total cost ${rep.pathCost.toFixed(3)}</p>`
      : `<p class="warn-inline">No path through the neighbour graph. These two are
         not merely distant, they are disconnected — usually a sign the pair is
         incoherent rather than novel. Raise <code>knn_k</code> if this happens often.</p>`}

    ${section('Balanced bridges', rep.balanced, 'genuinely between both')}
    ${section('Nearest the blend', rep.midpoint, 'closest to the midpoint vector')}
    ${section('Orthogonal', rep.orthogonal, 'sideways to the tension')}

    <details class="prompt-help">
      <summary>Use this as a generation prompt</summary>
      <pre id="prompt-text">${promptFor(rep)}</pre>
      <button id="copy-prompt" class="ghost">copy</button>
    </details>`;

  el.querySelectorAll('[data-idx]').forEach((node) => {
    node.addEventListener('click', () => focusNode(parseInt(node.dataset.idx, 10)));
  });
  const copy = document.getElementById('copy-prompt');
  if (copy) copy.addEventListener('click', () => {
    navigator.clipboard.writeText(document.getElementById('prompt-text').textContent);
    copy.textContent = 'copied';
    setTimeout(() => { copy.textContent = 'copy'; }, 1200);
  });
}

/*
 * The map finds the gap; a language model is what you actually want to walk
 * through it. Handing the user a filled-in prompt is the cheapest possible
 * version of that loop, and keeps the tool honest about which part is
 * geometry and which part is generation.
 */
function promptFor(rep) {
  const g = state.graph;
  const nm = (i) => `${g.nodes[i].label} (${g.nodes[i].definition})`;
  const bridges = rep.balanced.slice(0, 5).map((b) => g.nodes[b.idx].label).join(', ');
  return `Two concepts sit far apart in a concept map built from a language model:

A: ${nm(rep.aIdx)}
B: ${nm(rep.bIdx)}

Cosine distance ${rep.distance.toFixed(3)}. The shortest chain of related
concepts between them is: ${rep.path.map((i) => g.nodes[i].label).join(' -> ') || '(none found)'}.
Concepts sitting roughly equidistant from both: ${bridges}.

Propose three specific, non-obvious connections between A and B. For each:
name the shared underlying structure, state what it predicts or suggests that
neither concept alone does, and say plainly how it could be wrong. Reject any
connection that is merely a shared metaphor.`;
}

function focusNode(i) {
  const nd = state.graph.nodes[i];
  const r = canvas.getBoundingClientRect();
  state.view.x = r.width / 2 - nd.x * state.view.scale;
  state.view.y = r.height / 2 + nd.y * state.view.scale;
  state.hover = i;
  draw();
}

// ------------------------------------------------------------------ panels

function renderSuggested() {
  const g = state.graph;
  const el = document.getElementById('suggested');
  if (!g.pairs || !g.pairs.length) {
    el.innerHTML = '<p class="dim">No pairs in this build.</p>';
    return;
  }
  el.innerHTML = g.pairs.map((p, n) => {
    const A = state.byId.get(p.a), B = state.byId.get(p.b);
    if (!A || !B) return '';
    const f = p.features || {};
    const bars = ['distance', 'bridgeability', 'domain_gap', 'midpoint_gap', 'analogy']
      .map((k) => `<span class="fbar" title="${k} ${(f[k] ?? 0).toFixed(2)}">
                     <i style="height:${Math.round(100 * Math.min(1, f[k] ?? 0))}%"></i></span>`).join('');
    return `<button class="pair-card" data-a="${A.i}" data-b="${B.i}">
      <span class="rank">${n + 1}</span>
      <span class="pair-names">${A.label} <em>×</em> ${B.label}</span>
      <span class="pair-doms">${A.domain} · ${B.domain}</span>
      <span class="pair-stats">d ${p.distance.toFixed(2)} · ${f.hops ?? '?'} hops</span>
      <span class="fbars">${bars}</span>
    </button>`;
  }).join('');
  el.querySelectorAll('.pair-card').forEach((card) => {
    card.addEventListener('click', () => {
      state.slots.a = parseInt(card.dataset.a, 10);
      state.slots.b = parseInt(card.dataset.b, 10);
      renderSlots();
      runCollision();
    });
  });
}

// ------------------------------------------------------------------ wiring

document.getElementById('search').addEventListener('input', (e) => {
  const q = e.target.value.trim().toLowerCase();
  const box = document.getElementById('results');
  if (!q || !state.graph) { box.className = 'results hidden'; return; }
  const hits = state.graph.nodes
    .filter((nd) => nd.label.toLowerCase().includes(q) ||
                    nd.definition.toLowerCase().includes(q))
    .slice(0, 12);
  box.innerHTML = hits.map((nd) =>
    `<button data-idx="${nd.i}"><span class="dot" style="background:${state.domainColour.get(nd.domain)}"></span>
     ${nd.label}<em>${nd.domain}</em></button>`).join('') ||
    '<p class="dim pad">no matches</p>';
  box.className = 'results';
  box.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
    const i = parseInt(b.dataset.idx, 10);
    selectNode(i);
    focusNode(i);
    box.className = 'results hidden';
    document.getElementById('search').value = '';
  }));
});

document.querySelectorAll('.slot').forEach((s) => s.addEventListener('click', () => {
  state.slots[s.dataset.slot] = null;
  state.lastReport = null;
  document.getElementById('collision').classList.add('hidden');
  document.getElementById('collision-empty').classList.remove('hidden');
  renderSlots();
  draw();
}));

document.getElementById('collide-btn').addEventListener('click', runCollision);
document.getElementById('clear-btn').addEventListener('click', () => {
  state.slots = { a: null, b: null };
  state.lastReport = null;
  renderSlots();
  document.getElementById('collision').classList.add('hidden');
  document.getElementById('collision-empty').classList.remove('hidden');
  draw();
});
document.getElementById('reset-view').addEventListener('click', resetView);
document.getElementById('show-edges').addEventListener('change', (e) => {
  state.showEdges = e.target.checked; draw();
});
document.getElementById('show-labels').addEventListener('change', (e) => {
  state.showLabels = e.target.checked; draw();
});
document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach((x) => x.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach((x) => x.classList.remove('active'));
  t.classList.add('active');
  document.getElementById('tab-' + t.dataset.tab).classList.add('active');
}));

window.addEventListener('resize', resize);
resize();
load();
