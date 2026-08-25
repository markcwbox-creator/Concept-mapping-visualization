/**
 * The inspector: collision results, concept detail, and build diagnostics.
 *
 * The collision panel is the product. It answers four genuinely different
 * questions about a pair, and it is careful to say when the answer is weak —
 * a disconnected pair, an already-crowded blend point, or a distance that is
 * unremarkable for this map are all reported as such rather than dressed up.
 */

import { buildSummary } from '../data.js';
import { distancePercentile } from '../collide.js';
import { isPinned, on, pinCurrent, setSlot, state } from '../store.js';
import { beforeAfterBars, distributionChart, sweepChart } from './charts.js';
import { escapeHTML, focusOn } from './mapview.js';
import { download, downloadJSON, toast } from './shell.js';

let sweepData = null;

export function initInspector() {
  document.getElementById('inspector-tabs').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-panel]');
    if (!btn) return;
    selectPanel(btn.dataset.panel);
  });

  on('report', renderCollision);
  on('focus', renderConcept);
  on('slots', () => { if (!state.report) renderCollision(); });
  on('pinned', renderCollision);

  // The layer sweep is optional — only present if someone ran `collider sweep`
  // and copied the result next to the build. Absence is not an error.
  fetch('data/layer_sweep.json', { cache: 'no-cache' })
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => { sweepData = d; renderDiagnostics(); })
    .catch(() => {});

  renderCollision();
  renderConcept();
  renderDiagnostics();
}

export function selectPanel(name) {
  state.panelTab = name;
  document.querySelectorAll('#inspector-tabs [data-panel]').forEach((b) =>
    b.classList.toggle('active', b.dataset.panel === name));
  document.querySelectorAll('.inspector-body .panel').forEach((p) =>
    p.classList.toggle('active', p.id === 'panel-' + name));
}

/* ---------------------------------------------------------------- collision */

function bridgeCard(b) {
  const n = state.nodes[b.idx];
  // Position on the lean track: 0% hugs pole A, 100% hugs pole B.
  const lean = Math.round(100 * b.dA / (b.dA + b.dB));
  return `<li>
    <button class="bridge" data-idx="${b.idx}"
            title="distance to A ${b.dA.toFixed(3)} · to B ${b.dB.toFixed(3)}">
      <span class="bridge-top">
        <span class="swatch" style="background:${state.domainColour.get(n.domain)}"></span>
        <span class="blabel">${escapeHTML(n.label)}</span>
        <span class="bnum">${b.balance.toFixed(2)}</span>
      </span>
      <span class="lean"><b style="left:${lean}%"></b></span>
      <p class="bridge-def">${escapeHTML(n.definition)}</p>
    </button></li>`;
}

function section(title, hint, items) {
  if (!items.length) return '';
  return `<div class="section-title">${title}<span class="hint">${hint}</span></div>
          <ul class="bridges">${items.map(bridgeCard).join('')}</ul>`;
}

function renderCollision() {
  const host = document.getElementById('panel-collision');
  const r = state.report;

  if (!r) {
    const { a, b } = state.slots;
    const need = a === null ? 'Pick a first concept' :
      b === null ? 'Pick a second concept' : 'Press Collide';
    host.innerHTML = `<div class="empty">
      <div class="glyph" aria-hidden="true">◈</div>
      <h3>${need}</h3>
      <p>Pull two distant concepts together to see the chain between them, what
         sits genuinely in the middle, and whether anything names their blend.</p>
      <p class="subtle">Click the map, use the concept list, or press
         <kbd>⌘K</kbd> to search. <kbd>R</kbd> loads a suggested pair.</p>
    </div>`;
    return;
  }

  const s = buildSummary();
  const A = state.nodes[r.aIdx], B = state.nodes[r.bIdx];
  const pct = distancePercentile(s.percentiles, r.distance);
  const vac = r.midpointVacancy;

  // Vacancy is the headline judgement, so it gets a plain-language verdict
  // rather than making the reader interpret a ratio.
  const vacVerdict = vac > 1.5
    ? { kind: '', text: '<strong>Nothing in this map names that blend.</strong> The midpoint sits in empty space — a description without a word for it.' }
    : vac > 0.9
      ? { kind: '', text: 'The blend point is <strong>loosely occupied</strong>. Something is near it, but not squarely on it.' }
      : { kind: 'warn', text: 'The blend point is <strong>already densely occupied</strong> — this connection is probably well-trodden.' };

  const pinned = isPinned();

  host.innerHTML = `
    <div class="collision-head">
      <div class="chead-pole a">
        <div class="plabel"><span class="badge">A</span>${escapeHTML(A.label)}</div>
        <div class="pdom">${escapeHTML(A.domain)}</div>
      </div>
      <div class="chead-gap">
        <div class="gnum">${r.distance.toFixed(2)}</div>
        <div class="glab">COSINE DIST</div>
      </div>
      <div class="chead-pole b">
        <div class="plabel">${escapeHTML(B.label)}<span class="badge">B</span></div>
        <div class="pdom">${escapeHTML(B.domain)}</div>
      </div>
    </div>

    <div class="gauge">
      <div class="gauge-track">
        <div class="gauge-mark" style="left:${Math.min(100, Math.max(0, pct ?? 50))}%"></div>
      </div>
      <div class="gauge-legend">
        <span>typical</span>
        <span>${pct != null ? `~${Math.round(pct)}th percentile for this map` : 'distribution unknown'}</span>
        <span>extreme</span>
      </div>
    </div>

    <div class="callout ${vacVerdict.kind}">
      <div><span class="mono">${vac.toFixed(2)}×</span> blend vacancy — ${vacVerdict.text}</div>
    </div>

    <div class="section-title">Stepping stones
      <span class="hint">${r.path.length ? `${r.path.length - 1} hops · cost ${r.pathCost.toFixed(2)}` : 'none'}</span>
    </div>
    ${r.path.length ? `<ol class="stones">${r.path.map((i, k) => {
        const n = state.nodes[i];
        return `<li><button class="stone" data-idx="${i}">
          <span class="node"><i style="background:${state.domainColour.get(n.domain)}"></i></span>
          <span class="stext">${escapeHTML(n.label)}</span>
          <span class="shop">${k === 0 ? 'A' : k === r.path.length - 1 ? 'B' : k}</span>
        </button></li>`;
      }).join('')}</ol>`
      : `<div class="callout warn"><div><strong>No path through the neighbour graph.</strong>
           These two are not merely distant, they are disconnected — usually a sign the
           pair is incoherent rather than novel. Raise <code>knn_k</code> and rebuild if
           this happens often.</div></div>`}

    ${section('Balanced bridges', 'genuinely between both', r.balanced)}
    ${section('Nearest the blend', 'closest to the midpoint vector', r.midpoint)}
    ${section('Orthogonal', 'sideways to the tension', r.orthogonal)}

    <div class="section-title">Take it further</div>
    <textarea class="note-editor" id="collision-note"
              placeholder="What did you notice? Saved with the pin."></textarea>
    <div class="bridge-actions" style="margin-top:8px">
      <button class="btn sm ${pinned ? 'active' : ''}" id="btn-pin">
        ${pinned ? '★ pinned' : '☆ pin this'}
      </button>
      <button class="btn sm" id="btn-copy-prompt">copy prompt</button>
      <button class="btn sm" id="btn-export-collision">export JSON</button>
    </div>
    <details style="margin-top:12px">
      <summary class="subtle" style="cursor:pointer;font-size:var(--text-sm)">Generation prompt</summary>
      <pre id="prompt-text" style="white-space:pre-wrap;font-size:11px;color:var(--fg-muted);
           background:var(--bg-inset);padding:10px;border-radius:6px;max-height:240px;overflow:auto"
      >${escapeHTML(promptFor(r))}</pre>
    </details>`;

  host.querySelectorAll('[data-idx]').forEach((el) => {
    const idx = Number(el.dataset.idx);
    el.addEventListener('click', () => { state.focused = idx; renderConcept(); selectPanel('concept'); });
    el.addEventListener('dblclick', () => focusOn(idx, 3));
  });

  const noteEl = document.getElementById('collision-note');
  const key = [r.aIdx, r.bIdx].sort((x, y) => x - y).join(':');
  const existing = state.pinned.find((p) => p.key === key);
  if (existing) noteEl.value = existing.note || '';

  document.getElementById('btn-pin').addEventListener('click', () => {
    pinCurrent(noteEl.value);
    toast('Pinned. It will still be here next time you open this build.', 'success');
  });
  document.getElementById('btn-copy-prompt').addEventListener('click', async () => {
    await navigator.clipboard.writeText(promptFor(r));
    toast('Prompt copied to clipboard', 'success');
  });
  document.getElementById('btn-export-collision').addEventListener('click', () => {
    downloadJSON(`collision-${state.nodes[r.aIdx].id}--${state.nodes[r.bIdx].id}.json`,
      serialiseCollision(r));
  });
}

export function serialiseCollision(r) {
  const nm = (i) => ({ id: state.nodes[i].id, label: state.nodes[i].label,
                       domain: state.nodes[i].domain, definition: state.nodes[i].definition });
  const grp = (list) => list.map((b) => ({
    ...nm(b.idx), d_a: +b.dA.toFixed(4), d_b: +b.dB.toFixed(4),
    balance: +b.balance.toFixed(4), score: +b.score.toFixed(4),
  }));
  return {
    build: buildSummary().model,
    a: nm(r.aIdx), b: nm(r.bIdx),
    distance: +r.distance.toFixed(4),
    midpoint_vacancy: +r.midpointVacancy.toFixed(3),
    path: r.path.map(nm),
    path_cost: r.pathCost,
    balanced: grp(r.balanced), midpoint: grp(r.midpoint), orthogonal: grp(r.orthogonal),
    exported_at: new Date().toISOString(),
  };
}

/**
 * The map finds the gap; a language model is what walks through it. Handing the
 * user a filled-in prompt is the cheapest version of that loop, and it keeps
 * the tool honest about which half is geometry and which half is generation.
 */
export function promptFor(r) {
  const nm = (i) => `${state.nodes[i].label} (${state.nodes[i].definition})`;
  const bridges = r.balanced.slice(0, 5).map((b) => state.nodes[b.idx].label).join(', ');
  return `Two concepts sit far apart in a concept map built from a language model:

A: ${nm(r.aIdx)}
B: ${nm(r.bIdx)}

Cosine distance ${r.distance.toFixed(3)}. The shortest chain of related concepts
between them is: ${r.path.map((i) => state.nodes[i].label).join(' -> ') || '(none found)'}.
Concepts sitting roughly equidistant from both: ${bridges}.

Propose three specific, non-obvious connections between A and B. For each: name the
shared underlying structure, state what it predicts or suggests that neither concept
alone does, and say plainly how it could be wrong. Reject any connection that is
merely a shared metaphor.`;
}

/* ------------------------------------------------------------------ concept */

function renderConcept() {
  const host = document.getElementById('panel-concept');
  const i = state.focused;
  if (i === null || i === undefined || !state.nodes[i]) {
    host.innerHTML = `<div class="empty"><div class="glyph">◇</div>
      <h3>No concept selected</h3>
      <p>Click any point on the map or any row in the navigator to inspect it.</p></div>`;
    return;
  }
  const n = state.nodes[i];
  const k = state.graph.knn.k;
  const nbrs = [];
  for (let e = 0; e < Math.min(k, 12); e++) {
    const j = state.graph.knn.idx[i][e];
    nbrs.push({ idx: j, sim: state.graph.knn.sim[i][e] });
  }

  host.innerHTML = `
    <div class="collision-head" style="grid-template-columns:1fr">
      <div class="chead-pole">
        <div class="plabel">
          <span class="swatch" style="display:inline-block;width:9px;height:9px;border-radius:50%;
                background:${state.domainColour.get(n.domain)};margin-right:6px"></span>
          ${escapeHTML(n.label)}
        </div>
        <div class="pdom">${escapeHTML(n.domain)} · <span class="mono">${escapeHTML(n.id)}</span></div>
      </div>
    </div>
    <p style="font-size:var(--text-sm);color:var(--fg-muted);line-height:1.6;margin:12px 0">
      ${escapeHTML(n.definition)}</p>

    <div class="bridge-actions">
      <button class="btn sm" data-set="a">set as A</button>
      <button class="btn sm" data-set="b">set as B</button>
      <button class="btn sm" id="btn-locate">locate on map</button>
    </div>

    <div class="section-title">Nearest neighbours<span class="hint">cosine similarity</span></div>
    <ul class="bridges">${nbrs.map((x) => {
      const m = state.nodes[x.idx];
      return `<li><button class="bridge" data-idx="${x.idx}">
        <span class="bridge-top">
          <span class="swatch" style="background:${state.domainColour.get(m.domain)}"></span>
          <span class="blabel">${escapeHTML(m.label)}</span>
          <span class="bnum">${x.sim.toFixed(3)}</span>
        </span>
        <span class="microbar" style="width:100%;margin-top:6px">
          <i style="width:${Math.max(2, Math.min(100, x.sim * 100 / Math.max(0.01, nbrs[0].sim)))}%"></i>
        </span>
      </button></li>`;
    }).join('')}</ul>`;

  host.querySelectorAll('[data-set]').forEach((el) =>
    el.addEventListener('click', () => setSlot(el.dataset.set, i)));
  host.querySelector('#btn-locate').addEventListener('click', () => focusOn(i, 3));
  host.querySelectorAll('[data-idx]').forEach((el) =>
    el.addEventListener('click', () => { state.focused = Number(el.dataset.idx); renderConcept(); }));
}

/* -------------------------------------------------------------- diagnostics */

function renderDiagnostics() {
  const host = document.getElementById('panel-diagnostics');
  if (!state.graph) return;
  const s = buildSummary();
  const before = s.isoBefore, after = s.isoAfter;

  const isoBlock = before && after ? `
    <div class="metric-grid">
      <div class="metric">
        <div class="mlabel">Mean cosine</div>
        <div class="mvalue">${after.mean_cosine.toFixed(3)}</div>
        <div class="mdelta ${Math.abs(after.mean_cosine) < Math.abs(before.mean_cosine) ? 'good' : 'bad'}">
          was ${before.mean_cosine.toFixed(3)}</div>
      </div>
      <div class="metric">
        <div class="mlabel">Spread (std)</div>
        <div class="mvalue">${after.cosine_std.toFixed(3)}</div>
        <div class="mdelta ${after.cosine_std > before.cosine_std ? 'good' : 'bad'}">
          was ${before.cosine_std.toFixed(3)}</div>
      </div>
    </div>
    <div style="margin-top:12px">${beforeAfterBars([
      { label: 'mean cosine (random pairs)', before: before.mean_cosine, after: after.mean_cosine },
      { label: 'variance in top direction', before: before.top1_variance_ratio, after: after.top1_variance_ratio },
      { label: 'variance in top 8', before: before.top8_variance_ratio, after: after.top8_variance_ratio },
    ])}</div>
    <p class="subtle" style="font-size:var(--text-xs);line-height:1.6;margin-top:8px">
      Raw transformer activations are anisotropic — unrelated concepts sit at high
      cosine, so distance carries almost no signal. Mean cosine should fall toward
      zero <em>and</em> the spread should rise. If it did not, nothing on this map
      means very much.</p>`
    : '<p class="subtle">This build recorded no isotropy report.</p>';

  const cur = state.report ? state.report.distance : null;

  host.innerHTML = `
    <div class="section-title">Space quality<span class="hint">before → after correction</span></div>
    ${isoBlock}

    <div class="section-title">Distance distribution<span class="hint">cosine distance</span></div>
    ${distributionChart(s.percentiles, cur)}

    ${sweepData ? `<div class="section-title">Layer sweep<span class="hint">higher is better</span></div>
      ${sweepChart(sweepData, s.layer)}
      <p class="subtle" style="font-size:var(--text-xs);line-height:1.6">
        <strong>Probe accuracy</strong> uses hand-written triplets encoding relations
        (mechanism, analogy). <strong>Domain purity</strong> only measures topic, and
        typically keeps climbing into the last layers — a layer that wins on purity but
        loses on probes is doing subject classification, not concept geometry.</p>` : ''}

    <div class="section-title">Build</div>
    <table class="kv-table">
      <tr><td>encoder</td><td class="n">${escapeHTML(s.encoder)}</td></tr>
      <tr><td>model</td><td class="n">${escapeHTML(s.model)}</td></tr>
      <tr><td>layer</td><td class="n">${s.layer ?? '—'}${s.nLayers ? ' / ' + s.nLayers : ''}</td></tr>
      <tr><td>pooling</td><td class="n">${escapeHTML(s.pooling)}</td></tr>
      <tr><td>templates</td><td class="n">${escapeHTML(s.templates)}</td></tr>
      <tr><td>concepts</td><td class="n">${s.nConcepts}</td></tr>
      <tr><td>layout</td><td class="n">${escapeHTML(s.layout)}</td></tr>
      <tr><td>centred</td><td class="n">${s.spaceParams.center ? 'yes' : 'no'}</td></tr>
      <tr><td>components removed</td><td class="n">${s.spaceParams.remove_top_k ?? '—'}</td></tr>
      <tr><td>whitened</td><td class="n">${s.spaceParams.whiten ? 'yes' : 'no'}</td></tr>
      ${s.vectorMeta ? `<tr><td>client vectors</td><td class="n">${s.vectorMeta.dim}d int8${
        s.vectorMeta.variance_retained
          ? ` · ${(s.vectorMeta.variance_retained * 100).toFixed(1)}% var`
          : ''}</td></tr>` : ''}
    </table>

    <div class="bridge-actions" style="margin-top:12px">
      <button class="btn sm" id="btn-export-build">export build metadata</button>
    </div>`;

  const btn = document.getElementById('btn-export-build');
  if (btn) btn.addEventListener('click', () =>
    downloadJSON('build-metadata.json', state.graph.meta));
}

export { renderCollision, renderConcept, renderDiagnostics };
