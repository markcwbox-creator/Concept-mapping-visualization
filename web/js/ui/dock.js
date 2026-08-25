/**
 * Bottom dock: sortable data grids for suggested pairs, session history and
 * pinned collisions.
 *
 * The pairs grid is where the scoring model becomes arguable. Every feature
 * that fed the sampler is a sortable column with a micro-bar, so you can sort
 * by `bridgeability` and see whether the pairs it likes are actually the good
 * ones — which is the only honest way to tune the weights in `configs/*.yaml`.
 */

import { persist, setSlot, state, unpin } from '../store.js';
import { escapeHTML, focusOn } from './mapview.js';
import { downloadCSV, toast } from './shell.js';

let onCollide = () => {};

const PAIR_COLUMNS = [
  { key: 'rank', label: '#', width: 34, num: true },
  { key: 'pair', label: 'pair', grow: true },
  { key: 'score', label: 'score', num: true, bar: false },
  { key: 'distance', label: 'dist', num: true },
  { key: 'hops', label: 'hops', num: true },
  { key: 'bridgeability', label: 'bridge', num: true, bar: true },
  { key: 'midpoint_gap', label: 'gap', num: true, bar: true },
  { key: 'analogy', label: 'analogy', num: true, bar: true },
  { key: 'domain_gap', label: 'x-dom', num: true, bar: true },
];

export function initDock(collideFn) {
  onCollide = collideFn;

  document.getElementById('dock-tabs').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-dock]');
    if (!btn) return;
    selectDock(btn.dataset.dock);
  });

  const toggle = document.getElementById('btn-dock');
  toggle.addEventListener('click', () => {
    state.dockOpen = !state.dockOpen;
    document.getElementById('map-region').classList.toggle('dock-collapsed', !state.dockOpen);
    toggle.textContent = state.dockOpen ? 'collapse' : 'expand';
    persist();
  });
  document.getElementById('map-region').classList.toggle('dock-collapsed', !state.dockOpen);
  toggle.textContent = state.dockOpen ? 'collapse' : 'expand';

  selectDock(state.dockTab);
}

export function selectDock(name) {
  state.dockTab = name;
  persist();
  document.querySelectorAll('#dock-tabs [data-dock]').forEach((b) =>
    b.classList.toggle('active', b.dataset.dock === name));
  render();
}

export function render() {
  const body = document.getElementById('dock-body');
  document.getElementById('pairs-count').textContent =
    (state.graph && state.graph.pairs ? state.graph.pairs.length : 0);
  document.getElementById('history-count').textContent = state.history.length;
  document.getElementById('pinned-count').textContent = state.pinned.length;

  if (state.dockTab === 'pairs') renderPairs(body);
  else if (state.dockTab === 'history') renderHistory(body);
  else renderPinned(body);
}

/* -------------------------------------------------------------------- pairs */

function pairRows() {
  const pairs = (state.graph && state.graph.pairs) || [];
  return pairs.map((p, i) => {
    const A = state.byId.get(p.a), B = state.byId.get(p.b);
    if (!A || !B) return null;
    const f = p.features || {};
    return {
      rank: i + 1, a: A, b: B,
      score: p.score, distance: p.distance,
      hops: f.hops ?? -1,
      bridgeability: f.bridgeability ?? 0,
      midpoint_gap: f.midpoint_gap ?? 0,
      analogy: f.analogy ?? 0,
      domain_gap: f.domain_gap ?? 0,
      pathLength: (p.path || []).length,
    };
  }).filter(Boolean);
}

function renderPairs(host) {
  let rows = pairRows();
  if (!rows.length) {
    host.innerHTML = `<div class="empty"><div class="glyph">◌</div>
      <h3>No suggested pairs in this build</h3>
      <p>Run <code>python -m collider build</code> with <code>n_pairs</code> above zero.</p></div>`;
    return;
  }

  const { key, dir } = state.gridSort;
  if (key !== 'rank') {
    rows = rows.slice().sort((x, y) => (x[key] - y[key]) * dir);
  } else if (dir < 0) {
    rows = rows.slice().reverse();
  }

  const activeKey = state.report
    ? [state.nodes[state.report.aIdx].id, state.nodes[state.report.bIdx].id].sort().join(':')
    : null;

  host.innerHTML = `
    <table class="grid-table">
      <thead><tr>${PAIR_COLUMNS.map((c) => `
        <th data-key="${c.key}" class="${key === c.key ? 'sorted' : ''}"
            style="${c.width ? `width:${c.width}px` : ''}"
            title="sort by ${c.label}">${c.label}<span class="arrow">${
              key === c.key ? (dir < 0 ? '▾' : '▴') : '▾'}</span></th>`).join('')}
      </tr></thead>
      <tbody>${rows.map((r) => {
        const pk = [r.a.id, r.b.id].sort().join(':');
        return `<tr data-a="${r.a.i}" data-b="${r.b.i}" class="${pk === activeKey ? 'active' : ''}">
          <td class="n">${r.rank}</td>
          <td><span class="pair-cell">
            <span class="swatch" style="background:${state.domainColour.get(r.a.domain)}"></span>
            <span class="truncate">${escapeHTML(r.a.label)}</span>
            <span class="x">×</span>
            <span class="swatch" style="background:${state.domainColour.get(r.b.domain)}"></span>
            <span class="truncate">${escapeHTML(r.b.label)}</span>
          </span></td>
          <td class="n">${r.score.toFixed(2)}</td>
          <td class="n">${r.distance.toFixed(2)}</td>
          <td class="n">${r.hops < 0 ? '—' : r.hops}</td>
          ${['bridgeability', 'midpoint_gap', 'analogy', 'domain_gap'].map((k) =>
            `<td class="n"><span class="microbar"><i style="width:${
              Math.round(Math.min(1, Math.max(0, r[k])) * 100)}%"></i></span></td>`).join('')}
        </tr>`;
      }).join('')}</tbody>
    </table>`;

  host.querySelectorAll('th[data-key]').forEach((th) => th.addEventListener('click', () => {
    const k = th.dataset.key;
    state.gridSort = { key: k, dir: state.gridSort.key === k ? -state.gridSort.dir : -1 };
    persist();
    render();
  }));
  host.querySelectorAll('tbody tr').forEach((tr) => tr.addEventListener('click', () => {
    setSlot('a', Number(tr.dataset.a));
    setSlot('b', Number(tr.dataset.b));
    onCollide();
  }));
}

/* ------------------------------------------------------------------ history */

function renderHistory(host) {
  if (!state.history.length) {
    host.innerHTML = `<div class="empty"><div class="glyph">◷</div>
      <h3>No collisions yet this session</h3>
      <p>Everything you collide shows up here so you can walk back through it.</p></div>`;
    return;
  }
  host.innerHTML = `
    <table class="grid-table">
      <thead><tr><th>pair</th><th>dist</th><th>hops</th><th>vacancy</th><th>when</th></tr></thead>
      <tbody>${state.history.map((h) => `
        <tr data-a="${h.a}" data-b="${h.b}">
          <td><span class="pair-cell">
            <span class="truncate">${escapeHTML(state.nodes[h.a].label)}</span>
            <span class="x">×</span>
            <span class="truncate">${escapeHTML(state.nodes[h.b].label)}</span></span></td>
          <td class="n">${h.distance.toFixed(2)}</td>
          <td class="n">${h.hops}</td>
          <td class="n">${h.vacancy.toFixed(2)}×</td>
          <td class="n">${timeAgo(h.at)}</td>
        </tr>`).join('')}</tbody>
    </table>`;
  host.querySelectorAll('tbody tr').forEach((tr) => tr.addEventListener('click', () => {
    setSlot('a', Number(tr.dataset.a));
    setSlot('b', Number(tr.dataset.b));
    onCollide();
  }));
}

function timeAgo(ts) {
  const s = Math.round((Date.now() - ts) / 1000);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  return `${Math.round(s / 3600)}h`;
}

/* ------------------------------------------------------------------- pinned */

function renderPinned(host) {
  if (!state.pinned.length) {
    host.innerHTML = `<div class="empty"><div class="glyph">★</div>
      <h3>Nothing pinned</h3>
      <p>Pin a collision with a note when it produces a thought worth keeping.
         Pins persist in this browser across sessions and can be exported.</p></div>`;
    return;
  }
  host.innerHTML = `
    <div style="padding:12px">
      <div class="bridge-actions" style="margin-bottom:12px">
        <button class="btn sm" id="pins-csv">export CSV</button>
        <button class="btn sm" id="pins-clear">clear all</button>
      </div>
      ${state.pinned.map((p) => `
        <div class="pin-card">
          <div class="pc-head">
            <strong>${escapeHTML(p.aLabel)}</strong>
            <span class="x">×</span>
            <strong>${escapeHTML(p.bLabel)}</strong>
            <button class="btn sm ghost" style="margin-left:auto" data-unpin="${p.key}">remove</button>
          </div>
          ${p.note ? `<p class="pc-note">${escapeHTML(p.note)}</p>` : ''}
          <div class="pc-meta">d ${p.distance.toFixed(2)} · ${p.hops} hops ·
            vacancy ${p.vacancy.toFixed(2)}× · ${new Date(p.at).toLocaleString()}</div>
        </div>`).join('')}
    </div>`;

  host.querySelectorAll('[data-unpin]').forEach((b) =>
    b.addEventListener('click', () => unpin(b.dataset.unpin)));
  host.querySelector('#pins-csv').addEventListener('click', () => {
    downloadCSV('pinned-collisions.csv', state.pinned.map((p) => ({
      a: p.a, b: p.b, distance: p.distance.toFixed(4), hops: p.hops,
      vacancy: p.vacancy.toFixed(3), bridges: (p.bridges || []).join(' '),
      note: p.note || '', pinned_at: new Date(p.at).toISOString(),
    })));
  });
  host.querySelector('#pins-clear').addEventListener('click', () => {
    if (!confirm(`Remove all ${state.pinned.length} pinned collisions? This cannot be undone.`)) return;
    state.pinned.length = 0;
    persist();
    render();
    toast('All pins removed', 'warn');
  });
}
