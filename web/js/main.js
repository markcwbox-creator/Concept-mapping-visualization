/**
 * Bootstrap: wires the modules together, owns keyboard shortcuts, and runs
 * collisions.
 *
 * Everything above this file is either pure state (`store.js`), pure maths
 * (`collide.js`), or a view that reads state and emits actions. This module is
 * the only place that knows about all of them.
 */

import { collide } from './collide.js';
import { buildSummary, loadBuild, LoadError } from './data.js';
import {
  clearSlots, emit, isPinned, on, persist, pickConcept, setReport,
  setSlot, state, swapSlots,
} from './store.js';

import { fit, focusOn, initMap, exportPNG, requestDraw, escapeHTML } from './ui/mapview.js';
import { initNavigator } from './ui/navigator.js';
import { initInspector, promptFor, selectPanel, serialiseCollision } from './ui/inspector.js';
import { initDock, render as renderDock, selectDock } from './ui/dock.js';
import { openPalette, registerCommands } from './ui/palette.js';
import {
  applyTheme, closeModal, download, downloadJSON, initSplitters, modalIsOpen,
  renderStatus, showBanner, showModal, toast, toggleTheme,
} from './ui/shell.js';

const isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);

/* ------------------------------------------------------------------- boot */

async function boot() {
  applyTheme(state.theme);
  document.getElementById('omnibox-kbd').textContent = isMac ? '⌘K' : 'Ctrl K';

  try {
    await loadBuild();
  } catch (err) {
    fatal(err);
    return;
  }

  initMap();
  initSplitters();
  initNavigator();
  initInspector();
  initDock(runCollision);
  wireChrome();
  wireKeyboard();
  registerPaletteCommands();

  applyPanelVisibility();
  renderSlots();
  renderStatus();
  renderDock();
  fit();

  const s = buildSummary();
  if (s.provenance) {
    // A build made from a random-weight toy model must never be mistaken for a
    // result, however real the interface around it looks.
    showBanner(
      `<strong>Smoke-test data.</strong> ${escapeHTML(s.provenance)} ` +
      `The interface is real; the semantics are not.`, 'danger');
  }
  if (!state.space) {
    showBanner('This build has no embedded vectors, so live collisions are ' +
      'unavailable. Rebuild with <code>embed_vectors: true</code>.', 'warn');
  }

  on('slots', () => { renderSlots(); renderDock(); });
  on('report', renderDock);
  on('history', renderDock);
  on('pinned', renderDock);
  on('domains', requestDraw);
}

function fatal(err) {
  const msg = err instanceof LoadError ? err.message : String(err);
  document.getElementById('app').innerHTML = `
    <div class="empty" style="height:100vh">
      <div class="glyph" aria-hidden="true">⚠</div>
      <h3>Could not load the build</h3>
      <p>${escapeHTML(msg)}</p>
      <p class="subtle">Generate one with
        <code>python -m collider all --config configs/qwen3-1.7b-4bit.yaml</code>,
        then serve this directory with <code>python -m collider serve</code>.</p>
    </div>`;
}

/* -------------------------------------------------------------- collisions */

function runCollision() {
  const { a, b } = state.slots;
  if (a === null || b === null) return;
  if (!state.space) { toast('This build has no vectors to collide.', 'error'); return; }
  const t0 = performance.now();
  const report = collide(state.space, a, b, { topN: 8 });
  const ms = performance.now() - t0;
  setReport(report);
  selectPanel('collision');
  renderStatus(`collided in ${ms.toFixed(1)}ms`);
}

function renderSlots() {
  for (const which of ['a', 'b']) {
    const el = document.getElementById('slot-' + which);
    const idx = state.slots[which];
    el.classList.toggle('empty', idx === null);
    el.classList.toggle('filled', idx !== null);
    el.querySelector('.slot-text').textContent =
      idx === null ? 'pick a concept' : state.nodes[idx].label;
  }
  document.getElementById('btn-collide').disabled =
    state.slots.a === null || state.slots.b === null;
}

/** Load a random pair from the top of the suggested list. */
function surpriseMe() {
  const pairs = (state.graph && state.graph.pairs) || [];
  if (!pairs.length) { toast('This build has no suggested pairs.', 'warn'); return; }
  // Weighted toward the top of the ranking without always giving the same one.
  const pool = pairs.slice(0, Math.max(10, Math.ceil(pairs.length * 0.4)));
  const p = pool[Math.floor(Math.random() * pool.length)];
  const A = state.byId.get(p.a), B = state.byId.get(p.b);
  if (!A || !B) return;
  setSlot('a', A.i);
  setSlot('b', B.i);
  runCollision();
  focusOn(A.i, 1.4);
}

/* ------------------------------------------------------------------ chrome */

function applyPanelVisibility() {
  const ws = document.getElementById('workspace');
  ws.classList.toggle('rail-collapsed', !state.railOpen);
  ws.classList.toggle('inspector-collapsed', !state.inspectorOpen);
  document.getElementById('btn-rail').classList.toggle('active', state.railOpen);
  document.getElementById('btn-inspector').classList.toggle('active', state.inspectorOpen);
  requestDraw();
}

function wireChrome() {
  document.getElementById('omnibox').addEventListener('click', () => openPalette());
  document.getElementById('btn-collide').addEventListener('click', runCollision);
  document.getElementById('btn-swap').addEventListener('click', swapSlots);
  document.getElementById('btn-random').addEventListener('click', surpriseMe);
  document.getElementById('btn-export').addEventListener('click', exportSession);
  document.getElementById('btn-theme').addEventListener('click', toggleTheme);
  document.getElementById('btn-help').addEventListener('click', showShortcuts);
  document.getElementById('btn-fit').addEventListener('click', fit);

  document.getElementById('btn-rail').addEventListener('click', () => {
    state.railOpen = !state.railOpen; persist(); applyPanelVisibility();
  });
  document.getElementById('btn-inspector').addEventListener('click', () => {
    state.inspectorOpen = !state.inspectorOpen; persist(); applyPanelVisibility();
  });

  document.querySelectorAll('.slot').forEach((el) => el.addEventListener('click', (e) => {
    // Clicking the ✕ clears the slot; clicking the body opens search to refill it.
    if (e.target.closest('.slot-clear')) { setSlot(el.dataset.slot, null); return; }
    openPalette();
  }));

  const toggles = [
    ['tg-edges', 'showEdges'], ['tg-labels', 'showLabels'], ['tg-hulls', 'showHulls'],
  ];
  for (const [id, key] of toggles) {
    const btn = document.getElementById(id);
    btn.classList.toggle('active', state[key]);
    btn.addEventListener('click', () => {
      state[key] = !state[key];
      btn.classList.toggle('active', state[key]);
      persist();
      requestDraw();
    });
  }

  document.getElementById('btn-png').addEventListener('click', async () => {
    const blob = await exportPNG();
    if (blob) { download('concept-map.png', blob); toast('Map exported as PNG', 'success'); }
  });
}

function exportSession() {
  const payload = {
    build: state.graph.meta,
    exported_at: new Date().toISOString(),
    pinned: state.pinned,
    history: state.history.map((h) => ({
      a: state.nodes[h.a].id, b: state.nodes[h.b].id,
      distance: +h.distance.toFixed(4), hops: h.hops,
      vacancy: +h.vacancy.toFixed(3), at: new Date(h.at).toISOString(),
    })),
    current: state.report ? serialiseCollision(state.report) : null,
  };
  downloadJSON('collider-session.json', payload);
  toast('Session exported', 'success');
}

/* ---------------------------------------------------------------- keyboard */

const SHORTCUTS = [
  { group: 'Navigation', items: [
    { keys: [isMac ? '⌘' : 'Ctrl', 'K'], label: 'Command palette / search' },
    { keys: ['F'], label: 'Fit map to view' },
    { keys: ['Double-click'], label: 'Zoom to a concept' },
    { keys: ['Scroll'], label: 'Zoom about the cursor' },
  ]},
  { group: 'Collisions', items: [
    { keys: ['Enter'], label: 'Collide the current pair' },
    { keys: ['S'], label: 'Swap A and B' },
    { keys: ['R'], label: 'Load a suggested pair' },
    { keys: ['P'], label: 'Pin the current collision' },
    { keys: ['Esc'], label: 'Clear selection / close overlay' },
  ]},
  { group: 'Panels', items: [
    { keys: [isMac ? '⌘' : 'Ctrl', 'B'], label: 'Toggle navigator' },
    { keys: [isMac ? '⌘' : 'Ctrl', 'I'], label: 'Toggle inspector' },
    { keys: ['1', '2', '3'], label: 'Collision / Concept / Diagnostics' },
    { keys: ['E'], label: 'Toggle edges' },
    { keys: ['L'], label: 'Toggle labels' },
    { keys: ['?'], label: 'This help' },
  ]},
];

function showShortcuts() {
  showModal(`<h2>Keyboard shortcuts</h2>
    <div class="shortcuts">${SHORTCUTS.map((g) => `
      <dl><dt>${g.group}</dt>${g.items.map((s) => `
        <dd>${escapeHTML(s.label)}<span class="keys">${
          s.keys.map((k) => `<kbd>${escapeHTML(k)}</kbd>`).join('')}</span></dd>`).join('')}
      </dl>`).join('')}</div>`);
}

function wireKeyboard() {
  window.addEventListener('keydown', (e) => {
    const mod = isMac ? e.metaKey : e.ctrlKey;
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) || e.target.isContentEditable;

    if (mod && e.key.toLowerCase() === 'k') { e.preventDefault(); openPalette(); return; }
    if (e.key === 'Escape') {
      if (modalIsOpen()) closeModal();
      else if (!typing) clearSlots();
      return;
    }
    if (typing || modalIsOpen()) return;

    if (mod && e.key.toLowerCase() === 'b') { e.preventDefault(); document.getElementById('btn-rail').click(); return; }
    if (mod && e.key.toLowerCase() === 'i') { e.preventDefault(); document.getElementById('btn-inspector').click(); return; }
    if (mod) return;

    switch (e.key) {
      case 'Enter': runCollision(); break;
      case 'f': case 'F': fit(); break;
      case 's': case 'S': swapSlots(); break;
      case 'r': case 'R': surpriseMe(); break;
      case 'e': case 'E': document.getElementById('tg-edges').click(); break;
      case 'l': case 'L': document.getElementById('tg-labels').click(); break;
      case 'p': case 'P':
        if (state.report) { document.getElementById('btn-pin')?.click(); }
        break;
      case '1': selectPanel('collision'); break;
      case '2': selectPanel('concept'); break;
      case '3': selectPanel('diagnostics'); break;
      case '?': showShortcuts(); break;
      default: return;
    }
  });
}

/* ----------------------------------------------------------------- palette */

function registerPaletteCommands() {
  registerCommands([
    { title: 'Collide current pair', glyph: '◈', hint: 'Enter', keywords: 'run', run: runCollision },
    { title: 'Load a suggested pair', glyph: '⚡', hint: 'R', keywords: 'random surprise', run: surpriseMe },
    { title: 'Swap A and B', glyph: '⇄', hint: 'S', keywords: 'reverse', run: swapSlots },
    { title: 'Clear selection', glyph: '✕', hint: 'Esc', keywords: 'reset', run: clearSlots },
    { title: 'Fit map to view', glyph: '⤢', hint: 'F', keywords: 'zoom reset', run: fit },
    { title: 'Toggle theme', glyph: '◐', keywords: 'dark light', run: toggleTheme },
    { title: 'Toggle edges', glyph: '⋈', hint: 'E', keywords: 'graph lines', run: () => document.getElementById('tg-edges').click() },
    { title: 'Toggle labels', glyph: '⌶', hint: 'L', keywords: 'text names', run: () => document.getElementById('tg-labels').click() },
    { title: 'Toggle domain shading', glyph: '◍', keywords: 'hulls regions', run: () => document.getElementById('tg-hulls').click() },
    { title: 'Show diagnostics', glyph: '◔', hint: '3', keywords: 'isotropy quality build', run: () => selectPanel('diagnostics') },
    { title: 'Show suggested pairs', glyph: '☰', keywords: 'grid table', run: () => selectDock('pairs') },
    { title: 'Show pinned collisions', glyph: '★', keywords: 'saved notes', run: () => selectDock('pinned') },
    { title: 'Copy generation prompt', glyph: '⎘', keywords: 'llm prompt clipboard', run: async () => {
        if (!state.report) { toast('Collide a pair first.', 'warn'); return; }
        await navigator.clipboard.writeText(promptFor(state.report));
        toast('Prompt copied to clipboard', 'success');
      } },
    { title: 'Export session JSON', glyph: '↓', keywords: 'download save', run: exportSession },
    { title: 'Export map as PNG', glyph: '⛶', keywords: 'download image screenshot', run: () => document.getElementById('btn-png').click() },
    { title: 'Keyboard shortcuts', glyph: '?', hint: '?', keywords: 'help keys', run: showShortcuts },
  ]);
}

boot();
