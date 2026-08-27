/**
 * Application chrome: toasts, modal overlays, splitters, status bar, theme.
 * Small pieces, but they are most of what makes an app feel like a product
 * rather than a page.
 */

import { buildSummary, refreshColours } from '../data.js';
import { emit, persist, state } from '../store.js';
import { escapeHTML, requestDraw } from './mapview.js';

/* ------------------------------------------------------------------ toasts */

export function toast(message, kind = 'info', ms = 3200) {
  const host = document.getElementById('toasts');
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.innerHTML = `<span>${escapeHTML(message)}</span>`;
  host.appendChild(el);
  setTimeout(() => {
    el.style.transition = 'opacity 180ms, transform 180ms';
    el.style.opacity = '0';
    el.style.transform = 'translateY(4px)';
    setTimeout(() => el.remove(), 200);
  }, ms);
}

/* ------------------------------------------------------------------ modals */

let openModal = null;

export function showModal(html, { onClose } = {}) {
  closeModal();
  const root = document.getElementById('overlay-root');
  const scrim = document.createElement('div');
  scrim.className = 'scrim';
  scrim.innerHTML = `<div class="modal" role="dialog" aria-modal="true">${html}</div>`;
  scrim.addEventListener('mousedown', (e) => { if (e.target === scrim) closeModal(); });
  root.appendChild(scrim);
  openModal = { scrim, onClose };
  return scrim.querySelector('.modal');
}

export function closeModal() {
  if (!openModal) return;
  openModal.scrim.remove();
  if (openModal.onClose) openModal.onClose();
  openModal = null;
}

export const modalIsOpen = () => openModal !== null;

/* ------------------------------------------------------------------- theme */

export function applyTheme(theme) {
  state.theme = theme;
  document.documentElement.setAttribute('data-theme', theme);
  persist();
  // Domain colours are CSS variables, so they must be re-read after a swap.
  refreshColours();
  requestDraw();
  emit('domains');
}

export function toggleTheme() {
  applyTheme(state.theme === 'dark' ? 'light' : 'dark');
}

/* --------------------------------------------------------------- statusbar */

export function renderStatus(extra = '') {
  const s = buildSummary();
  const el = document.getElementById('statusbar');
  const iso = s.isoAfter ? s.isoAfter.mean_cosine : null;
  // Mean cosine near zero is the signal that the space is usable at all, so it
  // is the one diagnostic that lives permanently in the chrome.
  const isoOk = iso !== null && Math.abs(iso) < 0.05;
  const parts = [
    `<span class="status-item"><span class="dot ${s.provenance ? 'warn' : ''}"></span>${escapeHTML(s.model)}</span>`,
    `<span class="status-sep"></span><span class="status-item">layer ${s.layer ?? '—'}${s.nLayers ? '/' + s.nLayers : ''}</span>`,
    `<span class="status-sep"></span><span class="status-item">${escapeHTML(s.pooling)} pool</span>`,
    `<span class="status-sep"></span><span class="status-item">${s.nConcepts} concepts</span>`,
    `<span class="status-sep"></span><span class="status-item">${escapeHTML(s.layout)} layout</span>`,
  ];
  if (iso !== null) {
    parts.push(`<span class="status-sep"></span><span class="status-item" title="Mean cosine of random pairs after isotropy correction. Near zero is healthy.">`
      + `<span class="dot ${isoOk ? '' : 'err'}"></span>x&#772;cos ${iso.toFixed(3)}</span>`);
  }
  if (s.vectorMeta) {
    parts.push(`<span class="status-sep"></span><span class="status-item">${s.vectorMeta.dim}d int8</span>`);
  }
  if (extra) parts.push(`<span class="status-sep"></span><span class="status-item">${extra}</span>`);
  el.innerHTML = parts.join('');
}

/* ------------------------------------------------------------------ banner */

export function showBanner(html, kind = 'warn') {
  const app = document.getElementById('app');
  const existing = document.getElementById('build-banner');
  if (existing) existing.remove();
  const el = document.createElement('div');
  el.id = 'build-banner';
  el.className = `banner ${kind}`;
  el.innerHTML = `<span>${html}</span><button class="btn sm ghost" id="banner-dismiss">dismiss</button>`;
  app.insertBefore(el, app.firstChild);
  app.style.gridTemplateRows = `auto var(--topbar-h) 1fr var(--statusbar-h)`;
  el.querySelector('#banner-dismiss').addEventListener('click', () => {
    el.remove();
    app.style.gridTemplateRows = '';
  });
}

/* --------------------------------------------------------------- splitters */

/**
 * Draggable panel resizing.
 *
 * Sizes are written to CSS custom properties on the workspace grid rather than
 * to inline width on the panels, so the grid remains the single source of
 * layout truth and collapsed states keep working.
 */
export function initSplitters() {
  const ws = document.getElementById('workspace');
  const region = document.getElementById('map-region');

  const apply = () => {
    ws.style.setProperty('--rail-w', state.railWidth + 'px');
    ws.style.setProperty('--inspector-w', state.inspectorWidth + 'px');
    region.style.setProperty('--dock-h', state.dockHeight + 'px');
  };
  apply();

  const drag = (el, onMove) => {
    el.addEventListener('mousedown', (e) => {
      e.preventDefault();
      el.classList.add('dragging');
      document.body.style.userSelect = 'none';
      const move = (ev) => { onMove(ev); apply(); requestDraw(); };
      const up = () => {
        el.classList.remove('dragging');
        document.body.style.userSelect = '';
        window.removeEventListener('mousemove', move);
        window.removeEventListener('mouseup', up);
        persist();
      };
      window.addEventListener('mousemove', move);
      window.addEventListener('mouseup', up);
    });
  };

  drag(document.getElementById('split-rail'), (e) => {
    state.railWidth = Math.min(Math.max(e.clientX, 180), 460);
  });
  drag(document.getElementById('split-inspector'), (e) => {
    state.inspectorWidth = Math.min(Math.max(window.innerWidth - e.clientX, 300), 680);
  });
  drag(document.getElementById('split-dock'), (e) => {
    const r = region.getBoundingClientRect();
    state.dockHeight = Math.min(Math.max(r.bottom - e.clientY, 90), r.height - 200);
  });

  return apply;
}

/* --------------------------------------------------------------- downloads */

/** Trigger a client-side file download from a Blob. */
export function download(filename, blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function downloadJSON(filename, data) {
  download(filename, new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
}

export function downloadCSV(filename, rows) {
  if (!rows.length) return;
  const cols = Object.keys(rows[0]);
  const esc = (v) => {
    const s = v === null || v === undefined ? '' : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const body = [cols.join(','), ...rows.map((r) => cols.map((c) => esc(r[c])).join(','))].join('\n');
  download(filename, new Blob([body], { type: 'text/csv' }));
}
