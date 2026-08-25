/**
 * Left rail: domain filters and a virtualised concept list.
 *
 * The list is virtualised because the rail must behave identically at 150
 * concepts and at 150,000 — only the rows inside the scroll window exist in the
 * DOM, so filtering and scrolling stay instant regardless of map size.
 */

import { isVisible, on, pickConcept, setDomains, state, toggleDomain } from '../store.js';
import { escapeHTML, focusOn } from './mapview.js';

const ROW_H = 30;
const OVERSCAN = 6;

let listEl, sizerEl, windowEl, filtered = [];

export function initNavigator() {
  listEl = document.getElementById('concept-list');
  sizerEl = document.getElementById('concept-sizer');
  windowEl = document.getElementById('concept-window');

  document.getElementById('concept-filter').addEventListener('input', (e) => {
    state.filter = e.target.value.trim().toLowerCase();
    recompute();
  });
  listEl.addEventListener('scroll', renderWindow, { passive: true });
  document.getElementById('domains-all').addEventListener('click', () => setDomains([]));
  document.getElementById('domains-none').addEventListener('click',
    () => setDomains(state.graph.domains));

  on('domains', () => { renderDomains(); recompute(); });
  on('slots', renderWindow);
  on('concepts', recompute);

  renderDomains();
  recompute();
}

function renderDomains() {
  const host = document.getElementById('domain-list');
  host.innerHTML = state.graph.domains.map((d) => `
    <button class="domain-row ${state.hiddenDomains.has(d) ? 'off' : ''}" data-domain="${escapeHTML(d)}">
      <span class="swatch" style="background:${state.domainColour.get(d)}"></span>
      <span class="dname">${escapeHTML(d)}</span>
      <span class="dcount">${state.domainCounts.get(d) || 0}</span>
    </button>`).join('');
  host.querySelectorAll('[data-domain]').forEach((el) => {
    el.addEventListener('click', () => toggleDomain(el.dataset.domain));
  });

  // Legend on the map mirrors the rail, and clicking it filters too.
  const legend = document.getElementById('legend');
  legend.innerHTML = state.graph.domains.map((d) => `
    <button class="legend-item" data-domain="${escapeHTML(d)}"
            style="opacity:${state.hiddenDomains.has(d) ? .35 : 1}">
      <i style="background:${state.domainColour.get(d)}"></i>${escapeHTML(d)}
    </button>`).join('');
  legend.querySelectorAll('[data-domain]').forEach((el) => {
    el.addEventListener('click', () => toggleDomain(el.dataset.domain));
  });
}

function recompute() {
  const q = state.filter;
  filtered = [];
  for (let i = 0; i < state.nodes.length; i++) {
    if (!isVisible(i)) continue;
    if (q) {
      const n = state.nodes[i];
      if (!n.label.toLowerCase().includes(q) &&
          !n.definition.toLowerCase().includes(q) &&
          !n.domain.toLowerCase().includes(q)) continue;
    }
    filtered.push(i);
  }
  document.getElementById('concept-count').textContent = filtered.length;
  sizerEl.style.height = filtered.length * ROW_H + 'px';
  renderWindow();
}

function renderWindow() {
  const scroll = listEl.scrollTop;
  const height = listEl.clientHeight || 400;
  const first = Math.max(0, Math.floor(scroll / ROW_H) - OVERSCAN);
  const last = Math.min(filtered.length, Math.ceil((scroll + height) / ROW_H) + OVERSCAN);

  windowEl.style.transform = `translateY(${first * ROW_H}px)`;
  const html = [];
  for (let k = first; k < last; k++) {
    const i = filtered[k];
    const n = state.nodes[i];
    const cls = i === state.slots.a ? 'sel-a' : i === state.slots.b ? 'sel-b' : '';
    html.push(`<button class="concept-row ${cls}" data-idx="${i}" title="${escapeHTML(n.definition)}">
      <span class="swatch" style="background:${state.domainColour.get(n.domain)}"></span>
      <span class="cname">${escapeHTML(n.label)}</span>
      <span class="cdom">${escapeHTML(n.domain)}</span>
    </button>`);
  }
  windowEl.innerHTML = html.join('');
  windowEl.querySelectorAll('[data-idx]').forEach((el) => {
    const idx = Number(el.dataset.idx);
    el.addEventListener('click', () => pickConcept(idx));
    el.addEventListener('dblclick', () => focusOn(idx, 3));
  });
}
