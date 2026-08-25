/**
 * Command palette (⌘K / Ctrl-K).
 *
 * One input over two result kinds — commands and concepts — because a user
 * who wants "jump to entropy" and a user who wants "export the session"
 * are in the same mental mode: they know the name of the thing they want and
 * do not want to hunt for its button.
 *
 * Scoring is a small subsequence matcher rather than a substring test, so
 * "gradesc" finds "gradient descent" and "expjs" finds "Export session JSON".
 */

import { pickConcept, state } from '../store.js';
import { closeModal, showModal } from './shell.js';
import { escapeHTML, focusOn } from './mapview.js';

let commands = [];
let selected = 0;
let results = [];

export function registerCommands(list) { commands = list; }

/** Subsequence match: returns a score, or -1 for no match. Earlier and more
 *  contiguous matches score higher, so exact prefixes always win. */
function fuzzy(query, text) {
  if (!query) return 0;
  const q = query.toLowerCase(), t = text.toLowerCase();
  if (t.startsWith(q)) return 1000 - t.length;
  let qi = 0, score = 0, last = -1;
  for (let ti = 0; ti < t.length && qi < q.length; ti++) {
    if (t[ti] === q[qi]) {
      score += last === ti - 1 ? 6 : 1;
      if (ti === 0 || t[ti - 1] === ' ' || t[ti - 1] === '-') score += 4;
      last = ti; qi++;
    }
  }
  return qi === q.length ? score - t.length * 0.05 : -1;
}

export function openPalette(initial = '') {
  const modal = showModal(`
    <div class="palette" role="combobox" aria-expanded="true">
      <input id="palette-input" type="text" placeholder="Search concepts or run a command…"
             autocomplete="off" spellcheck="false" aria-label="Command palette" />
      <div class="palette-results" id="palette-results" role="listbox"></div>
    </div>`);
  modal.classList.remove('modal');

  const input = document.getElementById('palette-input');
  input.value = initial;
  input.focus();
  input.addEventListener('input', () => { selected = 0; update(input.value); });
  input.addEventListener('keydown', onKey);
  update(initial);
}

function onKey(e) {
  if (e.key === 'ArrowDown') { e.preventDefault(); selected = Math.min(selected + 1, results.length - 1); paint(); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); selected = Math.max(selected - 1, 0); paint(); }
  else if (e.key === 'Enter') { e.preventDefault(); run(results[selected]); }
  else if (e.key === 'Escape') { e.preventDefault(); closeModal(); }
}

function update(query) {
  const q = query.trim();
  const cmdHits = commands
    .map((c) => ({ kind: 'command', item: c, score: fuzzy(q, c.title + ' ' + (c.keywords || '')) }))
    .filter((r) => r.score >= 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 8);

  const conceptHits = q
    ? state.nodes
        .map((n, i) => ({ kind: 'concept', item: n, idx: i,
                          score: Math.max(fuzzy(q, n.label), fuzzy(q, n.domain) * 0.4) }))
        .filter((r) => r.score >= 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, 20)
    : [];

  results = [...cmdHits, ...conceptHits];
  selected = Math.min(selected, Math.max(0, results.length - 1));
  paint();
}

function paint() {
  const host = document.getElementById('palette-results');
  if (!host) return;
  if (!results.length) {
    host.innerHTML = `<p class="subtle" style="padding:16px;text-align:center">No matches</p>`;
    return;
  }
  let html = '';
  let lastKind = null;
  results.forEach((r, i) => {
    if (r.kind !== lastKind) {
      html += `<div class="palette-group">${r.kind === 'command' ? 'Commands' : 'Concepts'}</div>`;
      lastKind = r.kind;
    }
    if (r.kind === 'command') {
      html += `<button class="palette-item" role="option" data-i="${i}" aria-selected="${i === selected}">
        <span style="width:8px;text-align:center" aria-hidden="true">${r.item.glyph || '›'}</span>
        <span class="pi-main">${escapeHTML(r.item.title)}</span>
        ${r.item.hint ? `<kbd>${escapeHTML(r.item.hint)}</kbd>` : ''}
      </button>`;
    } else {
      html += `<button class="palette-item" role="option" data-i="${i}" aria-selected="${i === selected}">
        <span class="swatch" style="background:${state.domainColour.get(r.item.domain)}"></span>
        <span class="pi-main">${escapeHTML(r.item.label)}</span>
        <span class="pi-sub">${escapeHTML(r.item.domain)}</span>
      </button>`;
    }
  });
  host.innerHTML = html;
  host.querySelectorAll('[data-i]').forEach((el) => {
    el.addEventListener('click', () => run(results[Number(el.dataset.i)]));
    el.addEventListener('mousemove', () => {
      const i = Number(el.dataset.i);
      if (i !== selected) { selected = i; paint(); }
    });
  });
  const active = host.querySelector('[aria-selected="true"]');
  if (active) active.scrollIntoView({ block: 'nearest' });
}

function run(result) {
  if (!result) return;
  closeModal();
  if (result.kind === 'command') result.item.run();
  else { pickConcept(result.idx); focusOn(result.idx, 2.6); }
}
