/**
 * The review tool.
 *
 * Design principle: the bottleneck in this project is human attention, not
 * compute. So the only thing that matters here is seconds-per-judgement.
 * Everything follows from that:
 *
 * * **One item on screen, no scrolling.** Reading a list means re-orienting on
 *   every row.
 * * **Keyboard only.** 1 / 2 / space / u. The hand learns them in about six
 *   items and the eye stops leaving the content.
 * * **The proposal is hidden until you answer.** This is the important one. If
 *   the tool showed you what the model picked and asked "agree?", you would
 *   agree — anchoring is overwhelming on a judgement this fine. Instead you
 *   make the call independently and the tool compares afterwards. That turns
 *   rubber-stamping into a genuine second opinion, and it means disagreements
 *   are informative rather than embarrassing.
 * * **Option order is shuffled per item**, so position never leaks the answer.
 * * **Everything persists per batch.** Stop at item 40, come back tomorrow.
 */

const KEY = (batch) => `collider.review.${batch}`;

const state = {
  batch: null,
  mode: null,          // "triplet" | "concept"
  items: [],
  order: [],           // shuffled indices
  cursor: 0,
  decisions: {},       // itemId -> decision object
  concepts: new Map(), // id -> {label, definition, domain, structure}
  awaitingNext: false,
  startedAt: 0,
  times: [],
};

/* ------------------------------------------------------------------ helpers */

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function toast(msg, kind = 'info', ms = 2600) {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $('toasts').appendChild(el);
  setTimeout(() => el.remove(), ms);
}

/** Deterministic shuffle so a reload shows the same order — otherwise "undo"
 *  and resume become confusing. Seeded on the batch name. */
function seededShuffle(n, seed) {
  let s = 0;
  for (const ch of seed) s = (s * 31 + ch.charCodeAt(0)) >>> 0;
  const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function save() {
  try {
    localStorage.setItem(KEY(state.batch), JSON.stringify({
      decisions: state.decisions, cursor: state.cursor,
    }));
  } catch { /* private mode; the session still works, it just won't resume */ }
}

function load(batch) {
  try {
    const raw = localStorage.getItem(KEY(batch));
    return raw ? JSON.parse(raw) : { decisions: {}, cursor: 0 };
  } catch { return { decisions: {}, cursor: 0 }; }
}

/* --------------------------------------------------------------- data load */

async function loadManifest() {
  const res = await fetch('data/review/manifest.json', { cache: 'no-cache' });
  if (!res.ok) throw new Error('no manifest');
  return res.json();
}

async function loadConcepts() {
  // Definitions and structural signatures for every concept a triplet can
  // reference. Without the signature the reviewer is judging on topic alone,
  // which is exactly the confusion the whole project is trying to avoid.
  const res = await fetch('data/review/concepts.json', { cache: 'no-cache' });
  if (!res.ok) return;
  const rows = await res.json();
  for (const r of rows) state.concepts.set(r.id, r);
}

async function openBatch(batch) {
  const res = await fetch('data/review/' + batch.file, { cache: 'no-cache' });
  const text = await res.text();
  const items = text.split('\n')
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith('//'))
    .map((l, i) => { const o = JSON.parse(l); o._i = i; return o; });

  state.batch = batch.name;
  state.mode = batch.mode;
  state.items = items;
  state.order = seededShuffle(items.length, batch.name);
  const saved = load(batch.name);
  state.decisions = saved.decisions || {};
  state.cursor = Math.min(saved.cursor || 0, items.length);

  $('batch-name').textContent = batch.mode;
  $('picker').classList.add('hidden');
  render();
}

/* ----------------------------------------------------------------- rendering */

const itemId = (item) => item.id || item.label ||
  `${item.anchor}|${item.closer}|${item.farther}`;

function current() {
  if (state.cursor >= state.order.length) return null;
  return state.items[state.order[state.cursor]];
}

function render() {
  const total = state.items.length;
  const done = Object.keys(state.decisions).length;
  $('progress-bar').style.width = total ? (state.cursor / total * 100) + '%' : '0';
  $('progress-text').textContent = `${state.cursor} / ${total}`;

  if (state.times.length >= 3) {
    const med = [...state.times].sort((a, b) => a - b)[Math.floor(state.times.length / 2)];
    const left = Math.max(0, total - state.cursor);
    $('rate-text').textContent =
      `${(med / 1000).toFixed(1)}s/item · ~${Math.ceil(left * med / 60000)} min left`;
  }

  const item = current();
  if (!item) return renderDone(done);

  $('done').classList.add('hidden');
  state.awaitingNext = false;
  state.startedAt = performance.now();

  if (state.mode === 'triplet') renderTriplet(item);
  else renderConcept(item);
  renderStatus();
}

function conceptBits(id) {
  const c = state.concepts.get(id);
  return {
    label: c ? c.label : id,
    definition: c ? c.definition : '(definition not staged)',
    domain: c ? c.domain : '',
    structure: c && c.structure ? c.structure : '',
  };
}

function renderTriplet(item) {
  $('triplet').classList.remove('hidden');
  $('concept').classList.add('hidden');
  $('t-reveal').classList.add('hidden');

  const a = conceptBits(item.anchor);
  $('t-anchor-label').textContent = a.label;
  $('t-anchor-def').textContent = a.definition;
  $('t-anchor-struct').textContent = a.structure;

  // Shuffle which side the proposed answer appears on, per item, so position
  // carries no information.
  const flip = (state.order[state.cursor] % 2) === 1;
  const ids = flip ? [item.farther, item.closer] : [item.closer, item.farther];
  item._ids = ids;

  ids.forEach((id, k) => {
    const el = $(`t-opt-${k + 1}`);
    const b = conceptBits(id);
    el.classList.remove('picked');
    el.querySelector('h3').textContent = b.label;
    el.querySelector('.rv-def').textContent = b.definition;
    el.querySelector('.rv-struct').textContent = b.structure;
    el.querySelector('.rv-dom').textContent = b.domain;
  });
}

function renderConcept(item) {
  $('concept').classList.remove('hidden');
  $('triplet').classList.add('hidden');
  $('c-label').textContent = item.label;
  $('c-def').textContent = item.definition;
  $('c-domain').textContent = item.domain;

  // Rows the independent verifier could not settle get a visible flag and its
  // reasons, so the human's scarce attention lands on the genuinely borderline
  // cases rather than being spread evenly over things already checked twice.
  const flag = $('c-flag');
  if (item._verdict === 'uncertain') {
    flag.className = 'rv-reveal disagree';
    flag.innerHTML = '<strong>Verifier was unsure.</strong> ' +
      esc((item._reasons || []).join('; ') || 'no reason given');
  } else {
    flag.className = 'rv-reveal hidden';
  }

  const reasons = ['not a concept', 'duplicate', 'too vague', 'definition wrong',
                   'too famous/obvious', 'wrong domain'];
  $('c-reasons').innerHTML = reasons
    .map((r) => `<button class="rv-chip" data-reason="${esc(r)}">${esc(r)}</button>`).join('');
  $('c-reasons').querySelectorAll('[data-reason]').forEach((b) =>
    b.addEventListener('click', () => b.classList.toggle('on')));
}

function renderDone(done) {
  $('triplet').classList.add('hidden');
  $('concept').classList.add('hidden');
  $('done').classList.remove('hidden');

  const d = Object.values(state.decisions);
  let summary = `${done} decisions recorded.`;
  if (state.mode === 'triplet') {
    const answered = d.filter((x) => x.choice !== 'skip');
    const agree = answered.filter((x) => x.agrees).length;
    summary = `${answered.length} judged, ${d.length - answered.length} skipped. ` +
      `You agreed with the proposal on ${agree} of ${answered.length}` +
      (answered.length ? ` (${Math.round(agree / answered.length * 100)}%).` : '.');
  } else {
    const kept = d.filter((x) => x.decision === 'keep').length;
    summary = `${kept} kept, ${d.filter((x) => x.decision === 'drop').length} dropped, ` +
      `${d.filter((x) => x.decision === 'skip').length} unsure.`;
  }
  $('done-summary').textContent = summary;
  renderStatus();
}

function renderStatus() {
  const d = Object.values(state.decisions);
  const bits = [`batch ${esc(state.batch || '—')}`, `${d.length} recorded`];
  if (state.mode === 'triplet') {
    const answered = d.filter((x) => x.choice !== 'skip');
    const agree = answered.filter((x) => x.agrees).length;
    if (answered.length) bits.push(`agreement ${Math.round(agree / answered.length * 100)}%`);
  }
  $('statusbar').innerHTML = bits
    .map((b) => `<span class="status-item">${b}</span>`).join('<span class="status-sep"></span>');
}

/* ----------------------------------------------------------------- decisions */

function recordTriplet(choiceSlot) {
  const item = current();
  if (!item || state.awaitingNext) return;
  const elapsed = performance.now() - state.startedAt;
  state.times.push(elapsed);

  let choice, agrees = null;
  if (choiceSlot === 'skip') { choice = 'skip'; }
  else {
    const chosenId = item._ids[choiceSlot];
    choice = chosenId;
    agrees = chosenId === item.closer;
    $(`t-opt-${choiceSlot + 1}`).classList.add('picked');
  }

  state.decisions[itemId(item)] = {
    anchor: item.anchor, proposed_closer: item.closer, proposed_farther: item.farther,
    note: item.note || '', choice, agrees, ms: Math.round(elapsed),
  };
  save();

  // Reveal the proposal only now, so the judgement above was independent.
  const rev = $('t-reveal');
  if (choice === 'skip') {
    rev.className = 'rv-reveal';
    rev.innerHTML = `Skipped. Proposed answer was <strong>${esc(conceptBits(item.closer).label)}</strong>` +
      (item.note ? ` — ${esc(item.note)}` : '');
  } else if (agrees) {
    rev.className = 'rv-reveal';
    rev.innerHTML = `<strong>Agreed.</strong>` + (item.note ? ` ${esc(item.note)}` : '');
  } else {
    rev.className = 'rv-reveal disagree';
    rev.innerHTML = `<strong>You disagreed.</strong> The proposal was ` +
      `<strong>${esc(conceptBits(item.closer).label)}</strong>` +
      (item.note ? ` — ${esc(item.note)}` : '') +
      `. Your call is what gets recorded.`;
  }
  rev.classList.remove('hidden');

  // Agreement needs no reading — a brief flash confirms the keypress landed.
  // A disagreement is the informative case, so it gets long enough to actually
  // read the proposal you rejected; those are a minority, so the extra dwell
  // costs almost nothing across a whole batch.
  state.awaitingNext = true;
  setTimeout(next, choice === 'skip' ? 260 : agrees ? 320 : 1700);
}

function recordConcept(decision) {
  const item = current();
  if (!item) return;
  const elapsed = performance.now() - state.startedAt;
  state.times.push(elapsed);
  const edited = $('c-def').textContent.trim();
  const reasons = [...$('c-reasons').querySelectorAll('.rv-chip.on')]
    .map((b) => b.dataset.reason);

  state.decisions[itemId(item)] = {
    label: item.label, domain: item.domain,
    definition: edited, edited: edited !== item.definition,
    decision, reasons, ms: Math.round(elapsed),
  };
  save();
  next();
}

function next() { state.cursor++; save(); render(); }

function undo() {
  if (state.cursor === 0) return;
  state.cursor--;
  const item = current();
  if (item) delete state.decisions[itemId(item)];
  save();
  render();
}

/* -------------------------------------------------------------------- export */

function exportDecisions() {
  const rows = [];
  for (const idx of state.order) {
    const item = state.items[idx];
    const d = state.decisions[itemId(item)];
    if (!d) continue;
    if (state.mode === 'triplet') {
      if (d.choice === 'skip') continue;
      // Export in the probe-file format, using the HUMAN's choice as the label.
      rows.push(JSON.stringify({
        anchor: d.anchor,
        closer: d.choice,
        farther: d.choice === d.proposed_closer ? d.proposed_farther : d.proposed_closer,
        note: d.note,
        reviewed: true,
        agreed_with_proposal: d.agrees,
      }));
    } else {
      if (d.decision !== 'keep') continue;
      rows.push(JSON.stringify({
        label: d.label, definition: d.definition, domain: d.domain,
      }));
    }
  }
  const name = state.mode === 'triplet' ? 'reviewed_probes.jsonl' : 'reviewed_concepts.jsonl';
  const blob = new Blob([rows.join('\n') + '\n'], { type: 'application/x-ndjson' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast(`Exported ${rows.length} rows to ${name}`, 'success');
}

/* -------------------------------------------------------------------- wiring */

window.addEventListener('keydown', (e) => {
  if (e.target.isContentEditable) {
    if (e.key === 'Escape') e.target.blur();
    return;
  }
  if (state.mode === 'triplet') {
    if (e.key === '1') { e.preventDefault(); recordTriplet(0); }
    else if (e.key === '2') { e.preventDefault(); recordTriplet(1); }
    else if (e.key === ' ') { e.preventDefault(); recordTriplet('skip'); }
  } else if (state.mode === 'concept') {
    if (e.key === 'j' || e.key === 'J') { e.preventDefault(); recordConcept('keep'); }
    else if (e.key === 'k' || e.key === 'K') { e.preventDefault(); recordConcept('drop'); }
    else if (e.key === ' ') { e.preventDefault(); recordConcept('skip'); }
    else if (e.key === 'e' || e.key === 'E') { e.preventDefault(); $('c-def').focus(); }
  }
  if (e.key === 'u' || e.key === 'U') { e.preventDefault(); undo(); }
});

document.querySelectorAll('[data-act]').forEach((b) => b.addEventListener('click', () => {
  const act = b.dataset.act;
  if (act === 'undo') return undo();
  if (state.mode === 'triplet' && act === 'skip') return recordTriplet('skip');
  if (state.mode === 'concept') return recordConcept(act === 'skip' ? 'skip' : act);
}));
document.querySelectorAll('.rv-option').forEach((b) =>
  b.addEventListener('click', () => recordTriplet(Number(b.dataset.slot))));

$('btn-export').addEventListener('click', exportDecisions);
$('btn-export-2').addEventListener('click', exportDecisions);
$('btn-restart').addEventListener('click', () => {
  if (!state.batch) return;
  if (!confirm(`Discard all decisions for "${state.batch}"? This cannot be undone.`)) return;
  state.decisions = {}; state.cursor = 0; state.times = [];
  save(); render();
});
$('btn-theme').addEventListener('click', () => {
  const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  try { localStorage.setItem('collider.review.theme', next); } catch { /* ignore */ }
});

/* ---------------------------------------------------------------------- boot */

(async function boot() {
  try {
    const t = localStorage.getItem('collider.review.theme');
    if (t) document.documentElement.setAttribute('data-theme', t);
  } catch { /* ignore */ }

  await loadConcepts();
  let manifest;
  try {
    manifest = await loadManifest();
  } catch {
    $('batch-list').innerHTML =
      `<p class="subtle" style="text-align:center">Nothing staged yet.</p>`;
    return;
  }

  $('batch-list').innerHTML = manifest.batches.map((b, i) => {
    const saved = load(b.name);
    const done = Object.keys(saved.decisions || {}).length;
    return `<button class="rv-batch" data-i="${i}">
      <span class="rv-key">${i + 1}</span>
      <span class="rv-batch-name">${esc(b.name)}</span>
      <span class="rv-batch-meta">${b.count} ${esc(b.mode)}s${done ? ` · ${done} done` : ''}</span>
    </button>`;
  }).join('');
  $('batch-list').querySelectorAll('[data-i]').forEach((el) =>
    el.addEventListener('click', () => openBatch(manifest.batches[Number(el.dataset.i)])));
})();
