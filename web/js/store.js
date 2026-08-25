/**
 * Application state.
 *
 * A single observable store with topic-scoped subscriptions. Views subscribe to
 * the slices they care about and re-render only on those, which is what keeps
 * the map from repainting every time somebody types in the concept filter.
 *
 * Anything a user would be annoyed to lose on reload — panel sizes, theme,
 * pinned collisions, notes — is persisted to localStorage. Everything derived
 * from the build (vectors, kNN, layout) is not: it is reloaded from
 * `graph.json`, which is the source of truth.
 */

const PERSIST_KEY = 'collider.workspace.v1';

/** localStorage can throw outright (private mode, blocked site data), so every
 *  access is guarded and the app must work correctly with nothing stored. */
function readPersisted() {
  try {
    const raw = localStorage.getItem(PERSIST_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch { return {}; }
}

function writePersisted(data) {
  try { localStorage.setItem(PERSIST_KEY, JSON.stringify(data)); } catch { /* ignore */ }
}

const persisted = readPersisted();

export const state = {
  // ---- build data (never persisted) ------------------------------------
  graph: null,
  space: null,            // { vecs, n, dim, knnIdx, knnSim, k, typicalNN }
  nodes: [],
  byId: new Map(),
  domainColour: new Map(),
  domainCounts: new Map(),

  // ---- selection & results ---------------------------------------------
  slots: { a: null, b: null },
  report: null,
  hover: null,
  focused: null,          // concept shown in the Concept panel

  // ---- view -------------------------------------------------------------
  view: { x: 0, y: 0, scale: 1 },
  filter: '',
  hiddenDomains: new Set(persisted.hiddenDomains || []),

  // ---- session ----------------------------------------------------------
  history: [],            // recent collisions, newest first (session only)
  pinned: persisted.pinned || [],

  // ---- preferences (persisted) -----------------------------------------
  theme: persisted.theme || 'dark',
  showEdges: persisted.showEdges !== false,
  showLabels: persisted.showLabels !== false,
  showHulls: persisted.showHulls === true,
  railOpen: persisted.railOpen !== false,
  inspectorOpen: persisted.inspectorOpen !== false,
  dockOpen: persisted.dockOpen !== false,
  dockTab: persisted.dockTab || 'pairs',
  panelTab: 'collision',
  railWidth: persisted.railWidth || 264,
  inspectorWidth: persisted.inspectorWidth || 400,
  dockHeight: persisted.dockHeight || 240,
  gridSort: persisted.gridSort || { key: 'score', dir: -1 },
};

const subscribers = new Map();

/** Subscribe to one topic. Returns an unsubscribe function. */
export function on(topic, fn) {
  if (!subscribers.has(topic)) subscribers.set(topic, new Set());
  subscribers.get(topic).add(fn);
  return () => subscribers.get(topic).delete(fn);
}

/** Notify subscribers of one or more topics. */
export function emit(...topics) {
  for (const topic of topics) {
    const set = subscribers.get(topic);
    if (set) for (const fn of set) fn(state);
  }
}

export function persist() {
  writePersisted({
    theme: state.theme,
    showEdges: state.showEdges,
    showLabels: state.showLabels,
    showHulls: state.showHulls,
    railOpen: state.railOpen,
    inspectorOpen: state.inspectorOpen,
    dockOpen: state.dockOpen,
    dockTab: state.dockTab,
    railWidth: state.railWidth,
    inspectorWidth: state.inspectorWidth,
    dockHeight: state.dockHeight,
    gridSort: state.gridSort,
    hiddenDomains: [...state.hiddenDomains],
    pinned: state.pinned,
  });
}

/* ------------------------------------------------------------------ actions */

export function setSlot(which, idx) {
  state.slots[which] = idx;
  if (state.slots.a === null || state.slots.b === null) state.report = null;
  emit('slots', 'map');
}

/** Click behaviour on the map and in lists: fill A, then B, then start over. */
export function pickConcept(idx) {
  const { a, b } = state.slots;
  if (a !== null && b !== null) { state.slots = { a: idx, b: null }; state.report = null; }
  else if (a === null) state.slots.a = idx;
  else if (idx !== a) state.slots.b = idx;
  state.focused = idx;
  emit('slots', 'map', 'focus');
}

export function swapSlots() {
  const { a, b } = state.slots;
  state.slots = { a: b, b: a };
  emit('slots', 'map');
}

export function clearSlots() {
  state.slots = { a: null, b: null };
  state.report = null;
  emit('slots', 'map', 'report');
}

export function setReport(report) {
  state.report = report;
  if (report) {
    const key = [report.aIdx, report.bIdx].sort((x, y) => x - y).join(':');
    state.history = [
      { key, a: report.aIdx, b: report.bIdx, distance: report.distance,
        vacancy: report.midpointVacancy, hops: report.path.length - 1, at: Date.now() },
      ...state.history.filter((h) => h.key !== key),
    ].slice(0, 60);
  }
  emit('report', 'map', 'history');
}

export function toggleDomain(domain) {
  if (state.hiddenDomains.has(domain)) state.hiddenDomains.delete(domain);
  else state.hiddenDomains.add(domain);
  persist();
  emit('domains', 'map', 'concepts');
}

export function setDomains(hidden) {
  state.hiddenDomains = new Set(hidden);
  persist();
  emit('domains', 'map', 'concepts');
}

export function isVisible(idx) {
  return !state.hiddenDomains.has(state.nodes[idx].domain);
}

export function pinCurrent(note = '') {
  const r = state.report;
  if (!r) return null;
  const key = [r.aIdx, r.bIdx].sort((x, y) => x - y).join(':');
  const existing = state.pinned.find((p) => p.key === key);
  if (existing) { existing.note = note; }
  else {
    state.pinned.unshift({
      key,
      a: state.nodes[r.aIdx].id, b: state.nodes[r.bIdx].id,
      aLabel: state.nodes[r.aIdx].label, bLabel: state.nodes[r.bIdx].label,
      distance: r.distance, vacancy: r.midpointVacancy,
      hops: r.path.length - 1,
      bridges: r.balanced.slice(0, 5).map((x) => state.nodes[x.idx].id),
      note, at: Date.now(),
    });
  }
  persist();
  emit('pinned');
  return key;
}

export function unpin(key) {
  state.pinned = state.pinned.filter((p) => p.key !== key);
  persist();
  emit('pinned');
}

export function isPinned() {
  const r = state.report;
  if (!r) return false;
  const key = [r.aIdx, r.bIdx].sort((x, y) => x - y).join(':');
  return state.pinned.some((p) => p.key === key);
}
