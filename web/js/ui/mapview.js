/**
 * The map: a Canvas 2-D scatter of the whole concept space.
 *
 * Rendering strategy
 * ------------------
 * Coordinates are precomputed in Python (UMAP or PCA) and baked into
 * `graph.json`. A force simulation in the browser is slow above a few thousand
 * nodes and lands somewhere different on every reload, which destroys the
 * spatial memory that makes a map worth having.
 *
 * Draw order is back-to-front so the reader's eye lands on the right thing:
 * grid, domain hulls, edges, the collision path, dimmed nodes, highlighted
 * nodes, labels. When a collision is active everything unrelated drops to low
 * alpha — the map becomes a diagram of that one collision rather than a
 * uniform field of dots.
 *
 * Level of detail: edges fade out and labels are suppressed when zoomed out,
 * because at that scale they convey nothing and cost most of the frame budget.
 * Everything is redrawn through `requestAnimationFrame` so a drag never queues
 * more paints than the display can show.
 */

import { emit, isVisible, pickConcept, state } from '../store.js';

let canvas, ctx, pane, tooltip, minimap, miniCtx;
let dpr = 1;
let frame = null;
let dragging = null;
let miniDragging = false;

const LABEL_ZOOM = 2.0;      // labels for everything above this zoom factor
const EDGE_ZOOM_FADE = 0.55; // edges start fading below this

export function initMap() {
  canvas = document.getElementById('map-canvas');
  ctx = canvas.getContext('2d', { alpha: false });
  pane = document.getElementById('map-pane');
  tooltip = document.getElementById('tooltip');
  minimap = document.getElementById('minimap-canvas');
  miniCtx = minimap.getContext('2d');

  const ro = new ResizeObserver(() => resize());
  ro.observe(pane);
  resize();

  canvas.addEventListener('mousedown', onDown);
  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup', onUp);
  canvas.addEventListener('mouseleave', hideTooltip);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  canvas.addEventListener('dblclick', onDoubleClick);

  minimap.addEventListener('mousedown', (e) => { miniDragging = true; onMiniDrag(e); });
  window.addEventListener('mousemove', (e) => { if (miniDragging) onMiniDrag(e); });
  window.addEventListener('mouseup', () => { miniDragging = false; });

  return { fit, focusOn, requestDraw, exportPNG };
}

/* ------------------------------------------------------------------ geometry */

function resize() {
  const r = pane.getBoundingClientRect();
  dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(r.width * dpr));
  canvas.height = Math.max(1, Math.round(r.height * dpr));
  const m = minimap.getBoundingClientRect();
  minimap.width = Math.max(1, Math.round(m.width * dpr));
  minimap.height = Math.max(1, Math.round(m.height * dpr));
  requestDraw();
}

const viewSize = () => pane.getBoundingClientRect();

/** Zoom expressed as a multiple of the "fit" scale, so LOD thresholds mean the
 *  same thing regardless of window size. */
function zoomFactor() {
  const r = viewSize();
  return state.view.scale / (Math.min(r.width, r.height) * 0.44 || 1);
}

export function fit() {
  const r = viewSize();
  state.view = { scale: Math.min(r.width, r.height) * 0.44, x: r.width / 2, y: r.height / 2 };
  requestDraw();
  emit('view');
}

export function focusOn(idx, zoom = 2.4) {
  const n = state.nodes[idx];
  if (!n) return;
  const r = viewSize();
  const base = Math.min(r.width, r.height) * 0.44;
  state.view.scale = base * zoom;
  state.view.x = r.width / 2 - n.x * state.view.scale;
  state.view.y = r.height / 2 + n.y * state.view.scale;
  requestDraw();
  emit('view');
}

const sx = (n) => n.x * state.view.scale + state.view.x;
const sy = (n) => -n.y * state.view.scale + state.view.y;

function nodeAt(px, py) {
  let best = null, bestD = 15 * 15;
  for (let i = 0; i < state.nodes.length; i++) {
    if (!isVisible(i)) continue;
    const d = (sx(state.nodes[i]) - px) ** 2 + (sy(state.nodes[i]) - py) ** 2;
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

/* -------------------------------------------------------------------- input */

function localPoint(e) {
  const r = canvas.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

function onDown(e) {
  const p = localPoint(e);
  dragging = { ...p, moved: false };
  canvas.classList.add('grabbing');
}

function onMove(e) {
  if (dragging) {
    const p = localPoint(e);
    const dx = p.x - dragging.x, dy = p.y - dragging.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) dragging.moved = true;
    state.view.x += dx; state.view.y += dy;
    dragging.x = p.x; dragging.y = p.y;
    requestDraw();
    return;
  }
  if (e.target !== canvas) { hideTooltip(); return; }
  const p = localPoint(e);
  const i = nodeAt(p.x, p.y);
  if (i !== state.hover) { state.hover = i; requestDraw(); }
  if (i === null) { hideTooltip(); return; }
  showTooltip(i, p);
}

function onUp(e) {
  if (!dragging) return;
  const wasClick = !dragging.moved && e.target === canvas;
  dragging = null;
  canvas.classList.remove('grabbing');
  if (!wasClick) return;
  const p = localPoint(e);
  const i = nodeAt(p.x, p.y);
  if (i !== null) pickConcept(i);
}

function onDoubleClick(e) {
  const p = localPoint(e);
  const i = nodeAt(p.x, p.y);
  if (i !== null) focusOn(i, 3.2);
}

function onWheel(e) {
  e.preventDefault();
  const p = localPoint(e);
  // Trackpad pinch arrives as ctrlKey+wheel with much smaller deltas.
  const k = e.ctrlKey ? 0.012 : 0.0015;
  const f = Math.exp(-e.deltaY * k);
  const next = Math.min(Math.max(state.view.scale * f, 20), 40000);
  const applied = next / state.view.scale;
  // Zoom about the cursor so whatever is under the pointer stays put.
  state.view.x = p.x - (p.x - state.view.x) * applied;
  state.view.y = p.y - (p.y - state.view.y) * applied;
  state.view.scale = next;
  requestDraw();
  emit('view');
}

function onMiniDrag(e) {
  const r = minimap.getBoundingClientRect();
  const u = (e.clientX - r.left) / r.width;
  const v = (e.clientY - r.top) / r.height;
  const wx = (u * 2 - 1) * 1.12;
  const wy = -(v * 2 - 1) * 1.12;
  const vp = viewSize();
  state.view.x = vp.width / 2 - wx * state.view.scale;
  state.view.y = vp.height / 2 + wy * state.view.scale;
  requestDraw();
}

/* ------------------------------------------------------------------ tooltip */

function showTooltip(i, p) {
  const n = state.nodes[i];
  const r = viewSize();
  const inSlot = i === state.slots.a ? 'A' : i === state.slots.b ? 'B' : null;
  tooltip.innerHTML = `
    <h4><span class="swatch" style="display:inline-block;width:8px;height:8px;border-radius:50%;
        background:${state.domainColour.get(n.domain)}"></span>
      ${escapeHTML(n.label)}<em>${escapeHTML(n.domain)}</em></h4>
    <p>${escapeHTML(n.definition)}</p>
    <div class="tt-hint">${inSlot ? `in slot ${inSlot} — click to replace` :
      state.slots.a === null ? 'click to set as A' :
      state.slots.b === null ? 'click to set as B' : 'click to start a new pair'}</div>`;
  tooltip.classList.remove('hidden');
  const w = 272, h = tooltip.offsetHeight || 120;
  tooltip.style.left = Math.min(p.x + 16, r.width - w - 8) + 'px';
  tooltip.style.top = Math.min(p.y + 16, r.height - h - 8) + 'px';
}

function hideTooltip() {
  tooltip.classList.add('hidden');
  if (state.hover !== null) { state.hover = null; requestDraw(); }
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* ------------------------------------------------------------------- render */

export function requestDraw() {
  if (frame) return;
  frame = requestAnimationFrame(() => { frame = null; draw(); });
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function draw() {
  if (!state.graph) return;
  const r = viewSize();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = cssVar('--bg-canvas');
  ctx.fillRect(0, 0, r.width, r.height);

  const zoom = zoomFactor();
  const g = state.graph;

  // Sets that drive emphasis. A collision turns the map into a diagram of that
  // collision; without one, everything is drawn at full weight.
  const pathSet = new Set(state.report ? state.report.path : []);
  const hot = new Set(pathSet);
  if (state.report) {
    for (const key of ['balanced', 'midpoint', 'orthogonal']) {
      for (const b of state.report[key].slice(0, 6)) hot.add(b.idx);
    }
  }
  if (state.slots.a !== null) hot.add(state.slots.a);
  if (state.slots.b !== null) hot.add(state.slots.b);
  const focusMode = state.report !== null;

  drawGrid(r);
  if (state.showHulls) drawHulls(r);
  if (state.showEdges) drawEdges(g, zoom, focusMode, hot);
  if (pathSet.size > 1) drawPath(r);
  drawNodes(zoom, focusMode, hot, pathSet);
  if (state.showLabels) drawLabels(r, zoom, hot);

  drawMinimap(r);
  const readout = document.getElementById('zoom-readout');
  if (readout) readout.textContent = Math.round(zoom * 100) + '%';
}

function drawGrid(r) {
  // A faint world-space grid. Gives pan and zoom a sense of physicality that a
  // field of dots on flat colour does not have.
  const step = state.view.scale * 0.25;
  if (step < 12) return;
  ctx.strokeStyle = cssVar('--grid-line');
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = state.view.x % step; x < r.width; x += step) {
    ctx.moveTo(Math.round(x) + .5, 0); ctx.lineTo(Math.round(x) + .5, r.height);
  }
  for (let y = state.view.y % step; y < r.height; y += step) {
    ctx.moveTo(0, Math.round(y) + .5); ctx.lineTo(r.width, Math.round(y) + .5);
  }
  ctx.stroke();
}

function drawHulls() {
  // Soft blobs behind each domain's members: a cheap density cue that reads as
  // "region" without the visual lie of a hard convex hull boundary.
  const byDomain = new Map();
  for (let i = 0; i < state.nodes.length; i++) {
    if (!isVisible(i)) continue;
    const d = state.nodes[i].domain;
    if (!byDomain.has(d)) byDomain.set(d, []);
    byDomain.get(d).push(i);
  }
  ctx.save();
  ctx.globalAlpha = 0.1;
  for (const [domain, idxs] of byDomain) {
    if (idxs.length < 3) continue;
    let cx = 0, cy = 0;
    for (const i of idxs) { cx += sx(state.nodes[i]); cy += sy(state.nodes[i]); }
    cx /= idxs.length; cy /= idxs.length;
    let rad = 0;
    for (const i of idxs) {
      rad += Math.hypot(sx(state.nodes[i]) - cx, sy(state.nodes[i]) - cy);
    }
    rad = (rad / idxs.length) * 1.5;
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(rad, 12));
    grad.addColorStop(0, state.domainColour.get(domain));
    grad.addColorStop(1, 'transparent');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(rad, 12), 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawEdges(g, zoom, focusMode, hot) {
  const fade = Math.min(1, zoom / EDGE_ZOOM_FADE);
  const base = Math.min(0.4, 14 / Math.sqrt(Math.max(g.edges.length, 1))) * fade;
  if (base < 0.015) return;
  const stroke = cssVar('--edge-stroke') || '125,135,151';

  ctx.lineWidth = 0.7;
  ctx.strokeStyle = `rgba(${stroke},${focusMode ? base * 0.3 : base})`;
  ctx.beginPath();
  for (const e of g.edges) {
    if (!isVisible(e.s) || !isVisible(e.t)) continue;
    if (focusMode && (hot.has(e.s) || hot.has(e.t))) continue;
    ctx.moveTo(sx(g.nodes[e.s]), sy(g.nodes[e.s]));
    ctx.lineTo(sx(g.nodes[e.t]), sy(g.nodes[e.t]));
  }
  ctx.stroke();

  if (!focusMode) return;
  // Edges touching the collision keep full weight, so the local neighbourhood
  // of the result stays legible against the dimmed field.
  ctx.strokeStyle = `rgba(${stroke},${Math.min(0.5, base * 3)})`;
  ctx.beginPath();
  for (const e of g.edges) {
    if (!isVisible(e.s) || !isVisible(e.t)) continue;
    if (!hot.has(e.s) && !hot.has(e.t)) continue;
    ctx.moveTo(sx(g.nodes[e.s]), sy(g.nodes[e.s]));
    ctx.lineTo(sx(g.nodes[e.t]), sy(g.nodes[e.t]));
  }
  ctx.stroke();
}

function drawPath() {
  const path = state.report.path;
  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  // Halo first, then the dashed line on top: keeps the route readable over
  // both dense clusters and empty canvas.
  ctx.strokeStyle = cssVar('--bg-canvas');
  ctx.lineWidth = 6;
  ctx.beginPath();
  path.forEach((i, k) => {
    const n = state.nodes[i];
    k === 0 ? ctx.moveTo(sx(n), sy(n)) : ctx.lineTo(sx(n), sy(n));
  });
  ctx.stroke();

  ctx.strokeStyle = cssVar('--accent');
  ctx.lineWidth = 2.2;
  ctx.setLineDash([7, 5]);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
}

function drawNodes(zoom, focusMode, hot, pathSet) {
  const R = Math.max(2.2, 3.1 * Math.sqrt(Math.min(zoom, 6)));
  const { a, b } = state.slots;

  for (let i = 0; i < state.nodes.length; i++) {
    if (!isVisible(i)) continue;
    const n = state.nodes[i];
    const x = sx(n), y = sy(n);
    if (x < -30 || y < -30 || x > canvas.width / dpr + 30 || y > canvas.height / dpr + 30) continue;

    const isSlot = i === a || i === b;
    const isHot = hot.has(i) || pathSet.has(i);
    ctx.globalAlpha = focusMode && !isHot ? 0.18 : (state.hover === i ? 1 : 0.92);
    ctx.beginPath();
    ctx.arc(x, y, isSlot ? R * 2.2 : isHot ? R * 1.45 : R, 0, Math.PI * 2);
    ctx.fillStyle = state.domainColour.get(n.domain) || '#888';
    ctx.fill();

    if (isSlot) {
      ctx.globalAlpha = 1;
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = i === a ? cssVar('--pole-a') : cssVar('--pole-b');
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x, y, R * 3.4, 0, Math.PI * 2);
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.45;
      ctx.stroke();
    } else if (state.hover === i) {
      ctx.globalAlpha = 1;
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = cssVar('--fg');
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
}

function drawLabels(r, zoom, hot) {
  ctx.font = `500 11px ${cssVar('--font-sans') || 'system-ui'}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  const R = Math.max(2.2, 3.1 * Math.sqrt(Math.min(zoom, 6)));
  const { a, b } = state.slots;
  const halo = cssVar('--bg-canvas');
  const fg = cssVar('--fg');
  const muted = cssVar('--fg-muted');

  // Simple occlusion: reserve a horizontal band per label and skip any that
  // would collide. Cheap, stable, and far more readable than drawing all of
  // them and letting the text pile up.
  const placed = [];
  const fits = (x, y, w) => {
    for (const p of placed) {
      if (Math.abs(p.y - y) < 12 && Math.abs(p.x - x) < (p.w + w) / 2 + 6) return false;
    }
    return true;
  };

  const order = [];
  for (let i = 0; i < state.nodes.length; i++) {
    if (!isVisible(i)) continue;
    const priority = i === a || i === b ? 0 : hot.has(i) ? 1 : i === state.hover ? 0 : 2;
    if (priority === 2 && zoom < LABEL_ZOOM) continue;
    order.push([priority, i]);
  }
  order.sort((p, q) => p[0] - q[0]);

  for (const [priority, i] of order) {
    const n = state.nodes[i];
    const x = sx(n), y = sy(n) - R - 5;
    if (x < 0 || y < 10 || x > r.width || y > r.height) continue;
    const w = ctx.measureText(n.label).width;
    if (priority === 2 && !fits(x, y, w)) continue;
    placed.push({ x, y, w });
    ctx.lineWidth = 3.5;
    ctx.strokeStyle = halo;
    ctx.strokeText(n.label, x, y);
    ctx.fillStyle = priority === 0 ? fg : priority === 1 ? fg : muted;
    ctx.fillText(n.label, x, y);
  }
}

function drawMinimap(r) {
  const w = minimap.width / dpr, h = minimap.height / dpr;
  miniCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  miniCtx.clearRect(0, 0, w, h);
  const pad = 6, span = 1.12;
  const mx = (v) => pad + ((v / span + 1) / 2) * (w - pad * 2);
  const my = (v) => pad + ((-v / span + 1) / 2) * (h - pad * 2);

  for (let i = 0; i < state.nodes.length; i++) {
    if (!isVisible(i)) continue;
    miniCtx.fillStyle = state.domainColour.get(state.nodes[i].domain) || '#888';
    miniCtx.globalAlpha = 0.75;
    miniCtx.fillRect(mx(state.nodes[i].x) - 1, my(state.nodes[i].y) - 1, 2, 2);
  }
  miniCtx.globalAlpha = 1;

  // Viewport rectangle: where the main canvas currently is in world space.
  const wx0 = (0 - state.view.x) / state.view.scale;
  const wx1 = (r.width - state.view.x) / state.view.scale;
  const wy0 = -(0 - state.view.y) / state.view.scale;
  const wy1 = -(r.height - state.view.y) / state.view.scale;
  miniCtx.strokeStyle = cssVar('--accent');
  miniCtx.lineWidth = 1;
  miniCtx.strokeRect(mx(wx0), my(wy0), mx(wx1) - mx(wx0), my(wy1) - my(wy0));
}

/* ------------------------------------------------------------------ export */

/** Snapshot the map at 2x for a slide or an issue thread. */
export function exportPNG() {
  const scratch = document.createElement('canvas');
  scratch.width = canvas.width; scratch.height = canvas.height;
  const c = scratch.getContext('2d');
  c.fillStyle = cssVar('--bg-canvas');
  c.fillRect(0, 0, scratch.width, scratch.height);
  c.drawImage(canvas, 0, 0);
  return new Promise((resolve) => scratch.toBlob(resolve, 'image/png'));
}
