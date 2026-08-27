/**
 * Loading and shaping the build payload.
 *
 * `graph.json` plus a raw int8 vector blob is everything the client needs; there
 * is no API. That is deliberate — the whole app is servable from any static
 * host, and works offline once cached.
 */

import { dequantise } from './collide.js';
import { state } from './store.js';

const CAT_VARS = Array.from({ length: 18 }, (_, i) =>
  `--cat-${String(i + 1).padStart(2, '0')}`);

function catColour(i) {
  const css = getComputedStyle(document.documentElement);
  return css.getPropertyValue(CAT_VARS[i % CAT_VARS.length]).trim() || '#888';
}

export class LoadError extends Error {}

export async function loadBuild() {
  let graph;
  try {
    const res = await fetch('data/graph.json', { cache: 'no-cache' });
    if (!res.ok) throw new LoadError(`graph.json returned HTTP ${res.status}`);
    graph = await res.json();
  } catch (err) {
    throw new LoadError(
      err instanceof LoadError ? err.message :
      'Could not read data/graph.json. Is the page being served over HTTP ' +
      'rather than opened from the filesystem?');
  }

  if (!graph.format || !graph.format.startsWith('concept-collider/graph@')) {
    throw new LoadError(`Unrecognised payload format: ${graph.format || '(none)'}`);
  }

  state.graph = graph;
  state.nodes = graph.nodes;
  state.byId = new Map(graph.nodes.map((n) => [n.id, n]));

  state.domainColour = new Map(graph.domains.map((d, i) => [d, catColour(i)]));
  state.domainCounts = new Map(graph.domains.map((d) => [d, 0]));
  for (const n of graph.nodes) {
    state.domainCounts.set(n.domain, (state.domainCounts.get(n.domain) || 0) + 1);
  }

  if (graph.vectors) {
    const res = await fetch('data/' + graph.vectors.file, { cache: 'no-cache' });
    if (!res.ok) throw new LoadError(`${graph.vectors.file} returned HTTP ${res.status}`);
    const buf = await res.arrayBuffer();
    const expected = graph.vectors.n * graph.vectors.dim;
    if (buf.byteLength !== expected) {
      throw new LoadError(
        `${graph.vectors.file} is ${buf.byteLength} bytes but graph.json declares ` +
        `${graph.vectors.n}x${graph.vectors.dim} = ${expected}. The two files are ` +
        `from different builds — re-run the export.`);
    }
    const vecs = dequantise(buf, graph.vectors.n, graph.vectors.dim, graph.vectors.scale);
    const k = graph.knn.k;
    const nn = graph.knn.sim.map((row) => 1 - row[0]).sort((x, y) => x - y);
    state.space = {
      vecs, n: graph.vectors.n, dim: graph.vectors.dim, k,
      knnIdx: Int32Array.from(graph.knn.idx.flat()),
      knnSim: Float32Array.from(graph.knn.sim.flat()),
      typicalNN: nn[Math.floor(nn.length / 2)] || 0.1,
    };
  }

  return graph;
}

/** Re-resolve domain colours after a theme change (they are CSS variables). */
export function refreshColours() {
  if (!state.graph) return;
  state.domainColour = new Map(state.graph.domains.map((d, i) => [d, catColour(i)]));
}

/** Build metadata, flattened for display. */
export function buildSummary() {
  const m = (state.graph && state.graph.meta) || {};
  const cfg = m.config || {};
  return {
    model: m.model || cfg.model || 'unknown',
    encoder: m.encoder || cfg.encoder || '—',
    layer: m.layer ?? cfg.layer ?? null,
    nLayers: m.n_layers ?? null,
    pooling: m.pooling || cfg.pooling || '—',
    templates: (m.templates || []).join(', ') || '—',
    nConcepts: m.n_concepts ?? state.nodes.length,
    layout: (state.graph && state.graph.layout && state.graph.layout.method) || '—',
    isoBefore: m.isotropy_before || null,
    isoAfter: m.isotropy_after || null,
    percentiles: m.cosine_percentiles || {},
    spaceParams: m.space_params || {},
    provenance: m.provenance || null,
    vectorMeta: state.graph && state.graph.vectors,
  };
}
