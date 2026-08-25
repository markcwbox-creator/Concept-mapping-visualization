/*
 * Client-side collision maths — a direct port of src/collider/bridges.py.
 *
 * Kept in its own file, with no DOM access, so the two implementations can be
 * diffed against each other. `tests/test_parity.py` checks that the Python and
 * JavaScript versions agree on the seed build; if you change the scoring in
 * one, change it in the other or the test fails.
 *
 * Everything operates on a single flat Float32Array of L2-normalised vectors,
 * so cosine similarity is a plain dot product and a full pass over 10k
 * concepts is one tight loop of a few million multiply-adds — about 2 ms.
 */

(function (global) {
  'use strict';

  /** Dequantise the int8 blob the exporter writes into normalised floats. */
  function dequantise(buffer, n, dim, scale) {
    const raw = new Int8Array(buffer);
    const out = new Float32Array(n * dim);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      const off = i * dim;
      for (let d = 0; d < dim; d++) {
        const v = raw[off + d] * scale;
        out[off + d] = v;
        sum += v * v;
      }
      // Re-normalise: quantisation perturbs the norm slightly, and every
      // downstream formula assumes unit vectors.
      const inv = 1 / (Math.sqrt(sum) + 1e-8);
      for (let d = 0; d < dim; d++) out[off + d] *= inv;
    }
    return out;
  }

  function dot(vecs, dim, i, j) {
    let s = 0;
    const a = i * dim, b = j * dim;
    for (let d = 0; d < dim; d++) s += vecs[a + d] * vecs[b + d];
    return s;
  }

  /** Cosine distance from every concept to concept `i`. */
  function distancesTo(vecs, n, dim, i) {
    const out = new Float32Array(n);
    const off = i * dim;
    for (let j = 0; j < n; j++) {
      let s = 0;
      const b = j * dim;
      for (let d = 0; d < dim; d++) s += vecs[off + d] * vecs[b + d];
      out[j] = 1 - s;
    }
    return out;
  }

  /** Cosine distance from every concept to an arbitrary unit vector. */
  function distancesToVector(vecs, n, dim, q) {
    const out = new Float32Array(n);
    for (let j = 0; j < n; j++) {
      let s = 0;
      const b = j * dim;
      for (let d = 0; d < dim; d++) s += q[d] * vecs[b + d];
      out[j] = 1 - s;
    }
    return out;
  }

  /** Dijkstra over the kNN graph, edge weight = 1 - similarity. */
  function shortestPath(knnIdx, knnSim, k, n, source, target) {
    const dist = new Float64Array(n).fill(Infinity);
    const prev = new Int32Array(n).fill(-1);
    const done = new Uint8Array(n);
    dist[source] = 0;
    // Binary heap of [cost, node]; n is small enough that a simple array-based
    // heap comfortably outruns anything cleverer.
    const heap = [[0, source]];
    const push = (item) => {
      heap.push(item);
      let i = heap.length - 1;
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (heap[p][0] <= heap[i][0]) break;
        [heap[p], heap[i]] = [heap[i], heap[p]];
        i = p;
      }
    };
    const pop = () => {
      const top = heap[0];
      const last = heap.pop();
      if (heap.length) {
        heap[0] = last;
        let i = 0;
        for (;;) {
          const l = 2 * i + 1, r = l + 1;
          let m = i;
          if (l < heap.length && heap[l][0] < heap[m][0]) m = l;
          if (r < heap.length && heap[r][0] < heap[m][0]) m = r;
          if (m === i) break;
          [heap[m], heap[i]] = [heap[i], heap[m]];
          i = m;
        }
      }
      return top;
    };

    while (heap.length) {
      const [d, u] = pop();
      if (done[u]) continue;
      done[u] = 1;
      if (u === target) break;
      for (let e = 0; e < k; e++) {
        const v = knnIdx[u * k + e];
        const w = Math.max(0, 1 - knnSim[u * k + e]);
        if (d + w < dist[v]) {
          dist[v] = d + w;
          prev[v] = u;
          push([dist[v], v]);
        }
      }
    }
    if (!isFinite(dist[target])) return { path: [], cost: Infinity };
    const path = [target];
    while (path[path.length - 1] !== source) {
      const p = prev[path[path.length - 1]];
      if (p < 0) return { path: [], cost: Infinity };
      path.push(p);
    }
    path.reverse();
    return { path, cost: dist[target] };
  }

  function topN(scores, n, exclude) {
    const idx = [];
    for (let i = 0; i < scores.length; i++) {
      if (exclude.has(i) || !isFinite(scores[i])) continue;
      idx.push(i);
    }
    idx.sort((a, b) => scores[b] - scores[a]);
    return idx.slice(0, n);
  }

  /**
   * The four bridge views. Mirrors bridges.collide().
   *
   * `typicalNN` is the median nearest-neighbour distance of the whole map; it
   * is what turns "the blend point is 0.31 from anything" into the meaningful
   * statement "the blend point is 2.4x further from anything than a typical
   * concept is from its nearest neighbour" — i.e. nothing in this map names it.
   */
  function collide(space, aIdx, bIdx, opts) {
    const { vecs, n, dim, knnIdx, knnSim, k, typicalNN } = space;
    const options = Object.assign({ topN: 8, balancePower: 2.0 }, opts || {});

    const simAB = dot(vecs, dim, aIdx, bIdx);
    const distAB = 1 - simAB;
    const dA = distancesTo(vecs, n, dim, aIdx);
    const dB = distancesTo(vecs, n, dim, bIdx);
    const exclude = new Set([aIdx, bIdx]);

    const balance = new Float32Array(n);
    const balScore = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const imbalance = Math.abs(dA[i] - dB[i]) / (distAB + 1e-6);
      const bal = Math.min(1, Math.max(0, 1 - imbalance));
      balance[i] = bal;
      // See bridges.py: `directness`, not a max/sum "closeness" term, which
      // would score every equidistant point identically however far away.
      const directness = Math.min(1, Math.max(0, distAB / (dA[i] + dB[i] + 1e-6)));
      balScore[i] = directness * Math.pow(bal, options.balancePower);
    }

    // The blend: normalised midpoint of the two unit vectors.
    const mid = new Float32Array(dim);
    let midNorm = 0;
    for (let d = 0; d < dim; d++) {
      mid[d] = vecs[aIdx * dim + d] + vecs[bIdx * dim + d];
      midNorm += mid[d] * mid[d];
    }
    midNorm = Math.sqrt(midNorm) + 1e-8;
    for (let d = 0; d < dim; d++) mid[d] /= midNorm;
    const dMid = distancesToVector(vecs, n, dim, mid);
    const midScore = new Float32Array(n);
    for (let i = 0; i < n; i++) midScore[i] = 1 - dMid[i];

    let nearestMid = Infinity;
    for (let i = 0; i < n; i++) if (!exclude.has(i) && dMid[i] < nearestMid) nearestMid = dMid[i];

    // The orthogonal axis: concepts unaligned with the A-B tension and far
    // from both poles — the sideways answer rather than the in-between one.
    const axis = new Float32Array(dim);
    let axisNorm = 0;
    for (let d = 0; d < dim; d++) {
      axis[d] = vecs[aIdx * dim + d] - vecs[bIdx * dim + d];
      axisNorm += axis[d] * axis[d];
    }
    axisNorm = Math.sqrt(axisNorm);
    const orthoScore = new Float32Array(n);
    if (axisNorm > 1e-6) {
      for (let d = 0; d < dim; d++) axis[d] /= axisNorm;
      const proj = new Float32Array(n);
      let maxProj = 0;
      for (let i = 0; i < n; i++) {
        let s = 0;
        const off = i * dim;
        for (let d = 0; d < dim; d++) s += axis[d] * vecs[off + d];
        proj[i] = Math.abs(s);
        if (proj[i] > maxProj) maxProj = proj[i];
      }
      for (let i = 0; i < n; i++) {
        orthoScore[i] = (1 - proj[i] / (maxProj + 1e-8)) * Math.min(dA[i], dB[i]);
      }
    }

    const pack = (indices, scores, kind) => indices.map((i) => ({
      idx: i,
      dA: dA[i],
      dB: dB[i],
      balance: balance[i],
      score: scores[i],
      kind,
    }));

    const { path, cost } = shortestPath(knnIdx, knnSim, k, n, aIdx, bIdx);

    return {
      aIdx, bIdx,
      distance: distAB,
      similarity: simAB,
      path,
      pathCost: isFinite(cost) ? cost : null,
      balanced: pack(topN(balScore, options.topN, exclude), balScore, 'balanced'),
      midpoint: pack(topN(midScore, options.topN, exclude), midScore, 'midpoint'),
      orthogonal: axisNorm > 1e-6
        ? pack(topN(orthoScore, options.topN, exclude), orthoScore, 'orthogonal')
        : [],
      midpointVacancy: nearestMid / (typicalNN + 1e-8),
    };
  }

  /** Where a distance falls in the map's own distribution of distances. */
  function distancePercentile(percentiles, distance) {
    // `percentiles` maps a percentile of *similarity* to a value; convert.
    const entries = Object.entries(percentiles)
      .map(([q, sim]) => [parseFloat(q), 1 - sim])
      .sort((a, b) => a[1] - b[1]);
    if (!entries.length) return null;
    for (let i = 0; i < entries.length; i++) {
      if (distance <= entries[i][1]) {
        // percentiles were computed on similarity, so invert the quantile too
        return 100 - entries[i][0];
      }
    }
    return 100 - entries[entries.length - 1][0];
  }

  global.Collider = { dequantise, dot, distancesTo, shortestPath, collide, distancePercentile };
})(typeof window !== 'undefined' ? window : globalThis);
