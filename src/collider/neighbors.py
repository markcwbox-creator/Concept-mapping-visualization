"""Nearest-neighbour search and the kNN graph everything else is built on.

Backends
--------
* ``exact``  - numpy matmul. Exact, no dependencies, fine to ~50k concepts
               (50k x 50k float32 = 10 GB if materialised at once, so we chunk).
* ``faiss``  - `IndexFlatIP` for exact, `IndexHNSWFlat` for approximate.
               Use above ~100k.
* ``annoy``  - alternative ANN if faiss is awkward to install.

Because `space.prepare` L2-normalises everything, inner product *is* cosine
similarity and cosine *distance* is `1 - ip`. No backend needs to know about
normalisation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)

Backend = Literal["exact", "faiss", "annoy", "auto"]


@dataclass
class KNN:
    """k nearest neighbours for every concept.

    `idx[i]` are the neighbour rows of concept `i`, `sim[i]` their cosine
    similarities, both sorted descending and excluding `i` itself.
    """

    idx: np.ndarray  # (N, k) int32
    sim: np.ndarray  # (N, k) float32

    @property
    def k(self) -> int:
        return int(self.idx.shape[1])


def _exact_knn(x: np.ndarray, k: int, chunk: int = 2048) -> KNN:
    n = x.shape[0]
    k_eff = min(k, n - 1)
    out_i = np.empty((n, k_eff), dtype=np.int32)
    out_s = np.empty((n, k_eff), dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = x[start:end] @ x.T                      # (c, N)
        rows = np.arange(start, end)
        sims[np.arange(end - start), rows] = -np.inf   # drop self
        part = np.argpartition(-sims, k_eff - 1, axis=1)[:, :k_eff]
        part_s = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_s, axis=1)
        out_i[start:end] = np.take_along_axis(part, order, axis=1)
        out_s[start:end] = np.take_along_axis(part_s, order, axis=1)
    return KNN(out_i, out_s)


def _faiss_knn(x: np.ndarray, k: int, approximate: bool) -> KNN:
    import faiss  # type: ignore

    n, d = x.shape
    if approximate:
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 128
    else:
        index = faiss.IndexFlatIP(d)
    index.add(x)
    sim, idx = index.search(x, min(k + 1, n))
    # The first hit is the point itself (exact) or usually is (approximate);
    # strip it by identity rather than by position so ANN reordering is safe.
    k_eff = min(k, n - 1)
    out_i = np.empty((n, k_eff), dtype=np.int32)
    out_s = np.empty((n, k_eff), dtype=np.float32)
    for i in range(n):
        keep = [(j, s) for j, s in zip(idx[i], sim[i]) if j != i and j >= 0][:k_eff]
        while len(keep) < k_eff:                       # pad short ANN results
            keep.append((keep[-1] if keep else (i, 0.0)))
        out_i[i] = [j for j, _ in keep]
        out_s[i] = [s for _, s in keep]
    return KNN(out_i, out_s)


def _annoy_knn(x: np.ndarray, k: int, n_trees: int = 50) -> KNN:
    from annoy import AnnoyIndex  # type: ignore

    n, d = x.shape
    index = AnnoyIndex(d, "angular")
    for i in range(n):
        index.add_item(i, x[i])
    index.build(n_trees)
    k_eff = min(k, n - 1)
    out_i = np.empty((n, k_eff), dtype=np.int32)
    out_s = np.empty((n, k_eff), dtype=np.float32)
    for i in range(n):
        ids, dists = index.get_nns_by_item(i, k_eff + 1, include_distances=True)
        pairs = [(j, dd) for j, dd in zip(ids, dists) if j != i][:k_eff]
        out_i[i] = [j for j, _ in pairs]
        # annoy angular distance = sqrt(2 - 2*cos) -> cos = 1 - d^2/2
        out_s[i] = [1.0 - (dd * dd) / 2.0 for _, dd in pairs]
    return KNN(out_i, out_s)


def build_knn(
    vectors: np.ndarray, k: int = 24, backend: Backend = "auto", approximate: bool | None = None
) -> KNN:
    """Build the kNN graph.

    `k` controls how connected the graph is, and therefore how easily the
    bridge search can find a path between two distant concepts. Too small and
    the graph fragments into components with no path between them at all; too
    large and every path is short and uninteresting. 16-32 works well for
    1k-10k concepts.
    """
    x = np.ascontiguousarray(vectors, dtype=np.float32)
    n = x.shape[0]
    if approximate is None:
        approximate = n > 200_000
    if backend == "auto":
        backend = "faiss" if n > 100_000 else "exact"
        if backend == "faiss":
            try:
                import faiss  # noqa: F401
            except ImportError:
                log.warning("faiss not installed, falling back to exact search")
                backend = "exact"
    log.info("kNN: n=%d k=%d backend=%s approximate=%s", n, k, backend, approximate)
    if backend == "faiss":
        return _faiss_knn(x, k, approximate)
    if backend == "annoy":
        return _annoy_knn(x, k)
    return _exact_knn(x, k)


def similarity_percentiles(
    vectors: np.ndarray, qs: tuple[float, ...] = (1, 5, 25, 50, 75, 95, 99),
    sample: int = 3000, seed: int = 0,
) -> dict[float, float]:
    """Distribution of pairwise cosine over a random sample.

    Used to turn "far apart" into a concrete threshold on *this* space rather
    than a magic constant that only happened to work for one model.
    """
    rng = np.random.default_rng(seed)
    n = vectors.shape[0]
    idx = rng.choice(n, size=min(sample, n), replace=False)
    xs = vectors[idx]
    sims = xs @ xs.T
    off = sims[~np.eye(len(idx), dtype=bool)]
    return {q: float(np.percentile(off, q)) for q in qs}


def shortest_path(
    knn: KNN, source: int, target: int, max_nodes: int | None = None
) -> tuple[list[int], float]:
    """Cheapest path from `source` to `target` through the kNN graph.

    Edge weight is cosine *distance* (1 - similarity), so the cheapest path is
    the chain of smallest conceptual leaps. This is the "stepping stones"
    view: `photosynthesis -> energy conversion -> thermodynamics -> economics
    of scarcity`, which is far more useful to a human than a flat list of
    things near the midpoint.

    Returns `([], inf)` if the graph is disconnected between the two — which is
    itself a signal: an unbridgeable pair is usually incoherent rather than
    interesting, and `pairs.py` uses that.
    """
    import heapq

    n = knn.idx.shape[0]
    dist = np.full(n, np.inf, dtype=np.float64)
    prev = np.full(n, -1, dtype=np.int64)
    dist[source] = 0.0
    heap: list[tuple[float, int]] = [(0.0, source)]
    visited = 0
    seen = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in seen:
            continue
        seen.add(u)
        visited += 1
        if u == target:
            break
        if max_nodes and visited > max_nodes:
            break
        for v, s in zip(knn.idx[u], knn.sim[u]):
            v = int(v)
            w = max(0.0, 1.0 - float(s))
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if not np.isfinite(dist[target]):
        return [], float("inf")
    path = [target]
    while path[-1] != source:
        p = int(prev[path[-1]])
        if p < 0:
            return [], float("inf")
        path.append(p)
    return path[::-1], float(dist[target])
