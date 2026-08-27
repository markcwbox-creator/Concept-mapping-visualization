"""Sampling *distant but interesting* concept pairs — the collider's fuel.

The naive version of this ("take the pairs with the largest cosine distance")
does not work, and understanding why is most of the design.

The most distant pairs in any real concept space are not the most interesting
ones; they are the *most incoherent* ones — a technical term of art against a
mundane household object, two things with no shared frame at all. There is
nothing to build between them, so a human stares at the pair and shrugs. The
genuinely productive pairs sit slightly inside the tail: far enough apart that
nobody has already connected them, close enough that a chain of three or four
steps actually exists.

So "interesting" is scored as a weighted combination of five features, each
computed on the *empirical* distribution of this particular space (no magic
constants tuned to one model):

``distance``      Distance near a target percentile of the pairwise
                  distribution, penalised for being *too* extreme. A soft
                  window, not a threshold.
``bridgeability`` A path exists through the kNN graph, and its length sits in
                  the productive range — read off this graph's own hop
                  distribution rather than assumed. Unreachable scores zero:
                  that is the incoherence filter, and it is the feature that
                  does the most work.
``domain_gap``    How far apart the two concepts' *domains* sit, measured
                  between domain centroids. Same-domain pairs score 0.
``midpoint_gap``  The semantic midpoint of the pair is in a *sparse* region —
                  measured over every concept **except the two being paired**.
                  A crowded midpoint means the connection is already densely
                  occupied by existing concepts, i.e. already thought of. An
                  empty midpoint is an actual hole in the map.
``analogy``       The two concepts occupy structurally similar neighbourhoods
                  (similar local density profile), which is what makes an
                  analogy transfer rather than merely juxtapose.

Every returned pair carries its full feature breakdown, so you can re-weight
after seeing results instead of guessing weights up front. Treat the default
weights as a starting point to argue with — see `docs/DESIGN.md`.

A NOTE ON DEGENERACY, because three of these five features were once decorative
------------------------------------------------------------------------------
A weighted sum only ranks anything if its terms actually vary. Measured on the
first real build (Qwen3-1.7B, 153 concepts), three of the five did not:

* ``domain_gap`` was **1.000 on every one of the 200 shipped pairs** — zero
  variance. Binary "different domain?" saturates: 96% of random pairs already
  cross domains at this list size, and the distance pre-filter takes that to
  100%. It was a filter wearing a feature's clothes, contributing a constant
  0.6 to every score. It is now the distance between the two domain centroids,
  which spans 0.00-1.37 and correlates only 0.12 with the pair distance.

* ``midpoint_gap`` spanned 0.129-0.144 and correlated with pair distance at
  **r = 1.000**. That is not a tuning problem, it is algebra: for unit vectors
  the normalised midpoint is equidistant from a and b at sqrt((1+cos)/2), so
  whichever of the two is nearest sets the value, and the "sparsity" of the
  midpoint was a restatement of the pair's own separation. The midpoint's
  nearest neighbour was one of the pair's own endpoints in **4000 of 4000**
  candidates. Excluding the endpoints — the question was always "does a *third*
  concept name this blend?" — drops the correlation to 0.13 and widens the
  spread 2.6x. `bridges.py` masked the endpoints all along; this file did not.

* ``bridgeability`` took **two distinct values** across the shipped pairs,
  because ``target_hops`` was a hard-coded 4.0 while 97.6% of reachable
  candidates bridge in 2 or 3. The target sat in the far tail of the graph's
  own hop distribution, so the feature ranked "longest chain" rather than
  "productive chain". The target is now read off that distribution, and path
  *cost* — computed all along, never used — breaks ties within a hop stratum.

The lesson generalises past this file: a feature that never varies is
indistinguishable from a feature that works, because both produce a plausible
ranked list. `test_features_are_not_degenerate` exists to keep that honest.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from .neighbors import KNN, shortest_path
from .schema import ConceptSet

log = logging.getLogger(__name__)


@dataclass
class PairWeights:
    """Relative importance of each interestingness feature."""

    distance: float = 1.0
    bridgeability: float = 1.4
    domain_gap: float = 0.6
    midpoint_gap: float = 1.0
    analogy: float = 0.5

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class ConceptPair:
    a: str
    b: str
    a_idx: int
    b_idx: int
    distance: float
    score: float
    features: dict[str, float] = field(default_factory=dict)
    path: list[str] = field(default_factory=list)
    path_cost: float = float("inf")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not np.isfinite(d["path_cost"]):
            d["path_cost"] = None
        return d


def _soft_window(x: np.ndarray, target: float, width: float) -> np.ndarray:
    """Gaussian bump: 1 at `target`, falling off either side. Keeps the sampler
    away from both the boring middle and the incoherent extreme."""
    return np.exp(-0.5 * ((x - target) / max(width, 1e-6)) ** 2)


def _rank01(x: np.ndarray) -> np.ndarray:
    """Rank-normalise to [0, 1] over the candidate set.

    Used where the raw quantity is real but its *scale* is arbitrary, so a
    fixed divisor squashes every candidate into a sliver of the range and the
    feature stops ranking anything. Ranking is invariant to that choice: the
    best candidate scores 1, the worst 0, and the weights in `PairWeights`
    finally mean what they appear to mean, because every term spans the same
    interval.

    The cost is that the number is now relative to the candidate pool — a
    `midpoint_gap` of 0.9 means "sparser than 90% of candidates", not "empty in
    absolute terms". Where the absolute reading is the point, keep the raw
    value too; `midpoint_gap_raw` does exactly that, and `bridges.py` reports
    `midpoint_vacancy` on an absolute scale for the same reason.
    """
    if x.size == 0:
        return x
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    return (ranks / max(x.size - 1, 1)).astype(np.float32)


def _domain_separation(vectors: np.ndarray, domains: np.ndarray) -> dict[tuple[str, str], float]:
    """Cosine distance between every pair of domain centroids.

    Replaces the binary "are these different domains?", which sounds like a
    feature and behaves like a constant: with 18 domains over 153 concepts,
    96% of random pairs already cross a domain boundary and the distance
    pre-filter takes that to 100%. Centroid separation keeps the intent —
    reward reaching across the taxonomy — while retaining a gradient, so
    biology/chemistry and biology/jurisprudence are no longer the same move.

    Still your taxonomy, and still therefore your prior, which `docs/DESIGN.md`
    §5 flags as a known weakness. This makes it a graded prior, not a true one.
    """
    labels = sorted(set(domains.tolist()))
    cent = np.stack([vectors[domains == d].mean(axis=0) for d in labels])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-8
    sep = 1.0 - cent @ cent.T
    return {(x, y): float(sep[i, j]) for i, x in enumerate(labels) for j, y in enumerate(labels)}


def _density_profile(knn: KNN) -> np.ndarray:
    """Per-concept local density signature, L1-normalised.

    Uses the *shape* of the kNN similarity curve, not its position: a concept
    in a tight cluster has a flat, high profile; an outlier has a steeply
    decaying one. Two concepts with similar profiles sit in similarly organised
    parts of the space, which is a decent proxy for "an analogy between them
    would have parallel internal structure to work with".
    """
    prof = knn.sim.astype(np.float32)
    prof = prof - prof.min(axis=1, keepdims=True)
    return prof / (prof.sum(axis=1, keepdims=True) + 1e-8)


def sample_pairs(
    vectors: np.ndarray,
    concepts: ConceptSet,
    knn: KNN,
    *,
    n_pairs: int = 200,
    n_candidates: int = 20_000,
    distance_percentile: float = 92.0,
    window_percentile_width: float = 6.0,
    target_hops: float | None = None,
    hop_width: float = 1.5,
    weights: PairWeights | None = None,
    max_per_domain_pair: int = 6,
    max_per_concept: int = 4,
    path_search_budget: int = 4000,
    seed: int = 0,
) -> list[ConceptPair]:
    """Return the top `n_pairs` scored candidate pairs.

    Two-stage by necessity: scoring every pair is O(N^2) and the path search is
    the expensive part, so we sample `n_candidates` pairs, keep the cheap
    features for all of them, and only run Dijkstra on the survivors of a
    distance pre-filter. With N=10k this runs in well under a minute.
    """
    weights = weights or PairWeights()
    rng = np.random.default_rng(seed)
    n = vectors.shape[0]
    if n < 4:
        return []

    # ---- stage 1: candidate generation -----------------------------------
    n_candidates = min(n_candidates, n * (n - 1) // 2)
    a = rng.integers(0, n, size=n_candidates * 2)
    b = rng.integers(0, n, size=n_candidates * 2)
    keep = a != b
    a, b = a[keep][:n_candidates], b[keep][:n_candidates]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    uniq = np.unique(np.stack([lo, hi], axis=1), axis=0)
    a, b = uniq[:, 0], uniq[:, 1]

    sims = np.einsum("ij,ij->i", vectors[a], vectors[b])
    dists = 1.0 - sims

    target = float(np.percentile(dists, distance_percentile))
    width = float(
        np.percentile(dists, min(99.5, distance_percentile + window_percentile_width))
        - target
    ) or float(dists.std() * 0.25)
    f_distance = _soft_window(dists, target, width)

    # Pre-filter before the expensive stage: anything scoring near zero on the
    # distance window cannot recover, whatever its other features.
    order = np.argsort(-f_distance)
    short = order[: max(n_pairs * 20, 2000)]
    a, b, dists, f_distance = a[short], b[short], dists[short], f_distance[short]

    # ---- stage 2: expensive features -------------------------------------
    domains = np.array([c.domain for c in concepts])
    sep = _domain_separation(vectors, domains)
    raw_domain = np.array(
        [0.0 if da == db else sep[(da, db)] for da, db in zip(domains[a], domains[b])],
        dtype=np.float32,
    )
    # Same-domain pairs keep a hard zero — the original intent — and everything
    # above it is graded rather than lumped together at 1.0.
    f_domain = np.where(raw_domain > 0.0, _rank01(raw_domain), 0.0).astype(np.float32)

    # Midpoint sparsity: is there a *third* concept sitting at the blend?
    #
    # The two endpoints must be excluded, and forgetting to was a silent bug
    # for the life of this file. The normalised midpoint of two unit vectors is
    # equidistant from both at sqrt((1+cos)/2), which for any realistic concept
    # list is closer than anything else in the space — so the unmasked "nearest
    # neighbour of the midpoint" was one of the pair itself in 4000 of 4000
    # measured candidates, and the feature reduced to a monotone function of
    # the pair's own distance (r = 1.000 against `dists`). Masking them drops
    # that to 0.13. `bridges.py` has always masked; this did not.
    mid = vectors[a] + vectors[b]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True) + 1e-8
    typical_nn = float(np.median(1.0 - knn.sim[:, 0]))
    mid_nn = np.empty(len(a), dtype=np.float32)
    chunk = 512
    for s0 in range(0, len(a), chunk):
        block = mid[s0 : s0 + chunk] @ vectors.T
        rows = np.arange(block.shape[0])
        block[rows, a[s0 : s0 + chunk]] = -np.inf
        block[rows, b[s0 : s0 + chunk]] = -np.inf
        mid_nn[s0 : s0 + chunk] = 1.0 - block.max(axis=1)
    # Absolute reading, kept for the UI: a multiple of the typical nearest-
    # neighbour distance, where >1 means nothing in the list names this blend.
    raw_midgap = mid_nn / (typical_nn + 1e-8)
    f_midgap = _rank01(mid_nn)

    profiles = _density_profile(knn)
    # 1 - half the L1 distance between two L1-normalised profiles == overlap.
    f_analogy = 1.0 - 0.5 * np.abs(profiles[a] - profiles[b]).sum(axis=1)

    log.info("scoring paths for %d candidate pairs", len(a))
    paths: list[list[int]] = []
    costs: list[float] = []
    hops = np.full(len(a), -1.0, dtype=np.float32)
    for i in range(len(a)):
        path, cost = shortest_path(knn, int(a[i]), int(b[i]), max_nodes=path_search_budget)
        paths.append(path)
        costs.append(cost)
        if path:
            hops[i] = len(path) - 1

    # The productive hop range is a property of the graph, not a constant. A
    # hard-coded 4.0 sat in the far tail of this build's distribution (97.6% of
    # reachable candidates bridge in 2 or 3 hops), so the feature ranked the
    # longest chain rather than the most productive one. Take the target from
    # the upper-middle of what the graph actually offers.
    reachable = hops > 0
    if target_hops is None:
        target_hops = float(np.percentile(hops[reachable], 75.0)) if reachable.any() else 3.0
        log.info("target_hops derived from graph: %.1f", target_hops)
    f_bridge = np.where(reachable, _soft_window(hops, target_hops, hop_width), 0.0)

    # Within a hop stratum the path costs do not overlap at all, so cost adds
    # resolution the hop count cannot: among chains of equal length, prefer the
    # one whose steps are tighter. A multiplicative tie-break, deliberately not
    # a second opinion on range — the window above owns that.
    cost_arr = np.asarray(costs, dtype=np.float64)
    tightness = np.zeros(len(a), dtype=np.float32)
    if reachable.any():
        per_hop = np.where(reachable, cost_arr / np.maximum(hops, 1.0), np.inf)
        finite = np.isfinite(per_hop)
        t = np.zeros(len(a), dtype=np.float32)
        t[finite] = 1.0 - _rank01(per_hop[finite])
        tightness = t
    f_bridge = (f_bridge * (0.75 + 0.25 * tightness)).astype(np.float32)

    score = (
        weights.distance * f_distance
        + weights.bridgeability * f_bridge
        + weights.domain_gap * f_domain
        + weights.midpoint_gap * f_midgap
        + weights.analogy * f_analogy
    )

    # ---- stage 3: diversity-constrained selection -------------------------
    # Without this you get 40 variations on the same two domains, because a
    # single well-separated domain pair dominates the tail of the distribution.
    ranked = np.argsort(-score)
    chosen: list[ConceptPair] = []
    domain_pair_count: dict[tuple[str, str], int] = {}
    concept_count: dict[int, int] = {}
    for r in ranked:
        if len(chosen) >= n_pairs:
            break
        ia, ib = int(a[r]), int(b[r])
        if f_bridge[r] == 0.0:
            continue  # unreachable == incoherent; hard filter, not a penalty
        dk = tuple(sorted((domains[ia], domains[ib])))
        if domain_pair_count.get(dk, 0) >= max_per_domain_pair:
            continue
        if concept_count.get(ia, 0) >= max_per_concept:
            continue
        if concept_count.get(ib, 0) >= max_per_concept:
            continue
        domain_pair_count[dk] = domain_pair_count.get(dk, 0) + 1
        concept_count[ia] = concept_count.get(ia, 0) + 1
        concept_count[ib] = concept_count.get(ib, 0) + 1
        chosen.append(
            ConceptPair(
                a=concepts[ia].id,
                b=concepts[ib].id,
                a_idx=ia,
                b_idx=ib,
                distance=float(dists[r]),
                score=float(score[r]),
                features={
                    "distance": float(f_distance[r]),
                    "bridgeability": float(f_bridge[r]),
                    "domain_gap": float(f_domain[r]),
                    "midpoint_gap": float(f_midgap[r]),
                    "midpoint_gap_raw": float(raw_midgap[r]),
                    "domain_gap_raw": float(raw_domain[r]),
                    "analogy": float(f_analogy[r]),
                    "hops": float(len(paths[r]) - 1) if paths[r] else -1.0,
                },
                path=[concepts[p].id for p in paths[r]],
                path_cost=costs[r],
            )
        )
    log.info("selected %d pairs across %d domain pairs", len(chosen), len(domain_pair_count))
    return chosen
