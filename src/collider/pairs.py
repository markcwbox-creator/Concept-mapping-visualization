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
``bridgeability`` A path exists through the kNN graph, and its length is in the
                  productive range (3-5 hops). Unreachable scores zero — that
                  is the incoherence filter, and it is the feature that does
                  the most work.
``domain_gap``    The two concepts come from different labelled domains.
                  Cheap, but a good prior for "nobody has looked here".
``midpoint_gap``  The semantic midpoint of the pair is in a *sparse* region.
                  A crowded midpoint means the connection is already densely
                  occupied by existing concepts — i.e. already thought of. An
                  empty midpoint is an actual hole in the map.
``analogy``       The two concepts occupy structurally similar neighbourhoods
                  (similar local density profile), which is what makes an
                  analogy transfer rather than merely juxtapose.

Every returned pair carries its full feature breakdown, so you can re-weight
after seeing results instead of guessing weights up front. Treat the default
weights as a starting point to argue with — see `docs/DESIGN.md`.
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
    target_hops: float = 4.0,
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
    f_domain = (domains[a] != domains[b]).astype(np.float32)

    # Midpoint sparsity, measured against the typical nearest-neighbour
    # distance so the number means the same thing in any space.
    mid = vectors[a] + vectors[b]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True) + 1e-8
    typical_nn = float(np.median(1.0 - knn.sim[:, 0]))
    mid_nn = np.empty(len(a), dtype=np.float32)
    chunk = 512
    for s in range(0, len(a), chunk):
        block = mid[s : s + chunk] @ vectors.T
        mid_nn[s : s + chunk] = 1.0 - block.max(axis=1)
    f_midgap = np.clip(mid_nn / (typical_nn * 3.0 + 1e-8), 0.0, 1.0)

    profiles = _density_profile(knn)
    # 1 - half the L1 distance between two L1-normalised profiles == overlap.
    f_analogy = 1.0 - 0.5 * np.abs(profiles[a] - profiles[b]).sum(axis=1)

    log.info("scoring paths for %d candidate pairs", len(a))
    f_bridge = np.zeros(len(a), dtype=np.float32)
    paths: list[list[int]] = []
    costs: list[float] = []
    for i in range(len(a)):
        path, cost = shortest_path(knn, int(a[i]), int(b[i]), max_nodes=path_search_budget)
        paths.append(path)
        costs.append(cost)
        if path:
            hops = len(path) - 1
            f_bridge[i] = float(_soft_window(np.array([hops]), target_hops, hop_width)[0])

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
                    "analogy": float(f_analogy[r]),
                    "hops": float(len(paths[r]) - 1) if paths[r] else -1.0,
                },
                path=[concepts[p].id for p in paths[r]],
                path_cost=costs[r],
            )
        )
    log.info("selected %d pairs across %d domain pairs", len(chosen), len(domain_pair_count))
    return chosen
