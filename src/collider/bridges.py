"""What sits *between* two concepts — the payload of the collider workspace.

When a user pulls two distant concepts together, "here is a list of things
near the midpoint" is a weak answer. Four different questions are interesting,
and they have genuinely different answers:

1. **Stepping stones.** The cheapest chain through the kNN graph. This is a
   narrative: A leads to X leads to Y leads to B. It is the view humans find
   most immediately legible, and it never invents anything.
2. **Balanced bridges.** Concepts roughly *equidistant* from both, ranked by
   how balanced they are as well as how close. A concept 0.2 from A and 0.9
   from B is not a bridge, it is a neighbour of A — but a naive
   "sum of distances" ranking happily returns it. We penalise imbalance
   explicitly.
3. **The blend.** The normalised midpoint vector is a point in space that may
   have no concept sitting on it. Its nearest existing concepts describe what
   the blend "would be like", and the distance to the nearest one measures how
   *vacant* that spot is — a vacant blend point is the interesting case, since
   it is a description with no name.
4. **The axis.** A - B is a direction. Projecting the whole concept set onto it
   gives the spectrum the pair defines, and the concepts sitting near its
   middle *while being far from both endpoints* are a different, often
   stranger, kind of bridge: things that are orthogonal to the tension rather
   than between its poles.

None of these require an LLM at query time. Everything is a matmul against a
matrix already in memory, so the UI can stay interactive.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .neighbors import KNN, shortest_path
from .schema import ConceptSet


@dataclass
class Bridge:
    id: str
    idx: int
    d_a: float          # cosine distance to concept A
    d_b: float          # cosine distance to concept B
    balance: float      # 1.0 == perfectly equidistant, 0.0 == entirely one-sided
    score: float
    kind: str           # "balanced" | "midpoint" | "orthogonal"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CollisionReport:
    """Everything the front end needs to render one collision."""

    a: str
    b: str
    distance: float
    similarity: float
    distance_percentile: float | None
    path: list[str]
    path_cost: float | None
    balanced: list[Bridge] = field(default_factory=list)
    midpoint: list[Bridge] = field(default_factory=list)
    orthogonal: list[Bridge] = field(default_factory=list)
    midpoint_vacancy: float = 0.0
    shared_features: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("balanced", "midpoint", "orthogonal"):
            d[key] = [b if isinstance(b, dict) else b for b in d[key]]
        return d


def collide(
    vectors: np.ndarray,
    concepts: ConceptSet,
    knn: KNN,
    a_idx: int,
    b_idx: int,
    *,
    top_n: int = 10,
    balance_power: float = 2.0,
    typical_nn: float | None = None,
    distance_percentile_fn: Any = None,
) -> CollisionReport:
    """Compute all four bridge views for one pair.

    `balance_power` controls how hard imbalance is punished. At 1.0 you get
    something close to "sum of distances"; at 2.0 (default) a concept has to be
    genuinely between the two to rank. Raise it if your bridges keep looking
    like neighbours of one side.
    """
    a_vec, b_vec = vectors[a_idx], vectors[b_idx]
    sim_ab = float(a_vec @ b_vec)
    dist_ab = 1.0 - sim_ab

    d_a = 1.0 - vectors @ a_vec
    d_b = 1.0 - vectors @ b_vec

    mask = np.ones(len(vectors), dtype=bool)
    mask[[a_idx, b_idx]] = False

    # --- 1. stepping stones -------------------------------------------------
    path_idx, path_cost = shortest_path(knn, a_idx, b_idx)
    path = [concepts[p].id for p in path_idx]

    # --- 2. balanced bridges ------------------------------------------------
    # balance in [0, 1]: 1 when d_a == d_b, decaying with the imbalance
    # expressed as a fraction of the pair's own separation, so it is scale-free.
    imbalance = np.abs(d_a - d_b) / (dist_ab + 1e-6)
    balance = np.clip(1.0 - imbalance, 0.0, 1.0)
    # `directness` is how close the concept sits to the straight line between
    # the two: 1.0 means the detour through it costs nothing beyond the pair's
    # own separation, and it falls off as the detour grows.
    #
    # Note what this must NOT be. `1 - max(dA,dB)/(dA+dB)` looks like a
    # reasonable closeness term and is worthless: it equals 0.5 for *every*
    # equidistant point regardless of distance, so a concept 1.05 away from
    # both poles ranks level with one 0.6 away from both. Balance alone does
    # not make a bridge — the far-but-balanced points are the ones the
    # "orthogonal" view is for.
    directness = np.clip(dist_ab / (d_a + d_b + 1e-6), 0.0, 1.0)
    bal_score = directness * (balance**balance_power)
    bal_score[~mask] = -np.inf
    bal_top = np.argsort(-bal_score)[:top_n]
    balanced = [
        Bridge(
            id=concepts[int(i)].id, idx=int(i), d_a=float(d_a[i]), d_b=float(d_b[i]),
            balance=float(balance[i]), score=float(bal_score[i]), kind="balanced",
        )
        for i in bal_top if np.isfinite(bal_score[i])
    ]

    # --- 3. the blend -------------------------------------------------------
    mid = a_vec + b_vec
    mid /= np.linalg.norm(mid) + 1e-8
    mid_sim = vectors @ mid
    mid_sim_masked = np.where(mask, mid_sim, -np.inf)
    mid_top = np.argsort(-mid_sim_masked)[:top_n]
    midpoint = [
        Bridge(
            id=concepts[int(i)].id, idx=int(i), d_a=float(d_a[i]), d_b=float(d_b[i]),
            balance=float(balance[i]), score=float(mid_sim[i]), kind="midpoint",
        )
        for i in mid_top if np.isfinite(mid_sim_masked[i])
    ]
    # How empty is the blend point? Expressed as a multiple of the typical
    # nearest-neighbour distance: >1 means "nothing in your concept list
    # actually names this blend", which is the interesting outcome.
    if typical_nn is None:
        typical_nn = float(np.median(1.0 - knn.sim[:, 0]))
    nearest_mid_dist = float(1.0 - mid_sim_masked.max())
    vacancy = float(nearest_mid_dist / (typical_nn + 1e-8))

    # --- 4. the orthogonal axis --------------------------------------------
    axis = a_vec - b_vec
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 1e-6:
        axis = axis / axis_norm
        proj = np.abs(vectors @ axis)          # 0 == orthogonal to the tension
        far_from_both = np.minimum(d_a, d_b)   # and not simply near either pole
        ortho_score = (1.0 - proj / (proj.max() + 1e-8)) * far_from_both
        ortho_score[~mask] = -np.inf
        ortho_top = np.argsort(-ortho_score)[:top_n]
        orthogonal = [
            Bridge(
                id=concepts[int(i)].id, idx=int(i), d_a=float(d_a[i]), d_b=float(d_b[i]),
                balance=float(balance[i]), score=float(ortho_score[i]), kind="orthogonal",
            )
            for i in ortho_top if np.isfinite(ortho_score[i])
        ]
    else:
        orthogonal = []

    pct = None
    if distance_percentile_fn is not None:
        pct = float(distance_percentile_fn(dist_ab))

    return CollisionReport(
        a=concepts[a_idx].id,
        b=concepts[b_idx].id,
        distance=dist_ab,
        similarity=sim_ab,
        distance_percentile=pct,
        path=path,
        path_cost=None if not np.isfinite(path_cost) else float(path_cost),
        balanced=balanced,
        midpoint=midpoint,
        orthogonal=orthogonal,
        midpoint_vacancy=vacancy,
    )


def attach_sae_explanation(
    report: CollisionReport, sae_vectors: np.ndarray, a_idx: int, b_idx: int, top: int = 8
) -> CollisionReport:
    """Add "why are these related" feature evidence, when SAE vectors exist.

    Kept separate from `collide` because it needs a *different* vector set
    (sparse features) than the one used for geometry (dense residual). Both can
    coexist in a build; see `pipeline.py`.
    """
    from .encoders.sae import shared_features

    report.shared_features = [
        {"feature": f, "a_act": av, "b_act": bv}
        for f, av, bv in shared_features(sae_vectors, a_idx, b_idx, n=top)
    ]
    return report
