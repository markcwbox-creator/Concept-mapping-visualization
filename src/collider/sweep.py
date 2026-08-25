"""Pick the layer and pooling by measuring, not by folklore.

"Use a middle layer" is repeated everywhere and is roughly right, but the
actual best layer moves with the model, the pooling, and the kind of concepts
in your list. It costs one forward pass over a few hundred concepts to find
out, so there is no excuse for guessing.

Two metrics, deliberately different in character:

``domain_purity``     Of each concept's k nearest neighbours, what fraction
                      share its domain label? Easy to interpret, but it is a
                      *topicality* measure: a layer that scores well may simply
                      be good at coarse subject classification, which is not
                      the same as good conceptual geometry. Watch it, do not
                      worship it.
``triplet_accuracy``  Given (anchor, same-domain, different-domain), how often
                      is the same-domain one closer? Same caveat, less
                      sensitive to cluster size imbalance.
``probe_accuracy``    Optional and much better, if you will spend an hour on it:
                      hand-written triplets of the form "A should be closer to
                      B than to C" that encode the relations *you* care about
                      (analogy, mechanism, part-of), not just topic. 100-200 of
                      these are worth more than any amount of domain purity.
                      See `data/probes/example_probes.jsonl`.

Read the results as a curve, not a winner. Layers are highly correlated
neighbours of each other; if 16, 18 and 20 score within noise, take the middle
and move on. A sharp peak usually means your probe set is too small.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .neighbors import build_knn
from .schema import ConceptSet
from .space import SpaceTransform, isotropy_report

log = logging.getLogger(__name__)


def domain_purity(vectors: np.ndarray, concepts: ConceptSet, k: int = 10) -> float:
    knn = build_knn(vectors, k=k, backend="exact")
    domains = np.array([c.domain for c in concepts])
    same = domains[knn.idx] == domains[:, None]
    return float(same.mean())


def triplet_accuracy(
    vectors: np.ndarray, concepts: ConceptSet, n: int = 5000, seed: int = 0
) -> float:
    rng = np.random.default_rng(seed)
    domains = np.array([c.domain for c in concepts])
    by_domain: dict[str, np.ndarray] = {
        d: np.flatnonzero(domains == d) for d in set(domains.tolist())
    }
    usable = [d for d, ix in by_domain.items() if len(ix) >= 2]
    if len(usable) < 2:
        return float("nan")

    correct = 0
    total = 0
    for _ in range(n):
        d = usable[rng.integers(len(usable))]
        a, p = rng.choice(by_domain[d], size=2, replace=False)
        other = usable[rng.integers(len(usable))]
        while other == d:
            other = usable[rng.integers(len(usable))]
        neg = by_domain[other][rng.integers(len(by_domain[other]))]
        if vectors[a] @ vectors[p] > vectors[a] @ vectors[neg]:
            correct += 1
        total += 1
    return correct / max(total, 1)


def probe_accuracy(
    vectors: np.ndarray, concepts: ConceptSet, probes: Sequence[dict[str, str]]
) -> float:
    """Accuracy on hand-written `{anchor, closer, farther}` triplets."""
    ok = 0
    used = 0
    for t in probes:
        try:
            a = concepts.index_of(t["anchor"])
            p = concepts.index_of(t["closer"])
            n = concepts.index_of(t["farther"])
        except KeyError:
            continue
        used += 1
        if vectors[a] @ vectors[p] > vectors[a] @ vectors[n]:
            ok += 1
    return ok / used if used else float("nan")


def load_probes(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            out.append(json.loads(line))
    return out


def sweep_layers(
    encoder: Any,
    concepts: ConceptSet,
    layers: Sequence[int] | None = None,
    *,
    probes: Sequence[dict[str, str]] = (),
    k: int = 10,
    remove_top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Score a range of layers in a single forward pass over `concepts`.

    Pass a *subset* of your concept list (a few hundred, domain-balanced) —
    sweeping on 10k concepts wastes an hour to learn the same thing.
    """
    encoder.load()
    n_layers = encoder._n_layers  # noqa: SLF001 - intentional, same package
    if layers is None:
        # Every other layer over the middle 80% of the stack: enough resolution
        # to see the curve without paying for the flat, uninformative ends.
        lo, hi = max(1, n_layers // 10), n_layers
        layers = list(range(lo, hi + 1, max(1, n_layers // 14)))

    log.info("sweeping layers %s of %d on %d concepts", list(layers), n_layers, len(concepts))
    raw = encoder.encode_layers(concepts, layers)

    results = []
    for layer, vecs in sorted(raw.items()):
        tf = SpaceTransform.fit(vecs, remove_top_k=remove_top_k)
        prepared = tf.apply(vecs)
        iso = isotropy_report(prepared)
        row = {
            "layer": layer,
            "depth_frac": round(layer / n_layers, 3),
            "domain_purity": round(domain_purity(prepared, concepts, k=k), 4),
            "triplet_accuracy": round(triplet_accuracy(prepared, concepts), 4),
            "mean_cosine": round(iso["mean_cosine"], 4),
            "cosine_std": round(iso["cosine_std"], 4),
            "effective_rank": round(iso["effective_rank"], 1),
        }
        if probes:
            row["probe_accuracy"] = round(probe_accuracy(prepared, concepts, probes), 4)
        results.append(row)
        log.info("layer %2d: %s", layer, row)
    return results


def format_sweep_table(results: Sequence[dict[str, Any]]) -> str:
    if not results:
        return "(no results)"
    cols = list(results[0].keys())
    widths = {c: max(len(c), max(len(str(r[c])) for r in results)) for c in cols}
    head = "  ".join(c.rjust(widths[c]) for c in cols)
    lines = [head, "-" * len(head)]
    best = max(results, key=lambda r: r.get("probe_accuracy") or r["triplet_accuracy"])
    for r in results:
        mark = " <-" if r is best else ""
        lines.append("  ".join(str(r[c]).rjust(widths[c]) for c in cols) + mark)
    return "\n".join(lines)
