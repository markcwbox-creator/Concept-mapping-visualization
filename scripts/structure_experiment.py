#!/usr/bin/env python3
"""Does embedding a structural signature actually find cross-domain analogies?

The claim under test: for retrieving a known cross-field structural analogue,
embedding a domain-neutral *structural signature* beats embedding the concept's
*definition*.

Method. `data/probes/example_probes.jsonl` contains hand-labelled triplets
written before any of this existed, 21 of which carry an explicit note that the
pair is a structural analogy (`natural selection` / `gradient descent`,
`caching` / `immune memory`, `hysteresis` / `learned helplessness`, ...). For
each such pair, rank the target among all 152 other concepts and report where
it lands, under four regimes:

    topical        cosine on definition embeddings          (the current pipeline)
    structural     cosine on structural-signature embeddings
    dual a=0.5     structural - 0.5 * topical
    dual a=1.0     structural - 1.0 * topical               (the analogy finder)

Rank 1 is perfect; chance is ~76. The labels were written to test *layer choice*,
not to flatter this idea, and every concept in the list has a signature, so
there is no opportunity to select favourable cases.

    PYTHONPATH=src python scripts/structure_experiment.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from collider.config import Config                                  # noqa: E402
from collider.pipeline import build_encoder                         # noqa: E402
from collider.schema import ConceptSet, VectorSet                   # noqa: E402
from collider.space import prepare                                  # noqa: E402
from collider.structure import (                                    # noqa: E402
    STRUCTURE_TEMPLATES, DualSpace, as_concept_set, leaked_vocabulary,
    load_structures, missing_structures,
)

CONFIG = ROOT / "configs" / "qwen3-1.7b-cpu.yaml"
STRUCTURES = ROOT / "data" / "structures" / "seed_structures.jsonl"
PROBES = ROOT / "data" / "probes" / "example_probes.jsonl"
CACHE = ROOT / "data" / "build" / "structure_vectors"


def load_analogy_probes(concepts: ConceptSet) -> list[dict]:
    """The labelled pairs, restricted to those with an explicit analogy note."""
    out = []
    for line in PROBES.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        row = json.loads(line)
        if "note" not in row:
            continue
        try:
            a = concepts.index_of(row["anchor"])
            b = concepts.index_of(row["closer"])
        except KeyError:
            continue
        out.append({"anchor": row["anchor"], "target": row["closer"],
                    "a": a, "b": b, "note": row["note"]})
    return out


def ranks(score: np.ndarray, probes: list[dict]) -> np.ndarray:
    """1-based rank of each probe's target within its anchor's ranking.

    Scored symmetrically (better of the two directions) because a structural
    analogy is a symmetric claim, while retrieval is not: an abstract concept
    has many analogues and a specific one has few.
    """
    out = []
    for p in probes:
        ab = int((score[p["a"]] > score[p["a"], p["b"]]).sum()) + 1
        ba = int((score[p["b"]] > score[p["b"], p["a"]]).sum()) + 1
        out.append(min(ab, ba))
    return np.array(out)


def report(name: str, r: np.ndarray, n: int) -> dict:
    row = {
        "regime": name,
        "median_rank": float(np.median(r)),
        "mean_rank": float(r.mean()),
        "recall@1": float((r <= 1).mean()),
        "recall@5": float((r <= 5).mean()),
        "recall@10": float((r <= 10).mean()),
    }
    print(f"  {name:<16} median {row['median_rank']:>5.1f}   mean {row['mean_rank']:>6.1f}"
          f"   r@1 {row['recall@1']:.2f}   r@5 {row['recall@5']:.2f}   r@10 {row['recall@10']:.2f}")
    return row


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    cfg = Config.load(CONFIG)
    concepts = ConceptSet.from_jsonl(ROOT / cfg.concepts)
    structures = load_structures(STRUCTURES)

    missing = missing_structures(concepts, structures)
    if missing:
        print(f"!! {len(missing)} concepts have no signature: {missing[:5]}")
        return 1
    leaks = leaked_vocabulary(concepts, structures)
    print(f"concepts {len(concepts)} · signatures {len(structures)} · "
          f"vocabulary leaks {len(leaks)}"
          + (f" {list(leaks)[:4]}" if leaks else ""))

    # --- topical vectors: reuse the committed build -------------------------
    topical_raw = VectorSet.load(ROOT / cfg.build_dir)
    if topical_raw.concept_ids != concepts.ids:
        print("!! stored vectors do not match the concept file; re-run extract")
        return 1
    topical, _ = prepare(topical_raw, remove_top_k=cfg.remove_top_k)

    # --- structural vectors: same model, same layer, structure text ---------
    if (CACHE / "vectors.meta.json").exists():
        structural_raw = VectorSet.load(CACHE)
        print(f"structural vectors: cached ({structural_raw.dim}d)")
    else:
        print("structural vectors: extracting (same model and layer as the build)…")
        struct_concepts = as_concept_set(concepts, structures)
        encoder = build_encoder(cfg)
        encoder.templates = STRUCTURE_TEMPLATES
        structural_raw = encoder.encode(struct_concepts)
        structural_raw.meta["space_kind"] = "structural"
        structural_raw.save(CACHE)
    structural, _ = prepare(structural_raw, remove_top_k=cfg.remove_top_k)

    dual = DualSpace.from_vectorsets(topical, structural)
    probes = load_analogy_probes(concepts)
    n = len(concepts)
    print(f"\n{len(probes)} labelled cross-domain analogies · "
          f"{n} concepts · chance median rank ~{n // 2}\n")

    t_score = dual.topical @ dual.topical.T
    np.fill_diagonal(t_score, -np.inf)
    s_score = dual.structural @ dual.structural.T
    np.fill_diagonal(s_score, -np.inf)

    rt = ranks(t_score, probes)
    rs = ranks(s_score, probes)

    # Reciprocal rank fusion: the standard way to merge two rankings when you
    # have no held-out data to tune a weight on. A pair scores well if EITHER
    # channel ranks it highly, which is the right shape here — the two spaces
    # turn out to succeed on disjoint sets of pairs, so a subtraction (which
    # demands both) is exactly the wrong combiner.
    K = 60.0
    t_all = np.argsort(np.argsort(-t_score, axis=1), axis=1) + 1
    s_all = np.argsort(np.argsort(-s_score, axis=1), axis=1) + 1
    rrf = 1.0 / (K + t_all) + 1.0 / (K + s_all)
    np.fill_diagonal(rrf, -np.inf)

    rows = [
        report("topical", rt, n),
        report("structural", rs, n),
        report("dual a=0.5", ranks(dual.analogy_score(0.5), probes), n),
        report("dual a=1.0", ranks(dual.analogy_score(1.0), probes), n),
        report("fused (RRF)", ranks(rrf, probes), n),
    ]
    # The ceiling of any two-channel merge: take whichever space did better.
    # Not achievable without an oracle, but it bounds what fusion can buy.
    rows.append(report("union ceiling", np.minimum(rt, rs), n))

    # Is the structural space mushier than the topical one? Uniform phrasing
    # across signatures could compress the dynamic range and explain a chunk of
    # the gap without any claim about structure being the wrong idea.
    def spread(x):
        m = min(len(x), 150)
        sims = (x[:m] @ x[:m].T)[~np.eye(m, dtype=bool)]
        return float(sims.mean()), float(sims.std())
    tm, ts_ = spread(dual.topical)
    sm, ss_ = spread(dual.structural)
    print(f"\nspace spread   topical mean {tm:+.3f} std {ts_:.3f}"
          f"   ·   structural mean {sm:+.3f} std {ss_:.3f}")

    # --- per-probe detail ---------------------------------------------------
    print("\nper-pair rank (topical -> structural):")
    order = np.argsort(rs)
    for k in order:
        p = probes[k]
        arrow = "  " if rs[k] >= rt[k] else "->"
        print(f"  {arrow} {p['anchor'][:26]:<26} x {p['target'][:26]:<26} "
              f"{rt[k]:>4} -> {rs[k]:>3}")

    print("\ntop cross-domain analogies the dual score surfaces unprompted:")
    for row in dual.top_analogies(n=12, alpha=1.0, max_per_concept=2):
        da = concepts.get(row["a"]).domain
        db = concepts.get(row["b"]).domain
        print(f"  {row['a'][:24]:<24} x {row['b'][:24]:<24} "
              f"struct {row['structural_similarity']:+.3f}  topic {row['topical_similarity']:+.3f}"
              f"   [{da} x {db}]")

    out = ROOT / "data" / "build" / "structure_experiment.json"
    out.write_text(json.dumps({"regimes": rows, "n_probes": len(probes),
                               "n_concepts": n}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
