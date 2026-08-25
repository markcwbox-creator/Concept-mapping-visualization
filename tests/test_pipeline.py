"""Tests for the parts that can be wrong silently.

The failure mode this project has to guard against is not a crash — it is a
build that runs cleanly and produces a map whose distances mean nothing. So the
tests concentrate on the invariants that would let that happen: masking the
wrong tokens, the isotropy correction not actually correcting, kNN ordering,
and the two implementations of the collision maths drifting apart.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from collider.bridges import collide                       # noqa: E402
from collider.encoders.base import DEFAULT_TEMPLATES, Template  # noqa: E402
from collider.neighbors import build_knn, shortest_path     # noqa: E402
from collider.pairs import PairWeights, sample_pairs        # noqa: E402
from collider.schema import Concept, ConceptSet, VectorSet  # noqa: E402
from collider.space import SpaceTransform, isotropy_report, prepare  # noqa: E402
from smoke_model import make_smoke_encoder                  # noqa: E402

CONCEPTS = ROOT / "data" / "concepts" / "seed_concepts.jsonl"


@pytest.fixture(scope="module")
def concepts() -> ConceptSet:
    return ConceptSet.from_jsonl(CONCEPTS)


@pytest.fixture(scope="module")
def vectors(concepts) -> np.ndarray:
    enc = make_smoke_encoder(concepts, batch_size=16, layer=4)
    return enc.encode(concepts).vectors


# ------------------------------------------------------------------- schema


def test_concept_ids_are_unique_and_slugged(concepts):
    assert len(concepts) > 100
    assert len(set(concepts.ids)) == len(concepts)
    assert concepts.get("photosynthesis").domain == "biology"


def test_duplicate_ids_rejected():
    c = Concept(id="", label="x", definition="a thing")
    with pytest.raises(ValueError, match="duplicate"):
        ConceptSet([c, Concept(id="", label="x", definition="another thing")])


def test_empty_definition_rejected():
    with pytest.raises(ValueError, match="empty definition"):
        Concept(id="", label="x", definition="   ")


def test_vectorset_round_trip(tmp_path, concepts):
    vs = VectorSet(concepts.ids, np.random.rand(len(concepts), 8).astype(np.float32))
    vs.save(tmp_path)
    back = VectorSet.load(tmp_path)
    assert back.concept_ids == vs.concept_ids
    np.testing.assert_allclose(back.vectors, vs.vectors)


def test_artifact_version_mismatch_refuses(tmp_path, concepts):
    vs = VectorSet(concepts.ids, np.zeros((len(concepts), 4), dtype=np.float32))
    vs.save(tmp_path)
    meta = json.loads((tmp_path / "vectors.meta.json").read_text())
    meta["artifact_version"] = 999
    (tmp_path / "vectors.meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="version mismatch"):
        VectorSet.load(tmp_path)


# ---------------------------------------------------------------- templates


def test_template_span_excludes_scaffolding():
    c = Concept(id="x", label="entropy", definition="A measure of disorder.", domain="physics")
    t = Template("scaffolded", "Define the term.\n{label}: {definition}")
    text, (lo, hi) = t.render(c)
    assert text[lo:hi] == "entropy: A measure of disorder."
    assert "Define the term." not in text[lo:hi]


def test_all_default_templates_produce_a_valid_span():
    c = Concept(id="x", label="entropy", definition="A measure of disorder.", domain="physics")
    for t in DEFAULT_TEMPLATES:
        text, (lo, hi) = t.render(c)
        assert 0 <= lo < hi <= len(text)
        assert c.label in text[lo:hi]


# ------------------------------------------------------------- extraction


def test_extraction_shape_and_determinism(concepts):
    enc = make_smoke_encoder(concepts, batch_size=8, layer=3)
    a = enc.encode(concepts)
    b = enc.encode(concepts)
    assert a.vectors.shape == (len(concepts), 64)
    assert np.isfinite(a.vectors).all()
    # Same model, same input, same batching -> bitwise-stable output. If this
    # ever fails, something non-deterministic crept into pooling or batching.
    np.testing.assert_allclose(a.vectors, b.vectors, atol=1e-6)


def test_batching_does_not_change_results(concepts):
    """Padding must not leak into the pooled vector.

    This is the single most likely silent bug in the extractor: if pad tokens
    are included in the mean, a concept's vector depends on which other
    concepts happened to share its batch.
    """
    small = ConceptSet([concepts[i] for i in range(24)])
    # Both encoders build the same seeded model and the same tokenizer, so any
    # difference in output is attributable to batching alone.
    enc1 = make_smoke_encoder(small, batch_size=24, layer=3)
    enc2 = make_smoke_encoder(small, batch_size=3, layer=3)
    v1 = enc1.encode(small).vectors
    v2 = enc2.encode(small).vectors
    np.testing.assert_allclose(v1, v2, atol=1e-4)


@pytest.mark.parametrize("pooling", ["mean", "last", "max", "label_last"])
def test_all_pooling_modes_run(concepts, pooling):
    small = ConceptSet([concepts[i] for i in range(20)])
    enc = make_smoke_encoder(small, batch_size=8, layer=3, pooling=pooling)
    v = enc.encode(small).vectors
    assert v.shape == (20, 64)
    assert np.isfinite(v).all()


def test_layer_resolution(concepts):
    small = ConceptSet([concepts[i] for i in range(4)])
    enc = make_smoke_encoder(small, layers=6)
    assert enc.resolve_layer(-1) == 6
    assert enc.resolve_layer(2) == 2
    enc.layer, enc.layer_frac = None, 0.5
    assert enc.resolve_layer() == 3
    with pytest.raises(ValueError, match="out of range"):
        enc.resolve_layer(99)


def test_different_layers_give_different_vectors(concepts):
    small = ConceptSet([concepts[i] for i in range(16)])
    enc = make_smoke_encoder(small, batch_size=16)
    out = enc.encode_layers(small, [1, 5])
    assert not np.allclose(out[1], out[5])


# ------------------------------------------------------------------- space


def test_isotropy_correction_actually_corrects(vectors):
    before = isotropy_report(vectors)
    tf = SpaceTransform.fit(vectors, remove_top_k=2)
    after = isotropy_report(tf.apply(vectors))
    # The whole point of space.py: random pairs should stop looking similar,
    # and the spread of cosines should widen so thresholds mean something.
    assert abs(after["mean_cosine"]) < abs(before["mean_cosine"])
    assert abs(after["mean_cosine"]) < 0.05
    assert after["top1_variance_ratio"] < before["top1_variance_ratio"]


def test_prepare_normalises(vectors, concepts):
    vs = VectorSet(concepts.ids, vectors, {})
    prepared, _ = prepare(vs, remove_top_k=2)
    norms = np.linalg.norm(prepared.vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_transform_round_trip(tmp_path, vectors):
    tf = SpaceTransform.fit(vectors, remove_top_k=2)
    tf.save(tmp_path / "space.npz")
    back = SpaceTransform.load(tmp_path / "space.npz")
    np.testing.assert_allclose(tf.apply(vectors), back.apply(vectors), atol=1e-6)


def test_sae_vectors_skip_isotropy(concepts):
    """SAE features must not be centred — it would destroy their sparsity."""
    fake = np.abs(np.random.RandomState(0).rand(len(concepts), 32)).astype(np.float32)
    vs = VectorSet(concepts.ids, fake, {"skip_isotropy": True})
    prepared, tf = prepare(vs)
    assert tf.mean is None and tf.components is None
    assert (prepared.vectors >= 0).all()


# --------------------------------------------------------------- neighbours


def test_knn_excludes_self_and_is_sorted(vectors):
    x = SpaceTransform.fit(vectors, remove_top_k=2).apply(vectors)
    knn = build_knn(x, k=10, backend="exact")
    assert knn.idx.shape == (len(x), 10)
    for i in range(len(x)):
        assert i not in knn.idx[i]
        assert np.all(np.diff(knn.sim[i]) <= 1e-6)


def test_knn_matches_brute_force(vectors):
    x = SpaceTransform.fit(vectors, remove_top_k=2).apply(vectors)
    knn = build_knn(x, k=5, backend="exact")
    sims = x @ x.T
    np.fill_diagonal(sims, -np.inf)
    for i in (0, 7, 42):
        expected = np.argsort(-sims[i])[:5]
        np.testing.assert_array_equal(knn.idx[i], expected)


def test_shortest_path_is_symmetric_in_cost(vectors):
    x = SpaceTransform.fit(vectors, remove_top_k=2).apply(vectors)
    knn = build_knn(x, k=12, backend="exact")
    p1, c1 = shortest_path(knn, 0, 100)
    if p1:
        assert p1[0] == 0 and p1[-1] == 100
        assert c1 >= 0


# -------------------------------------------------------------------- pairs


def test_sampled_pairs_respect_diversity_caps(vectors, concepts):
    x = SpaceTransform.fit(vectors, remove_top_k=2).apply(vectors)
    knn = build_knn(x, k=12, backend="exact")
    pairs = sample_pairs(
        x, concepts, knn, n_pairs=30, n_candidates=3000,
        max_per_domain_pair=2, max_per_concept=2, weights=PairWeights(),
    )
    assert pairs
    from collections import Counter

    dom = Counter(tuple(sorted((concepts.get(p.a).domain, concepts.get(p.b).domain))) for p in pairs)
    assert max(dom.values()) <= 2
    per_concept = Counter(cid for p in pairs for cid in (p.a, p.b))
    assert max(per_concept.values()) <= 2


def test_sampled_pairs_are_all_bridgeable(vectors, concepts):
    """Unreachable pairs are filtered, not merely down-weighted."""
    x = SpaceTransform.fit(vectors, remove_top_k=2).apply(vectors)
    knn = build_knn(x, k=12, backend="exact")
    pairs = sample_pairs(x, concepts, knn, n_pairs=20, n_candidates=3000)
    assert all(p.path for p in pairs)
    assert all(np.isfinite(p.path_cost) for p in pairs)


# ------------------------------------------------------------------ bridges


def test_balanced_bridge_prefers_near_over_far(vectors, concepts):
    """The bug this guards: `1 - max/(dA+dB)` scores every equidistant point
    identically, so a balanced-but-distant concept outranks a balanced-and-close
    one. Directness must break that tie."""
    x = SpaceTransform.fit(vectors, remove_top_k=2).apply(vectors)
    knn = build_knn(x, k=12, backend="exact")
    rep = collide(x, concepts, knn, 0, 60, top_n=10)
    assert rep.balanced
    top = rep.balanced[0]
    # Among near-perfectly balanced candidates, the top one must not be beaten
    # on total distance by another equally balanced candidate.
    equally_balanced = [b for b in rep.balanced if abs(b.balance - top.balance) < 0.02]
    assert top.d_a + top.d_b <= min(b.d_a + b.d_b for b in equally_balanced) + 1e-6


def test_collide_excludes_the_poles(vectors, concepts):
    x = SpaceTransform.fit(vectors, remove_top_k=2).apply(vectors)
    knn = build_knn(x, k=12, backend="exact")
    rep = collide(x, concepts, knn, 3, 90, top_n=8)
    ids = {b.id for b in rep.balanced + rep.midpoint + rep.orthogonal}
    assert rep.a not in ids and rep.b not in ids


def test_collide_is_symmetric(vectors, concepts):
    x = SpaceTransform.fit(vectors, remove_top_k=2).apply(vectors)
    knn = build_knn(x, k=12, backend="exact")
    ab = collide(x, concepts, knn, 5, 77, top_n=6)
    ba = collide(x, concepts, knn, 77, 5, top_n=6)
    assert ab.distance == pytest.approx(ba.distance)
    assert {b.id for b in ab.balanced} == {b.id for b in ba.balanced}
    assert ab.midpoint_vacancy == pytest.approx(ba.midpoint_vacancy)


# ------------------------------------------------------- python / js parity


@pytest.mark.skipif(
    subprocess.run(["which", "node"], capture_output=True).returncode != 0,
    reason="node not available",
)
def test_js_matches_python_collision(tmp_path):
    """The browser and the CLI must agree.

    Two implementations of the same scoring drift apart the moment someone
    tunes one of them, and the drift is invisible — both sides keep producing
    plausible lists. This pins them together on the committed smoke build.
    """
    graph_path = ROOT / "web" / "data" / "graph.json"
    # `web/data` is committed so the front end works out of the box, but
    # `data/build` is gitignored — on a fresh clone the intermediate artifacts
    # this test needs are absent until someone runs a build.
    needed = [
        graph_path,
        ROOT / "web" / "data" / "vectors.i8",
        ROOT / "data" / "build" / "vectors.meta.json",
        ROOT / "data" / "build" / "knn.npz",
    ]
    if not all(p.exists() for p in needed):
        pytest.skip("no build present; run scripts/make_smoke_build.py")

    graph = json.loads(graph_path.read_text())
    concepts = ConceptSet.from_jsonl(CONCEPTS)
    vs = VectorSet.load(ROOT / "data" / "build")
    cfg_top_k = graph["meta"].get("config", {}).get("remove_top_k")
    prepared, _ = prepare(vs, remove_top_k=cfg_top_k)

    knn_blob = np.load(ROOT / "data" / "build" / "knn.npz")
    from collider.neighbors import KNN

    knn = KNN(knn_blob["idx"], knn_blob["sim"])
    a, b = 0, 60
    py = collide(prepared.vectors, concepts, knn, a, b, top_n=6)

    script = tmp_path / "run.mjs"
    script.write_text(f"""
import {{ readFileSync }} from 'node:fs';
const src = readFileSync({str(ROOT / 'web' / 'collide.js')!r}, 'utf8');
const g = {{}};
new Function('globalThis', src)(g);
const C = g.Collider;
const graph = JSON.parse(readFileSync({str(graph_path)!r}, 'utf8'));
const buf = readFileSync({str(ROOT / 'web' / 'data' / 'vectors.i8')!r});
const vecs = C.dequantise(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
                          graph.vectors.n, graph.vectors.dim, graph.vectors.scale);
const k = graph.knn.k;
const space = {{
  vecs, n: graph.vectors.n, dim: graph.vectors.dim, k,
  knnIdx: Int32Array.from(graph.knn.idx.flat()),
  knnSim: Float32Array.from(graph.knn.sim.flat()),
  typicalNN: 0.1,
}};
const rep = C.collide(space, {a}, {b}, {{ topN: 6 }});
console.log(JSON.stringify({{
  distance: rep.distance,
  path: rep.path,
  balanced: rep.balanced.map(x => x.idx),
  midpoint: rep.midpoint.map(x => x.idx),
}}));
""")
    out = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, cwd=str(ROOT)
    )
    assert out.returncode == 0, out.stderr
    js = json.loads(out.stdout)

    # int8 quantisation and the PCA projection mean the two will not agree to
    # machine precision; they must agree on the ranking and to ~2 decimals.
    assert js["distance"] == pytest.approx(py.distance, abs=0.02)
    assert js["path"] == [concepts.index_of(p) for p in py.path]
    assert js["balanced"][:3] == [b.idx for b in py.balanced][:3]
    assert js["midpoint"][:3] == [b.idx for b in py.midpoint][:3]


# ------------------------------------------------------------------- export


def test_graph_json_is_well_formed():
    path = ROOT / "web" / "data" / "graph.json"
    if not path.exists():
        pytest.skip("no build present")
    g = json.loads(path.read_text())
    assert g["format"].startswith("concept-collider/graph@")
    assert len(g["nodes"]) == g["meta"]["n_concepts"]
    for i, nd in enumerate(g["nodes"]):
        assert nd["i"] == i
        assert {"id", "label", "definition", "domain", "x", "y"} <= set(nd)
    n = len(g["nodes"])
    for e in g["edges"]:
        assert 0 <= e["s"] < n and 0 <= e["t"] < n and e["s"] != e["t"]
    for p in g["pairs"]:
        assert p["a"] != p["b"]
        assert p["path"], "every exported pair must be bridgeable"
