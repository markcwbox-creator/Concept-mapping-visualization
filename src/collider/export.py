"""Write the front-end payload: `graph.json` plus a quantised vector blob.

The key decision here is that the browser gets *vectors*, not just a
precomputed graph. If the front end only had edges and a fixed list of
pairs, the user could explore what we already decided was interesting — which
defeats the point. Shipping vectors means any pair the user drags together can
be collided live, at 60 fps, with no server.

Making that affordable takes two steps:

* **Project down.** A PCA projection to `reduced_dim` (256 by default) keeps
  cosine similarities very close to their full-dimensional values while cutting
  the payload by ~8x for a 2048-d model. Ranking of neighbours is essentially
  unchanged at this scale; exact scores drift in the third decimal.
* **Quantise to int8.** One global scale factor, symmetric. Another 4x. The
  added error is well below the difference between two adjacent neighbours.

10k concepts x 256 dims x 1 byte = 2.5 MB, which loads instantly and lets the
browser do a full 10k-row matmul per collision in a few milliseconds.

For very large maps (>50k), flip `embed_vectors=False` and run collisions
server-side via `serve.py`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .neighbors import KNN
from .pairs import ConceptPair
from .schema import ConceptSet

log = logging.getLogger(__name__)


def quantise_int8(vectors: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric int8 quantisation with a single global scale."""
    scale = float(np.abs(vectors).max()) or 1.0
    q = np.clip(np.round(vectors / scale * 127.0), -127, 127).astype(np.int8)
    return q, scale / 127.0


def reduce_dims(vectors: np.ndarray, dim: int, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Project onto the top `dim` singular directions, preserving cosine.

    Deliberately does **not** re-centre. The input has already been through
    `space.prepare`, so it is centred and L2-normalised; subtracting the mean a
    second time (the textbook PCA step) shifts every vector and changes the
    cosines the browser computes, so the map would quietly disagree with the
    CLI about how far apart two concepts are.

    Projecting the uncentred matrix onto its own right singular vectors is an
    orthogonal transform of the span, so inner products — and therefore every
    cosine downstream — are preserved exactly whenever the retained directions
    cover the data (always true when `dim >= n_concepts`, and to within
    `variance_retained` otherwise). `test_js_matches_python_collision` pins
    this: it is the only thing standing between the two implementations and a
    silent disagreement.
    """
    if dim >= vectors.shape[1]:
        return vectors, {"reduced": False, "dim": int(vectors.shape[1])}
    _, s, vt = np.linalg.svd(vectors, full_matrices=False)
    # `full_matrices=False` yields only min(n_concepts, d_model) directions, so
    # a concept list smaller than the requested width silently produces a
    # narrower matrix. Take the true width from the array rather than from the
    # request, or graph.json ends up declaring a dimension the blob does not
    # have and the browser reads garbage.
    dim = min(dim, vt.shape[0])
    proj = vectors @ vt[:dim].T
    # Norms are preserved by the projection up to the discarded tail; renormalise
    # so any residual drift cannot accumulate into the quantisation step.
    proj /= np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8
    kept = float((s[:dim] ** 2).sum() / (s**2).sum())
    log.info("projected %d -> %d dims, %.2f%% of variance retained",
             vectors.shape[1], proj.shape[1], kept * 100)
    return proj.astype(np.float32), {
        "reduced": True, "dim": int(proj.shape[1]), "variance_retained": kept,
    }


def build_edges(
    knn: KNN, *, per_node: int = 8, min_similarity: float = 0.0
) -> list[dict[str, Any]]:
    """Deduplicated kNN edges for the map view.

    `per_node` is a *display* cap, deliberately smaller than the k used for
    path-finding: the graph you compute over should be denser than the graph
    you draw, or the picture turns into a hairball.
    """
    seen: set[tuple[int, int]] = set()
    edges: list[dict[str, Any]] = []
    for i in range(knn.idx.shape[0]):
        for j, s in zip(knn.idx[i, :per_node], knn.sim[i, :per_node]):
            j = int(j)
            if s < min_similarity:
                continue
            key = (i, j) if i < j else (j, i)
            if key in seen:
                continue
            seen.add(key)
            edges.append({"s": key[0], "t": key[1], "w": round(float(s), 4)})
    return edges


def export_graph(
    out_dir: str | Path,
    concepts: ConceptSet,
    vectors: np.ndarray,
    knn: KNN,
    coords: np.ndarray,
    *,
    pairs: Sequence[ConceptPair] = (),
    meta: dict[str, Any] | None = None,
    edges_per_node: int = 8,
    reduced_dim: int = 256,
    embed_vectors: bool = True,
    layout_method: str = "pca",
) -> Path:
    """Write `graph.json` (+ `vectors.i8`) into `out_dir`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    nodes = []
    for i, c in enumerate(concepts):
        node: dict[str, Any] = {
            "i": i,
            "id": c.id,
            "label": c.label,
            "definition": c.definition,
            "domain": c.domain,
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
        }
        if coords.shape[1] > 2:
            node["z"] = round(float(coords[i, 2]), 4)
        nodes.append(node)

    payload: dict[str, Any] = {
        "format": "concept-collider/graph@1",
        "meta": dict(meta or {}),
        "layout": {"method": layout_method, "dims": int(coords.shape[1])},
        "domains": concepts.domains,
        "nodes": nodes,
        "edges": build_edges(knn, per_node=edges_per_node),
        "pairs": [p.to_dict() for p in pairs],
        "knn": {
            "k": knn.k,
            # The full kNN index is what lets the browser draw local
            # neighbourhoods without a matmul; it is small (N*k int32).
            "idx": knn.idx.astype(np.int32).tolist(),
            "sim": np.round(knn.sim, 4).tolist(),
        },
    }

    if embed_vectors:
        reduced, red_meta = reduce_dims(vectors, reduced_dim)
        q, scale = quantise_int8(reduced)
        (out / "vectors.i8").write_bytes(q.tobytes())
        payload["vectors"] = {
            **red_meta,
            # These four are measured from the array actually written, and are
            # listed last so they win over anything red_meta reports. The blob
            # is raw bytes with no header: if `n` or `dim` here disagree with
            # it by even one, every vector the client reads is shifted.
            "file": "vectors.i8",
            "dtype": "int8",
            "n": int(q.shape[0]),
            "dim": int(q.shape[1]),
            "scale": scale,
        }
        log.info("wrote vectors.i8 (%.2f MB)", q.nbytes / 1e6)

    path = out / "graph.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s (%.2f MB, %d nodes, %d edges)",
             path, path.stat().st_size / 1e6, len(nodes), len(payload["edges"]))
    return path
