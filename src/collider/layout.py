"""2-D / 3-D coordinates for the map view.

Layout is computed in Python and baked into `graph.json` rather than run in the
browser. Two reasons: a force-directed layout of 10k nodes in JavaScript is
both slow and non-deterministic (the map looks different every reload, which
destroys the user's spatial memory of it), and UMAP's structure is far more
faithful to the actual geometry than a spring simulation on a thresholded
graph.

Backends, in order of preference:

* ``umap``  - best local structure; needs `umap-learn`. Use `metric="cosine"`.
* ``pca``   - numpy only, deterministic, instant. Globally faithful but
              squashes local structure. Always available, and honestly good
              enough below ~1k concepts.
* ``tsne``  - via scikit-learn if you have it and prefer its cluster
              separation. Worse global structure than UMAP.

Whatever runs, the output is scaled into a stable [-1, 1] box so the front end
never has to guess a viewport.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def _rescale(coords: np.ndarray) -> np.ndarray:
    """Centre and scale into [-1, 1] preserving aspect ratio."""
    c = coords - coords.mean(axis=0)
    scale = np.abs(c).max()
    if scale < 1e-8:
        return c.astype(np.float32)
    return (c / scale).astype(np.float32)


def pca_layout(vectors: np.ndarray, dims: int = 2) -> np.ndarray:
    x = vectors - vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return _rescale(x @ vt[:dims].T)


def umap_layout(
    vectors: np.ndarray,
    dims: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    import umap  # type: ignore

    reducer = umap.UMAP(
        n_components=dims,
        n_neighbors=min(n_neighbors, max(2, len(vectors) - 1)),
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    return _rescale(reducer.fit_transform(vectors))


def tsne_layout(vectors: np.ndarray, dims: int = 2, seed: int = 0) -> np.ndarray:
    from sklearn.manifold import TSNE  # type: ignore

    perplexity = min(30, max(5, len(vectors) // 4))
    ts = TSNE(
        n_components=dims, metric="cosine", init="pca",
        perplexity=perplexity, random_state=seed,
    )
    return _rescale(ts.fit_transform(vectors))


def compute_layout(
    vectors: np.ndarray, method: str = "auto", dims: int = 2, seed: int = 0
) -> tuple[np.ndarray, str]:
    """Returns (coords, method_actually_used)."""
    if method == "auto":
        try:
            import umap  # noqa: F401

            method = "umap"
        except ImportError:
            log.info("umap-learn not installed; using PCA layout")
            method = "pca"
    try:
        if method == "umap":
            return umap_layout(vectors, dims=dims, seed=seed), "umap"
        if method == "tsne":
            return tsne_layout(vectors, dims=dims, seed=seed), "tsne"
    except Exception as exc:  # noqa: BLE001 - layout must never fail a build
        log.warning("%s layout failed (%s); falling back to PCA", method, exc)
    return pca_layout(vectors, dims=dims), "pca"
