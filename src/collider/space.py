"""Turning raw LLM activations into a space where cosine distance means something.

**This module is the single biggest quality lever in the project.** Skipping it
is the most common reason a concept map "sort of works but every distance looks
the same".

The problem: hidden states of a trained transformer are strongly *anisotropic*.
They occupy a narrow cone rather than filling the sphere, driven by

  1. a large shared mean vector every token has a big component along, and
  2. a handful of "rogue dimensions" with variance orders of magnitude above
     the rest, which end up dominating any dot product.

The practical symptom is that raw cosine similarities between unrelated
concepts sit around 0.7-0.95 instead of near 0. The ranking is still weakly
informative, but the *magnitudes* are useless, the dynamic range is tiny, and
"pick pairs above the 90th percentile of distance" degenerates into noise —
which is exactly the operation this project is built around.

The fix, in order:

  1. **Centre** — subtract the corpus mean. Removes (1).
  2. **All-but-the-top** — project out the top-`k` principal components of the
     centred set (Mu & Viswanath, 2018). Removes (2). k = 2-8 is typical;
     `d_model // 100` is a decent rule of thumb.
  3. **Optional whitening** — equalise remaining variance across directions.
     Sharpens distances further but throws away the "importance" ordering of
     directions, and can amplify noise on small concept sets. Off by default;
     turn it on above ~5k concepts, where the covariance estimate is stable.
  4. **L2-normalise** — so cosine == dot product, and FAISS inner-product
     search is exact cosine search.

The transform is *fitted on the concept set itself* and its parameters are
stored, so new concepts added later can be projected into the same space
without re-fitting (which would silently move every existing point).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .schema import VectorSet


@dataclass
class SpaceTransform:
    """A fitted centre / de-anisotropy / whiten / normalise pipeline."""

    mean: np.ndarray | None = None
    components: np.ndarray | None = None   # (k, D) directions to remove
    scale: np.ndarray | None = None        # (D,) whitening scale, or None
    normalise: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ fit

    @classmethod
    def fit(
        cls,
        vectors: np.ndarray,
        *,
        center: bool = True,
        remove_top_k: int | None = None,
        whiten: bool = False,
        normalise: bool = True,
    ) -> "SpaceTransform":
        x = np.asarray(vectors, dtype=np.float32)
        n, d = x.shape
        if remove_top_k is None:
            # Mu & Viswanath's rule of thumb, clamped to something sane for
            # small concept sets where you cannot afford to throw away much.
            remove_top_k = int(np.clip(d // 100, 2, 8))

        mean = x.mean(axis=0) if center else None
        xc = x - mean if mean is not None else x

        components = None
        if remove_top_k and remove_top_k > 0 and n > remove_top_k + 1:
            # Economy SVD on the centred matrix: right singular vectors are the
            # principal directions. Cheaper and better conditioned than forming
            # the DxD covariance when N << D, which is the usual case here.
            _, _, vt = np.linalg.svd(xc, full_matrices=False)
            components = np.ascontiguousarray(vt[:remove_top_k], dtype=np.float32)
            xc = xc - (xc @ components.T) @ components

        scale = None
        if whiten:
            std = xc.std(axis=0)
            # Guard against dead dimensions (exactly zero variance) blowing up.
            scale = (1.0 / np.maximum(std, 1e-6)).astype(np.float32)

        return cls(
            mean=mean.astype(np.float32) if mean is not None else None,
            components=components,
            scale=scale,
            normalise=normalise,
            params={
                "center": center,
                "remove_top_k": int(remove_top_k or 0),
                "whiten": bool(whiten),
                "normalise": normalise,
                "fitted_on": int(n),
                "dim": int(d),
            },
        )

    # ------------------------------------------------------------- transform

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
            squeeze = True
        else:
            squeeze = False
        if self.mean is not None:
            x = x - self.mean
        if self.components is not None:
            x = x - (x @ self.components.T) @ self.components
        if self.scale is not None:
            x = x * self.scale
        if self.normalise:
            x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        return np.ascontiguousarray(x.astype(np.float32)[0] if squeeze else x.astype(np.float32))

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p,
            **{
                k: v
                for k, v in {
                    "mean": self.mean,
                    "components": self.components,
                    "scale": self.scale,
                }.items()
                if v is not None
            },
        )
        p.with_suffix(".json").write_text(json.dumps(self.params, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SpaceTransform":
        p = Path(path)
        blob = np.load(p if p.suffix == ".npz" else p.with_suffix(".npz"))
        params = json.loads(p.with_suffix(".json").read_text())
        return cls(
            mean=blob["mean"] if "mean" in blob else None,
            components=blob["components"] if "components" in blob else None,
            scale=blob["scale"] if "scale" in blob else None,
            normalise=params.get("normalise", True),
            params=params,
        )


def prepare(
    vs: VectorSet,
    *,
    center: bool = True,
    remove_top_k: int | None = None,
    whiten: bool = False,
) -> tuple[VectorSet, SpaceTransform]:
    """Fit and apply the transform, returning a new VectorSet.

    SAE feature vectors opt out (`skip_isotropy` in their meta): they are
    non-negative and sparse by construction, centring destroys the sparsity
    that makes them cheap and interpretable, and they do not suffer from the
    rogue-dimension problem in the first place. They still get L2-normalised.
    """
    if vs.meta.get("skip_isotropy"):
        tf = SpaceTransform(normalise=True, params={"skipped": "sae features"})
    else:
        tf = SpaceTransform.fit(
            vs.vectors, center=center, remove_top_k=remove_top_k, whiten=whiten
        )
    out = VectorSet(
        concept_ids=list(vs.concept_ids),
        vectors=tf.apply(vs.vectors),
        meta={**vs.meta, "space": "prepared", "space_params": tf.params},
    )
    return out, tf


# --------------------------------------------------------------- diagnostics


def isotropy_report(vectors: np.ndarray, sample: int = 4000, seed: int = 0) -> dict:
    """Numbers that tell you whether the space is usable.

    Run this before and after `prepare` — the difference is the whole argument
    for this module, and it is worth printing in your logs every build.

    * `mean_cosine` of random pairs should drop from ~0.5-0.9 to near 0.
    * `cosine_std` should *rise*: that is dynamic range you can threshold on.
    * `top1_variance_ratio` is how much of the variance sits in one direction;
      well above ~0.3 means rogue dimensions are still in charge.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(vectors, dtype=np.float32)
    n = x.shape[0]
    idx = rng.choice(n, size=min(sample, n), replace=False)
    xs = x[idx]
    xn = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-8)

    m = min(len(xn), 1500)
    sims = xn[:m] @ xn[:m].T
    off = sims[~np.eye(m, dtype=bool)]

    xc = xs - xs.mean(axis=0)
    sv = np.linalg.svd(xc, compute_uv=False)
    var = sv**2
    return {
        "n_sampled": int(len(idx)),
        "mean_cosine": float(off.mean()),
        "cosine_std": float(off.std()),
        "cosine_p05": float(np.percentile(off, 5)),
        "cosine_p95": float(np.percentile(off, 95)),
        "top1_variance_ratio": float(var[0] / var.sum()),
        "top8_variance_ratio": float(var[:8].sum() / var.sum()),
        "effective_rank": float(np.exp(-(lambda p: (p * np.log(p + 1e-12)).sum())(var / var.sum()))),
    }
