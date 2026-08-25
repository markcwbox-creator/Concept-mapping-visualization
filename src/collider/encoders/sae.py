"""Sparse-autoencoder feature vectors — the interpretable alternative to raw
residual-stream vectors.

Why bother
----------
Residual-stream vectors are dense and entangled: a high cosine tells you two
concepts are "similar" but not *in what respect*. SAE features are sparse and
(often) monosemantic, so a concept becomes a short list of named-ish features.
That buys you three things this project actually wants:

1. **Explainable edges.** "These two are close because they share features
   #4471 (cyclical processes) and #9902 (energy transfer)" beats "cosine 0.71".
2. **Better blends.** A blend of two concepts in SAE space is a *union of
   features*, which is much closer to what a human means by "combine these
   two ideas" than the arithmetic midpoint of two dense vectors.
3. **Controllable distance.** You can mask out feature groups (say, syntax or
   register features) and re-measure distance on the remainder, which is how
   you find pairs that are far apart *conceptually* rather than far apart
   *stylistically*.

Cost
----
Feature dimension is typically 16x-65x d_model (e.g. 2048 -> 65k+), so dense
storage for 10k concepts is gigabytes. Hence `top_k`: we keep only the largest
activations per concept and store them sparsely. Cosine on sparse vectors is
cheap and exact.

Where to get SAEs
-----------------
* `sae_lens` (`SAE.from_pretrained`) — broadest coverage, includes Gemma Scope
  and several Llama-3.1-8B releases.
* Gemma Scope (`google/gemma-scope-*`) — very well documented, JumpReLU.
* Llama Scope (`fnlp/Llama-Scope-*`) — Llama-3.1-8B, every layer.
* EleutherAI `sae` / `sparsify` — TopK SAEs for several open models.

**The one thing you must get right**: an SAE is trained at a specific hook
point (e.g. "residual stream after block 20", or "MLP output at block 12").
Feeding it activations from anywhere else produces confident nonsense. Set
`layer` and `hook_point` to exactly what the SAE release specifies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..schema import ConceptSet, VectorSet
from .base import DEFAULT_TEMPLATES, Template
from .hf_residual import HFResidualEncoder

log = logging.getLogger(__name__)


@dataclass
class SAEWeights:
    """Minimal SAE parameterisation: encoder matrix, biases, activation."""

    W_enc: Any          # (d_model, n_features)
    b_enc: Any          # (n_features,)
    b_dec: Any          # (d_model,) - subtracted before encoding
    activation: str = "relu"        # "relu" | "jumprelu" | "topk"
    threshold: Any = None           # (n_features,) for jumprelu
    k: int | None = None            # for topk

    @property
    def n_features(self) -> int:
        return int(self.W_enc.shape[1])


def load_sae(
    source: str,
    *,
    release: str | None = None,
    sae_id: str | None = None,
    device: str = "cuda",
) -> SAEWeights:
    """Load SAE weights either via `sae_lens` or from a local npz/safetensors.

    `source` is either the literal string ``"sae_lens"`` (then `release` and
    `sae_id` are required) or a path to a file containing `W_enc`, `b_enc`,
    `b_dec` and optionally `threshold`.
    """
    import torch

    if source == "sae_lens":
        from sae_lens import SAE  # type: ignore

        if not release or not sae_id:
            raise ValueError("sae_lens source requires release= and sae_id=")
        sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
        if isinstance(sae, tuple):  # older sae_lens returns (sae, cfg, sparsity)
            sae = sae[0]
        cfg = getattr(sae, "cfg", None)
        activation = getattr(cfg, "activation_fn_str", "relu") or "relu"
        return SAEWeights(
            W_enc=sae.W_enc.detach(),
            b_enc=sae.b_enc.detach(),
            b_dec=sae.b_dec.detach(),
            activation=activation,
            threshold=getattr(sae, "threshold", None),
            k=getattr(cfg, "k", None),
        )

    path = Path(source)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file  # type: ignore

        blob = load_file(str(path))
    else:
        npz = np.load(path)
        blob = {k: torch.from_numpy(npz[k]) for k in npz.files}

    missing = {"W_enc", "b_enc", "b_dec"} - set(blob)
    if missing:
        raise ValueError(f"SAE file {path} is missing {sorted(missing)}")
    return SAEWeights(
        W_enc=blob["W_enc"].to(device),
        b_enc=blob["b_enc"].to(device),
        b_dec=blob["b_dec"].to(device),
        activation=str(blob.get("activation", "relu")),
        threshold=blob.get("threshold"),
    )


@dataclass
class SAEEncoder:
    """Concept vectors as sparse SAE feature activations.

    Produces a dense (N, F) matrix if `dense_out=True` (only sane for small F),
    otherwise a top-k sparse representation saved alongside a dense projection
    used for layout. In practice you want:

        distances  -> sparse features (exact, interpretable)
        2-D layout -> dense residual vectors (SAE space is too sparse for UMAP
                      to behave well without a lot of tuning)
    """

    model_name: str = "Qwen/Qwen3-1.7B"
    layer: int = 20
    sae_source: str = "sae_lens"
    sae_release: str | None = None
    sae_id: str | None = None
    top_k: int = 128
    pooling: str = "mean"
    templates: Sequence[Template] = DEFAULT_TEMPLATES
    batch_size: int = 8
    load_in_4bit: bool = False
    device: str | None = None
    _sae: SAEWeights | None = field(default=None, init=False, repr=False)

    def _transform(self, sae: SAEWeights):
        import torch

        def fn(acts: Any) -> Any:
            # acts: (B, T, d_model) -> (B, T, F)
            x = acts.to(sae.W_enc.dtype) - sae.b_dec
            pre = x @ sae.W_enc + sae.b_enc
            if sae.activation == "jumprelu" and sae.threshold is not None:
                return pre * (pre > sae.threshold)
            if sae.activation == "topk" and sae.k:
                vals, idx = pre.topk(sae.k, dim=-1)
                out = torch.zeros_like(pre)
                return out.scatter_(-1, idx, torch.relu(vals))
            return torch.relu(pre)

        return fn

    def encode(self, concepts: ConceptSet) -> VectorSet:
        import torch

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._sae = load_sae(
            self.sae_source,
            release=self.sae_release,
            sae_id=self.sae_id,
            device=device,
        )
        log.info("SAE loaded: %d features", self._sae.n_features)

        base = HFResidualEncoder(
            model_name=self.model_name,
            layer=self.layer,
            pooling=self.pooling,
            templates=self.templates,
            batch_size=self.batch_size,
            load_in_4bit=self.load_in_4bit,
            device=device,
            activation_transform=self._transform(self._sae),
        )
        dense = base.encode_layers(concepts, [self.layer])[self.layer]  # (N, F)

        # Sparsify: keep the top-k features per concept. Everything downstream
        # works on L2-normalised vectors, and zeroing the tail changes cosines
        # by very little (SAE activations are heavy-tailed by construction)
        # while cutting storage by ~500x.
        if self.top_k and self.top_k < dense.shape[1]:
            keep = np.argpartition(-dense, self.top_k, axis=1)[:, : self.top_k]
            sparse = np.zeros_like(dense)
            rows = np.arange(dense.shape[0])[:, None]
            sparse[rows, keep] = dense[rows, keep]
            dense = sparse

        return VectorSet(
            concept_ids=concepts.ids,
            vectors=dense,
            meta={
                "encoder": "sae",
                "model": self.model_name,
                "layer": self.layer,
                "sae_release": self.sae_release,
                "sae_id": self.sae_id,
                "n_features": int(dense.shape[1]),
                "top_k": self.top_k,
                "pooling": self.pooling,
                "templates": [t.name for t in self.templates],
                "space": "raw",
                # SAE activations are non-negative and already sparse; centring
                # and whitening (space.py) are *not* appropriate here and are
                # skipped automatically for this encoder.
                "skip_isotropy": True,
            },
        )


def top_features(
    vectors: np.ndarray, row: int, n: int = 10
) -> list[tuple[int, float]]:
    """The n strongest features for one concept — the basis of an explainable edge."""
    v = vectors[row]
    idx = np.argsort(-v)[:n]
    return [(int(i), float(v[i])) for i in idx if v[i] > 0]


def shared_features(
    vectors: np.ndarray, i: int, j: int, n: int = 10
) -> list[tuple[int, float, float]]:
    """Features active in *both* concepts, ranked by their weaker activation.

    This is the "why are these two related" explanation, and equally the "what
    survives the collision" list when blending two distant concepts.
    """
    a, b = vectors[i], vectors[j]
    both = np.minimum(a, b)
    idx = np.argsort(-both)[:n]
    return [(int(k), float(a[k]), float(b[k])) for k in idx if both[k] > 0]
