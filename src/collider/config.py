"""Build configuration.

One flat config object, loadable from YAML, overridable from the command line.
The point is that a build is fully described by this object plus the concept
file — so `configs/*.yaml` is the reproducibility unit you commit, and the
config is copied verbatim into `graph.json` so any map can be traced back to
the settings that made it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class Config:
    # --- inputs -----------------------------------------------------------
    concepts: str = "data/concepts/seed_concepts.jsonl"
    probes: str = "data/probes/example_probes.jsonl"
    build_dir: str = "data/build"
    web_data_dir: str = "web/data"

    # --- encoder ----------------------------------------------------------
    encoder: str = "hf_residual"           # hf_residual | sae | sentence
    model: str = "Qwen/Qwen3-1.7B"
    layer: int | None = None               # None -> layer_frac
    layer_frac: float = 0.65
    pooling: str = "mean"                  # mean | last | max | label_last
    templates: str = "default"             # default | fast
    batch_size: int = 16
    max_length: int = 128
    dtype: str = "float16"
    load_in_4bit: bool = False
    device: str | None = None
    trust_remote_code: bool = False

    # --- SAE (only used when encoder == "sae") ----------------------------
    sae_source: str = "sae_lens"
    sae_release: str | None = None
    sae_id: str | None = None
    sae_top_k: int = 128

    # --- space ------------------------------------------------------------
    center: bool = True
    remove_top_k: int | None = None        # None -> dim//100, clamped to [2, 8]
    whiten: bool = False

    # --- graph ------------------------------------------------------------
    knn_k: int = 24
    knn_backend: str = "auto"
    edges_per_node: int = 8

    # --- pairs ------------------------------------------------------------
    n_pairs: int = 200
    n_candidates: int = 20_000
    distance_percentile: float = 92.0
    target_hops: float = 4.0
    max_per_domain_pair: int = 6
    max_per_concept: int = 4
    pair_weights: dict[str, float] = field(default_factory=dict)

    # --- layout / export --------------------------------------------------
    layout: str = "auto"                   # auto | umap | pca | tsne
    layout_dims: int = 2
    reduced_dim: int = 256
    embed_vectors: bool = True
    seed: int = 0

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def merge_cli(self, overrides: dict[str, Any]) -> "Config":
        """Apply non-None CLI overrides, returning a new Config."""
        data = asdict(self)
        for k, v in overrides.items():
            if v is not None and k in data:
                data[k] = v
        return Config(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
