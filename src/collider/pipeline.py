"""Stage orchestration: concepts -> vectors -> space -> graph -> web payload.

Stages are separate functions and each writes its artifact to `build_dir`, so
re-running the cheap end of the pipeline (pairs, layout, export) after tweaking
weights does not re-run the expensive end (extraction). In practice you extract
once and then iterate on the last three stages dozens of times.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .encoders.base import DEFAULT_TEMPLATES, FAST_TEMPLATES
from .export import export_graph
from .layout import compute_layout
from .neighbors import build_knn, similarity_percentiles
from .pairs import PairWeights, sample_pairs
from .schema import ConceptSet, VectorSet
from .space import isotropy_report, prepare

log = logging.getLogger(__name__)


def build_encoder(cfg: Config) -> Any:
    templates = FAST_TEMPLATES if cfg.templates == "fast" else DEFAULT_TEMPLATES
    if cfg.encoder == "sae":
        from .encoders.sae import SAEEncoder

        if cfg.layer is None:
            raise ValueError("encoder=sae requires an explicit `layer` matching the SAE")
        return SAEEncoder(
            model_name=cfg.model, layer=cfg.layer, sae_source=cfg.sae_source,
            sae_release=cfg.sae_release, sae_id=cfg.sae_id, top_k=cfg.sae_top_k,
            pooling=cfg.pooling, templates=templates, batch_size=cfg.batch_size,
            load_in_4bit=cfg.load_in_4bit, device=cfg.device,
        )
    if cfg.encoder == "sentence":
        from .encoders.sentence import SentenceEncoder

        return SentenceEncoder(model_name=cfg.model, batch_size=cfg.batch_size,
                               device=cfg.device)
    from .encoders.hf_residual import HFResidualEncoder

    return HFResidualEncoder(
        model_name=cfg.model, layer=cfg.layer, layer_frac=cfg.layer_frac,
        pooling=cfg.pooling, templates=templates, batch_size=cfg.batch_size,
        max_length=cfg.max_length, device=cfg.device, dtype=cfg.dtype,
        load_in_4bit=cfg.load_in_4bit, trust_remote_code=cfg.trust_remote_code,
    )


# --------------------------------------------------------------------- stages


def stage_extract(cfg: Config) -> VectorSet:
    """Expensive stage: run the model. Everything else is seconds."""
    concepts = ConceptSet.from_jsonl(cfg.concepts)
    log.info("loaded %d concepts across %d domains", len(concepts), len(concepts.domains))
    encoder = build_encoder(cfg)
    vs = encoder.encode(concepts)
    vs.meta["config"] = cfg.to_dict()
    vs.save(cfg.build_dir)
    log.info("raw vectors: %s", vs.vectors.shape)
    log.info("isotropy BEFORE prepare: %s", json.dumps(isotropy_report(vs.vectors), indent=2))
    return vs


def stage_build(cfg: Config) -> Path:
    """Cheap stages: space -> kNN -> pairs -> layout -> export."""
    concepts = ConceptSet.from_jsonl(cfg.concepts)
    vs = VectorSet.load(cfg.build_dir)
    if vs.concept_ids != concepts.ids:
        raise ValueError(
            "concept file and stored vectors disagree. Re-run `extract` after "
            "editing the concept list."
        )

    prepared, transform = prepare(
        vs, center=cfg.center, remove_top_k=cfg.remove_top_k, whiten=cfg.whiten
    )
    transform.save(Path(cfg.build_dir) / "space.npz")
    before = isotropy_report(vs.vectors)
    after = isotropy_report(prepared.vectors)
    log.info(
        "isotropy: mean cosine %.3f -> %.3f, std %.3f -> %.3f",
        before["mean_cosine"], after["mean_cosine"],
        before["cosine_std"], after["cosine_std"],
    )

    x = prepared.vectors
    knn = build_knn(x, k=cfg.knn_k, backend=cfg.knn_backend)
    np.savez(Path(cfg.build_dir) / "knn.npz", idx=knn.idx, sim=knn.sim)

    pcts = similarity_percentiles(x)
    log.info("cosine percentiles: %s", {k: round(v, 3) for k, v in pcts.items()})

    weights = PairWeights(**cfg.pair_weights) if cfg.pair_weights else PairWeights()
    pairs = sample_pairs(
        x, concepts, knn,
        n_pairs=cfg.n_pairs, n_candidates=cfg.n_candidates,
        distance_percentile=cfg.distance_percentile, target_hops=cfg.target_hops,
        weights=weights, max_per_domain_pair=cfg.max_per_domain_pair,
        max_per_concept=cfg.max_per_concept, seed=cfg.seed,
    )
    (Path(cfg.build_dir) / "pairs.json").write_text(
        json.dumps([p.to_dict() for p in pairs], indent=2), encoding="utf-8"
    )

    coords, method = compute_layout(x, method=cfg.layout, dims=cfg.layout_dims, seed=cfg.seed)

    meta = {
        **{k: v for k, v in vs.meta.items() if k != "config"},
        "config": cfg.to_dict(),
        "space_params": transform.params,
        "isotropy_before": before,
        "isotropy_after": after,
        "cosine_percentiles": {str(k): v for k, v in pcts.items()},
        "n_concepts": len(concepts),
    }
    path = export_graph(
        cfg.build_dir, concepts, x, knn, coords,
        pairs=pairs, meta=meta, edges_per_node=cfg.edges_per_node,
        reduced_dim=cfg.reduced_dim, embed_vectors=cfg.embed_vectors,
        layout_method=method,
    )

    web = Path(cfg.web_data_dir)
    web.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, web / "graph.json")
    blob = Path(cfg.build_dir) / "vectors.i8"
    if blob.exists():
        shutil.copy(blob, web / "vectors.i8")
    log.info("front-end payload copied to %s", web)
    return path


def stage_all(cfg: Config) -> Path:
    stage_extract(cfg)
    return stage_build(cfg)
