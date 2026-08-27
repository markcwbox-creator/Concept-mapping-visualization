"""A sentence-embedding baseline.

Not the point of the project, but you want it available for two reasons:

1. **A control.** If your residual-stream map is not measurably better than
   `all-MiniLM-L6-v2` at the probe triplets in `sweep.py`, something in the
   extraction is wrong (bad layer, scaffolding leaking into the pooled span,
   isotropy correction skipped). This is the fastest way to find that out.
2. **A cheap large map.** MiniLM embeds 10k definitions on CPU in a couple of
   minutes with no VRAM at all, which is a good way to iterate on the pairing
   heuristics and the front end before spending GPU time.

The trade is that you lose the thing that makes this project interesting:
sentence encoders are trained to collapse paraphrases, so their space is tuned
for retrieval similarity rather than for the internal conceptual structure an
LLM actually uses. Distant pairs from a retrieval encoder are mostly "different
topic"; distant pairs from a mid-layer residual stream are more often
"different *kind of thing*".
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schema import ConceptSet, VectorSet


@dataclass
class SentenceEncoder:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    device: str | None = None

    def encode(self, concepts: ConceptSet) -> VectorSet:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(self.model_name, device=self.device)
        texts = [f"{c.label}: {c.definition}" for c in concepts]
        vecs = model.encode(
            texts, batch_size=self.batch_size, convert_to_numpy=True,
            show_progress_bar=True, normalize_embeddings=False,
        )
        return VectorSet(
            concept_ids=concepts.ids,
            vectors=vecs,
            meta={"encoder": "sentence", "model": self.model_name, "space": "raw"},
        )
