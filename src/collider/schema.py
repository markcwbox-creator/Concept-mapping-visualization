"""Core data types and on-disk artifact contracts.

Everything downstream (distances, pairing, bridges, the web front end) is
defined against the small set of types in this module.  Keeping the contract
in one place is what makes the encoder swappable: a residual-stream encoder,
an SAE-feature encoder and a plain sentence-embedding baseline all produce
the same `VectorSet`, so nothing downstream has to care which one ran.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

# Bump when the meaning of an artifact field changes in a way that would make
# an older build silently misinterpreted.  `load_vectors` refuses mismatches.
ARTIFACT_VERSION = 1

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Stable id from a human label. `Bayesian inference` -> `bayesian-inference`."""
    return _SLUG_RE.sub("-", text.strip().lower()).strip("-")


@dataclass
class Concept:
    """One node in the map.

    A concept is represented by a *short definition*, not a bare token.  This is
    deliberate: single tokens are at the mercy of BPE fragmentation and are
    wildly polysemous ("bank", "cell", "field").  A canonical phrase plus one
    sentence of definition pins down which sense we mean and gives the model
    enough context to place it.
    """

    id: str
    label: str
    definition: str
    domain: str = "misc"
    aliases: list[str] = field(default_factory=list)
    # Free-form provenance: which source list this came from, licence, etc.
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = slugify(self.label)
        self.label = self.label.strip()
        self.definition = " ".join(self.definition.split())
        if not self.label:
            raise ValueError(f"concept {self.id!r} has an empty label")
        if not self.definition:
            raise ValueError(f"concept {self.id!r} has an empty definition")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Concept":
        known = {f for f in ("id", "label", "definition", "domain", "aliases", "meta")}
        extra = {k: v for k, v in d.items() if k not in known}
        base = {k: v for k, v in d.items() if k in known}
        base.setdefault("id", "")
        c = cls(**base)  # type: ignore[arg-type]
        if extra:
            c.meta.update(extra)
        return c

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConceptSet:
    """An ordered, de-duplicated collection of concepts.

    Order matters everywhere else in the codebase: row `i` of the vector matrix
    is `concept_set[i]`.  All persistence goes through this class so that
    ordering can never drift between an extraction run and a later analysis run.
    """

    def __init__(self, concepts: Iterable[Concept]):
        self._concepts: list[Concept] = []
        self._index: dict[str, int] = {}
        for c in concepts:
            self.add(c)

    def add(self, concept: Concept) -> None:
        if concept.id in self._index:
            # Duplicate ids are a data bug, not something to silently coalesce:
            # two different senses that collide on a slug need distinct ids.
            raise ValueError(f"duplicate concept id: {concept.id!r}")
        self._index[concept.id] = len(self._concepts)
        self._concepts.append(concept)

    def __len__(self) -> int:
        return len(self._concepts)

    def __iter__(self) -> Iterator[Concept]:
        return iter(self._concepts)

    def __getitem__(self, i: int) -> Concept:
        return self._concepts[i]

    def index_of(self, concept_id: str) -> int:
        return self._index[concept_id]

    def get(self, concept_id: str) -> Concept:
        return self._concepts[self._index[concept_id]]

    @property
    def ids(self) -> list[str]:
        return [c.id for c in self._concepts]

    @property
    def domains(self) -> list[str]:
        return sorted({c.domain for c in self._concepts})

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ConceptSet":
        concepts = []
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                try:
                    concepts.append(Concept.from_dict(json.loads(line)))
                except Exception as exc:  # noqa: BLE001 - want the line number
                    raise ValueError(f"{path}:{lineno}: {exc}") from exc
        return cls(concepts)

    def to_jsonl(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for c in self._concepts:
                fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


@dataclass
class VectorSet:
    """A matrix of concept vectors plus the provenance needed to reproduce it.

    `vectors` is (N, D) float32, row-aligned to `concept_ids`.  `meta` records
    exactly how it was produced (model, layer, pooling, templates, whether the
    space has been centred/whitened) so that two builds are never silently
    compared against each other.
    """

    concept_ids: list[str]
    vectors: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {self.vectors.shape}")
        if len(self.concept_ids) != self.vectors.shape[0]:
            raise ValueError(
                f"{len(self.concept_ids)} ids but {self.vectors.shape[0]} vectors"
            )
        self.vectors = np.ascontiguousarray(self.vectors, dtype=np.float32)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "vectors.npy", self.vectors)
        payload = {
            "artifact_version": ARTIFACT_VERSION,
            "concept_ids": self.concept_ids,
            "dim": self.dim,
            "meta": self.meta,
        }
        (out / "vectors.meta.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, out_dir: str | Path) -> "VectorSet":
        out = Path(out_dir)
        payload = json.loads((out / "vectors.meta.json").read_text(encoding="utf-8"))
        version = payload.get("artifact_version")
        if version != ARTIFACT_VERSION:
            raise ValueError(
                f"artifact version mismatch: file is v{version}, code expects "
                f"v{ARTIFACT_VERSION}. Re-run extraction."
            )
        vectors = np.load(out / "vectors.npy")
        return cls(payload["concept_ids"], vectors, payload.get("meta", {}))
