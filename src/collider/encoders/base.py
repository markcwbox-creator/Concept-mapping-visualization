"""Encoder interface and the prompt templates concepts are rendered into.

An encoder maps `ConceptSet -> VectorSet`.  That is the whole contract.  Swapping
residual-stream vectors for SAE features, or for an off-the-shelf sentence
embedder, is a one-line change in the config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..schema import Concept, ConceptSet, VectorSet


@dataclass(frozen=True)
class Template:
    """A way of writing a concept down for the model to read.

    `render` returns `(full_text, content_span)`.  `content_span` is the
    character range of the part we actually want to pool over — this lets us
    average over the definition itself while excluding scaffolding like
    "Define the following term:" that would otherwise contribute a large,
    concept-independent component to every vector.
    """

    name: str
    pattern: str  # uses {label}, {definition}, {domain}
    # Which field the pooled span covers: "content" (label + definition),
    # "definition", or "all" (the whole rendered string, scaffolding included).
    span: str = "content"

    def render(self, concept: Concept) -> tuple[str, tuple[int, int]]:
        text = self.pattern.format(
            label=concept.label,
            definition=concept.definition,
            domain=concept.domain,
        )
        if self.span == "all":
            return text, (0, len(text))

        target = concept.definition if self.span == "definition" else None
        if target is None:
            # "content" = from the first occurrence of the label through the
            # end of the definition.  Falls back to the whole string if the
            # template reorders things unexpectedly.
            start = text.find(concept.label)
            end_anchor = text.find(concept.definition)
            if start == -1 or end_anchor == -1:
                return text, (0, len(text))
            return text, (start, end_anchor + len(concept.definition))

        start = text.find(target)
        if start == -1:
            return text, (0, len(text))
        return text, (start, start + len(target))


# Averaging a concept's vector across several phrasings cancels out a
# surprising amount of prompt-specific noise.  With a single template you are
# partly measuring the template; with four you are mostly measuring the concept.
# Cheap win: cost scales linearly, quality of the neighbourhood structure
# improves noticeably (see docs/DESIGN.md, "Why template averaging").
DEFAULT_TEMPLATES: tuple[Template, ...] = (
    Template("bare", "{label}: {definition}"),
    Template("concept_of", "The concept of {label}. {definition}"),
    Template("domain_gloss", 'In {domain}, "{label}" means: {definition}'),
    Template("dictionary", "Entry: {label}\nDefinition: {definition}"),
)

# A single-template setting for fast iteration / layer sweeps, where relative
# comparisons matter more than absolute quality.
FAST_TEMPLATES: tuple[Template, ...] = (DEFAULT_TEMPLATES[0],)


@runtime_checkable
class Encoder(Protocol):
    """Anything that can turn concepts into vectors."""

    def encode(self, concepts: ConceptSet) -> VectorSet:  # pragma: no cover - protocol
        ...
