"""Structural signature vectors — the "same shape, different subject" space.

Why this exists
---------------
Residual-stream cosine measures *semantic* similarity: whether two concepts are
talked about in similar terms. That is the wrong instrument for finding
`antibiotic resistance` ~ `SEO spam adaptation` ~ `tax-loophole evolution`. All
three are variation-under-selection, and semantic distance puts them far apart
*precisely because* their subject matter differs. The thing that makes them the
same is invisible to a measure built out of topical content.

So this module builds a second, orthogonal space. Instead of asking "what is
this concept about", it asks a fixed inventory of questions about the concept's
*form* — is there a feedback loop, a threshold, a conserved quantity, a
bottleneck, an arms race — and represents the concept by its answers.

Two concepts with nothing topical in common land in the same place in this
space if and only if they have the same underlying structure. That is the
whole point.

How the answers are obtained
----------------------------
Not by sampling text. For each (concept, predicate) pair we run one forward
pass and read the model's logits for " Yes" versus " No" at the answer
position. The score is `log P(yes) - log P(no)` — a continuous, deterministic,
calibratable quantity. Sampling would give a coin flip; the logit difference
gives a graded judgement, which is what a vector needs.

Calibration matters as much as it did for isotropy
--------------------------------------------------
Raw yes/no logits are badly biased per predicate: some questions get "yes" for
almost every concept ("does it involve a trade-off?"), others almost never.
Left uncorrected, those predicates contribute a large constant to every vector
and swamp the discriminative ones — exactly the anisotropy problem in a
different costume. `standardise()` z-scores each predicate across the concept
set, so every axis contributes on the basis of how much it *distinguishes*
concepts rather than how agreeable the question is.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..schema import ConceptSet, VectorSet

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Predicate:
    id: str
    probe: str
    family: str = "misc"


def load_predicates(path: str | Path) -> list[Predicate]:
    out: list[Predicate] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        d = json.loads(line)
        out.append(Predicate(id=d["id"], probe=d["probe"], family=d.get("family", "misc")))
    if not out:
        raise ValueError(f"no predicates found in {path}")
    return out


# Asking the same question two ways and averaging cancels a surprising amount
# of phrasing bias, the same trick template averaging plays in hf_residual.
PROMPT_TEMPLATES: tuple[str, ...] = (
    "{label}: {definition}\n\n"
    "Question: Structurally, does this involve {probe}?\n"
    "Answer with one word, yes or no.\n"
    "Answer:",

    "Consider the following idea.\n{label} — {definition}\n\n"
    "Ignoring its subject matter and thinking only about its underlying "
    "structure, does it involve {probe}?\n"
    "Answer:",
)


@dataclass
class StructureEncoder:
    """Score every concept against every structural predicate.

    Cost is `n_concepts x n_predicates x n_templates` short forward passes.
    For the 153-concept seed list and 52 predicates that is ~16k passes of
    about 70 tokens — a few minutes on a GPU, well under an hour on CPU.
    It scales linearly, so 10k concepts wants a GPU.
    """

    model_name: str = "Qwen/Qwen3-1.7B"
    predicates_path: str = "data/structure/predicates.jsonl"
    templates: Sequence[str] = PROMPT_TEMPLATES
    batch_size: int = 16
    device: str | None = None
    dtype: str = "float16"
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _yes_ids: list[int] = field(default_factory=list, init=False, repr=False)
    _no_ids: list[int] = field(default_factory=list, init=False, repr=False)

    # ---------------------------------------------------------------- loading

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=self.trust_remote_code
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # Left padding: we read the logits at the final position, and with
        # right padding that position is a pad token for every sequence except
        # the longest one in the batch.
        self._tokenizer.padding_side = "left"

        kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["torch_dtype"] = getattr(torch, self.dtype, torch.float32)

        # Unlike the residual encoder, this one genuinely needs the LM head.
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if "device_map" not in kwargs:
            model = model.to(self.device)
        model.eval()
        self._model = model

        # Collect every plausible surface form of yes/no. Which of these are
        # single tokens varies by tokenizer, so gather the ids that are and
        # sum their probabilities rather than betting on one spelling.
        self._yes_ids = self._token_ids([" Yes", " yes", "Yes", "yes", " YES"])
        self._no_ids = self._token_ids([" No", " no", "No", "no", " NO"])
        if not self._yes_ids or not self._no_ids:
            raise RuntimeError("could not resolve yes/no token ids for this tokenizer")
        log.info("yes ids %s / no ids %s", self._yes_ids, self._no_ids)

    def _token_ids(self, candidates: Sequence[str]) -> list[int]:
        ids = []
        for c in candidates:
            enc = self._tokenizer.encode(c, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
        return sorted(set(ids))


    def _last_position_logits(self, enc: dict[str, Any]) -> Any:
        """Logits for the final token only, without materialising the rest.

        `model(**enc).logits` allocates (batch, seq_len, vocab_size). For a
        151k-vocabulary model at batch 8 and 192 tokens that is ~930 MB in a
        single fp32 tensor, and it is the first thing to exhaust memory on a
        small machine — for no benefit, since we read exactly one position.

        Running the base model and applying the LM head to the last hidden
        state allocates (batch, vocab) instead: ~5 MB. The base module includes
        the final norm on every architecture this supports, so the result is
        numerically identical to slicing the full logits.
        """
        base = getattr(self._model, "model", None)
        head = getattr(self._model, "lm_head", None)
        if base is None or head is None:
            # Unusual architecture: fall back to the memory-hungry path rather
            # than failing, and let the caller lower the batch size.
            return self._model(**enc).logits[:, -1, :].float()
        hidden = base(**enc).last_hidden_state[:, -1, :]
        return head(hidden).float()

    # ------------------------------------------------------------------ main

    def score_matrix(self, concepts: ConceptSet) -> tuple[np.ndarray, list[Predicate]]:
        """Return the raw (n_concepts, n_predicates) log-odds matrix."""
        import torch

        self.load()
        predicates = load_predicates(self.predicates_path)
        n, m = len(concepts), len(predicates)
        log.info("scoring %d concepts x %d predicates x %d templates = %d passes",
                 n, m, len(self.templates), n * m * len(self.templates))

        totals = np.zeros((n, m), dtype=np.float32)
        for t_i, template in enumerate(self.templates):
            # Flatten to one long list so batches stay full across the
            # concept/predicate boundary.
            jobs = [(i, j) for i in range(n) for j in range(m)]
            scores = np.zeros(len(jobs), dtype=np.float32)
            for start in range(0, len(jobs), self.batch_size):
                chunk = jobs[start : start + self.batch_size]
                prompts = [
                    template.format(
                        label=concepts[i].label,
                        definition=concepts[i].definition,
                        probe=predicates[j].probe,
                    )
                    for i, j in chunk
                ]
                # Left padding (set in load()) guarantees the last position is
                # the real end of every prompt, which is what makes reading a
                # single position correct across a ragged batch.
                enc = self._tokenizer(
                    prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=192,
                )
                enc = {k: v.to(self._model.device) for k, v in enc.items()}
                with torch.inference_mode():
                    logits = self._last_position_logits(enc)
                logprobs = torch.log_softmax(logits, dim=-1)
                # Sum probability mass over all spellings, then take the
                # log-odds. This is the concept's score on this axis.
                yes = torch.logsumexp(logprobs[:, self._yes_ids], dim=-1)
                no = torch.logsumexp(logprobs[:, self._no_ids], dim=-1)
                scores[start : start + len(chunk)] = (yes - no).cpu().numpy()
                if start % (self.batch_size * 40) == 0:
                    log.info("template %d/%d: %d/%d", t_i + 1, len(self.templates),
                             start + len(chunk), len(jobs))
            totals += scores.reshape(n, m)

        return totals / len(self.templates), predicates

    def encode(self, concepts: ConceptSet) -> VectorSet:
        raw, predicates = self.score_matrix(concepts)
        return VectorSet(
            concept_ids=concepts.ids,
            vectors=raw,
            meta={
                "encoder": "structure",
                "model": self.model_name,
                "space": "raw",
                "predicates": [p.id for p in predicates],
                "predicate_families": [p.family for p in predicates],
                "n_predicates": len(predicates),
                "templates": len(self.templates),
                # This space gets its own calibration (per-predicate z-scoring),
                # not the residual-stream one.
                "skip_isotropy": True,
            },
        )


def standardise(raw: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Z-score each predicate across concepts, then L2-normalise each concept.

    The per-predicate centring is the important half. Without it, an agreeable
    predicate ("does it involve a trade-off?") that scores +4 for everything
    contributes a large constant component to every concept vector, and the
    cosine between any two concepts is dominated by how agreeable the question
    set is rather than by which questions actually separate them. Same failure
    mode as residual-stream anisotropy, same fix.
    """
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    # A predicate with no variance carries no information about *differences*
    # between concepts; zero it rather than dividing by ~0 and amplifying noise.
    dead = std < 1e-6
    z = (raw - mean) / np.where(dead, 1.0, std)
    z[:, dead] = 0.0
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    return z.astype(np.float32), {
        "predicate_mean": mean.tolist(),
        "predicate_std": std.tolist(),
        "dead_predicates": int(dead.sum()),
    }
