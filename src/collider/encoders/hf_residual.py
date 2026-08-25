"""Residual-stream concept vectors from any Hugging Face causal LM.

Design notes that matter for quality (the full argument is in docs/DESIGN.md):

* **Hook, don't `output_hidden_states=True`.** Asking for all hidden states
  materialises `n_layers` tensors of shape (batch, seq, d_model) at once. On a
  4 GB card that is what kills you first. A forward hook on the one layer you
  want holds a single tensor, and lets you capture several specific layers
  during a sweep without paying for all of them.
* **`AutoModel`, not `AutoModelForCausalLM`.** We never sample, so the LM head
  (often ~10% of parameters, and for tied-embedding models a big matmul) is
  dead weight. Loading the base model alone saves real VRAM.
* **Mid-to-late layers.** Early layers are dominated by token identity and
  surface form; the last couple of layers are bent toward next-token
  prediction and get noticeably worse as general-purpose semantics. Somewhere
  around 60-70% of depth is the usual sweet spot — but do not take that on
  faith, run `collider.sweep` and measure it on your own concept list.
* **Pool over content tokens only.** Template scaffolding contributes a large
  constant direction to every vector, which compresses the dynamic range of
  every cosine you later compute.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..schema import ConceptSet, VectorSet
from .base import DEFAULT_TEMPLATES, Template

log = logging.getLogger(__name__)


def _resolve_layer_list(model: Any) -> Sequence[Any]:
    """Find the list of transformer blocks across common HF architectures."""
    for path in (
        ("layers",),                 # Llama/Qwen/Mistral via AutoModel
        ("model", "layers"),         # ...ForCausalLM wrappers
        ("transformer", "h"),        # GPT-2 / GPT-J / Falcon
        ("gpt_neox", "layers"),      # NeoX / Pythia
        ("encoder", "layer"),        # BERT-likes, if someone points this at one
    ):
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__") and len(obj) > 0:
            return obj
    raise RuntimeError(
        "Could not locate the transformer block list on this model. "
        "Add its attribute path to _resolve_layer_list()."
    )


@dataclass
class HFResidualEncoder:
    """Batch-embed concept definitions from a chosen residual-stream layer.

    Parameters
    ----------
    model_name:
        Any HF causal LM. Sensible picks for a 4 GB card, in 4-bit:
        `Qwen/Qwen3-1.7B` (28 layers, d=2048) or `Qwen/Qwen3-4B`. For
        cloud/Colab runs `meta-llama/Llama-3.1-8B` in fp16 is comfortable on a
        T4/A100 and has public SAEs available.
    layer:
        Absolute block index to read. `hidden_states` indexing convention:
        layer `k` means "the output of block `k`", with `0` meaning the token
        embeddings before any block. Negative indexes count from the end.
        Leave `None` to use `layer_frac`.
    layer_frac:
        Fractional depth, used when `layer` is None. 0.65 by default.
    pooling:
        ``mean``   - average over content tokens (default; most robust)
        ``last``   - the final content token (natural for a causal LM, but
                     over-weights whatever word happens to end the definition)
        ``max``    - element-wise max over content tokens (spikier, occasionally
                     better at picking out a distinctive feature)
        ``label_last`` - the last token of the *label* mention, i.e. "what the
                     model thinks this term is having just read it in context".
    templates:
        Phrasings to average over. See `base.DEFAULT_TEMPLATES`.
    load_in_4bit:
        Use bitsandbytes NF4. This is the flag that makes a 1.7B-4B model fit
        in 4 GB. Requires `bitsandbytes`; ignored on CPU.
    """

    model_name: str = "Qwen/Qwen3-1.7B"
    layer: int | None = None
    layer_frac: float = 0.65
    pooling: str = "mean"
    templates: Sequence[Template] = DEFAULT_TEMPLATES
    batch_size: int = 16
    max_length: int = 128
    device: str | None = None
    dtype: str = "float16"
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    # Optional per-token map applied to the captured residual stream *before*
    # pooling: (B, T, D) -> (B, T, F). This is the SAE swap-in point. Order
    # matters — an SAE is trained on individual token activations, so encoding
    # per token and then pooling feature activations is faithful, while pooling
    # the residual stream first and encoding the average is not (the mean of a
    # set of activations is off the manifold the SAE was fit to).
    activation_transform: Any = None
    # Populated on first use.
    _model: Any = field(default=None, repr=False, init=False)
    _tokenizer: Any = field(default=None, repr=False, init=False)
    _n_layers: int = field(default=0, repr=False, init=False)

    # ---------------------------------------------------------------- loading

    def load(self) -> None:
        """Load tokenizer + base model. Idempotent."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        log.info("loading %s on %s", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=self.trust_remote_code
        )
        if not self._tokenizer.is_fast:
            raise RuntimeError(
                "A fast tokenizer is required (we need offset mappings to pool "
                "over content tokens only)."
            )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                # Double quantisation trims another ~0.4 bits/param; on a 4 GB
                # card that is the difference between 4B fitting and not.
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["torch_dtype"] = getattr(torch, self.dtype, torch.float32)

        model = AutoModel.from_pretrained(self.model_name, **kwargs)
        if "device_map" not in kwargs:
            model = model.to(self.device)
        model.eval()
        self._model = model
        self._n_layers = len(_resolve_layer_list(model))
        log.info("model has %d blocks", self._n_layers)

    def resolve_layer(self, layer: int | None = None) -> int:
        """Turn the layer spec into a concrete non-negative block index."""
        self.load()
        spec = self.layer if layer is None else layer
        if spec is None:
            spec = max(1, round(self.layer_frac * self._n_layers))
        if spec < 0:
            # Layers are 1-indexed here (layer k == "output of block k", with 0
            # reserved for the raw embeddings), so -1 must land on the last
            # block, not the second-to-last.
            spec = self._n_layers + spec + 1
        if not 0 < spec <= self._n_layers:
            raise ValueError(
                f"layer {spec} out of range for a {self._n_layers}-block model"
            )
        return int(spec)

    # -------------------------------------------------------------- internals

    def _content_mask(
        self, encoded: Any, spans: list[tuple[int, int]], label_spans: list[tuple[int, int]]
    ) -> tuple[Any, Any]:
        """Boolean masks over tokens: (content tokens, label-final token).

        Built from character offsets so it is tokenizer-agnostic — no assumptions
        about BOS tokens, sentencepiece prefix spaces or byte-level quirks.
        """
        import torch

        offsets = encoded["offset_mapping"]  # (B, T, 2)
        attn = encoded["attention_mask"].bool()
        b, t, _ = offsets.shape
        starts = offsets[..., 0]
        ends = offsets[..., 1]
        # Special tokens get (0, 0); drop them explicitly.
        real = attn & (ends > starts)

        span_lo = torch.tensor([s[0] for s in spans], device=offsets.device).view(b, 1)
        span_hi = torch.tensor([s[1] for s in spans], device=offsets.device).view(b, 1)
        content = real & (starts >= span_lo) & (ends <= span_hi)

        # Guard: if a template rendered oddly and nothing matched, fall back to
        # all real tokens rather than emitting a NaN row.
        empty = ~content.any(dim=1)
        if bool(empty.any()):
            content[empty] = real[empty]

        lab_lo = torch.tensor([s[0] for s in label_spans], device=offsets.device).view(b, 1)
        lab_hi = torch.tensor([s[1] for s in label_spans], device=offsets.device).view(b, 1)
        in_label = real & (starts >= lab_lo) & (ends <= lab_hi)
        # Keep only the last token of the label mention.
        idx = torch.arange(t, device=offsets.device).view(1, t).expand(b, t)
        last_idx = torch.where(in_label, idx, torch.full_like(idx, -1)).max(dim=1).values
        label_last = torch.zeros_like(content)
        has_label = last_idx >= 0
        label_last[has_label, last_idx[has_label]] = True
        label_last[~has_label] = content[~has_label]
        return content, label_last

    def _pool(self, hidden: Any, content: Any, label_last: Any) -> Any:
        """(B, T, D) hidden states -> (B, D) using the configured pooling."""
        import torch

        mask = label_last if self.pooling == "label_last" else content
        m = mask.unsqueeze(-1).to(hidden.dtype)
        if self.pooling == "mean":
            return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
        if self.pooling in ("last", "label_last"):
            if self.pooling == "last":
                # index of the final content token in each row
                idx = torch.where(
                    content,
                    torch.arange(content.shape[1], device=content.device).view(1, -1),
                    torch.full_like(content, -1, dtype=torch.long),
                ).max(dim=1).values.clamp(min=0)
                return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]
            return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
        if self.pooling == "max":
            neg_inf = torch.finfo(hidden.dtype).min
            return hidden.masked_fill(~mask.unsqueeze(-1), neg_inf).max(dim=1).values
        raise ValueError(f"unknown pooling: {self.pooling!r}")

    # ------------------------------------------------------------------- main

    def encode_layers(
        self, concepts: ConceptSet, layers: Sequence[int]
    ) -> dict[int, np.ndarray]:
        """Embed every concept at several layers in one pass.

        Used by both `encode` (single layer) and the layer sweep. Capturing N
        layers in one forward pass costs N tensors of activations instead of
        one, but only a single pass of compute — worth it for the sweep,
        pointless for a normal run.
        """
        import torch

        self.load()
        resolved = [self.resolve_layer(l) for l in layers]
        blocks = _resolve_layer_list(self._model)
        embed_out: dict[int, Any] = {}
        handles = []

        def make_hook(idx: int):
            def hook(_module, _inp, out):
                # HF blocks return either a tensor or a tuple whose first
                # element is the residual stream.
                embed_out[idx] = out[0] if isinstance(out, tuple) else out
            return hook

        try:
            for l in resolved:
                # block index l-1 produces "output of block l"
                handles.append(blocks[l - 1].register_forward_hook(make_hook(l)))

            n = len(concepts)
            per_template: dict[int, list[np.ndarray]] = {l: [] for l in resolved}

            for template in self.templates:
                rows = {l: [] for l in resolved}
                for start in range(0, n, self.batch_size):
                    batch = [concepts[i] for i in range(start, min(start + self.batch_size, n))]
                    texts, spans, label_spans = [], [], []
                    for c in batch:
                        text, span = template.render(c)
                        texts.append(text)
                        spans.append(span)
                        lpos = text.find(c.label)
                        label_spans.append(
                            (lpos, lpos + len(c.label)) if lpos >= 0 else span
                        )
                    enc = self._tokenizer(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_offsets_mapping=True,
                    )
                    enc = {k: v.to(self._model.device) for k, v in enc.items()}
                    model_inputs = {
                        k: v for k, v in enc.items() if k != "offset_mapping"
                    }
                    embed_out.clear()
                    with torch.inference_mode():
                        self._model(**model_inputs)
                    content, label_last = self._content_mask(enc, spans, label_spans)
                    for l in resolved:
                        acts = embed_out[l].float()
                        if self.activation_transform is not None:
                            acts = self.activation_transform(acts)
                        pooled = self._pool(acts, content, label_last)
                        rows[l].append(pooled.cpu().numpy().astype(np.float32))
                    if start % (self.batch_size * 20) == 0:
                        log.info(
                            "template=%s %d/%d", template.name, start + len(batch), n
                        )
                for l in resolved:
                    per_template[l].append(np.concatenate(rows[l], axis=0))

            out: dict[int, np.ndarray] = {}
            for l in resolved:
                stack = np.stack(per_template[l], axis=0)  # (T, N, D)
                # Normalise each template's vector before averaging, so that a
                # template that happens to produce larger-norm activations does
                # not dominate the average.
                stack /= np.linalg.norm(stack, axis=-1, keepdims=True) + 1e-8
                out[l] = stack.mean(axis=0).astype(np.float32)
            return out
        finally:
            for h in handles:
                h.remove()

    def encode(self, concepts: ConceptSet) -> VectorSet:
        layer = self.resolve_layer()
        vecs = self.encode_layers(concepts, [layer])[layer]
        return VectorSet(
            concept_ids=concepts.ids,
            vectors=vecs,
            meta={
                "encoder": "hf_residual",
                "model": self.model_name,
                "layer": layer,
                "n_layers": self._n_layers,
                "pooling": self.pooling,
                "templates": [t.name for t in self.templates],
                "quantised_4bit": bool(self.load_in_4bit),
                "space": "raw",  # not yet centred/whitened; see space.py
            },
        )
