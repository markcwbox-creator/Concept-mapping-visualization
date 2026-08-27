"""Build a tiny, randomly-initialised model + tokenizer with no network access.

This exists so the extraction machinery — forward hooks, layer indexing,
offset-based content masking, every pooling mode, template averaging, padded
batching — can be exercised in CI on a CPU in seconds, without downloading
weights.

What it validates: shapes, indexing, masking, determinism, the artifact
contract.

What it emphatically does NOT validate: semantics. The weights are random, so
every "concept vector" it produces is noise. Any graph built from it is a
structural fixture, never a result. Anything derived from it is stamped with a
`provenance` field that the front end renders as a red banner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def build_tokenizer(corpus: Iterable[str], vocab_size: int = 2000):
    """Train a byte-level BPE on the given text. Fast tokenizer, offsets included."""
    from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        show_progress=False,
    )
    tok.train_from_iterator(list(corpus), trainer=trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
    )


def build_model(vocab_size: int, hidden: int = 64, layers: int = 6, seed: int = 0):
    """A real Qwen3 architecture at toy scale, randomly initialised."""
    import torch
    from transformers import AutoModel, Qwen3Config

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden // 4,
        max_position_embeddings=512,
    )
    return AutoModel.from_config(cfg)


def make_smoke_encoder(concepts, *, hidden: int = 64, layers: int = 6, **kwargs):
    """An `HFResidualEncoder` pre-loaded with the toy model and tokenizer."""
    from collider.encoders.hf_residual import HFResidualEncoder

    corpus = [f"{c.label}: {c.definition} {c.domain}" for c in concepts]
    tokenizer = build_tokenizer(corpus)
    model = build_model(len(tokenizer), hidden=hidden, layers=layers)
    model.eval()

    enc = HFResidualEncoder(model_name="<smoke>", device="cpu", dtype="float32", **kwargs)
    enc._model = model          # noqa: SLF001 - deliberate injection for tests
    enc._tokenizer = tokenizer  # noqa: SLF001
    enc._n_layers = layers      # noqa: SLF001
    return enc
