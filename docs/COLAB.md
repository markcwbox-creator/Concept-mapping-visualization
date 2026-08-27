# Running extraction on Colab

Extraction is the only stage that needs a GPU. Everything after it —
space correction, kNN, pairing, layout, export — is CPU-only and runs in
seconds, so the split is clean: extract in the cloud, iterate locally.

Paste this into a Colab cell (T4 runtime is enough for Qwen3-4B in 4-bit, or
Llama-3.1-8B in fp16 on an A100/L4):

```python
!git clone https://github.com/markcwbox-creator/Concept-mapping-visualization.git
%cd Concept-mapping-visualization
!pip install -q -r requirements.txt bitsandbytes

# Optional but worth the five minutes: measure the layer instead of guessing.
!PYTHONPATH=src python -m collider sweep \
    --config configs/qwen3-1.7b-4bit.yaml --limit 300

# Then extract at the layer the sweep picked.
!PYTHONPATH=src python -m collider extract \
    --config configs/qwen3-1.7b-4bit.yaml --layer 18

from google.colab import files
files.download('data/build/vectors.npy')
files.download('data/build/vectors.meta.json')
```

Locally, drop both files into `data/build/` and run:

```bash
python -m collider build --config configs/qwen3-1.7b-4bit.yaml
python -m collider serve
```

`vectors.meta.json` records the model, layer, pooling and templates that
produced the matrix, and `build` refuses to proceed if the concept list has
changed since extraction — so a stale pairing of vectors and concepts cannot
silently produce a plausible-looking wrong map.

## Notes

* **Gated models.** Llama-3.1 needs `huggingface_hub.login()` and an accepted
  licence before `from_pretrained` will work.
* **Session limits.** 10k concepts x 4 templates is comfortably inside a free
  Colab session. Above ~50k, extract in shards (split the concept JSONL, run
  each shard, `np.concatenate` the results in list order) — row order must
  match the concept file exactly.
* **Don't re-sweep every run.** The layer is a property of the model and your
  concept style, not of an individual build. Sweep once, write the number into
  your config, move on.
