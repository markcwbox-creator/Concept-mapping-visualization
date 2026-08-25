# Concept-mapping-visualization

An open source side project in an attempt to reverse engineer a map of concepts
from LLMs — and to collide distant parts of that map together looking for
connections nobody has made yet.

Extract concept vectors from an open-weight LLM's residual stream (or its SAE
features), correct the geometry so distances actually mean something, find pairs
that are far apart *but still bridgeable*, and explore them in a dependency-free
web front end where any two concepts can be pulled together and collided live.

![the collider workspace](docs/screenshot.png)

---

## Quick start

```bash
pip install -r requirements.txt

# 1. See the interface immediately, with no GPU and no downloads.
#    (Structurally real; semantically meaningless — random-weight toy model.)
python scripts/make_smoke_build.py
python -m collider serve          # -> http://127.0.0.1:8000

# 2. Measure which layer to use, rather than guessing.
python -m collider sweep --config configs/qwen3-1.7b-4bit.yaml --limit 300

# 3. Build for real. Extraction is the only stage that needs a GPU.
python -m collider all --config configs/qwen3-1.7b-4bit.yaml
python -m collider serve
```

Collide two concepts without leaving the terminal:

```bash
python -m collider collide --a photosynthesis --b monetary-policy
```

### On 4 GB of VRAM

`configs/qwen3-1.7b-4bit.yaml` loads Qwen3-1.7B in 4-bit NF4 and fits
comfortably. Two details do most of that work: the extractor captures
activations with a **forward hook** rather than `output_hidden_states=True`
(which would materialise every layer at once), and it loads `AutoModel` rather
than `AutoModelForCausalLM`, since the LM head is dead weight when you never
sample. Roughly 10k concepts in 20-30 minutes.

### Splitting across Colab

Extraction is the only expensive stage:

```bash
# on Colab, with a bigger GPU
python -m collider extract --config configs/llama31-8b-sae.yaml
# copy data/build/vectors.npy + vectors.meta.json back, then locally:
python -m collider build --config configs/llama31-8b-sae.yaml
```

`build` is pure CPU and takes seconds, so you can iterate on pairing weights,
layout and the front end dozens of times per extraction.

---

## What it does

**Concepts are short definitions, not tokens.** Single tokens fragment under
BPE and average over senses; `photosynthesis: The process by which plants
convert light energy...` pins down what you mean and gives the model something
to condition on.

**The geometry gets corrected before anything is measured.** Raw LLM activations
are anisotropic — random unrelated concepts sit at cosine 0.7-0.95, so
"distance" carries almost no signal. Centring plus all-but-the-top removal
takes the seed build from mean cosine **0.257 → -0.006** with a *wider* spread.
Every build prints this before/after, because if it doesn't happen the map is
decoration. See [docs/DESIGN.md §4](docs/DESIGN.md).

**"Distant" is not the same as "interesting".** The most distant pairs are the
most incoherent ones. Pairs are scored on distance *near a target percentile*,
bridgeability through the kNN graph (unreachable pairs are filtered outright),
domain gap, midpoint sparsity, and neighbourhood-structure similarity — with
per-domain and per-concept diversity caps. Every pair carries its feature
breakdown so you can re-weight after seeing results.

**A collision answers four different questions.**

| view | question |
|---|---|
| Stepping stones | the cheapest real chain between them — nothing invented |
| Balanced bridges | what sits genuinely *between*, not merely near one side |
| Nearest the blend | what is closest to the midpoint vector, and how vacant that spot is |
| Orthogonal | far from both poles and sideways to the tension between them |

A **blend vacancy above 1** means nothing in your map names that blend — a
description without a word, which is the case worth looking at.

**The front end has no dependencies.** No build step, no CDN, no framework.
Layout is precomputed in Python (a browser force simulation lands somewhere
different every reload, destroying spatial memory), but PCA-reduced int8
vectors ship to the client, so *any* pair collides live in a couple of
milliseconds without a server.

---

## Layout

```
src/collider/
  schema.py            Concept / ConceptSet / VectorSet — the artifact contract
  config.py            one flat config; committed in configs/, copied into graph.json
  encoders/
    base.py            Encoder protocol + prompt templates and pooling spans
    hf_residual.py     residual-stream extraction (hooks, 4-bit, template averaging)
    sae.py             SAE features — the interpretable swap-in
    sentence.py        MiniLM baseline / control
  space.py             centring, all-but-the-top, whitening, isotropy diagnostics
  neighbors.py         exact / FAISS / Annoy kNN, Dijkstra over the kNN graph
  pairs.py             distant-but-interesting pair sampling
  bridges.py           the four bridge views
  layout.py            UMAP / PCA / t-SNE, with graceful fallback
  export.py            graph.json + quantised vector blob
  sweep.py             layer & pooling selection by measurement
  pipeline.py          stage orchestration
  __main__.py          CLI: sweep | extract | build | all | collide | serve

web/                   index.html, app.js, collide.js, style.css — zero dependencies
data/concepts/         seed_concepts.jsonl (153 concepts, 18 domains)
data/probes/           hand-written triplets for honest evaluation
configs/               qwen3-1.7b-4bit · llama31-8b-sae · minilm-cpu
docs/DESIGN.md         the arguments, and where this is most likely wrong
tests/                 29 tests, including Python↔JavaScript parity
```

---

## Growing the concept list

The seed list is 153 concepts across 18 domains — enough to demonstrate the
mechanism, not enough to find anything. To reach 1k-10k, the honest options are
WordNet synsets with glosses (~120k, lexical and heavily nominal) or Wikidata
descriptions (millions, entity-heavy and full of proper nouns that are not
really concepts).

Watch for the seam: definitions from different sources share stylistic
structure that shows up as similarity. If source predicts distance, that is a
bug to fix, not a discovery. Curation is the substance here, not a chore to
automate away.

---

## Tests

```bash
PYTHONPATH=src:tests python -m pytest tests/ -q
```

Tests run on a randomly-initialised toy model built offline, so the whole suite
runs on CPU in seconds with no downloads. They target the failure modes that are
*silent* — padding leaking into pooled vectors, isotropy correction not
correcting, kNN ordering, and the Python and JavaScript collision maths drifting
apart.

## Status

Beta. The pipeline, the geometry corrections, the pair scoring and the front end
all work and are tested. What has not been validated is the part that matters
most: whether the pairs it surfaces are *actually* generative for a human. See
[docs/DESIGN.md](docs/DESIGN.md) for where this is most likely to be wrong.

## Licence

Apache 2.0.
