# Findings from the first real build

Qwen3-1.7B, fp32 on CPU, 153 seed concepts, four templates, layer 18 of 28.
Reproduce with `python -m collider all --config configs/qwen3-1.7b-cpu.yaml`.

These are measurements from one small build, not conclusions about concept maps
in general. They are written down because each one changed a decision.

---

## 1. Anisotropy is far worse than the literature's framing suggests

| | before correction | after |
|---|---|---|
| mean cosine, random pairs | **0.838** | **−0.007** |
| spread (std) | **0.027** | **0.070** |

A mean of 0.838 with a standard deviation of 0.027 means every pair of concepts
in the raw space sat between roughly 0.76 and 0.92 similarity. Nothing was far
from anything. The core operation this project is built on — "sample pairs above
the 90th percentile of distance" — would have been sampling noise in the third
decimal place.

Centring plus removing the top 8 principal directions fixes it: the mean drops
to zero and the usable spread grows 2.6×. **This single step is the difference
between a working map and a decorative one**, and it is invisible if you only
look at neighbour rankings, which stay roughly sensible either way.

## 2. Domain purity and probe accuracy disagree about which layer is best

From `docs/layer_sweep_qwen3-1.7b.txt`:

| layer | depth | domain purity | probe accuracy |
|---|---|---|---|
| 8 | 0.29 | 0.150 | 0.808 |
| **16** | **0.57** | 0.126 | **0.923** |
| 18 | 0.64 | 0.143 | 0.885 |
| 24 | 0.86 | 0.211 | 0.808 |
| 28 | 1.00 | **0.216** | 0.846 |

`domain_purity` climbs almost monotonically into the final layer.
`probe_accuracy` — hand-written triplets encoding *relations* (mechanism,
analogy, part-of) — peaks in the middle of the stack and falls away.

This is the design doc's caveat showing up in real data: **a late layer is
better at deciding what subject something belongs to and worse at conceptual
geometry.** Anyone who picks a layer by domain purity, which is the obvious
metric because it needs no hand-labelling, will pick close to the worst layer
for this project's purpose. The 26-triplet probe set is far too small to
resolve 16 from 18 — that gap is one triplet — but it comfortably separates
mid-stack from late.

## 3. The concept list, not the model, is the binding constraint

Nearest neighbours at layer 18, after correction:

```
monetary-policy  -> control-system(0.24), inflation(0.21), phase-transition(0.16)
counterpoint     -> polyrhythm(0.36), due-process(0.19), consensus-protocol(0.17)
triage           -> differential-diagnosis(0.18), load-balancing(0.16)
entropy          -> differential-diagnosis(0.15), vernacular-architecture(0.14)
recursion        -> teleology(0.16), metaphor(0.14), comparative-advantage(0.14)
```

The first three contain exactly what the project is for: `monetary-policy →
control-system`, `triage → load-balancing` and `counterpoint → consensus-protocol`
are real cross-domain mechanism analogies that no topic model would produce. The
last two are close to noise.

The tell is the magnitudes. Top-neighbour similarities of 0.15–0.36 in a space
whose overall spread is 0.070 means the nearest neighbour is only two to four
standard deviations from a random concept. With 153 points in a 2048-dimensional
space, almost everything is almost orthogonal to almost everything, so "the
nearest concept" is often just the least arbitrary of many arbitrary options.

Aggregate quality is nonetheless real and well above chance:

* `probe_accuracy` **0.885** (chance 0.5, 26 triplets)
* `domain_purity@10` **0.143** (chance ≈ 0.056, 18 domains) — 2.5× chance

**Implication for scaling:** more concepts will help far more than a bigger
model. The bridges are thin because the map is sparse, not because 1.7B
parameters cannot represent these ideas. Getting to 5–10k curated concepts is
the highest-value next step; swapping to Llama-3.1-8B is not.

## 4. Two silent export bugs, both caught by the parity test

Neither would have crashed anything. Both would have made the browser quietly
disagree with the CLI about how far apart two concepts are.

1. **Double centring.** `reduce_dims` subtracted the mean before projecting —
   the textbook PCA step. But the input had already been centred and
   L2-normalised by `space.prepare`, so the second subtraction shifted every
   vector and changed every cosine the client computed.
2. **A declared dimension the blob did not have.** `np.linalg.svd(...,
   full_matrices=False)` on a (153, 2048) matrix returns only 153 singular
   vectors, so requesting 256 produced a 153-wide matrix — while `graph.json`
   reported 256, because the metadata took the dimension from the *request*. The
   client would have read every vector at the wrong stride.

Both were invisible to the toy-model test fixture, whose 64-dimensional vectors
skip the projection path entirely. They only appeared once a real 2048-dimensional
build existed. That is an argument for keeping at least one real build in the
loop rather than trusting a fast synthetic fixture alone.

## 5. What is still unvalidated

Whether any of this is *generative for a human*. The map surfaces pairs; nothing
here shows that exploring them produces thoughts a person would not otherwise
have had. Until that is measured — the pin-and-note workflow in the UI exists to
collect exactly that data — every weight in `pairs.py` is a guess with a
plausible story attached.
