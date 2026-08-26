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
`probe_accuracy` — triplets encoding *relations* (mechanism,
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

---

## 5. Structural signatures: a hypothesis, tested and mostly refuted

**The hypothesis.** The pipeline measures similarity between *definitions*,
which is a **topical** measure. Cross-domain analogy needs a **structural** one.
`immune memory` and `caching` are the same structure — pay once, store the
result, answer the repeat instantly, fail when the key goes stale — but their
definitions share almost no vocabulary. So (the argument went) write a
domain-neutral *structural signature* for each concept, embed that instead, and
analogies become neighbours instead of distant pairs.

**The test.** Signatures written for all 153 concepts against a fixed template,
with one hard rule: no word naming the field. Scored against the 21 labelled
cross-domain analogies in `data/probes/example_probes.jsonl` — written earlier,
to test layer choice, not this. Rank the target among the other 152; chance
median is ~76. Reproduce with `scripts/structure_experiment.py`.

| regime | median rank | mean | r@1 | r@5 | r@10 |
|---|---|---|---|---|---|
| topical (definitions) | **8** | 20.9 | 0.24 | 0.48 | **0.52** |
| structural (signatures) | 36 | 39.3 | 0.14 | 0.33 | 0.33 |
| dual, structural − 0.5·topical | 73 | 68.0 | 0.00 | 0.10 | 0.24 |
| dual, structural − 1.0·topical | 88 | 95.0 | 0.00 | 0.00 | 0.05 |
| fused (reciprocal rank fusion) | 8 | 21.1 | 0.24 | 0.38 | 0.52 |
| union ceiling (oracle) | **2** | 13.7 | 0.33 | 0.62 | 0.62 |

**Three results, in descending order of how much they cost to learn the hard way.**

**(a) The intuitive score is the worst thing you can build.** "Structurally near
*and* topically far" — subtract one similarity from the other — is the obvious
formalisation of "same structure, unrelated fields", and it performs at chance.
It is worse than either input alone. The reason is that real analogies are
often *somewhat* topically related, so the subtraction actively punishes correct
answers, and subtracting a noisy quantity adds its noise. Anyone building this
from intuition writes this scorer first. It does not work.

**(b) Replacing definitions with signatures makes things worse on average, but
that average hides the real finding.** The two spaces succeed on disjoint sets
of pairs:

```
structure rescues what topic cannot reach     topic holds what structure loses
  placebo-effect  x priming       81 ->   2     moral-hazard x principal-agent   1 ->  37
  hysteresis x learned-helpless   38 ->   4     improvisation x flow-state       1 ->  39
  chronic-inflammation x feedback 59 ->  44     caching x immune-memory          1 ->  14
  compression x abstraction-layer 47 ->  36     network-effect x cultural-trans 11 -> 139
```

The pattern is consistent: signatures win when the pair is topically remote, and
lose when it is topically close. That is exactly the complementarity you would
want — and the oracle union confirms real headroom (median 8 → **2**, r@5
0.48 → 0.62).

This is not an artefact of the signature space being mushier. Both spaces have
identical spread after correction (mean cosine −0.006, std 0.070 vs 0.071). The
signature space is just as discriminative; it discriminates on something else.

**(c) Naive fusion cannot capture the headroom.** Reciprocal rank fusion — the
standard untuned combiner — lands exactly on topical's numbers (median 8, r@10
0.52) and buys nothing, because when one channel is badly wrong it drags the
merged score down. The gap between RRF (8) and the oracle (2) is the size of the
prize available to a *learned* combiner, and it is large.

**What this means for training.** It locates precisely where fitting parameters
earns its cost. Not the language model — that stays frozen and pre-trained, and
nothing else is affordable or desirable. But a small combiner over the two
spaces, fitted on labelled analogy pairs, is seconds of CPU and has a measured
median-rank-8-to-2 ceiling. That is the highest-return training in the project,
and it converts the bottleneck into a labelling problem: 21 labelled pairs is
enough to measure with, nowhere near enough to fit with.

**Caveats.** 21 probes is a small sample and rank differences of a few places
are noise (81 → 2 and 1 → 139 are not). The signatures were written by a large
model that had seen the project's aims and the probe file, so the structural
space may partly encode its own analogy judgements — though note this
contamination would inflate the structural result, and the structural result
*lost*. The unprompted pairs the dual score surfaces (`diffusion` ×
`affordance`, `resonance` × `planned-obsolescence`) read as noise, which is
consistent with (a) rather than a separate failure.

---

## 6. What is still unvalidated

Whether any of this is *generative for a human*. The map surfaces pairs; nothing
here shows that exploring them produces thoughts a person would not otherwise
have had. Until that is measured — the pin-and-note workflow in the UI exists to
collect exactly that data — every weight in `pairs.py` is a guess with a
plausible story attached.
