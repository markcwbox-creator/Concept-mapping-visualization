# Design notes

The arguments behind the choices in the code, and the places where the
project is most likely to be wrong. Read this before changing defaults.

---

## 1. Why definitions, not tokens

Single tokens fail three ways at once:

* **BPE fragmentation.** "photosynthesis" is several tokens; the vector you get
  from any one of them is a fragment, not a concept.
* **Polysemy.** "bank", "cell", "field", "spring" — a token vector is an average
  over senses weighted by corpus frequency, which is a sense nobody means.
* **No frame.** A bare noun gives the model nothing to condition on, so the
  representation is dominated by surface statistics.

A canonical phrase plus one sentence of definition fixes the sense, gives the
model context, and gives *you* something readable in the UI. The cost is that
you are now partly measuring your own prose. Template averaging
(§3) is the mitigation, and the probe set (§6) is how you check it worked.

**Known weakness.** Definitions written by one author in one voice share
stylistic structure that shows up as similarity. If you expand the concept
list from mixed sources (WordNet glosses, Wikipedia first sentences, your own
prose), expect a visible seam in the map along source lines. Normalise the
phrasing, or add source as a metadata field and check whether it predicts
distance — if it does, you have a problem to fix, not a discovery.

---

## 2. Which layer, and why not to trust "the middle"

Layer choice matters more than model choice at this scale. The received wisdom
— "use a middle layer" — is directionally right for a reason:

* **Early layers** are dominated by token identity and surface form. Two
  concepts sharing a rare word look similar; two paraphrases do not.
* **Late layers** are increasingly shaped by the next-token prediction task.
  Representations get organised around what comes *next* rather than what the
  thing *is*, which is a different geometry than you want.
* **Roughly 60-75% of depth** usually holds the most task-general semantic
  structure.

But "usually" is doing real work in that sentence, and the actual optimum moves
with the model, the pooling and the concept list. `python -m collider sweep`
measures it in one forward pass over a few hundred concepts. Take the curve,
not the argmax: adjacent layers are highly correlated, so if 16, 18 and 20 are
within noise, pick the middle one and stop.

Do not read a sharp peak as a strong result. It usually means the probe set is
too small to resolve the differences.

---

## 3. Pooling and template averaging

**Pool over content tokens only.** The template scaffolding ("Define the
following term:") contributes a large, concept-independent component to every
vector. Including it compresses the dynamic range of every cosine you later
compute — everything gets pulled toward a shared direction. The extractor
builds the pooling mask from character offsets, so only the label and
definition contribute.

**Mean over the span** is the robust default. `last` is the natural choice for
a causal model and is defensible, but it over-weights whatever word happens to
end the definition, which is an artefact of your prose, not the concept.
`label_last` — the hidden state at the final token of the label, having read it
in context — is the most "interpretability-native" option and worth trying;
it is noisier but sometimes sharper.

**Average over several templates.** With one phrasing you are measuring the
concept *and* the template. With four you are mostly measuring the concept.
Cost is linear; the improvement in neighbourhood quality is easy to see and
easy to verify with the probe set. Each template's vector is normalised before
averaging so a phrasing that happens to produce larger activations cannot
dominate.

---

## 4. Isotropy correction: the thing that makes or breaks this

**This is the highest-leverage part of the pipeline and the easiest to skip.**

Hidden states of a trained transformer are strongly anisotropic. They occupy a
narrow cone rather than the sphere, because of

1. a large shared mean vector that every token has a big component along, and
2. a few "rogue dimensions" whose variance is orders of magnitude above the
   rest, which dominate every dot product.

The symptom: raw cosine between *unrelated* concepts sits around 0.7-0.95
instead of near 0. Rankings are still weakly informative, so the map looks like
it works — but the magnitudes are meaningless, the dynamic range is tiny, and
"sample pairs above the 90th percentile of distance" degenerates into sampling
noise. That operation is the entire point of this project.

The correction, in `space.py`, in order: centre → project out the top-k
principal components (Mu & Viswanath's "all-but-the-top") → optional whitening
→ L2 normalise.

Measured on the real Qwen3-1.7B build (layer 18, 153 concepts): mean cosine
**0.838 → −0.007**, spread **0.027 → 0.070**. A mean of 0.838 with a standard
deviation of 0.027 puts every pair in the space between roughly 0.76 and 0.92 —
percentile-based pair sampling would have been sampling noise in the third
decimal. `isotropy_report` prints before and after on every build; if that gap
does not appear, stop and find out why before reading anything into the map.
Full numbers in [FINDINGS.md](FINDINGS.md).

Whitening is off by default. It sharpens distances further but discards the
importance ordering of directions and amplifies noise when the covariance
estimate is poor. Turn it on above ~5k concepts.

SAE features opt out entirely (`skip_isotropy`): they are non-negative and
sparse by construction, centring destroys the sparsity that makes them cheap,
and they do not have the rogue-dimension problem.

---

## 5. What "distant but interesting" means

The naive version — take the largest distances — does not work, and why it
fails is most of the design.

The most distant pairs in a real concept space are the most *incoherent* ones:
a technical term of art against a mundane object, two things with no shared
frame. There is nothing to build between them. The productive pairs sit
slightly inside the tail: far enough that nobody has connected them, close
enough that a chain of three or four steps exists.

`pairs.py` scores five features, all computed against the empirical
distribution of *your* space rather than against magic constants:

| feature | what it captures | why |
|---|---|---|
| `distance` | near a target percentile, penalised for extremity | a soft window, not a threshold |
| `bridgeability` | a kNN path exists, at a hop count read off this graph's own distribution, with path cost breaking ties | **the incoherence filter; does the most work** |
| `domain_gap` | distance between the two domain centroids | graded prior for "nobody has looked here" |
| `midpoint_gap` | the semantic midpoint sits in a sparse region, **excluding the pair itself** | a crowded midpoint means already-thought-of |
| `analogy` | similar local density profiles | structure to transfer, not just juxtaposition |

Unreachable pairs are filtered out entirely, not down-weighted. Selection is
diversity-constrained (caps per domain-pair and per concept), because without
that a single well-separated domain pair dominates the whole tail.

**The defaults are a starting point to argue with.** Every pair carries its
full feature breakdown into `graph.json` and the UI, so you can re-weight after
seeing results rather than guessing up front. Expect to change them.

**Three of these five were once decorative, and that is the instructive part.**
Measured on the first real build, `domain_gap` was **1.000 on every one of the
200 shipped pairs**, `midpoint_gap` spanned 0.129–0.144, and `bridgeability`
took **two** distinct values. A weighted sum with three constant terms is a
two-feature scorer wearing a five-feature label, and it still produced a
plausible ranked list every time — which is precisely why it survived so long.

The causes were different in kind, and only one was a tuning mistake:

* `domain_gap` was **saturated**. Binary "different domain?" is 96% true for
  random pairs at this list size and 100% true after the distance pre-filter.
  It was a filter presented as a feature. Now graded by centroid separation
  (range 0.00–1.37, correlation with pair distance 0.12).
* `midpoint_gap` was **algebraically circular**. The normalised midpoint of two
  unit vectors sits at `sqrt((1+cos)/2)` from both endpoints — nearer than
  anything else in a 153-concept list — so "how empty is the midpoint" was
  measuring the pair's own separation. It correlated with `distance` at
  **r = 1.000**, and the midpoint's nearest neighbour was one of the pair
  itself in **4000 of 4000** candidates. Masking the endpoints, which
  `bridges.py` had done all along, drops the correlation to 0.13.
* `bridgeability` was **mis-targeted**. `target_hops = 4.0` sat in the far tail
  of a graph where 97.6% of reachable candidates bridge in 2 or 3 hops, so the
  feature rewarded the longest chain rather than the most productive one. The
  target now comes from the graph's own hop distribution, and path cost — which
  the code computed and then discarded — breaks ties within a hop stratum.

The general lesson is worth more than the three fixes: **a degenerate feature
and a working feature produce the same-looking output.** Nothing crashes,
nothing looks wrong, and the ranked list stays plausible. Only measuring the
spread finds it, which is what `test_features_are_not_degenerate` now does on
every run.

**What the hop fix does not buy you.** Deriving the target stops the feature
aiming outside the graph; it cannot put information into a graph that has none.
At 153 concepts with `knn_k = 24`, every reachable pair bridges in 2 or 3 hops,
so the derived target is 2.0 and hop count discriminates almost nothing — the
path-cost tie-break carries most of what `bridgeability` now contributes. That
is the honest state of it: the metric is correctly aimed and still nearly
uninformative, and only a bigger concept list changes that. Lowering `knn_k`
lengthens paths artificially and buys nothing real.

**Remaining known weakness.** `domain_gap` still rewards your own taxonomy —
grading it makes it a graded prior, not a true one. `analogy` remains a crude
proxy: local density-curve overlap is a long way from structural analogy in the
Gentner sense, and the name claims more than the mathematics delivers. Both are
cheap stand-ins for what SAE feature overlap (§8) would measure directly.

---

## 6. Evaluating without fooling yourself

`domain_purity` and `triplet_accuracy` in `sweep.py` use domain labels, which
makes them *topicality* measures. A layer that maximises them may simply be
good at coarse subject classification — which is not the same as good
conceptual geometry, and is arguably the opposite of what this project wants.

This is not hypothetical. On the real build, `domain_purity` climbs almost
monotonically to the final layer (0.216 at layer 28) while `probe_accuracy`
peaks mid-stack (0.923 at layer 16) and falls away. Choosing a layer by domain
purity — the obvious metric, because it needs no hand-labelling — picks close to
the worst layer for this project's purpose. See [FINDINGS.md §2](FINDINGS.md).

`probe_accuracy` is the one that matters *once a human has reviewed it*:
triplets of the form
"A should be closer to B than to C", encoding the relations you actually care
about (mechanism, analogy, part-of) rather than shared topic. 100-200 of these
are worth more than any amount of domain purity.
`data/probes/example_probes.jsonl` has two dozen as a template.

The MiniLM config (`configs/minilm-cpu.yaml`) is the control. If your
residual-stream build does not beat an off-the-shelf sentence encoder on the
probe triplets, the problem is in extraction — not in the idea.

---

## 7. Why layout is precomputed and vectors are shipped

Layout runs in Python and is baked into `graph.json`. A force simulation in the
browser is slow above a few thousand nodes and, worse, lands somewhere
different on every reload — which destroys the spatial memory that makes a map
worth having.

But the browser *does* get vectors, PCA-projected to 256 dims and quantised to
int8 (2.5 MB for 10k concepts). If the front end only had precomputed pairs,
users could only explore what we already decided was interesting, which defeats
the purpose. With vectors on the client, any pair can be collided live in a
couple of milliseconds with no server.

The two implementations of the collision maths (`bridges.py` and
`collide.js`) are pinned together by `test_js_matches_python_collision`.
Two copies of the same scoring drift apart the moment someone tunes one, and
the drift is invisible because both keep producing plausible-looking lists.

---

## 8. Swapping in SAE features

Residual vectors tell you two concepts are similar but not *in what respect*.
SAE features are sparse and often monosemantic, which buys three things:

1. **Explainable edges** — "they share feature #4471 and #9902" instead of
   "cosine 0.71".
2. **Better blends** — a blend in SAE space is a union of features, much closer
   to what a person means by "combine these two ideas" than the arithmetic
   midpoint of two dense vectors.
3. **Controllable distance** — mask out feature groups (register, syntax) and
   re-measure, to find pairs that are far apart *conceptually* rather than
   *stylistically*.

Two things to get right:

* **Hook point.** An SAE is trained at one specific site (residual stream after
  block 20, MLP output at block 12, …). Feeding it activations from anywhere
  else produces confident nonsense. Match `layer` to the release card exactly.
* **Order of operations.** SAEs are trained on individual token activations, so
  encode per token and then pool the feature activations. Pooling the residual
  stream first and encoding the average is wrong — the mean of a set of
  activations is off the manifold the SAE was fit to. `hf_residual.py` takes an
  `activation_transform` applied before pooling for exactly this reason.

Sources: `sae_lens` (`SAE.from_pretrained`), Gemma Scope, Llama Scope,
EleutherAI's TopK SAEs. Feature dimension is typically 16-65x d_model, so
`top_k` sparsification is what keeps 10k concepts affordable.

---

## 9. Scaling

| concepts | what changes |
|---|---|
| **1k** (here) | exact kNN, PCA layout, everything instant |
| **10k** | exact kNN still fine (~0.4 GB for the full similarity matrix, chunked); switch layout to UMAP; extraction ~20-30 min on a 4 GB card |
| **100k** | FAISS `IndexFlatIP`; turn whitening on; consider dropping `embed_vectors` and serving collisions from Python; extraction wants a cloud GPU |
| **1M+** | FAISS HNSW, sharded extraction, and the front end needs level-of-detail rendering (WebGL, not Canvas 2-D) — the current renderer will not carry it |

The stage boundaries exist for this: extraction is the only expensive step, and
everything after it re-runs in seconds. Extract on Colab, copy back
`vectors.npy` + `vectors.meta.json`, iterate locally.

Growing the concept list is the more interesting scaling axis. WordNet gives
~120k synsets with glosses for free; Wikidata gives millions of entities with
descriptions. Both will change the character of the map — WordNet is
lexical-semantic and heavily nominal, Wikidata is entity-heavy and full of
proper nouns that are not really "concepts". Curation is not a chore to
automate away here; it is the substance of what makes the map worth having.
