# Red team

An adversarial audit of this repository, written to be published with it rather
than answered when someone else finds it first. The brief was: what would let a
motivated journalist or a rival researcher dismiss this project inside five
minutes, and how much of that dismissal would be fair?

Everything below is reproducible from the committed data. Where a claim is
tested, the test is named.

---

## The most damaging finding, in one paragraph

**Every artefact in the evaluation chain was written by a language model, and
five places in the repository describe some of them as "hand-written".** The
concept list, the definitions, the structural signatures, the analogy probes,
the candidate batch and its "independent verification" were all produced by
Claude (`git log` shows `Author: Claude <noreply@anthropic.com>` on
`cb070ca`, `afff390`, `c49d219`, `7fa7b68` — i.e. on every data file in the
repo). `data/probes/example_probes.jsonl:1` nevertheless opens
`// Hand-written triplets`, `README.md:173` says `data/probes/ hand-written
triplets for honest evaluation`, `docs/DESIGN.md:171` and `docs/FINDINGS.md:42`
repeat it, and `scripts/structure_experiment.py:8` says "hand-labelled triplets
written before any of this existed". No human wrote them. The load-bearing
number in the project — `probe_accuracy 0.885` — is therefore *the rate at
which Qwen3-1.7B's geometry agrees with Claude's analogy judgements, on an exam
Claude wrote, over definitions Claude also wrote*. That is a real and
interesting measurement, but it is not the measurement the docs say it is, and
the word "hand-written" is the single quotable line that would end a reader's
trust in the rest. It is also the cheapest thing here to fix.

---

## Findings, by severity

### 1. Provenance is misstated; the circularity is total (blocker)

**The attack.** "They benchmarked an LLM against an LLM's opinions using an
LLM's embeddings and called the labels hand-written. There is no ground truth
anywhere in this repository."

**Evidence.**

| artefact | file | who wrote it |
|---|---|---|
| concept labels + definitions | `data/concepts/seed_concepts.jsonl` | LLM (`cb070ca`) |
| candidate concepts | `data/concepts/candidates_batch1.jsonl` | LLM (`1ab066d`/`c49d219`) |
| "independent verification" | `data/concepts/candidates_batch1.verdicts.jsonl` | a second LLM pass |
| structural signatures | `data/structures/seed_structures.jsonl` | LLM (stated openly in the file header and `structure.py:52`) |
| analogy probes | `data/probes/example_probes.jsonl` | LLM, **described as hand-written** |
| analogy triplets | `data/probes/candidates_triplets.jsonl` | LLM (`c49d219` says so) |
| embeddings | Qwen3-1.7B, layer 18 | model |

The tightest illustration is `caching` / `immune-memory`, the project's flagship
analogy. The probe note in `example_probes.jsonl` says *"storing a past result to
answer faster next time"*. The signature written later for `caching` says *"The
result of an expensive operation is retained so that repeat requests are
answered from the store instead of recomputed… Fails when the stored copy no
longer matches what it stands for."* and for `immune-memory` *"A costly first
response is stored so a repeat of the same input is answered far faster… Fails
when the input drifts past the stored key."* The same author, working from the
same intuition, wrote the question and both halves of the answer. Cosine
similarity between those two sentences is not evidence that the two ideas share
a structure; it is evidence that one writer paraphrased themselves.

**How much is fair.** The misdescription is entirely fair and indefensible. The
circularity charge is fair as stated but does not void everything:

* **Does not survive circularity:** `FINDINGS §3`'s `probe_accuracy 0.885` as a
  measure of "conceptual geometry"; any reading of `§2` as evidence about
  concepts rather than about two LLM-derived metrics; the entire claim that the
  map "reverse engineers a map of concepts from LLMs" — the ontology is
  exogenous and author-written, the model only supplies distances between the
  author's own prose. `DESIGN §1` half-concedes this ("you are now partly
  measuring your own prose") and then the README headline does not.
* **Survives circularity intact:** `FINDINGS §1` (anisotropy 0.838 → −0.007) —
  pure geometry, indifferent to who wrote the text; `FINDINGS §4` (the two
  export bugs) — facts about code; the negative result in `FINDINGS §5`
  (signatures lose to definitions). A negative result under contamination that
  would have *inflated* the losing side is the most credible number in the
  repository, and `FINDINGS §5` already says so.

**The fix.** Delete the word "hand-written" from all five sites and replace it
with a provenance table like the one above. Rename `probe_accuracy` in the docs
to what it measures — *agreement with an LLM-authored analogy key*. Then get
even 30 triplets adjudicated by a person who did not write them; that single
step converts the headline number from circular to merely small.

---

### 2. The 26-probe set is used three times, is much easier than advertised, and cannot support the conclusions drawn from it

**The attack.** "Twenty-six items, written by the system's own author, chose the
layer, scored the layer, and judged the structural experiment. Nothing was held
out, and the exam is easier than they claim."

**Evidence.**

* `docs/DESIGN.md:171` says probe triplets encode "relations (mechanism,
  analogy, part-of) rather than shared topic". They largely do not. Of the 26
  probes in `example_probes.jsonl`, **10 are solvable by the rule "pick the
  candidate in the anchor's own domain"** (`moral-hazard → principal-agent-problem`
  not `diffusion`; `monetary-policy → inflation` not `phoneme`;
  `entropy → conservation-of-energy` not `taboo`; …). Only **1 of 26**
  (`antibiotic-resistance → natural-selection`, distractor `triage`) is hard in
  the sense the newer file defines — cross-domain target with a same-domain
  distractor. The newer `candidates_triplets.jsonl` is 250/250 on that
  criterion, so the project already knows what a hard probe looks like; the
  numbers in FINDINGS just were not computed on one.
* A **TF-IDF bag-of-words baseline over the definitions** — no model at all —
  scores **17/26 = 0.654** on `example_probes.jsonl`, against Qwen's 0.885.
  The honest headline is "0.885 vs a 0.654 lexical baseline, a difference of
  six items", not "0.885 vs chance 0.5". On the 250-triplet set the same
  baseline scores **125/250 = 0.500**, exactly chance — that file is a far
  better evaluation set and is currently unused.
* Every layer distinction in `FINDINGS §2` is one or two items.
  0.923/0.885/0.846/0.808 are 24, 23, 22 and 21 correct out of 26; Wilson 95%
  intervals are [0.76,0.98], [0.71,0.96], [0.66,0.94] and [0.62,0.91]. FINDINGS
  says the set "comfortably separates mid-stack from late" — that separation is
  **two triplets** (layer 16's 24/26 against layer 28's 22/26), and layer 28
  actually scores *above* layers 8 and 24. The word "comfortably" is not
  supportable.
* `FINDINGS §5`'s table has n=21. r@5 0.476 vs 0.333 vs oracle 0.619 is 10 vs 7
  vs 13 items; the intervals [0.28,0.68], [0.17,0.55], [0.41,0.79] all overlap.
  The doc's caveat ("21 probes is a small sample") is present but the prose
  around it — "the oracle union confirms real headroom", "a measured
  median-rank-8-to-2 ceiling" — is written with more confidence than three items
  of separation license.
* **Unflagged confound in §5.** Layer 18 was selected using these probes in the
  *topical* space. `scripts/structure_experiment.py` then embeds the structural
  signatures at the same layer 18 without a separate sweep. The comparison
  "topical beats structural" is therefore run at a layer tuned for one of the
  two contestants. That does not overturn the result, but it is not stated, and
  a rival would find it in ten minutes.

**How much is fair.** All of it. None of it is fatal — the conclusions are
plausible and the direction of the §5 result is probably right — but every
number needs a confidence interval next to it and the word "comfortably" needs
to go.

**The fix.** Adjudicate and use the 250-triplet set; report the TF-IDF and
MiniLM baselines alongside every accuracy; hold out a probe subset from layer
selection; sweep layers separately for the structural space.

---

### 3. Overclaiming: statements the shipped build does not support

Checked line by line against `data/build/pairs.json`, `graph.json`,
`layer_sweep.json` and `structure_experiment.json`.

| claim | where | what the build actually shows |
|---|---|---|
| "A **blend vacancy above 1** means nothing in your map names that blend… which is the case worth looking at" | `README.md:118` | Vacancy computed as in `bridges.py:158` over all 11,628 concept pairs: median **1.010**, range 0.90–1.12, **58% of all pairs exceed 1**. The threshold selects the majority, not the interesting minority. In a 153-concept map every blend point is vacant; the statistic is a tautology at this scale. |
| bridgeability targets "3-5 hops"; "a chain of three or four steps actually exists" | `DESIGN.md:138`, `pairs.py:12,22` | In the 200 shipped pairs, hop counts are **2 or 3 only** — max 3, median 3. `target_hops=4.0` is unreachable in this build, so the feature never scores above 0.801 and is effectively binary. **FIXED** — see §11. |
| pairs "scored on distance, bridgeability, domain gap, midpoint sparsity, and neighbourhood-structure similarity" | `README.md:103` | Across the 200 shipped pairs: `domain_gap` is **1.000 for every pair** (zero variance — it is a filter, not a feature); `midpoint_gap` spans **0.129–0.144**; `analogy` spans 0.75–0.98 with median 0.91. Only `distance` (0.49–1.00) and the two-valued `bridgeability` actually rank anything. "Five features" is presentation; two do the work. **FIXED** — see §11, and the cause of the `midpoint_gap` case turned out to be a bug, not a weighting choice. |
| "real cross-domain mechanism analogies **that no topic model would produce**" | `FINDINGS §3` | No topic model, and no baseline of any kind, was ever run. The MiniLM control that `DESIGN §6` calls the decisive test ("If your residual-stream build does not beat an off-the-shelf sentence encoder on the probe triplets, the problem is in extraction") appears in `configs/` and `requirements.txt` and **is not reported anywhere**. |
| "Anisotropy is far worse than the literature's framing suggests" | `FINDINGS §1` heading | Mean raw cosine 0.838 is squarely inside the range the anisotropy literature reports for mid/late transformer layers. The measurement is right; the comparative framing is not supported and is not needed — the finding stands on its own. |
| "Roughly 10k concepts in 20-30 minutes" on 4 GB; "2.5 MB for 10k concepts"; "under 6 ms" per collision | `README.md` §"On 4 GB of VRAM", §"Layout" | The only build in the repo is 153 concepts, fp32, **on CPU** (`graph.json` meta: `quantised_4bit: false`, `device: cpu`). Every 4-bit, GPU and 10k-scale figure is extrapolation presented in the indicative mood. |
| "collide distant parts of that map together looking for **connections nobody has made yet**" | `README.md:4-5` | Novelty is never checked, by any means, for any pair. The top-scoring shipped pair is `sampling-bias × consensus-protocol`. |
| `scripts/generate_structures.py` | cited by `DESIGN`/`structure.py:52` as how signatures scale | **The file does not exist in the repo.** The one generation step that requires a large model is unreproducible. |

**How much is fair.** All fair, all fixable by editing prose or adding one
script. None of it implies bad faith — the pattern is a README written for the
design as intended and not re-checked against the build as shipped.

---

### 4. Ideology: no net partisan lean, but a real hedging asymmetry and two opposite disciplinary captures

Tested in both directions, over all 405 concepts (153 seed + 252 candidates).

**Where a conservative or market-liberal reader would object.**

* `structural violence` (`candidates_batch1.jsonl`, anthropology): *"Harm
  inflicted not by any actor but by the ordinary arrangement of institutions."*
  Galtung's contested theoretical construct, asserted as a plain fact. The
  critique writes itself: the definition classifies institutional outcomes as
  *violence* by fiat, which is the entire point in dispute.
* `cultural capital`: *"Tastes and competences that convert into advantage
  because institutions recognise them"*; `habitus`: *"Durable dispositions laid
  down by upbringing that generate behaviour without deliberate choice."* Two
  Bourdieusian constructs stated as mechanism, the second strongly deterministic
  about individual agency.
* `planned obsolescence` (seed, engineering): *"Designing a product with a
  deliberately limited useful life to drive replacement."* A contested claim
  about corporate intent, asserted twice more downstream — its signature says
  *"replacement demand is created rather than met, aligning the producer's
  interest against durability"*, and it is used as the correct answer in two
  probes (`moral-hazard → planned-obsolescence`, `apoptosis →
  planned-obsolescence`).
* `ethnocentrism`: *"Judging another way of life by the standards of one's own,
  taken as the default."* A relativist framing in which cross-cultural
  evaluation is itself the error.

**Where a leftist or heterodox economist would object.**

* `tragedy of the commons` (seed): *"The depletion of a shared resource by
  individuals acting in their own rational interest"*, with a signature adding
  *"Resolved only by changing who bears the cost."* Hardin without Ostrom —
  whose Nobel-winning work showed commons are routinely self-governed *without*
  changing ownership. "Resolved only by" is the contested half, asserted.
* The economics list is a straight neoclassical syllabus: comparative
  advantage, deadweight loss, rent seeking, natural monopoly, price elasticity,
  signalling, adverse selection, transaction cost, arbitrage, discounting,
  time inconsistency, hold-up, externality. Absent from all 405: exploitation,
  monopsony, market power, unemployment, aggregate demand, class, inequality,
  distribution, taxation, labour, unions. The map presents economics as a
  science of efficiency with no distributive content.
* `moral hazard` and `creative destruction` are given in their strongest,
  least-qualified forms, both of which do political work outside the seminar.

**The asymmetry that makes this quotable.** The dataset demonstrably *knows how
to hedge*. Four definitions do it — `linguistic relativity`: "**The claim
that** the categories a language provides shape…"; `reductionism`: "**The view
that**…"; `social contract`: "**The idea that**…"; `form follows function`:
"**The principle that**…". None of the eight contested items above gets that
treatment. A critic does not need to prove intent; they only need to put
`linguistic relativity` and `structural violence` side by side.

**Verdict.** There is **no net left or right lean**. What there is is
*disciplinary capture that points in opposite directions in different domains*
— Chicago-flavoured economics next to Bourdieu-flavoured anthropology —
because each field was sampled from its own canon without a step that asks
"is this contested?". That is a more defensible position than a partisan lean,
and it is worth saying out loud rather than being discovered.

**The fix.** One pass with a single rule: any concept that names a contested
empirical or normative claim gets an explicit hedge ("The claim that…",
"On one account…"), the same one `linguistic relativity` already gets.

---

### 5. Absence: what a 405-concept map of "human concepts" leaves out

Keyword sweep over all 405 labels and definitions. Zero occurrences of:
religion, God, faith, prayer, the sacred, the soul; democracy, voting,
elections, sovereignty, revolution, the state as such; war, military, deterrence,
weapons; marriage, parenthood, childhood, grief, love, friendship; food,
cooking, agriculture, farming; sport, games, play as such; craft, trade skills,
navigation; poverty, inequality, taxation; climate, weather, geology, astronomy,
the ocean; humour; sex; almost all emotion (`attachment` and `flow state` are
the nearest things to a feeling in the entire set).

The 18 domains are, without exception, university departments — and not even
all of them: no history, no political science, no theology, no education, no
business, no earth science. Religion appears only as an object of study, in the
etic voice: `ritual` is *"A prescribed sequence of actions whose meaning exceeds
its practical effect"*, alongside `taboo`, `totemism`, `commensality`,
`syncretism`. A believer would say the map has defined the referent away. A
historian would note that nothing in the set is older than the modern academy's
framing of it.

The implied worldview is legible and worth owning: **secular, technocratic,
Anglophone, modern, and mechanism-first** — the world as a set of processes with
transferable dynamics, not as a set of ends, values, powers or lived
experiences. Supporting details: 50% of concepts are STEM-domain; every
definition is in one voice and one dialect (British spellings throughout);
`wabi sabi` is close to the only non-Western entry in 405; the philosophy
section is Kant, Hobbes and analytic epistemology; `instrumental convergence`
and `epistemic humility` (defined as *"Holding beliefs in proportion to
evidence"*, which is calibration, not humility) place the author precisely in
the AI-safety/rationalist milieu.

**How much is fair.** Fair as a description, unfair as an accusation. A
153-concept demonstrator has to omit almost everything. What is *not* defensible
is the framing: `README.md` and `DESIGN §9` present list growth as a scaling
problem (WordNet, Wikidata) rather than a coverage problem. The honest sentence
is: *this is a map of mechanisms in the modern research academy, and the domains
absent from it are absent by choice.*

Also worth pre-empting: `candidates_batch1.jsonl` is exactly **14 concepts per
domain** for all 18 domains — a quota that the project's own verifier flagged in
`c49d219` as having "padded the tail of the thinner domains with technique names
rather than mechanisms". Good that it was caught; bad that the file shipped with
the quota intact.

---

### 6. Construct validity: contested empirical claims used as ground truth

The probes do not merely describe structures; they assert that specific,
disputed empirical claims are true, and then score a model on agreeing.

* **`priming`.** Seed definition: *"When exposure to one stimulus changes the
  response to a later one without conscious awareness."* Social/behavioural
  priming is the central casualty of psychology's replication crisis. It is not
  a marginal entry: `placebo-effect → priming` is one of the 21 analogy probes,
  and it is the **headline example of FINDINGS §5(b)** ("structure rescues what
  topic cannot reach: `placebo-effect x priming 81 -> 2`"). The flagship
  positive result in the structural experiment rests on a pair whose second
  member may not be a real effect.
* **`placebo effect`, defined two incompatible ways in the same repository.**
  The definition asserts improvement (*"Improvement produced by expectation of
  treatment"*) and the signature doubles down (*"produces a real measured
  change"*), while a probe asserts the sceptical reading — `regression-to-the-mean
  → placebo-effect`, note: *"improvement that would have happened anyway is
  credited to the intervention"*. Both cannot be the structure.
* **`learned helplessness`** is given in its 1967 form; Maier & Seligman (2016)
  reversed the mechanism (passivity is the default; control is what is learned).
  It is the `closer` in `hysteresis → learned-helplessness`.
* **`attachment`.** Two probes (`attachment → hysteresis`, `attachment →
  overfitting`, *"a model locked to one narrow early sample"*) encode strong
  early-determinism about adult relating, which the longitudinal literature
  supports far more weakly than the note implies.
* **`triage → opportunity-cost`**, with the signature *"Optimises the aggregate
  at the cost of the individual"*, encodes one contested position in medical
  ethics (utilitarian allocation) as the structure of triage, where
  egalitarian and lottery protocols are live alternatives.

**How much is fair.** Fully fair, and cheap to fix: these are ~6 rows out of
276 probes. Either drop them or mark them `"contested": true` and report
accuracy with and without.

---

### 7. `pairs.py`'s "interesting" is a value judgement, and it is the author's

`docs/DESIGN §5` already concedes that `domain_gap` "rewards your own taxonomy"
and that `analogy` "is a crude proxy". Two things it does not concede:

* `midpoint_gap` encodes **"unnamed ⇒ unexplored ⇒ valuable"**. The opposite
  reading — unnamed because incoherent, or because the blend is not worth a
  word — is never entertained, and as shown in §3 the statistic cannot
  discriminate at this map size anyway.
* `analogy` is the name of a feature that computes **kNN-similarity-curve shape
  overlap**. Two concepts score high because their neighbourhoods decay at a
  similar rate, which is a density statistic and has no relation to structural
  analogy in the Gentner sense the docs invoke. Its shipped range is
  0.75–0.98, so it mostly re-ranks noise. The name does work the mathematics
  does not do — that is the definition of smuggling a value judgement, even if
  the docstring is candid about the proxy.

**The fix.** Rename `analogy` to `density_profile_overlap`; state the
`midpoint_gap` assumption as an assumption; or drop both until the structural
space (`structure.py`) can measure the thing directly, which is what it exists
for.

---

### 8. Signature discipline: the "hard rule" is broken in ~8% of rows and the automated check cannot see it

`structure.py`'s `SIGNATURE_INSTRUCTION` states a hard rule: *"do not use any
word that names or implies the field"*. Running `leaked_vocabulary()` over
`seed_structures.jsonl` returns **one** hit (`phase-space` / "space"). A
30-second word-boundary regex over the same file returns **13**:
`information-asymmetry` ("price", "market"), `liquidity` ("price"),
`opportunity-cost` ("price"), `cultural-transmission` ("genetic",
"biological"), `kinship-system` ("biological"), `herd-immunity`,
`antibiotic-resistance`, `diaspora`, `natural-selection` ("population"),
`confirmation-bias` ("immune"), `grammatical-case` ("economy"),
`symmetry-breaking` ("law").

The checker only compares against the concept's *own label words* and its domain
*name*, so an economics signature saying "price" and "market" is invisible to
it. `structure.py` says "treat a clean result as necessary but not sufficient" —
correct, and then the sufficient check was never run. Since the structural space
exists precisely to strip topical signal, residual field vocabulary in 8% of
rows is a direct confound on `FINDINGS §5`.

**The fix.** Add a domain-vocabulary lexicon to `leaked_vocabulary()` and make
it a test.

---

### 9. Probe independence: 250 triplets, 159 distinct claims

`candidates_triplets.jsonl` contains 250 rows over **159 distinct
anchor/closer pairs**; **91 pairs appear in both directions** (`A closer to B
than C` and `B closer to A than D`). The commit message for `7fa7b68` measures
this honestly and collapses them *for review staging* — but the file as shipped
still contains all 250, and anyone computing an accuracy over it will report an
n that is 57% larger than the number of independent claims. Also worth noting:
despite living beside `candidates_batch1.jsonl`, **zero of the 250 triplets
reference any of the 252 candidate concepts** — the candidate batch has no
evaluation coverage at all.

**The fix.** Ship the collapsed file, or add a `"reverse_of"` field so nobody
computes an inflated n.

---

## What I tried to attack and could not

This section is the reason to believe the rest.

* **Fabricated or massaged numbers — none found.** I recomputed
  `FINDINGS §3`'s nearest-neighbour lists from the committed vectors
  (`vectors.npy` + `space.prepare(center=True, remove_top_k=8)`) and reproduced
  every figure to the digit: `monetary-policy → control-system (0.24),
  inflation (0.21), phase-transition (0.16)`; `counterpoint → polyrhythm
  (0.36)`; `triage → differential-diagnosis (0.18)`. The `FINDINGS §2` and
  `§5` tables match `data/build/layer_sweep.json` and
  `structure_experiment.json` exactly. The isotropy numbers match
  `graph.json`. Nothing is rounded in a flattering direction.
* **Cherry-picked neighbour examples — a mild charge at worst.** FINDINGS §3
  shows 5 of 153 neighbour lists and calls 2 of the 5 "close to noise". A
  seeded random sample of 20 concepts gives roughly the same picture (good:
  `metamorphosis → phase-transition`, `critical-mass → activation-energy`,
  `fiduciary-duty → principal-agent-problem`, `inflation → semantic-drift`;
  junk: `photosynthesis → creative-destruction`, `carrying-capacity →
  epistemic-humility`, `chirality → fiduciary-duty`, `leitmotif →
  keystone-species`). The examples are the best ones, but the doc's own
  surrounding text — "the nearest concept is often just the least arbitrary of
  many arbitrary options" — states the true hit rate more honestly than the
  examples imply. Recommendation: print a random sample instead. Not a scandal.
* **The headline claims in `Status` are not overclaimed.** "153 concepts is too
  few", "top-neighbour similarities sit only two to four standard deviations
  above random" and "Nothing here shows the pairs are generative for a human"
  are the two strongest attacks available on the project's *purpose*, and the
  README makes both of them itself, in the README, above the fold. A critic
  cannot claim to have discovered them.
* **The tests are real.** `PYTHONPATH=src:tests pytest tests/ -q` → **29
  passed**, exactly as claimed, in 94 s with no downloads. They target silent
  failures (padding leaking into pooled vectors, isotropy correction not
  correcting, Python↔JavaScript collision parity), which is the right target
  set. `test_sampled_pairs_are_all_bridgeable` and
  `test_sampled_pairs_respect_diversity_caps` genuinely enforce what `pairs.py`
  documents.
* **The verification pass is what it says it is (minus "independent" meaning
  human).** The commit claims 213 keep / 16 uncertain / 23 drop; the file
  contains exactly 213 / 16 / 23. The claim "100% cross-domain anchor/closer,
  100% same-domain hard distractors, 147 anchors" over the 250 triplets is
  **exactly true** on recount, as is the "91 reverse-direction pairs" figure.
  The six factual corrections listed in `c49d219` (bioaccumulation described
  biomagnification, etc.) are real errors, really fixed.
* **`FINDINGS §5` is a model of how to report a failed hypothesis.** It leads
  with the refutation, names the intuitive scorer that does not work, reports
  wins *and* losses in a balanced 4-and-4 table, and volunteers the
  contamination risk with the correct sign analysis ("this contamination would
  inflate the structural result, and the structural result *lost*"). The
  attack I expected here — "they buried the negative result" — does not land at
  all.
* **No evidence of a political filter on inclusion.** I looked for one-sided
  omission (e.g. market concepts present and their critiques absent, or vice
  versa) and found the opposite: both economics and anthropology are
  over-represented in their own orthodoxies. The problem is unhedged canon, not
  selection.
* **The geometry work is sound and is the strongest thing here.** Centring plus
  all-but-the-top on a real 2048-d build, with before/after printed on every
  run, is a correct implementation of a correct idea; the two export bugs in
  `§4` (double centring, declared-vs-actual blob width) are exactly the class of
  bug that kills projects like this silently, and they were caught by a parity
  test rather than by luck.

---

## Blockers for publishing this as a proof of concept

Only one is a blocker.

1. **The word "hand-written."** Five sites, one commit to fix, and until it is
   fixed every other honest statement in the repository is discounted by a
   reader who finds it. Replace with a provenance table.

Everything else is a "publish with these edits":

2. Confidence intervals on every accuracy, and "comfortably" struck from
   `FINDINGS §2`.
3. `README` claims about blend vacancy, 3-5 hops, five features, "no topic
   model would produce", and the 4-bit/10k figures brought into line with the
   shipped build — or marked as untested projections.
4. A one-line hedge on the eight contested definitions, matching the treatment
   `linguistic relativity` already gets.
5. `scripts/generate_structures.py` committed, or the reference removed.
6. The MiniLM control run and reported, since `DESIGN §6` declares it decisive.
   (Interim data point from this audit: a TF-IDF bag-of-words baseline gets
   0.654 on the probe set against Qwen's 0.885, and exactly 0.500 on the harder
   250-triplet set.)

---

## Addendum: the structural-ontology commitment (added after the initial audit)

The audit above, and an earlier scan of the corpus, both looked for bias in
**vocabulary and claims**. That is the wrong place to look. The load-bearing
ideological commitment in this dataset is *structural*, it sits upstream of any
wording, and it was imposed deliberately by a single design decision.

`SIGNATURE_INSTRUCTION` in `src/collider/structure.py` requires every concept
to be described as:

> the mechanism (what acts on what, and with what effect); what drives it or
> what it is limited by; and how it characteristically fails.

That is a **functionalist, systems-engineering ontology**. It presupposes that a
concept is a causal machine with inputs, a governing dynamic, and a breakdown
state. Concepts that really are causal machines fit it exactly. Concepts that
are normative, constitutive or interpretive do not — and get rewritten as
machines anyway.

`taboo` becomes "a prohibition so strong that violating it threatens the
surrounding order… Marks a structural boundary". That is a Durkheimian
functionalist reading, produced by the template rather than chosen. A different
tradition would say a taboo is *constitutive* of the sacred and has no failure
mode at all. `categorical imperative`, a normative principle, is slotted in as
"a consistency test on proposed policies" — an instrumentalist reading. The
template made those choices, not an author.

**The effect is measurable.** Counting signatures that contain a genuine
failure clause, by domain:

| | with failure clause | rate |
|---|---|---|
| biology | 10/10 | 100% |
| computing | 6/12 | 50% |
| physics | 4/10 | 40% |
| psychology | 2/10 | 20% |
| philosophy | 1/10 | 10% |
| economics, law, art, chemistry, linguistics, ecology | 0 | **0%** |
| **STEM overall** | 30/77 | **39%** |
| **non-STEM overall** | 8/76 | **11%** |

A 28-point gap. Where the template's third slot has nothing real to hold, it is
filled with padding or quietly dropped — so non-STEM concepts carry a weaker,
partial signature and are systematically disadvantaged in the structural space.
This is a plausible contributor to the §5 result, and it was not controlled for.

**This is not a left or right lean.** It is modernist and technocratic, and it
privileges the fields whose concepts happen to be machines. It is also the
hardest finding in this document to fix, because the template is what makes
cross-domain comparison possible at all — a signature format that accommodated
constitutive and normative concepts equally well would be a different, and
probably weaker, instrument.

Honest options, none free:
1. Keep one template and state the commitment plainly. Cheapest; leaves the
   distortion in place.
2. Add a second signature type for non-mechanistic concepts and compare within
   type rather than across. More faithful; halves the cross-domain reach that
   is the point of the project.
3. Drop the failure slot, making the template mechanism + constraint only.
   Reduces the distortion, loses the slot that best discriminates causal
   analogies.

No option is taken here. The measurement is recorded so the choice is made
knowingly.


---

## Addendum 2 — the three degenerate features, resolved

The audit's §3 was right that three of the five pair-scoring features carried no
information. Fixing them turned up something the audit did not have: one of the
three was not a tuning choice at all.

### `midpoint_gap` was measuring the pair against itself

The feature asks "is the semantic midpoint of these two concepts in an empty
region?" — the idea being that an unoccupied blend is a hole in the map. It was
computed as the distance from the normalised midpoint to its nearest neighbour
**over the whole concept list, the two endpoints included**.

For unit vectors that is circular. The normalised midpoint of `a` and `b` sits
at `sqrt((1+cos(a,b))/2)` from each of them, and for any realistic concept list
that is nearer than anything else in the space. So the "nearest neighbour of the
midpoint" was `a` or `b` — in **4000 of 4000** measured candidates — and the
feature was a monotone restatement of the pair's own distance. Correlation with
`distance` across the candidate set: **r = 1.000**.

Masking the two endpoints, which the question always implied ("does a *third*
concept name this blend?"), drops that correlation to **0.13** and widens the
spread 2.6×. `bridges.py` had masked the endpoints since it was written; this
file never did. Two implementations of the same idea, one correct, and the
divergence was invisible because both produced plausible lists.

### `domain_gap` was a filter wearing a feature's clothes

Binary "different domain?" is true for 96% of random pairs at 153 concepts over
18 domains, and 100% after the distance pre-filter — hence zero variance. It now
measures the distance between the two **domain centroids**: range 0.00–1.37,
correlation with pair distance 0.12, so biology↔chemistry and
biology↔jurisprudence are no longer scored as the same move. Same-domain pairs
still score a hard zero.

The audit's deeper objection stands unchanged: this rewards the author's
taxonomy. Grading it makes it a *graded* prior, not a true one.

### `bridgeability` was aimed outside the graph

`target_hops = 4.0` was a hard-coded claim about graph diameter, and a wrong one:
97.6% of reachable candidates bridge in 2 or 3 hops. Aiming at 4 made the feature
rank the *longest* chain rather than the most productive one — the opposite of
the documented intent. The target is now read off the graph's own hop
distribution (75th percentile). Path cost, which the code computed on every
candidate and then threw away, breaks ties within a hop stratum: among chains of
equal length, prefer the one whose steps are tighter.

### Measured before and after, on the 200 shipped pairs

| feature | before (min–max, std) | after (min–max, std) | distinct values |
|---|---|---|---|
| `domain_gap` | 1.000–1.000, **0.000** | 0.142–0.993, 0.181 | 1 → **200** |
| `midpoint_gap` | 0.131–0.142, 0.002 | 0.465–1.000, 0.116 | 66 → **200** |
| `bridgeability` | 0.801–1.000, 0.079 | 0.700–0.994, 0.062 | **2** → **195** |
| `midpoint_gap` × `distance` correlation | **+1.000** | **−0.005** | — |

`test_features_are_not_degenerate`, `test_midpoint_gap_excludes_the_pair_itself`
and `test_target_hops_adapts_to_the_graph` now fail the build if any of the five
collapses again.

### What is still open from §3 and §7

* **Blend vacancy is still a tautology at this scale.** 58% of all pairs exceed
  the ">1 means nothing names this blend" threshold, and after the fix every
  selected pair sits at 1.01–1.12. At 153 concepts the whole space is sparse.
  This is a concept-count problem, not a formula problem, and it does not go
  away until the list is an order of magnitude bigger.
* **`analogy` still claims more than it computes.** It is kNN-similarity-curve
  shape overlap, which is a density statistic, not structural analogy in the
  Gentner sense the docs invoke. Renaming it to `density_profile_overlap`
  remains the honest move and has not been made.
* **No baseline has been run.** The MiniLM control that `DESIGN §6` calls the
  decisive test is still unreported.

---

## Addendum 3 — the structural-ontology finding is now a scope decision, not a defect

Addendum 1 measured that `SIGNATURE_INSTRUCTION`'s mechanism/driver/**failure**
template fits STEM subject matter far better than it fits the humanities: a
failure clause appears in 39% of STEM signatures against 11% of non-STEM ones
(biology 10/10, computing 6/12, philosophy 1/10, and 0% across economics, law,
art, chemistry, linguistics and ecology). Three remedies were laid out and none
was chosen.

**The owner has now chosen: the STEM skew is accepted and intended.** The
project's purpose is transferring solved methods to unsolved problems, and
mechanism/driver/failure is the right decomposition for the fields where that
transfer is tractable. A structural template that describes a chemical reaction
and a legal doctrine equally well would describe neither usefully.

This converts a finding into a documented scope boundary, and the boundary
should be read plainly: **for non-STEM concepts this map's structural layer is
weaker, and signatures there will under-specify.** That is a known cost of the
decomposition, not evidence that the pipeline is malfunctioning. The finding
stays on the record above so the trade is legible rather than accidental.
