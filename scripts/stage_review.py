#!/usr/bin/env python3
"""Stage candidate files for the review tool, and merge reviewed output back.

    python scripts/stage_review.py              # stage everything it can find
    python scripts/stage_review.py --merge ~/Downloads/reviewed_probes.jsonl

Staging copies candidate batches into `web/data/review/` alongside a manifest
and a concept lookup (definitions + structural signatures), because the browser
can only fetch from the served directory. Merging validates a reviewed export
and appends it to the real data files.

The reviewer is deliberately a separate page from the map: reviewing is a
different job, done in a different mood, and mixing the two would clutter both.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "data" / "review"

# (path, batch name, mode) — mode drives which UI the reviewer shows.
CANDIDATES = [
    ("data/probes/candidates_triplets.jsonl", "analogy-triplets", "triplet"),
    ("data/concepts/candidates_batch1.jsonl", "concepts-batch1", "concept"),
]

SLUG = re.compile(r"[^a-z0-9]+")
slugify = lambda s: SLUG.sub("-", s.strip().lower()).strip("-")  # noqa: E731


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"  !! {path.name}:{lineno} is not valid JSON: {exc}")
    return rows


def build_concept_lookup() -> list[dict]:
    """Definitions + structural signatures for everything a triplet can cite.

    The signature matters: without it the reviewer is comparing topics, which is
    precisely the confusion these labels exist to resolve.
    """
    concepts = read_jsonl(ROOT / "data" / "concepts" / "seed_concepts.jsonl")
    structures = {r["id"]: r["structure"]
                  for r in read_jsonl(ROOT / "data" / "structures" / "seed_structures.jsonl")}
    out = []
    for c in concepts:
        cid = c.get("id") or slugify(c["label"])
        out.append({
            "id": cid, "label": c["label"], "definition": c["definition"],
            "domain": c.get("domain", ""), "structure": structures.get(cid, ""),
        })
    return out


def validate_triplets(rows: list[dict], known: set[str]) -> tuple[list[dict], list[str]]:
    """Drop rows the reviewer could not render, and say why.

    A triplet citing a concept that does not exist wastes a human judgement, so
    it is cheaper to catch it here than to let it reach the screen.
    """
    good, problems = [], []
    seen = set()
    for i, r in enumerate(rows, 1):
        missing = [k for k in ("anchor", "closer", "farther") if k not in r]
        if missing:
            problems.append(f"row {i}: missing {missing}")
            continue
        unknown = [r[k] for k in ("anchor", "closer", "farther") if r[k] not in known]
        if unknown:
            problems.append(f"row {i}: unknown concept id(s) {unknown}")
            continue
        if len({r["anchor"], r["closer"], r["farther"]}) != 3:
            problems.append(f"row {i}: anchor/closer/farther are not three distinct concepts")
            continue
        key = (r["anchor"], r["closer"], r["farther"])
        if key in seen:
            problems.append(f"row {i}: duplicate of an earlier row")
            continue
        seen.add(key)
        good.append(r)
    return good, problems


def collapse_reverse_pairs(rows: list[dict]) -> tuple[list[dict], int]:
    """Keep one direction per concept pair.

    A triplet asserting `A is closer to B than to C` and one asserting `B is
    closer to A than to D` make the SAME underlying claim — that A and B share
    a structure — with different distractors. They are not identical
    judgements, but once a reviewer has affirmed A~B the reverse is largely
    predictable, so the second one buys little and costs the same attention.

    On the first generated batch this was 36% of rows, so collapsing is worth
    real minutes. It is also harmless downstream: `scripts/structure_experiment.py`
    already scores retrieval symmetrically, taking the better of the two
    directions, so it never needed both.

    Where both directions exist, keep the one whose anchor has FEWER triplets
    overall. That spends the saved attention on widening anchor coverage rather
    than deepening it on concepts already well represented.
    """
    from collections import Counter
    anchor_counts = Counter(r["anchor"] for r in rows)
    best: dict[frozenset, dict] = {}
    for r in rows:
        key = frozenset((r["anchor"], r["closer"]))
        incumbent = best.get(key)
        if incumbent is None or anchor_counts[r["anchor"]] < anchor_counts[incumbent["anchor"]]:
            best[key] = r
    kept = [r for r in rows if best.get(frozenset((r["anchor"], r["closer"]))) is r]
    return kept, len(rows) - len(kept)


def apply_verdicts(rows: list[dict], src: Path) -> tuple[list[dict], dict[str, int]]:
    """Filter candidates through an independent verifier's judgements.

    The point of the two-agent split is that a human should only ever see rows
    that survived a second, cold opinion. Anything the verifier dropped never
    reaches the screen; anything it could not settle is passed through *flagged*,
    so the human's attention lands on the genuinely borderline cases instead of
    being spread evenly across rows that were already checked twice.

    Absent a verdicts file this is a no-op, so an unverified batch still works —
    it just costs the reviewer more.
    """
    vpath = src.with_suffix(".verdicts.jsonl")
    if not vpath.exists():
        return rows, {}

    verdicts = {v["label"]: v for v in read_jsonl(vpath) if "label" in v}
    counts = {"keep": 0, "uncertain": 0, "drop": 0, "unjudged": 0}
    kept = []
    for r in rows:
        v = verdicts.get(r.get("label"))
        if v is None:
            counts["unjudged"] += 1
            kept.append(r)
            continue
        verdict = v.get("verdict", "uncertain")
        counts[verdict] = counts.get(verdict, 0) + 1
        if verdict == "drop":
            continue
        # A verifier that proposes a fix has done work worth keeping; the human
        # sees the corrected text and can still edit or reject it.
        if v.get("suggested_definition"):
            r = {**r, "definition": v["suggested_definition"]}
        if v.get("suggested_domain"):
            r = {**r, "domain": v["suggested_domain"]}
        r = {**r, "_verdict": verdict, "_reasons": v.get("reasons", [])}
        kept.append(r)
    return kept, counts


def validate_concepts(rows: list[dict], existing: set[str]) -> tuple[list[dict], list[str]]:
    good, problems = [], []
    seen = set()
    for i, r in enumerate(rows, 1):
        if not r.get("label") or not r.get("definition"):
            problems.append(f"row {i}: missing label or definition")
            continue
        slug = slugify(r["label"])
        if slug in existing:
            problems.append(f"row {i}: '{r['label']}' already exists in the seed list")
            continue
        if slug in seen:
            problems.append(f"row {i}: '{r['label']}' duplicated within this batch")
            continue
        seen.add(slug)
        good.append(r)
    return good, problems


def stage(keep_reverse: bool = False) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lookup = build_concept_lookup()
    (OUT / "concepts.json").write_text(json.dumps(lookup), encoding="utf-8")
    known = {c["id"] for c in lookup}

    batches = []
    for rel, name, mode in CANDIDATES:
        src = ROOT / rel
        if not src.exists():
            print(f"  – {rel} not present yet, skipping")
            continue
        rows = read_jsonl(src)
        rows, verdicts = apply_verdicts(rows, src)
        if mode == "triplet":
            if not keep_reverse:
                rows, collapsed = collapse_reverse_pairs(rows)
                if collapsed:
                    print(f"  · collapsed {collapsed} reverse-direction rows "
                          f"(same claim, different distractor) — pass --keep-reverse to review them")
            good, problems = validate_triplets(rows, known)
        else:
            good, problems = validate_concepts(rows, known)

        dest = OUT / f"{name}.jsonl"
        dest.write_text("\n".join(json.dumps(r) for r in good) + "\n", encoding="utf-8")
        batches.append({"name": name, "file": dest.name, "mode": mode, "count": len(good)})

        if verdicts:
            print(f"  · verifier: {verdicts.get('keep', 0)} keep, "
                  f"{verdicts.get('uncertain', 0)} uncertain, "
                  f"{verdicts.get('drop', 0)} dropped before review"
                  + (f", {verdicts['unjudged']} unjudged" if verdicts.get("unjudged") else ""))
        print(f"  ✓ {name}: {len(good)} staged for review")
        if problems:
            print(f"    {len(problems)} rejected before review:")
            for p in problems[:8]:
                print(f"      - {p}")
            if len(problems) > 8:
                print(f"      … and {len(problems) - 8} more")

    if not batches:
        print("\nNothing staged. Generate candidates first.")
        return 1

    (OUT / "manifest.json").write_text(json.dumps({"batches": batches}, indent=2),
                                       encoding="utf-8")
    print(f"\nStaged {sum(b['count'] for b in batches)} items into {OUT.relative_to(ROOT)}")
    print("Review them at:  python -m collider serve   ->  http://127.0.0.1:8000/review.html")
    return 0


def merge(path: Path) -> int:
    """Append a reviewed export to the real data files, with validation."""
    rows = read_jsonl(path)
    if not rows:
        print(f"{path} contained no usable rows.")
        return 1

    if "anchor" in rows[0]:
        target = ROOT / "data" / "probes" / "example_probes.jsonl"
        known = {c["id"] for c in build_concept_lookup()}
        good, problems = validate_triplets(rows, known)
        kind = "triplets"
    else:
        target = ROOT / "data" / "concepts" / "seed_concepts.jsonl"
        existing = {slugify(c["label"]) for c in read_jsonl(target)}
        good, problems = validate_concepts(rows, existing)
        kind = "concepts"

    for p in problems:
        print(f"  ! {p}")
    if not good:
        print("Nothing valid to merge.")
        return 1

    backup = target.with_suffix(target.suffix + ".bak")
    shutil.copy(target, backup)
    with open(target, "a", encoding="utf-8") as fh:
        for r in good:
            r.pop("reviewed", None)
            r.pop("agreed_with_proposal", None)
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Merged {len(good)} {kind} into {target.relative_to(ROOT)} "
          f"(backup at {backup.name})")
    if kind == "concepts":
        print("Concepts changed — re-run `python -m collider all` before using the map,\n"
              "and add structural signatures for the new entries.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merge", metavar="FILE",
                    help="merge a reviewed export back into the data files")
    ap.add_argument("--keep-reverse", action="store_true",
                    help="also review triplets that restate an earlier pair in reverse")
    args = ap.parse_args()
    if args.merge:
        return merge(Path(args.merge).expanduser())
    return stage(keep_reverse=args.keep_reverse)


if __name__ == "__main__":
    sys.exit(main())
