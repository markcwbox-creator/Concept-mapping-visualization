"""Command-line entry point.

    python -m collider sweep    --config configs/qwen3-1.7b.yaml --limit 300
    python -m collider extract  --config configs/qwen3-1.7b.yaml
    python -m collider build    --config configs/qwen3-1.7b.yaml
    python -m collider all      --config configs/qwen3-1.7b.yaml
    python -m collider collide  --a photosynthesis --b monetary-policy
    python -m collider serve

`extract` is the only stage that needs a GPU. Run it on Colab, copy
`data/build/vectors.npy` + `vectors.meta.json` back, and run everything else
locally on CPU in seconds.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from .config import Config
from .neighbors import KNN, build_knn
from .pipeline import build_encoder, stage_all, stage_build, stage_extract
from .schema import ConceptSet, VectorSet
from .space import prepare
from .sweep import format_sweep_table, load_probes, sweep_layers


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="YAML config file (see configs/)")
    p.add_argument("--concepts", help="override concept jsonl path")
    p.add_argument("--model", help="override model name")
    p.add_argument("--layer", type=int, help="override residual layer")
    p.add_argument("--pooling", choices=["mean", "last", "max", "label_last"])
    p.add_argument("--batch-size", type=int, dest="batch_size")
    p.add_argument("--device", help="cuda | cpu")
    p.add_argument("--load-in-4bit", action="store_true", default=None, dest="load_in_4bit")
    p.add_argument("--build-dir", dest="build_dir")


def _cfg(args: argparse.Namespace) -> Config:
    cfg = Config.load(args.config)
    return cfg.merge_cli(vars(args))


def cmd_sweep(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    concepts = ConceptSet.from_jsonl(cfg.concepts)
    if args.limit and args.limit < len(concepts):
        # Domain-balanced subsample: sweeping on an accidentally lopsided
        # subset makes domain_purity meaningless.
        rng = np.random.default_rng(cfg.seed)
        by_domain: dict[str, list] = {}
        for c in concepts:
            by_domain.setdefault(c.domain, []).append(c)
        per = max(2, args.limit // max(1, len(by_domain)))
        picked = []
        for items in by_domain.values():
            idx = rng.permutation(len(items))[:per]
            picked.extend(items[i] for i in idx)
        concepts = ConceptSet(picked)
        logging.info("swept on a %d-concept balanced subsample", len(concepts))

    encoder = build_encoder(cfg)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    results = sweep_layers(
        encoder, concepts, layers,
        probes=load_probes(cfg.probes), remove_top_k=cfg.remove_top_k,
    )
    print(format_sweep_table(results))
    out = Path(cfg.build_dir) / "layer_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    print("Pick a layer from the curve, then set `layer:` in your config.")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    stage_extract(_cfg(args))
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    stage_build(_cfg(args))
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    stage_all(_cfg(args))
    return 0


def cmd_collide(args: argparse.Namespace) -> int:
    """Collide two concepts from the terminal — the same computation the web UI runs."""
    from .bridges import collide

    cfg = _cfg(args)
    concepts = ConceptSet.from_jsonl(cfg.concepts)
    vs = VectorSet.load(cfg.build_dir)
    prepared, _ = prepare(vs, center=cfg.center, remove_top_k=cfg.remove_top_k,
                          whiten=cfg.whiten)
    knn_path = Path(cfg.build_dir) / "knn.npz"
    if knn_path.exists():
        blob = np.load(knn_path)
        knn = KNN(blob["idx"], blob["sim"])
    else:
        knn = build_knn(prepared.vectors, k=cfg.knn_k)

    report = collide(
        prepared.vectors, concepts, knn,
        concepts.index_of(args.a), concepts.index_of(args.b), top_n=args.top,
    )
    print(f"\n{args.a}  <->  {args.b}")
    print(f"  cosine distance {report.distance:.3f}   "
          f"blend vacancy {report.midpoint_vacancy:.2f}x typical NN distance")
    print("\n  stepping stones:")
    print("    " + " -> ".join(report.path) if report.path else "    (no path)")
    for title, group in (("balanced bridges", report.balanced),
                         ("nearest the blend", report.midpoint),
                         ("orthogonal to the tension", report.orthogonal)):
        print(f"\n  {title}:")
        for b in group[: args.top]:
            print(f"    {b.id:<34} dA={b.d_a:.3f} dB={b.d_b:.3f} bal={b.balance:.2f}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import http.server
    import socketserver

    root = Path(args.root).resolve()
    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(  # noqa: E731
        *a, directory=str(root), **kw
    )
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"serving {root} at http://127.0.0.1:{args.port}/  (ctrl-c to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser("collider", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sweep", help="measure which layer to use")
    _common(p)
    p.add_argument("--layers", help="comma-separated layer indices")
    p.add_argument("--limit", type=int, default=300, help="concepts to sweep on")
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("extract", help="run the model, write vectors.npy")
    _common(p)
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("build", help="space, kNN, pairs, layout, export (no GPU)")
    _common(p)
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("all", help="extract + build")
    _common(p)
    p.set_defaults(fn=cmd_all)

    p = sub.add_parser("collide", help="collide two concepts in the terminal")
    _common(p)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--top", type=int, default=8)
    p.set_defaults(fn=cmd_collide)

    p = sub.add_parser("serve", help="static server for the web front end")
    p.add_argument("--root", default="web")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
