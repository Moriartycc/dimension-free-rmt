"""Portable experiment entry points; plot, smoke, or full computation."""

from pathlib import Path
import argparse, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]


def main(name, here):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", choices=("plot", "smoke", "full", "theory"), default="plot"
    )
    ap.add_argument("--q", type=int, default=16)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--side", choices=("absolute", "relative"), default="absolute")
    ap.add_argument("--appendix", action="store_true")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--features", type=Path)
    args = ap.parse_args()

    def call(file, *extra):
        subprocess.run(
            [sys.executable, str(here / file), *map(str, extra)], cwd=here, check=True
        )

    if args.mode == "smoke":
        subprocess.run(
            [sys.executable, str(ROOT / "tests" / "numerical_checks.py")], check=True
        )
        return
    if args.mode in ("theory", "full"):
        if name == "gram":
            if args.features is None:
                ap.error("--features is required for a fresh CIFAR experiment")
            call("extract_features.py", args.features.resolve())
            call("simulation.py", args.features.resolve())
        elif name == "directional":
            call("compute_directional.py")
        elif name in ("bulk", "spectral", "ridge"):
            call("compute.py")
        elif name == "gf":
            call("compute_supplement.py" if args.appendix else "compute.py")
        elif name == "paths":
            for law in ("power", "critical"):
                call("prepare.py", law)
                call("run.py", "preflight", law)
                for col in range(25):
                    call("run.py", "production", law, "--column", col)
            call("aggregate.py")
        if args.mode == "full":
            if name == "spectral":
                for group, sizes, seed in [
                    ("small", (2, 8), 202611160000),
                    ("large", (32, 128), 202611160001000),
                ]:
                    call(
                        "worker.py",
                        "--device",
                        args.device,
                        "--sizes",
                        *sizes,
                        "--seed",
                        seed,
                        "--output",
                        here / "samples" / group,
                    )
                call("aggregate.py")
            elif name == "ridge":
                call("worker.py", "--trials", 400)
                call("aggregate.py")
            elif name == "gf" and not args.appendix:
                call("postprocess.py")
    if name == "ridge":
        if args.mode in ("theory", "full"):
            call("summarize.py")
        call("plot.py", "--q", args.q, "--k", args.k, "--side", args.side)
    elif name == "gf":
        call("plot_supplement.py" if args.appendix else "plot.py")
    else:
        call("plot.py")
