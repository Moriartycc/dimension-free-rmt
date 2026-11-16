"""Rebuild a paper figure (or all figures) from the shipped numerical results."""

from pathlib import Path
import argparse, json, subprocess, sys

ROOT = Path(__file__).resolve().parent
if __name__ == "__main__":
    figures = json.loads((ROOT / "figures.json").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("figure", choices=["all", *figures])
    a = ap.parse_args()
    for number, entry in figures.items():
        if a.figure not in ("all", number):
            continue
        print("Figure", number, flush=True)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / entry["directory"] / "main.py"),
                *entry["arguments"],
            ],
            check=True,
        )
