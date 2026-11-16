from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "shared").is_dir())
sys.path.insert(0, str(ROOT / "shared"))
from runner import main

if __name__ == "__main__":
    main("bulk", Path(__file__).resolve().parent)
