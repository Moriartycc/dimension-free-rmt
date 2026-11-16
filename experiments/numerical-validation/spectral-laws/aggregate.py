"""Join independent trial spectra; fail on incomplete runs."""

from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CASES = (("power", 0.2), ("critical", 0.2), ("power", 0.05), ("critical", 0.05))
if __name__ == "__main__":
    arrays = {f"{law}_{p:g}_n{n}": [] for law, p in CASES for n in (2, 8, 32, 128)}
    for group, ns, seed in [
        ("small", (2, 8), 202611160000),
        ("large", (32, 128), 202611160001000),
    ]:
        for rep in range(2000):
            with np.load(HERE / "samples" / group / f"r{rep:04d}.npz") as data:
                assert (
                    int(data["d"]) == 2**24 and int(data["seed"]) == seed + 104729 * rep
                )
                for law, p in CASES:
                    for n in ns:
                        key = f"{law}_{p:g}_n{n}"
                        v = data[key]
                        assert (
                            v.shape == (n,)
                            and np.isfinite(v).all()
                            and np.all(np.diff(v) >= 0)
                        )
                        arrays[key].append(v)
    np.savez_compressed(
        HERE / "data" / "spectra.npz", **{k: np.stack(v) for k, v in arrays.items()}
    )
