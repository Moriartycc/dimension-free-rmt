from model import HERE
import numpy as np
from model import N
from model import D
from model import TRIALS
from model import FIT_COUNT


def collect(name):
    folder = HERE / name
    population = dict(np.load(folder / "population.npz"))
    residual = np.empty((3, 425, TRIALS), complex)
    fit_se = np.empty(425)
    crossing = np.zeros((3, 425))
    runtime = 0.0
    max_residual = 0.0
    for point in range(425):
        with np.load(folder / "results" / f"cell_{point:03d}.npz") as data:
            assert (
                int(data["D"]),
                int(data["M"]),
                int(data["trials"]),
                int(data["n"]),
            ) == (D, FIT_COUNT, TRIALS, N)
            assert np.allclose(data["zeta"], population["zeta"].flat[point])
            residual[:, point] = data["errors"]
            fit_se[point] = float(data["fit_se"])
            crossing[2, point] = np.mean(data["first_cross"] >= 0)
            runtime += float(data["seconds"])
            max_residual = max(max_residual, float(data["max_residual"]))
    assert np.isfinite(residual).all()
    mean = residual.mean(-1)
    drift = abs(mean)
    rms = np.sqrt(np.mean(abs(residual) ** 2, axis=-1))
    se = residual.std(-1, ddof=1) / np.sqrt(TRIALS)
    se[0] = np.hypot(se[0], fit_se)
    assert np.all(rms >= drift * (1 - 1e-12))
    result = dict(
        population,
        mean=mean.reshape(3, 17, 25),
        drift=drift.reshape(3, 17, 25),
        rms=rms.reshape(3, 17, 25),
        se=se.reshape(3, 17, 25),
        crossing=crossing.reshape(3, 17, 25),
        fit_se=fit_se.reshape(17, 25),
        residual=residual.reshape(3, 17, 25, TRIALS),
        M=FIT_COUNT,
        trials=TRIALS,
        total_cell_gpu_hours=runtime / 3600,
        max_solver_residual=max_residual,
    )
    np.savez_compressed(folder / "heatmap_data.npz", **result)
    return result


if __name__ == "__main__":
    for name in ("power", "critical"):
        data = collect(name)
        folder = HERE / "completed" / name
        folder.mkdir(parents=True, exist_ok=True)
        residual = np.abs(data["residual"])
        np.savez_compressed(folder / "heatmap_data.npz", **data)
        np.savez_compressed(
            folder / "endpoint_norms.npz",
            x=data["x"],
            y=data["y"],
            support=data["support"],
            l1=residual.mean(-1),
            l2=np.sqrt((residual**2).mean(-1)),
        )
    values = np.concatenate(
        [
            np.load(HERE / "completed" / law / "endpoint_norms.npz")[key].ravel()
            for law in ("power", "critical")
            for key in ("l1", "l2")
        ]
    )
    import json

    (HERE / "completed" / "l1_l2_diagnostics.json").write_text(
        json.dumps({"shared_color_limits": [float(values.min()), float(values.max())]})
    )
