from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "shared").is_dir())
sys.path.insert(0, str(ROOT / "shared"))
from common import np, json, CASES

HERE = Path(__file__).resolve().parent


def summarize(q=16):
    out = HERE / "data"
    raw = np.load(HERE / "ridge_aggregated.npz")
    theory = np.load(HERE / "data" / "theory.npz")
    qi = int(np.flatnonzero(raw["q"] == q)[0])
    tqi = int(np.flatnonzero(theory["q"] == q)[0])
    saved = {key: raw[key] for key in ("n", "c", "k", "q", "d", "noise_sd")}
    saved["theory_n"] = theory["n"]
    checks = {}
    for law, p in CASES:
        name = f"{law}_{p:g}"
        values = raw[name][:, :, :, :, :, qi]
        mean = values.mean(0)
        se = values.std(axis=0, ddof=1) / np.sqrt(400)
        baseline = values[:, :, 1:2, :, :]
        base_mean = baseline.mean(0)
        assert (base_mean > 0).all()
        ratio = mean / base_mean
        influence = (values - ratio[None] * baseline) / base_mean[None]
        ratio_se = influence.std(axis=0, ddof=1) / np.sqrt(400)
        v = theory[name][:, :, :, :, tqi]
        pred = np.stack(
            [1e-4 * v[:, :, :, 0], v[:, :, :, 1], 1e-4 * v[:, :, :, 0] + v[:, :, :, 1]],
            axis=-1,
        )
        pred_ratio = pred / pred[:, 1:2, :, :]
        assert np.allclose(ratio[:, 1], 1) and np.max(ratio_se[:, 1]) < 1e-14
        assert np.allclose(
            mean[:, :, :, 0] + mean[:, :, :, 1], mean[:, :, :, 2], rtol=1e-12, atol=0
        )
        saved.update(
            {
                name + "_mean": mean,
                name + "_se": se,
                name + "_ratio": ratio,
                name + "_ratio_se": ratio_se,
                name + "_prediction": pred,
                name + "_prediction_ratio": pred_ratio,
            }
        )
        checks[name] = dict(
            minimum_component_baseline=float(base_mean.min()),
            maximum_ratio_95_percent_half_width=float(1.96 * ratio_se.max()),
            component_ridgeless_ratios_identically_one=True,
            total_risk_ratios_match_previous_figures=True,
        )
    saved["selected_q"] = q
    np.savez_compressed(out / f"ridge_q{q}_combined_summary.npz", **saved)
    (out / f"ridge_q{q}_combined_checks.json").write_text(json.dumps(checks, indent=2))


if __name__ == "__main__":
    summarize(16)
    summarize(24)
