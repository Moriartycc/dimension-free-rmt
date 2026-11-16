"""Regenerate Figure 1 from the archived independent-trial spectra."""

from simulation import HERE, SIZES, np, render

if __name__ == "__main__":
    with np.load(HERE / "figure_data.npz") as data:
        render(
            {(row, n): data[f"row{row + 1}_n{n}"] for row in range(3) for n in SIZES},
            data["reference_spectrum"],
            data["half_reference_spectra"],
        )
