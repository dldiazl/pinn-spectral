"""Plot known-data MSE versus epoch for one manually trained NAS architecture.

This is a standalone diagnostic script outside the numbered pipeline. It reads
the training history saved by a manual run of ``tanh_8x90`` and is used to
inspect convergence without rerunning the full NAS study.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    """Plot the supervised known-data MSE versus training epoch."""
    input_path = Path(
        r"data\nas\histories\nas_t1_e1000_n100_seed0\manual\manual_tanh_8x90_e1000.parquet"
    )
    output_path = Path(
        r"results\figures\nas\manual_tanh_8x90_mse_vs_epoch.png"
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_parquet(input_path)

    required_columns = {"Epoch", "LossBC"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in history file: {sorted(missing)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    epoch = df["Epoch"].to_numpy()
    mse = df["LossBC"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.semilogy(epoch, mse, linewidth=1.4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Known-data MSE")
    ax.set_title("Manual NAS training: tanh_8x90")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.show()

    print(f"Final known-data MSE: {mse[-1]:.6e}")
    print(f"Minimum known-data MSE: {mse.min():.6e}")
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()