"""Descriptive P10 learning curves; no superiority test or model selection."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path("outputs/rclat_p10")
    source = root / "execution_closeout.json"
    data = json.loads(source.read_text())
    colors = {"mel": "#2466ad", "combined": "#cb5926"}
    labels = {"mel": "Log-mel", "combined": "RCLAT combined"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for arm, summary in data["arms"].items():
        history = summary["epoch_history"]
        epochs = [r["epoch"] for r in history]
        color = colors[arm]
        axes[0, 0].plot(
            epochs, [r["mean_batch_training_loss"] for r in history], label=labels[arm], color=color
        )
        axes[0, 1].plot(
            epochs, [100 * r["speaker_macro_cer"] for r in history], label=labels[arm], color=color
        )
        axes[0, 1].scatter(
            [summary["selected_epoch"]],
            [100 * summary["speaker_macro_cer"]],
            color=color,
            s=45,
            zorder=5,
        )
        axes[1, 0].plot(
            epochs, [100 * r["nonempty_fraction"] for r in history], label=labels[arm], color=color
        )
    axes[0, 0].set(title="Training loss", xlabel="Epoch", ylabel="Mean batch CTC loss")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].axhline(
        50, color="0.4", linestyle="--", linewidth=1, label="Learning gate: CER <50%"
    )
    axes[0, 1].set(title="Validation transcription", xlabel="Epoch", ylabel="Speaker-macro CER (%)")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 0].axhline(95, color="0.4", linestyle="--", linewidth=1)
    axes[1, 0].set(
        title="Validation output support",
        xlabel="Epoch",
        ylabel="Nonempty utterances (%)",
        ylim=(-2, 102),
    )
    with (root / "speaker_results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    speakers = {
        arm: {r["speaker"]: float(r["cer"]) * 100 for r in rows if r["arm"] == arm}
        for arm in ["mel", "combined"]
    }
    ids = sorted(speakers["mel"])
    x = np.array([speakers["mel"][i] for i in ids])
    y = np.array([speakers["combined"][i] for i in ids])
    limit = max(100, np.max(x), np.max(y)) * 1.03
    axes[1, 1].plot([0, limit], [0, limit], color="0.6", linestyle="--", linewidth=1)
    axes[1, 1].scatter(x, y, color="#7552a0", alpha=0.8, s=28)
    axes[1, 1].set(
        title="33 validation speakers, selected checkpoints",
        xlabel="Log-mel speaker CER (%)",
        ylabel="Combined speaker CER (%)",
        xlim=(0, limit),
        ylim=(0, limit),
    )
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    fig.suptitle("RCLAT P10 — development validation only", fontsize=14)
    for suffix in ["png", "svg"]:
        target = root / f"learning_results.{suffix}"
        if target.exists():
            raise RuntimeError("Refusing to overwrite completed figure")
        fig.savefig(target, dpi=180)
    plt.close(fig)
    print("Saved descriptive learning-results PNG and SVG; no inferential comparison added.")


if __name__ == "__main__":
    main()
