"""Standalone plots of saved diagnostic measurements; no inference."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rclat_ctc_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = Path("outputs/rclat_ctc_learning_diagnostic_v1")
d = json.loads((p / "summary.json").read_text())
fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
for split, label, color in [
    ("training", "Training", "#2463a6"),
    ("validation", "Validation", "#c56524"),
]:
    rows = d["D1"]
    steps = [r["checkpoint"] for r in rows]
    axes[0, 0].plot(
        steps,
        [100 * r[split]["speaker_macro_cer"] for r in rows],
        marker="o",
        color=color,
        label=label,
    )
    axes[0, 1].plot(
        steps,
        [100 * r[split]["nonempty_fraction"] for r in rows],
        marker="o",
        color=color,
        label=label,
    )
    axes[1, 0].plot(
        steps,
        [100 * r[split]["mean_blank_probability"] for r in rows],
        color=color,
        label=label + " mean probability",
    )
    axes[1, 0].plot(
        steps,
        [100 * r[split]["blank_argmax_fraction"] for r in rows],
        color=color,
        linestyle="--",
        label=label + " argmax fraction",
    )
    axes[1, 1].plot(
        steps,
        [
            100 * r[split]["E_share_decoded_characters"]
            if r[split]["E_share_decoded_characters"] is not None
            else np.nan
            for r in rows
        ],
        marker="o",
        color=color,
        label=label,
    )
axes[0, 0].axhline(
    10, color="#777777", linestyle=":", label="Fixed-final-step training fit indicator"
)
axes[0, 0].set_title("Speaker-macro CER")
axes[0, 1].set_title("Nonempty decoded outputs")
axes[1, 0].set_title("Free-running blank predictions")
axes[1, 1].set_title("E share of decoded characters\nUndefined when no characters emitted")
for ax in axes.flat:
    ax.set_xlabel("Diagnostic update")
    ax.set_ylabel("Percent")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
fig.savefig(p / "D1_trajectories.svg")
fig.savefig(p / "D1_trajectories.png", dpi=160)
plt.close(fig)
