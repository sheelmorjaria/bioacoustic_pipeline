"""Plot saved synthetic measurements; no model execution."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rclat_synthetic_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path("scripts").resolve()))
from rclat_p10_train import learning_rate

p = Path("outputs/rclat_ctc_synthetic_diagnostic_v1")
d = json.loads((p / "summary.json").read_text())
rows = d["checkpoints"]
x = [r["step"] for r in rows]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout="constrained")
for split, label, color in [
    ("training", "Training", "#2463a6"),
    ("validation", "Validation", "#c56524"),
]:
    axes[0].plot(x, [100 * r[split]["CER"] for r in rows], marker="o", color=color, label=label)
    axes[1].plot(
        x,
        [100 * r[split]["exact_message_recovery"] for r in rows],
        marker="o",
        color=color,
        label=label,
    )
axes[0].axhline(10, color="#777777", linestyle=":", label="Fixed-final-step training fit indicator")
axes[0].set_title("Character error rate")
axes[0].set_ylabel("Percent")
axes[1].set_title("Exact message recovery")
axes[1].set_ylabel("Percent")
axes[2].plot(range(1, 2001), [learning_rate(i, 21120) for i in range(1, 2001)], color="#555555")
axes[2].set_title("Unchanged optimisation prefix")
axes[2].set_ylabel("Learning rate")
for ax in axes:
    ax.set_xlabel("Update")
    ax.grid(alpha=0.2)
axes[0].legend(fontsize=7)
axes[1].legend(fontsize=8)
fig.savefig(p / "trajectories.svg")
fig.savefig(p / "trajectories.png", dpi=160)
plt.close(fig)
