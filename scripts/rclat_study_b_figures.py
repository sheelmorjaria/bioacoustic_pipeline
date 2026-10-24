"""Export completed Study B results without changing the frozen analysis."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    path = Path("outputs/rclat_study_b_results.json")
    results = json.loads(path.read_text())
    output = Path("outputs/rclat_study_b_speaker_results.csv")
    figure = Path("outputs/rclat_study_b_results.png")
    provenance = Path("outputs/rclat_study_b_figure_provenance.json")
    if any(p.exists() for p in [output, figure, provenance]):
        raise RuntimeError("Refusing to overwrite result exports")
    rows = []
    for arm, result in results["arms"].items():
        for speaker, metrics in result["evaluations"]["evaluation"]["per_speaker"].items():
            rows.append({"arm": arm, "speaker": speaker, **metrics})
    with output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["arm", "speaker", "wer", "cer"])
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    speakers = sorted({r["speaker"] for r in rows})
    for arm, result in results["arms"].items():
        axes[0].plot(
            speakers,
            [result["evaluations"]["evaluation"]["per_speaker"][s]["wer"] * 100 for s in speakers],
            marker="o",
            label=arm,
        )
        losses = result["training_loss_trace"]
        axes[1].plot(range(1, len(losses) + 1), losses, label=arm)
    axes[0].set(
        xlabel="Held-out speaker", ylabel="Word error rate (%)", title="All eight speaker outcomes"
    )
    axes[1].set(
        xlabel="Fixed training epoch",
        ylabel="Mean CTC loss",
        title="Training trace (no epoch selection)",
    )
    axes[0].tick_params(axis="x", rotation=45)
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle("RCLAT Study B: bounded English transcription pilot")
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    provenance.write_text(
        json.dumps(
            {
                "source": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "new_inference_or_model_selection": False,
                "exports": [str(output), str(figure)],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
