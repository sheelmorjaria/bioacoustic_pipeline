"""Read synthetic diagnostic results; no model execution."""

from collections import Counter
import json
from pathlib import Path
import numpy as np

OUT = Path("outputs/rclat_ctc_synthetic_diagnostic_v1")


def main():
    result = dict(checkpoints=[], real_speech_learning_established=False)
    for path in sorted((OUT / "SYNTHETIC").glob("step_*/evaluation.json")):
        checkpoint = int(path.parent.name.split("_")[-1])
        data = json.loads(path.read_text())
        row = dict(step=checkpoint)
        for split in ["training", "validation"]:
            r = data[split]
            predictions = r["predictions"]
            c = Counter(x["hypothesis"] for x in predictions)
            row[split] = dict(
                CER=r["metrics"]["pooled_cer"],
                nonempty_fraction=r["metrics"]["nonempty_fraction"],
                exact_message_recovery=sum(x["reference"] == x["hypothesis"] for x in predictions)
                / len(predictions),
                median_character_length_ratio=r[
                    "hypothesis_reference_character_length_ratio_quantiles"
                ][2],
                distinct_hypotheses=len(c),
                top_hypotheses=c.most_common(3),
                mean_utterance_normalized_CTC_loss=float(
                    np.mean(
                        [
                            -x["conditioned_statistics"]["ctc_log_probability"] / x["target_length"]
                            for x in predictions
                        ]
                    )
                ),
                **r["frame_weighted_statistics"],
            )
        result["checkpoints"].append(row)
    steps = []
    f = OUT / "SYNTHETIC_journal.jsonl"
    if f.exists():
        for line in f.open():
            r = json.loads(line)
            if r.get("event") == "step_complete":
                steps.append(r)
    result["completed_update_count"] = len(steps)
    result["loss_first100_mean"] = (
        float(np.mean([r["loss"] for r in steps[:100]])) if steps else None
    )
    result["loss_last100_mean"] = (
        float(np.mean([r["loss"] for r in steps[-100:]])) if steps else None
    )
    result["failures"] = [f.name for f in OUT.glob("*FAILURE.json")]
    result["completed"] = (OUT / "result.json").exists()
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for r in result["checkpoints"]:
        print(
            "step",
            r["step"],
            "CER train/val",
            round(r["training"]["CER"], 4),
            round(r["validation"]["CER"], 4),
            "exact train/val",
            r["training"]["exact_message_recovery"],
            r["validation"]["exact_message_recovery"],
        )
    print("completed updates", len(steps), "failures", result["failures"])


if __name__ == "__main__":
    main()
