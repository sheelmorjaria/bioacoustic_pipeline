"""Read completed/checkpoint diagnostic artifacts; never runs a model."""

import json
from pathlib import Path
import numpy as np

OUT = Path("outputs/rclat_ctc_learning_diagnostic_v1")


def main():
    result = dict(D0=[], D1=[], new_inference=False)
    for stage in ["D0", "D1"]:
        for path in sorted((OUT / stage).glob("step_*/evaluation.json")):
            data = json.loads(path.read_text())
            step = int(path.parent.name.split("_")[-1])
            row = dict(checkpoint=step)
            for split in ["training", "validation"]:
                r = data[split]
                character_count = sum(v for k, v in r["character_distribution"])
                frequency = dict(r["character_distribution"])
                preds = r["predictions"]
                row[split] = dict(
                    **{
                        k: r["metrics"][k]
                        for k in [
                            "speaker_macro_cer",
                            "speaker_macro_wer",
                            "pooled_cer",
                            "nonempty_fraction",
                        ]
                    },
                    **r["frame_weighted_statistics"],
                    E_share_decoded_characters=frequency.get("E", 0) / character_count
                    if character_count
                    else None,
                    decoded_character_count=character_count,
                    median_character_length_ratio=r[
                        "hypothesis_reference_character_length_ratio_quantiles"
                    ][2],
                    distinct_decoded_characters=len(frequency),
                    mean_utterance_normalized_CTC_loss=float(
                        np.mean(
                            [
                                -p["conditioned_statistics"]["ctc_log_probability"]
                                / p["target_length"]
                                for p in preds
                            ]
                        )
                    ),
                    single_character_word_fraction=r["single_character_word_fraction"],
                )
            result[stage].append(row)
    journal = OUT / "D1_journal.jsonl"
    steps = []
    if journal.exists():
        for line in journal.open():
            r = json.loads(line)
            if r.get("event") == "step_complete":
                steps.append(r)
    result["D1_completed_update_count"] = len(steps)
    result["D1_loss_first100_mean"] = (
        float(np.mean([r["loss"] for r in steps[:100]])) if steps else None
    )
    result["D1_loss_last100_mean"] = (
        float(np.mean([r["loss"] for r in steps[-100:]])) if steps else None
    )
    result["D1_completed"] = (OUT / "D1_result.json").exists()
    result["failures"] = [f.name for f in OUT.glob("*FAILURE.json")]
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for stage in ["D0", "D1"]:
        for r in result[stage]:
            print(
                stage,
                r["checkpoint"],
                "CER train/val",
                round(r["training"]["speaker_macro_cer"], 4),
                round(r["validation"]["speaker_macro_cer"], 4),
                "blank argmax train/val",
                round(r["training"]["blank_argmax_fraction"], 4),
                round(r["validation"]["blank_argmax_fraction"], 4),
            )
    print("D1 completed updates", len(steps), "failures", result["failures"])


if __name__ == "__main__":
    main()
