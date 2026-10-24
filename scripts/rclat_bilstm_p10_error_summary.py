"""Saved validation analysis plus fixed-checkpoint full-training diagnostics."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
import rclat_bilstm_p10_error_probe as probe

ROOT = probe.ROOT
core = probe.core
STUDY = Path("outputs/rclat_bilstm_p10_v1")


def describe_predictions(rows):
    metrics = core.predecessor.original.aggregate(rows)
    speakers = sorted({r["speaker"] for r in rows})
    decomposition = {}
    for unit, den in [("character", "reference_characters"), ("word", "reference_words")]:
        decomposition[unit] = {}
        for component in ["substitutions", "deletions", "insertions"]:
            decomposition[unit][component] = float(
                np.mean(
                    [
                        sum(
                            r[unit + "_decomposition"][component] for r in rows if r["speaker"] == s
                        )
                        / sum(r[den] for r in rows if r["speaker"] == s)
                        for s in speakers
                    ]
                )
            )
    hypotheses = Counter(r["hypothesis"] for r in rows)
    return {
        "metrics": metrics,
        "speaker_macro_decomposition": decomposition,
        "exact_utterances": sum(r["reference"] == r["hypothesis"] for r in rows),
        "distinct_hypotheses": len(hypotheses),
        "most_common_hypothesis_count": hypotheses.most_common(1)[0][1],
        "median_output_reference_character_length": float(
            np.median(
                [len(r["hypothesis"].replace(" ", "")) / r["reference_characters"] for r in rows]
            )
        ),
    }


def main():
    plan = json.loads((ROOT / "protocol.json").read_text())
    for p, d in plan["dependencies"].items():
        assert core.sha(p) == d
    analyses = {}
    raw = 0
    for device, entry in plan["cases"].items():
        out = ROOT / device.replace(":", "_")
        result = json.loads((out / "result.json").read_text())
        assert result["completed"] and result["validation_anchor_hypotheses_reproduced"]
        epoch = entry["epoch"]
        evaluation = json.loads((out / f"epoch_probe/step_{epoch:04}/evaluation.json").read_text())
        for data in evaluation.values():
            for r in data["predictions"]:
                assert core.sha(r["raw_path"]) == r["raw_sha256"]
                raw += 1
        saved = json.loads(Path(entry["saved_validation"]).read_text())
        train = evaluation["training"]
        validation = describe_predictions(saved["predictions"])
        analyses[str(epoch)] = {
            "training": {
                **describe_predictions(train["predictions"]),
                "frame_statistics": train["frame_weighted_statistics"],
            },
            "validation": validation,
            "validation_minus_training_CER": validation["metrics"]["speaker_macro_cer"]
            - train["metrics"]["speaker_macro_cer"],
        }
    trajectories = []
    for e in range(1, 31):
        x = json.loads((STUDY / "xpu_0" / f"epoch_{e:02d}.json").read_text())
        trajectories.append(
            {
                "epoch": e,
                "training_loss": x["mean_batch_training_loss"],
                "validation_CER": x["speaker_macro_cer"],
                "validation_WER": x["speaker_macro_wer"],
                "nonempty": x["nonempty_fraction"],
            }
        )
    grounding = json.loads((STUDY / "xpu_0/secondary_grounding.json").read_text())
    positive = sum(
        b["cer"] > a["cer"]
        for a, b in zip(
            grounding["matched"]["speaker_results"], grounding["native_swapped"]["speaker_results"]
        )
    )
    summary = {
        "completed": True,
        "analyses": analyses,
        "trajectory": trajectories,
        "secondary_grounding": {k: grounding[k] for k in ["matched", "native_swapped", "interval"]},
        "grounding_positive_speakers": positive,
        "raw_records_verified": raw,
        "optimizer_updates": 0,
        "original_study_gate_remains_failed": True,
        "P100_authorized": False,
        "P960_authorized": False,
        "stage2_ready": False,
    }
    core.write(ROOT / "summary.json", summary)
    lines = [
        "# BiLSTM-P10-v1 closed-study error diagnostic",
        "",
        "Inference-only evaluation of the original full2814-utterance training population at fixed epochs18(selected) and30(final). Full2864-utterance validation predictions are the original saved outputs, not a new evaluation or selection opportunity.",
        "",
        "| Epoch | Training CER | Validation CER | Gap | Training WER | Validation WER |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for epoch, r in analyses.items():
        a = r["training"]["metrics"]
        b = r["validation"]["metrics"]
        lines.append(
            f"| {epoch} | {a['speaker_macro_cer']:.2%} | {b['speaker_macro_cer']:.2%} | {r['validation_minus_training_CER'] * 100:.2f} pp | {a['speaker_macro_wer']:.2%} | {b['speaker_macro_wer']:.2%} |"
        )
    lines += [
        "",
        "## Remaining validation errors",
        "",
        "| Epoch | Deletions | Substitutions | Insertions | Exact utterances |",
        "|---|---:|---:|---:|---:|",
    ]
    for epoch, r in analyses.items():
        x = r["validation"]
        d = x["speaker_macro_decomposition"]["character"]
        lines.append(
            f"| {epoch} | {d['deletions']:.2%} | {d['substitutions']:.2%} | {d['insertions']:.2%} | {x['exact_utterances']}/2864 |"
        )
    ci = grounding["interval"]
    lines += [
        "",
        "## Prespecified secondary grounding",
        "",
        f"Selected epoch18, fixed384-utterance matched inventory: matched CER {grounding['matched']['speaker_macro_cer']:.2%}; mismatched CER {grounding['native_swapped']['speaker_macro_cer']:.2%}. Difference {100 * ci['point_estimate']:.2f} percentage points, descriptive speaker-cluster percentile95 interval [{100 * ci['percentile95'][0]:.2f},{100 * ci['percentile95'][1]:.2f}] points. Positive speaker contrasts: {positive}/24.",
        "",
        "This secondary subset overlaps checkpoint-selection validation and remains exposed development evidence. It cannot rescue the failed primary gate. The inference probes do not alter weights or scores of the completed study. All source checkpoint hashes, saved prediction files and new raw outputs are preserved and verified. No optimiser updates occurred. P10-C3 and BiLSTM-P10-v1 remain closed; P100/P960 and MACDE–CAVEM Stage2 remain blocked.",
        "",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    core.write(
        ROOT / "closeout.json",
        {
            "completed": True,
            "raw_records_verified": raw,
            "training_utterances_per_checkpoint": 2814,
            "validation_anchors_per_checkpoint": 4,
            "optimizer_updates": 0,
            "original_learning_gate_unchanged": True,
            "P100_authorized": False,
            "P960_authorized": False,
            "stage2_ready": False,
        },
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
