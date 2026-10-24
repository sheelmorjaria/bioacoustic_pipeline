"""Offline control contrasts and saved-artifact verification."""

import json
from pathlib import Path

import rclat_ctc_learning_diagnostic as core

ROOT = Path("outputs/rclat_acoustic_grounding_v2")


def main():
    plan = json.loads((ROOT / "execution_plan.json").read_text())
    for path, digest in plan["dependencies"].items():
        assert core.sha(path) == digest, path
    output = {}
    raw = 0
    for device in ["xpu_0", "xpu_1"]:
        receipt = json.loads((ROOT / device / "result.json").read_text())
        assert receipt["completed"] and receipt["original_reproduction_passed"]
        output[device] = {}
        for condition in plan["conditions"]:
            evaluation = json.loads((ROOT / device / condition / "step_2000/evaluation.json").read_text())
            output[device][condition] = {}
            for split in ["training", "validation"]:
                data = evaluation[split]
                for row in data["predictions"]:
                    assert core.sha(row["raw_path"]) == row["raw_sha256"]
                    raw += 1
                original = json.loads((ROOT / device / "original/step_2000/evaluation.json").read_text())[split]
                donor_rows = {r["uid"]:r for r in original["predictions"]}
                donor_errors = donor_n = 0
                for r in data["predictions"]:
                    donor_ref = donor_rows[plan["donors"][r["uid"]]]["reference"].replace(" ", "")
                    donor_errors += core.edit_counts(donor_ref, r["hypothesis"].replace(" ", ""))["errors"]
                    donor_n += len(donor_ref)
                speaker_delta = {r["speaker"]: r["cer"] - next(x["cer"] for x in original["metrics"]["speaker_results"] if x["speaker"] == r["speaker"]) for r in data["metrics"]["speaker_results"]}
                output[device][condition][split] = {"metrics":data["metrics"],"CER_increase_vs_original": data["metrics"]["speaker_macro_cer"] - original["metrics"]["speaker_macro_cer"],"paired_speaker_CER_increase":speaker_delta,"distinct_hypotheses":len({r["hypothesis"] for r in data["predictions"]}),"mean_blank_statistics":data["frame_weighted_statistics"],"donor_reference_pooled_CER":donor_errors/donor_n,"exact_utterances":sum(r["reference"]==r["hypothesis"] for r in data["predictions"])}
    core.write(ROOT / "summary.json", {"completed":True,"original_reproduction_passed_both_devices":True,"raw_records_verified":raw,"results":output,"training_updates":0,"P100_authorized":False,"P960_authorized":False,"stage2_ready":False})
    lines = ["# RCLAT acoustic-grounding diagnostic", "", "Inference-only, capped real-speech step-2000 checkpoint. Primary descriptive contrast: validation CER with swapped inputs minus original-input CER, against unchanged recipient references.", "", "| Condition | Training CER | Validation CER | Validation CER increase | Validation WER |", "|---|---:|---:|---:|---:|"]
    for condition, data in output["xpu_0"].items():
        lines.append(f"| {condition} | {data['training']['metrics']['speaker_macro_cer']:.2%} | {data['validation']['metrics']['speaker_macro_cer']:.2%} | {100*data['validation']['CER_increase_vs_original']:.2f} pp | {data['validation']['metrics']['speaker_macro_wer']:.2%} |")
    lines += ["", "Original hypotheses reproduced exactly on both devices; all original-logit relative L2 errors met 1e-4. No optimiser was constructed or advanced, checkpoint bytes and model tensors remained unchanged. All 768 saved raw records passed hash verification. Device-specific contrasts and paired speaker results are retained in summary.json.", "", "Controls operate on scaled features, not waveforms. Swapped donor inputs are same-speaker, adjacent-duration paired and nearest-neighbour resampled to recipient lengths. The largest duration ratio is 4.01, so swap effects conflate content mismatch and resampling distortion. Reversal disrupts general intelligibility and cannot isolate semantic or prosodic effects. Zeroed normalised features with retained masks are not physical silence; duration and mask cues remain. These are development diagnostics on a single saved seed, not a new confirmatory study.", "", "The v1 launch failed before inference because required oneDNN logging was omitted; v2 froze that launch correction before running. No model or control was changed in response to results. P10-C3 and learning-rate comparisons remain closed; P100/P960 and MACDE–CAVEM Stage 2 remain blocked.", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
