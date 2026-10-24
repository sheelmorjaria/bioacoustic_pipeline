"""Validate all fixed-endpoint artifacts and report the frozen LR comparison."""

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path("outputs/rclat_ctc_lr_diagnostic_v1")
STEPS = [0, 100, 250, 500, 1000, 1500, 2000]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    plan = json.loads((ROOT / "execution_plan.json").read_text())
    for path, digest in plan["dependencies"].items():
        assert sha(path) == digest, path
    cases = {}
    raw_count = 0
    for policy in plan["policies"]:
        for device in plan["execution_order"]:
            out = ROOT / policy / device.replace(":", "_")
            result = json.loads((out / "result.json").read_text())
            assert result["completed"] and result["updates"] == 2000
            assert json.loads((out / "preflight_result.json").read_text())["passed"]
            journal = [json.loads(x) for x in (out / "SYNTHETIC_journal.jsonl").read_text().splitlines()]
            completed = [x for x in journal if x.get("event") == "step_complete"]
            assert [x["step"] for x in completed] == list(range(1, 2001))
            trajectory = []
            for step in STEPS:
                receipt = json.loads((out / f"step_{step:04}.json").read_text())
                assert sha(receipt["checkpoint"]) == receipt["checkpoint_sha256"]
                evaluation = json.loads((out / "SYNTHETIC" / f"step_{step:04}" / "evaluation.json").read_text())
                point = {"step": step}
                for split, count in [("training", 64), ("validation", 32)]:
                    data = evaluation[split]
                    assert len(data["predictions"]) == count
                    for row in data["predictions"]:
                        assert sha(row["raw_path"]) == row["raw_sha256"]
                        raw_count += 1
                    point[split] = {
                        "metrics": data["metrics"],
                        "frame_statistics": data["frame_weighted_statistics"],
                        "character_distribution": data["character_distribution"],
                        "length_ratio_quantiles": data["hypothesis_reference_character_length_ratio_quantiles"],
                        "distinct_hypotheses": len({x["hypothesis"] for x in data["predictions"]}),
                        "exact_fraction": sum(x["hypothesis"] == x["reference"] for x in data["predictions"]) / count,
                    }
                trajectory.append(point)
            cases[f"{policy}/{device}"] = {"result": result, "trajectory": trajectory, "first100_loss": sum(x["loss"] for x in completed[:100]) / 100, "last100_loss": sum(x["loss"] for x in completed[-100:]) / 100}
    prefix = {}
    for device in plan["execution_order"]:
        prefix[device] = {}
        for step in [0, 100, 250]:
            models = []
            for policy in plan["policies"]:
                receipt = json.loads((ROOT / policy / device.replace(":", "_") / f"step_{step:04}.json").read_text())
                models.append(torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)["model"])
            prefix[device][str(step)] = all(torch.equal(models[0][k], models[1][k]) for k in models[0])
            del models
    contrasts = {}
    for device in plan["execution_order"]:
        original = cases[f"original/{device}"]["result"]
        capped = cases[f"capped250/{device}"]["result"]
        contrasts[device] = {
            "training_CER_original_minus_capped": original["final_training_metrics"]["pooled_cer"] - capped["final_training_metrics"]["pooled_cer"],
            "validation_CER_original_minus_capped": original["final_validation_metrics"]["pooled_cer"] - capped["final_validation_metrics"]["pooled_cer"],
        }
    summary = {"completed": True, "plan_sha256": sha(ROOT / "execution_plan.json"), "raw_records_verified": raw_count, "common_prefix_exact_model_equality": prefix, "paired_interpretation_prerequisite_passed": all(all(x.values()) for x in prefix.values()), "cases": cases, "contrasts": contrasts, "single_seed_not_independent_statistical_replication": True, "real_speech_learning_established": False, "P100_authorized": False, "P960_authorized": False, "stage2_ready": False}
    with (ROOT / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
    lines = ["# RCLAT synthetic learning-rate comparison", "", "Fixed endpoint: 2,000 updates. Only the learning-rate function changed; fresh seed-0 initialisation and all synthetic fixtures were preserved.", "", "| Policy | Device | Training CER | Validation CER | Training nonempty | Fit indicator |", "|---|---|---:|---:|---:|---|"]
    for key, case in cases.items():
        r = case["result"]
        policy, device = key.split("/")
        lines.append(f"| {policy} | {device} | {r['final_training_metrics']['pooled_cer']:.2%} | {r['final_validation_metrics']['pooled_cer']:.2%} | {r['final_training_metrics']['nonempty_fraction']:.2%} | {'PASS' if r['training_fit_indicator_passed'] else 'FAIL'} |")
    lines.extend(["", f"Common-prefix exact model equality at steps 0, 100 and 250: **{summary['paired_interpretation_prerequisite_passed']}**. All 2,688 raw posterior records and 28 checkpoint hashes were verified; all four runs completed exactly 2,000 updates. All frozen dependency hashes remain unchanged.", "", "Intermediate measurements are explanatory only and do not replace the fixed-final-step indicator. Two devices with one seed provide hardware reproducibility, not independent statistical replication. A synthetic result does not establish real-speech learning or authorise P100/P960. Prior scientific studies remain closed and MACDE–CAVEM Stage 2 remains blocked.", ""])
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
