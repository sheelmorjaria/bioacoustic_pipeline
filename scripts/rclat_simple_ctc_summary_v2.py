"""Verify and report the fixed small-encoder diagnostic."""

import json
from pathlib import Path

import rclat_ctc_learning_diagnostic as core

ROOT = Path("outputs/rclat_simple_ctc_diagnostic_v2")
STEPS = [0, 100, 250, 500, 1000, 1500, 2000]


def main():
    plan = json.loads((ROOT / "execution_plan.json").read_text())
    for path, digest in plan["dependencies"].items():
        assert core.sha(path) == digest, path
    cases = {}
    raw = 0
    for device in ["xpu_0", "xpu_1"]:
        out = ROOT / device
        result = json.loads((out / "D1_result.json").read_text())
        assert result["completed"] and result["updates"] == 2000
        preflight = json.loads((out / f"preflight_xpu{device[-1]}_result.json").read_text())
        assert preflight["passed"]
        journal = [json.loads(x) for x in (out / "D1_journal.jsonl").read_text().splitlines()]
        updates = [x for x in journal if x.get("event") == "step_complete"]
        assert [x["step"] for x in updates] == list(range(1, 2001))
        trajectory = []
        for step in STEPS:
            receipt = json.loads((out / f"D1_step_{step:04}.json").read_text())
            assert core.sha(receipt["checkpoint"]) == receipt["checkpoint_sha256"]
            evaluation = json.loads((out / "D1" / f"step_{step:04}/evaluation.json").read_text())
            point = {"step":step}
            for split, n in [("training",64),("validation",32)]:
                data = evaluation[split]
                assert len(data["predictions"]) == n
                for row in data["predictions"]:
                    assert core.sha(row["raw_path"]) == row["raw_sha256"]
                    raw += 1
                point[split] = {"metrics":data["metrics"],"frame_statistics":data["frame_weighted_statistics"],"distinct_hypotheses":len({r["hypothesis"] for r in data["predictions"]}),"exact_utterances":sum(r["reference"]==r["hypothesis"] for r in data["predictions"]),"length_quantiles":data["hypothesis_reference_character_length_ratio_quantiles"],"character_distribution":data["character_distribution"]}
            trajectory.append(point)
        cases[device] = {"result":result,"preflight":preflight,"trajectory":trajectory,"first100_loss":sum(x["loss"] for x in updates[:100])/100,"last100_loss":sum(x["loss"] for x in updates[-100:])/100}
    original = json.loads(Path("outputs/rclat_ctc_real_lr_diagnostic_v1/summary.json").read_text())
    core.write(ROOT / "summary.json", {"completed":True,"cases":cases,"raw_records_verified":raw,"historical_capped_Conformer":{d:original["cases"]["capped250/"+d.replace("_",":")]["result"] for d in cases},"single_seed_hardware_reproducibility_only":True,"P100_authorized":False,"P960_authorized":False,"stage2_ready":False})
    lines = ["# RCLAT small-encoder trainability diagnostic", "", "Fixed final endpoint: 2,000 updates. Same small real-speech cohort, capped learning rate, AdamW and CTC decoder as the completed Conformer diagnostic; encoder replaced by a 781,725-parameter convolutional–BiLSTM.", "", "| Encoder / device | Training CER | Validation CER | Training nonempty | Fit gate |", "|---|---:|---:|---:|---|"]
    for device, case in cases.items():
        r=case["result"]
        lines.append(f"| Conv–BiLSTM / {device} | {r['final_training_metrics']['speaker_macro_cer']:.2%} | {r['final_validation_metrics']['speaker_macro_cer']:.2%} | {r['final_training_metrics']['nonempty_fraction']:.2%} | {'PASS' if r['training_fit_indicator_passed'] else 'FAIL'} |")
    lines += ["| Prior capped Conformer / both XPUs | 74.42% | 77.12% | 100% | FAIL |", "", "Both fresh CPU–XPU preflights passed. All 1,344 raw posterior records and 14 checkpoint hashes were verified; all 2,000 updates per device were sequential and all frozen dependencies unchanged. No intermediate checkpoint replaced the registered endpoint.", "", "This is a single-seed diagnostic with two hardware executions, not independent statistical replication. The earlier Conformer runs are historical controls. Encoder capacity, recurrence, positional processing and initialisation differ together, so the contrast cannot identify one architectural mechanism. No final-test data were accessed. P10-C3 and previous diagnostics remain closed; P100/P960 and MACDE–CAVEM Stage 2 remain blocked.", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
