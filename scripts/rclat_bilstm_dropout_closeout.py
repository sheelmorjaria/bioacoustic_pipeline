"""Observe terminal state, verify completed runs and preserve incomplete results."""

import json
import time

import rclat_bilstm_dropout_train as runner

ROOT = runner.ROOT
core = runner.core


def main():
    while True:
        terminal = True
        states = {}
        for device in ["xpu_0", "xpu_1"]:
            out = ROOT / device
            failures = sorted(out.glob("*FAILURE.json"))
            complete = (out / "result.json").exists()
            terminal = terminal and (complete or bool(failures))
            states[device] = {"complete": complete, "failure_records": [str(p) for p in failures]}
        if terminal:
            break
        time.sleep(10)
    time.sleep(2)
    both = all(x["complete"] for x in states.values())
    results = {}
    if both:
        for device in states:
            out = ROOT / device
            result = json.loads((out / "result.json").read_text())
            assert result["completed"] and result["updates"] == 21120
            history = []
            for epoch in range(1, 31):
                r = json.loads((out / f"epoch_{epoch:02d}.json").read_text())
                assert core.sha(r["checkpoint"]) == r["checkpoint_sha256"]
                assert len(r["predictions"]) == 2864
                assert r["step"] == epoch * 704
                calculated = core.predecessor.original.aggregate(r["predictions"])
                assert calculated["speaker_macro_cer"] == r["speaker_macro_cer"]
                history.append(r)
            assert runner.select_checkpoint(history)["epoch"] == result["selected"]["epoch"]
            assert runner.learning_gate(result["selected"]) == result["learning_gate_passed"]
            completed = []
            with (out / "steps.jsonl").open() as f:
                for line in f:
                    row = json.loads(line)
                    if row.get("event") == "step_complete":
                        completed.append(row["step"])
            assert completed == list(range(1, 21121))
            results[device] = result
    passed = both and all(r["learning_gate_passed"] for r in results.values())
    comparison = None
    if both:
        baseline = results["xpu_0"]["selected"]
        regularised = results["xpu_1"]["selected"]
        b = {r["speaker"]: r for r in baseline["speaker_results"]}
        d = {r["speaker"]: r for r in regularised["speaker_results"]}
        assert set(b) == set(d) and len(b) == 33
        differences = [b[s]["cer"] - d[s]["cer"] for s in sorted(b)]
        comparison = runner.paired_interval(differences, 20261023, 10000)
        comparison["positive_speakers"] = sum(v > 0 for v in differences)
        comparison["speaker_effects"] = {s: b[s]["cer"] - d[s]["cer"] for s in sorted(b)}
        comparison["baseline_minus_dropout_WER"] = (
            baseline["speaker_macro_wer"] - regularised["speaker_macro_wer"]
        )
        for device in results:
            measured = json.loads((ROOT / device / "training_metrics.json").read_text())
            results[device]["training_metrics"] = {k: v["metrics"] for k, v in measured.items()}
    close = {
        "terminal": True,
        "primary_dropout_comparison": comparison,
        "both_completed": both,
        "paired_learning_gate_passed": passed,
        "gate_evaluated": both,
        "states": states,
        "results": results,
        "P100_authorized": False,
        "P960_authorized": False,
        "stage2_ready": False,
    }
    core.write(ROOT / "scientific_closeout.json", close)
    lines = [
        "# RCLAT BiLSTM dropout-only comparison closeout",
        "",
        f"Both executions completed: {both}. Joint learning gate passed: {passed}.",
        "",
    ]
    if both:
        lines += [
            "| Device | Selected epoch | Validation CER | Nonempty | Gate |",
            "|---|---:|---:|---:|---|",
        ]
        for d, r in results.items():
            lines.append(
                f"| {d} | {r['selected']['epoch']} | {r['selected']['speaker_macro_cer']:.2%} | {r['selected']['nonempty_fraction']:.2%} | {'PASS' if r['learning_gate_passed'] else 'FAIL'} |"
            )
        lines += [
            "",
            "All60checkpoint hashes, all30epochs of2864predictions perdevice and all21120sequential updates perdevice verified. Selected and final metrics and the secondary grounding analysis are retained in device result files.",
        ]
    else:
        lines += [
            "At least one execution is incomplete. The paired learning gate is not evaluated as a completed learning result. Failure artifacts are preserved; no automatic repair, retry or resumption occurred.",
            json.dumps(states, indent=2),
        ]
    if comparison is not None:
        lines += [
            "",
            f"Primary descriptive baseline-minus-dropout selected validation CER: {comparison['point_estimate'] * 100:.2f} percentage points; paired speaker-cluster percentile95 interval [{comparison['percentile95'][0] * 100:.2f}, {comparison['percentile95'][1] * 100:.2f}]. Positive speaker contrasts: {comparison['positive_speakers']}/33.",
        ]
    lines += [
        "",
        "The arms are different dropout policies, not same-policy hardware replicates. One seed and fixed device assignment leave device/model-seed uncertainty unresolved; fresh numerical checks and prior device reproduction mitigate but do not remove that limitation. Same-seed hardware reproduction does not provide independent statistical replication. Development-exposed validation and selected checkpoint results are not confirmatory final-test accuracy. This is a log-mel baseline, not a RCLAT representation-superiority comparison. Prior results remain closed; P100/P960 and MACDE–CAVEM Stage2 remain blocked.",
        "",
    ]
    with (ROOT / "RESULTS.md").open("x") as f:
        f.write("\n".join(lines))
    print("Scientific closeout written.", flush=True)


if __name__ == "__main__":
    main()
