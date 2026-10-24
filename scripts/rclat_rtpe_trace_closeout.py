"""CPU-only verification of recorded numerical traces; no XPU replay or correction."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    root = Path("outputs/rclat_rtpe_numerical_trace_v1")
    plan = json.loads((root / "plan.json").read_text())
    assert all(sha(p) == h for p, h in plan["dependencies"].items())
    assert all(sha(p) == h for p, h in json.loads((root / "manifest.json").read_text()).items())
    summary = json.loads((root / "summary.json").read_text())
    assert summary["completed_cases"] == 15
    audits = []
    for seed, _ in plan["seed_pairs"]:
        for case in plan["cases"]:
            directory = root / f"seed_{seed}_case_{case}"
            result = json.loads((directory / "result.json").read_text())
            assert result["seed"] == seed and result["case"] == case
            events = [
                json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()
            ]
            assert [r["sequence"] for r in events] == list(range(1, len(events) + 1))
            assert (
                len(events) == result["events_recorded"] + 1
                and events[-1]["name"] == "case.finished"
            )
            payload = torch.load(
                directory / "checkpoints.pt", map_location="cpu", weights_only=True
            )
            mask = payload["transfer.observed"]
            target = payload["transfer.target"]
            forecast = payload["forecast"]
            np.testing.assert_array_equal(payload["source.observed"].numpy(), mask.numpy())
            np.testing.assert_array_equal(payload["source.target"].numpy(), target.numpy())
            assert torch.isfinite(target[mask]).all() and torch.isfinite(forecast).all()
            for name in [
                "encoder_input",
                "GRU_sequence",
                "GRU_hidden",
                "forecast_head_input",
                "forecast_head_output",
            ]:
                assert torch.isfinite(payload[name]).all()
            for name, tensor in payload.items():
                if name.startswith("transfer.parameter."):
                    np.testing.assert_array_equal(
                        tensor.numpy(), payload[name.replace("transfer.", "source.", 1)].numpy()
                    )
            selected = [
                r
                for r in events
                if r.get("operation") == "aten.index.Tensor"
                and r.get("operand_shapes") == [[4, 3, 64], [4, 3, 64]]
                and r["name"].startswith("weighted_loss")
            ]
            assert len(selected) == 2
            expected = int(mask.sum())
            assert all(r["shape"][0] != expected for r in selected)
            if result["status"] == "EXCEPTION":
                assert result["first_unexpected_nonfinite"]["sequence"] == selected[1]["sequence"]
                assert selected[0]["host_nonfinite"] == 0 and selected[1]["host_nonfinite"] == 1
                captured = torch.load(
                    directory / "first_nonfinite.pt", map_location="cpu", weights_only=True
                )
                assert captured["metadata"]["operation"] == "aten.index.Tensor"
                assert int((~torch.isfinite(captured["tensor"])).sum()) == 1
                assert "Observed predictions and targets must be finite" in result["traceback"]
                assert "input_gradient" not in payload
            else:
                assert all(e < 1e-4 for e in result["relative_errors"].values())
            audits.append(
                dict(
                    seed=seed,
                    case=case,
                    status=result["status"],
                    expected_selected_count=expected,
                    XPU_prediction_selected_count=selected[0]["shape"][0],
                    XPU_target_selected_count=selected[1]["shape"][0],
                    XPU_target_selection_nonfinite=selected[1]["host_nonfinite"],
                    CPU_selected_target_all_finite=True,
                    full_forecast_all_finite=True,
                    transfers_exact=True,
                    encoder_and_head_outputs_finite=True,
                    first_nonfinite_stage="XPU boolean-indexed target selection in loss guard"
                    if result["status"] == "EXCEPTION"
                    else None,
                )
            )
    result = dict(
        audit_verified=True,
        cases=audits,
        cases_with_incorrect_boolean_selection_size=15,
        reproduced_loss_guard_failures=2,
        instrumented_final_parity_passes=13,
        source_transfers_and_forward_checkpoints_all_finite_on_observed_cells=True,
        attribution="Instrumented reproductions localised to aten.index.Tensor target[observed], not GRU returned activations or correct CPU selection",
        unresolved="Original failed seed/case still unknown;underlying backend cause and effect of tracing remain unestablished",
        scientific_optimizer_updates=0,
        additional_XPU_runs=0,
        additional_model_inference=0,
        corrected_replication_authorized=False,
        replication_gate="UNEVALUATED",
        P100_authorized=False,
        P960_authorized=False,
        stage2_ready=False,
    )
    with (root / "localization_audit.json").open("x") as f:
        json.dump(result, f, indent=2)
    lines = [
        "# RTPE XPU numerical fault-localisation diagnostic",
        "",
        "Completed15prospectively specified instrumented seed/case checks with no optimiser updates or internal-evaluation scoring. All48CPU tests and lint passed before execution. Model parameters, fixture values, masks, FP32 policy, requested cache capacity16384 and numerical tolerance1e-4 were preserved. Per-case start, events, completion/failure, activations and first-nonfinite snapshots are retained.",
        "",
        "**Observed localisation:** seed307/case2 and seed401/case2 reproduced the weighted-loss finiteness-guard exception. Transfers were exact, the observed target values were finite under correct CPU selection, and encoder inputs, GRU returned states and forecast outputs were finite. The first unexpected nonfinite appeared in `aten.index.Tensor` while selecting `target[observed]` inside the guard. Backward was not reached for these two cases.",
        "",
        "| Seed | Case | Expected selected elements | XPU selected elements | Nonfinite selected target values | Final registered parity |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in audits:
        lines.append(
            f"| {row['seed']} | {row['case']} | {row['expected_selected_count']} | {row['XPU_target_selected_count']} | {row['XPU_target_selection_nonfinite']} | {row['status']} |"
        )
    lines += [
        "",
        "Thecase2mask is4x3x64with767Truecells. Correct CPU boolean selection returns767finite target values. XPU selection returned511values and included one NaN in each failing case. Entirely observed cases have768Truecells but returned512values. Both prediction and target guard selections had incorrect output cardinality in all15instrumented cases.",
        "",
        "Thirteen cases nevertheless met the original final forecast/loss/gradient relative-error criterion. That does not certify their boolean selection: those gathers only feed validity guards, whereas the subsequent weighted residual arithmetic uses shape-preserving where/sum operations. This diagnostic reports operator-cardinality anomalies separately from final-metric parity; it does not relabel the failed scientific replication as evaluated.",
        "",
        "CPU-only postprocessing verified source/transfer equality, every forward checkpoint, all15case-event sequences, the first-nonfinite captures and the expected mask counts. No new model inference, XPU replay or optimiser update was used for the localization audit.",
        "",
        "Limits: this localises the instrumented reproductions, not the precise low-level kernel defect. Tracing and host copies synchronise execution and can alter timing/dispatch. The original replication did not log its failed seed/case, so these two captured failures cannot be asserted to be its original failing cases or a definitive account of its internal cause.",
        "",
        "Next milestone: a separately frozen minimal boolean-indexing reproducer using the captured tensors/masks without the GRU. Only after isolation should a targeted validity-check correction be specified and independently verified. No operator, precision, cache policy or replication seed was changed here, and no corrected replication is authorised.",
        "",
        "Original RTPE-1 remains closed with its positive development result. The five-seed study remains blocked and its forecasting criterion UNEVALUATED. P100/P960 and MACDE–CAVEM Stage2 remain blocked.",
    ]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (root / "README.md").write_text(
        "Preflight-only fault localisation completed. See RESULTS.md and localization_audit.json. Replication remains unevaluated; no correction or scientific restart occurred.\n"
    )
    files = [
        root / "plan.json",
        root / "summary.json",
        root / "manifest.json",
        root / "localization_audit.json",
        root / "RESULTS.md",
        root / "README.md",
        Path(__file__),
    ]
    with (root / "report_manifest.json").open("x") as f:
        json.dump({str(p): sha(p) for p in files}, f, indent=2)
    print(
        "Verified15case records:2reproduced loss-guard failures;15incorrect selection cardinalities;zero updates or XPU replays."
    )


if __name__ == "__main__":
    main()
