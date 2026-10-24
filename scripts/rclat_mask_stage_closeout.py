"""CPU-only audit of isolated XPU stage records; no new device computation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("outputs/rclat_mask_stages_v1")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    plan = json.loads((ROOT / "plan.json").read_text())
    summary = json.loads((ROOT / "summary.json").read_text())
    assert all(sha(p) == h for p, h in plan["dependencies"].items())
    assert all(sha(p) == h for p, h in json.loads((ROOT / "manifest.json").read_text()).items())
    assert (
        summary["supervised_workers"] == 24
        and summary["workers_with_result"] == 24
        and summary["timeouts"] == 0
    )
    checks = []
    for row in summary["workers"]:
        directory = (
            ROOT
            / f"{next(i for i, c in enumerate(plan['cases']) if c['name'] == row['case']):02d}_{row['case']}_{row['operation']}"
        )
        events = [
            json.loads(line) for line in (directory / "stages.jsonl").read_text().splitlines()
        ]
        assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))
        pending = []
        for event in events:
            if event["event"] == "started":
                pending.append(event["stage"])
            if event["event"] in ["completed", "failed"]:
                pending.remove(event["stage"])
        assert not pending
        completed = [e["stage"] for e in events if e["event"] == "completed"]
        assert row["worker_result"]["input_roundtrip_verified"]
        for name in [
            "pre_operation.values_roundtrip",
            "pre_operation.mask_roundtrip",
            "post_operation.values_roundtrip",
            "post_operation.mask_roundtrip",
        ]:
            assert name in completed
            values = torch.load(directory / (name + ".pt"), map_location="cpu", weights_only=True)
            np.testing.assert_array_equal(values["actual"].numpy(), values["expected"].numpy())
        index_expected = None
        index_actual = None
        if row["operation"] == "index":
            assert completed.index("pre_operation.values_roundtrip") < completed.index(
                "boolean_index"
            )
            assert completed.index("pre_operation.mask_roundtrip") < completed.index(
                "boolean_index"
            )
            assert row["worker_result"]["status"] == "OPERATOR_MISMATCH"
            data = torch.load(
                directory / "boolean_index.result_roundtrip.pt",
                map_location="cpu",
                weights_only=True,
            )
            index_actual = data["actual"].numel()
            index_expected = data["expected"].numel()
            assert index_expected - index_actual == 256
        else:
            assert "boolean_index" not in completed
            assert row["worker_result"]["status"] == "PASS"
            if row["operation"] in ["and", "where"]:
                assert row["worker_result"]["verdict"] == row["worker_result"]["reference_verdict"]
                comparisons = [
                    e
                    for e in events
                    if e["event"] == "comparison" and ("counts_roundtrip" in e["stage"])
                ]
                assert comparisons and all(e["exact_equal"] for e in comparisons)
        checks.append(
            dict(
                case=row["case"],
                operation=row["operation"],
                status=row["worker_result"]["status"],
                pre_and_post_input_equivalence_verified=True,
                all_operations_durably_completed=True,
                expected_index_count=index_expected,
                actual_index_count=index_actual,
                worker_elapsed_seconds=row["process"]["elapsed_seconds"],
            )
        )
    audit = dict(
        verified=True,
        workers_audited=24,
        operator_mismatches=6,
        transfer_only_passes=6,
        AND_passes=6,
        WHERE_passes=6,
        pre_and_post_input_equivalence_verified=True,
        prior_stall_reproduced=False,
        prior_input_corruption_reproduced=False,
        remaining_scope="Fresh processes,explicit synchronisation and retained dense transfer buffers differ from earlier same-process unobserved execution;no universal correction or kernel-cause claim",
        guard_certified=False,
        scientific_optimizer_updates=0,
        additional_device_operations=0,
        corrected_replication_authorized=False,
        workers=checks,
    )
    with (ROOT / "localization_audit.json").open("x") as f:
        json.dump(audit, f, indent=2)
    lines = [
        "# Isolated XPU mask-stage diagnostic closeout",
        "",
        "All24prospectively registered fresh-process workers completed within their individual20-second limits; no timeout or unreaped child occurred. Total supervisor execution was54.51seconds. The55related CPU tests and lint passed before the run. No model forward, optimiser update, production-guard change or replication restart occurred.",
        "",
        "| Input/layout | Transfer only | Native indexing: actual / expected | AND predicate | WHERE predicate |",
        "|---|---|---:|---|---|",
    ]
    for case in plan["cases"]:
        c = case["name"]
        by_mode = {r["operation"]: r for r in checks if r["case"] == c}
        idx = by_mode["index"]
        lines.append(
            f"| {c} | PASS | {idx['actual_index_count']} / {idx['expected_index_count']} | PASS | PASS |"
        )
    lines += [
        "",
        "**Observed localisation:** device values and masks were exactly equal to CPU reference values before native indexing. The indexed output nevertheless had incorrect cardinality in all six indexing workers. Post-index values/masks still matched the source. This isolates a native boolean-selection discrepancy under these verified input/layout conditions, rather than an already incorrect input at the observed pre-index boundary.",
        "",
        "Each worker recorded and synchronised individual device initialisation, transfer, backing allocation/fill, layout-view construction, strided/offset copy, roundtrip, selection or predicate operation. CPU-only audit verified all event sequences, completed boundaries, snapshots, selection counts and predicate-count comparisons. The six transfer-only workers and twelve separate predicate workers passed their defined comparisons.",
        "",
        "The previously stalled masked-Inf strided case completed in all four fresh-process modes. No previous post-gather source/mask discrepancy recurred. This does not locate the earlier stall: fresh processes, explicit synchronisation, per-operation observations and retained dense transfer tensors change execution history and temporary lifetimes. Those factors were not independently separated.",
        "",
        "Candidate predicates were tested here on six reference-valid input/layout cases, including masked NaN/Inf. This was fault localisation rather than complete semantic validation: observed-invalid and empty-origin cases were not included in this matrix. Neither candidate is certified or deployed, and the exact low-level kernel cause remains unidentified.",
        "",
        "Next milestone: a separately frozen guard-validation matrix with observed NaN/Inf, unsupported nonfinite values, empty per-origin support and verified contiguous/strided/offset inputs, retaining process isolation and stage-level evidence. Scientific restart requires its own prospective corrective specification after numerical prerequisites pass.",
        "",
        "Original RTPE-1 remains closed with its positive development result. The failed five-seed replication remains UNEVALUATED. Earlier incomplete diagnostics, frozen code and raw artifacts are preserved. P100/P960 and MACDE–CAVEM Stage2 remain blocked.",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (ROOT / "README.md").write_text(
        "Fresh-process stage localisation completed; six verified-input indexing discrepancies reproduced. Candidate guards remain uncertified. See RESULTS.md, plan.json and localization_audit.json.\n"
    )
    files = [
        ROOT / "plan.json",
        ROOT / "summary.json",
        ROOT / "manifest.json",
        ROOT / "localization_audit.json",
        ROOT / "RESULTS.md",
        ROOT / "README.md",
        Path(__file__),
    ]
    with (ROOT / "report_manifest.json").open("x") as f:
        json.dump({str(p): sha(p) for p in files}, f, indent=2)
    print(
        "Audited24workers:6native indexing mismatches with verified pre/post inputs;18passing transfer/predicate workers;no timeouts."
    )


if __name__ == "__main__":
    main()
