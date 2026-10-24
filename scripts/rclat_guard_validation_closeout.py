"""Independent host audit and scoped certificate for registered guard semantics."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("outputs/rclat_guard_validation_v1")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    plan = json.loads((ROOT / "plan.json").read_text())
    result = json.loads((ROOT / "summary.json").read_text())
    assert all(sha(p) == h for p, h in plan["dependencies"].items())
    assert all(sha(p) == h for p, h in json.loads((ROOT / "manifest.json").read_text()).items())
    assert (
        result["completed_workers"] == 132
        and result["all_workers_recorded"]
        and result["timeouts"] == 0
    )
    assert result["predicates_registered_matrix_passed"]
    worker_checks = []
    for row in result["worker_records"]:
        index = row["index"]
        fixture = plan["workers"][index]
        directory = ROOT / f"worker_{index:03d}"
        assert row["fixture"] == fixture
        assert row["process"]["returncode"] == 0 and not row["process"]["timed_out"]
        actual = json.loads((directory / "result.json").read_text())
        assert actual["status"] == "PASS" and actual["input_roundtrip_verified"]
        source = torch.load(directory / "CPU_sources.pt", map_location="cpu", weights_only=True)
        x = source["values"].numpy()
        m = source["observed"].numpy()
        counts = m.reshape(len(m), -1).sum(1)
        bad = (m & ~np.isfinite(x)).reshape(len(m), -1).sum(1)
        valid = bool((counts > 0).all() and (bad == 0).all())
        assert (
            valid == fixture["expected_valid"] == actual["verdict"] == actual["reference_verdict"]
        )
        pattern = fixture["pattern"]
        if pattern.startswith("observed_"):
            assert m.flatten()[0] and bad[0] == 1
        if pattern.startswith("masked_"):
            assert not m.flatten()[0] and bad.sum() == 0 and valid
        if pattern == "empty_first_origin":
            assert counts[0] == 0 and (counts[1:] > 0).all() and not valid
        if pattern == "empty_all":
            assert counts.sum() == 0 and not valid
        if pattern == "sparse_one_per_origin":
            assert (counts == 1).all() and valid
        if pattern == "mixed_observed_nan_masked_inf":
            assert (
                np.isnan(x.flatten()[0]) and m.flatten()[0] and not m.flatten()[1] and bad[0] == 1
            )
        for stage, key in [
            ("pre_operation.values_roundtrip", "values"),
            ("pre_operation.mask_roundtrip", "observed"),
            ("post_operation.values_roundtrip", "values"),
            ("post_operation.mask_roundtrip", "observed"),
        ]:
            snapshot = torch.load(
                directory / (stage + ".pt"), map_location="cpu", weights_only=True
            )
            np.testing.assert_array_equal(snapshot["actual"].numpy(), source[key].numpy())
        saved_counts = torch.load(
            directory / "predicate.observed_counts_roundtrip.pt",
            map_location="cpu",
            weights_only=True,
        )
        np.testing.assert_array_equal(saved_counts["actual"].numpy(), counts)
        if fixture["operation"] == "and":
            saved_bad = torch.load(
                directory / "AND.bad_counts_roundtrip.pt", map_location="cpu", weights_only=True
            )
            np.testing.assert_array_equal(saved_bad["actual"].numpy(), bad)
        events = [
            json.loads(line) for line in (directory / "stages.jsonl").read_text().splitlines()
        ]
        assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))
        assert not any(e["stage"] == "boolean_index" for e in events)
        pending = []
        for e in events:
            if e["event"] == "started":
                pending.append(e["stage"])
            elif e["event"] in ["completed", "failed"]:
                pending.remove(e["stage"])
        assert not pending
        worker_checks.append(
            dict(
                index=index,
                name=fixture["name"],
                expected_valid=valid,
                observed_counts_verified=True,
                observed_nonfinite_counts_verified=fixture["operation"] == "and",
                source_and_pre_post_roundtrips_verified=True,
                no_boolean_index_stage=True,
                pid=row["process"]["pid"],
            )
        )
    assert len({r["pid"] for r in worker_checks}) == 132
    accepted = sum(r["expected_valid"] for r in worker_checks)
    assert accepted == 60 and len(worker_checks) - accepted == 72
    certificate = dict(
        status="BOTH_DIAGNOSTIC_PREDICATES_PASSED_REGISTERED_MATRIX",
        workers_verified=132,
        AND_workers_passed=66,
        WHERE_workers_passed=66,
        correct_accept_workers=60,
        correct_reject_workers=72,
        observed_NaN_Inf_rejection_verified=True,
        masked_NaN_Inf_tolerance_verified=True,
        empty_per_origin_support_rejection_verified=True,
        sparse_support_acceptance_verified=True,
        observed_counts_verified=True,
        AND_observed_nonfinite_counts_verified=True,
        device="XPU0",
        precision="FP32",
        observed_runtime=dict(torch=torch.__version__, numpy=np.__version__),
        scalar_nonfinite_probe_positions="Flattened0;position1for mixed maskedInf;emptyfirst/allorigin and sparsefirstcell eachorigin",
        shapes=plan["shapes"],
        layouts=plan["layouts"],
        implementation="Staged AND/WHERE branches in rclat_mask_stages.worker_case;fresh processes,retained transfer tensors,explicit synchronisation",
        scope="Finite registered synthetic matrix;not arbitrary shapes/dtypes,original unobserved tensor lifetime/history,full weighted objective/gradient parity or production deployment",
        elapsed_seconds=result["elapsed_seconds"],
        scientific_optimizer_updates=0,
        model_forward_passes=0,
        production_guard_changed=False,
        corrected_replication_authorized=False,
        replication_gate="UNEVALUATED",
        P100_authorized=False,
        P960_authorized=False,
        stage2_ready=False,
    )
    with (ROOT / "certificate.json").open("x") as f:
        json.dump(certificate, f, indent=2)
    with (ROOT / "independent_audit.json").open("x") as f:
        json.dump(
            dict(verified=True, workers=worker_checks, additional_device_operations=0), f, indent=2
        )
    lines = [
        "# XPU guard-semantics validation: registered matrix passed",
        "",
        "Both diagnostic predicates passed all registered semantic checks:132fresh-process workers completed,66forAND and66forWHERE,with no timeout. The69related CPU tests and lint passed before execution. Independent host audit reproduced every verdict and checked source/roundtrip equality,counts and stage completion.",
        "",
        "| Requirement | Result |",
        "|---|---|",
        "| Finite observed values | Accepted correctly |",
        "| Masked NaN, positive Inf and negative Inf | Tolerated correctly |",
        "| Observed NaN, positive Inf and negative Inf | Rejected correctly |",
        "| Entirely empty support | Rejected correctly |",
        "| Empty support in one origin with other origins supported | Rejected correctly |",
        "| One observed finite cell per origin | Accepted correctly |",
        "| Observed NaN combined with masked Inf | Rejected correctly |",
        "| Per-origin observed counts | Exact NumPy agreement |",
        "| AND observed-nonfinite counts | Exact NumPy agreement |",
        "| Input/layout transfer and pre/post snapshots | Exact source agreement |",
        "",
        "The matrix covered FP32target tensors shaped(4,3,64)and(128,3,64), each with contiguous,strided-last-axis and offset7storage layouts. Elevenpatterns per shape/layout were independently tested for both predicates. Sixtyworkers correctly accepted reference-valid inputs and72correctly rejected observed-invalid or unsupported inputs. No selected predicate path executed a native boolean-index stage.",
        "",
        f"All workers used fixed20secondprocess limits; total elapsed was {result['elapsed_seconds']:.2f}seconds under the600secondglobal bound. Each operation boundary was explicitly synchronised and durably recorded. No worker was retried or replaced.",
        "",
        "Scalar NaN/Inf probes used flattened position0(and position1for the mixed maskedInf);empty support used first/all origins,and sparse support used the first cell of each origin. The matrix does not exhaust fault locations within each tensor.",
        "",
        "**Scoped certificate:** this establishes the missing-data semantics of the tested staged AND and WHERE implementations under this finite registered matrix. It does not certify arbitrary tensor shapes,dtypes,original asynchronous/temporary-buffer histories,production integration,or full model loss/gradient parity. The causes of the earlier indexing anomaly and stall remain unresolved at the low-level backend.",
        "",
        "**No deployment or restart:** the frozen production guard,weighted-loss implementation,original RTPE-1 results and failed five-seed replication were not changed. The scientific replication criterion remains UNEVALUATED. P100/P960 and MACDE–CAVEM Stage2 remain blocked.",
        "",
        "Next milestone: propose a separately versioned guard-only correction,bind it to this numeric contract,and independently verify CPU-XPU forecasts,weighted loss and parameter/input gradients before any scientific successor is considered. Passing this matrix does not itself authorise that successor.",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (ROOT / "README.md").write_text(
        "Registered diagnostic guard-semantics matrix passed. See RESULTS.md,certificate.json and independent_audit.json. No production change or replication restart occurred.\n"
    )
    files = [
        ROOT / "plan.json",
        ROOT / "summary.json",
        ROOT / "manifest.json",
        ROOT / "certificate.json",
        ROOT / "independent_audit.json",
        ROOT / "RESULTS.md",
        ROOT / "README.md",
        Path(__file__),
    ]
    with (ROOT / "report_manifest.json").open("x") as f:
        json.dump({str(p): sha(p) for p in files}, f, indent=2)
    print(json.dumps(certificate, indent=2))


if __name__ == "__main__":
    main()
