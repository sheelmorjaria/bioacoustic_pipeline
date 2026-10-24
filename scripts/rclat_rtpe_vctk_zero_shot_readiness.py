"""Independent preparation audit and synthetic scoring smoke;no VCTK prediction."""

from collections import Counter
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha, write
from bioacoustic_pipeline.diagnostics.rclat_vctk_transfer_execution import run_inference_only

ROOT = Path("outputs/rclat_rtpe_vctk_zero_shot_execution_v1")


def main():
    spec = importlib.util.spec_from_file_location("runner", "scripts/rclat_rtpe_vctk_zero_shot.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    p, e = entry.bindings()
    verification = json.loads((ROOT / "source_verification.json").read_text())
    preflight = json.loads((ROOT / "preflight.json").read_text())
    assert verification["passed"] and preflight["passed"]
    for record in [verification, preflight]:
        assert record["execution_sha256"] == sha(ROOT / "execution_manifest.json")
    assert sha(ROOT / "source_arrays.npz") == verification["source_arrays_sha256"]
    assert sha(ROOT / "origin_ids.json") == verification["origin_ids_sha256"]
    inventory = json.loads((entry.SPEC / "selected_inventory.json").read_text())
    ids = json.loads((ROOT / "origin_ids.json").read_text())
    expected = [
        dict(speaker=row["speaker"], event=row["event"], origin=o)
        for row in inventory
        for o in row["selected_origins"]
    ]
    assert ids == expected and len(ids) == 27648
    assert (
        len({row["event"] for row in ids}) == 3456 and len({row["speaker"] for row in ids}) == 108
    )
    assert set(Counter(row["speaker"] for row in ids).values()) == {256}
    with np.load(ROOT / "source_arrays.npz", allow_pickle=False) as f:
        assert f["x"].shape == (27648, 32, 64) and f["v"].all() and np.isfinite(f["x"]).all()
        assert f["y"].shape == (27648, 3, 64) and f["m"][:, 1:].all() and not f["m"][:, 0].any()
        assert np.isfinite(f["y"][:, 1:]).all()
    maximum = 0.0
    cases = []
    for directory in sorted((ROOT / "numerical_preflight").glob("seed_*_case_*")):
        record = json.loads((directory / "result.json").read_text())
        assert record["passed"]
        cases.append((record["seed"], record["fixture"]))
        t = torch.load(directory / "tensors.pt", map_location="cpu", weights_only=True)
        errors = []
        for a, b in zip(t["CPU"], t["XPU"]):
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            error = float((a.double() - b.double()).norm() / a.double().norm().clamp_min(1e-30))
            assert error < 1e-4
            maximum = max(maximum, error)
            errors.append(error)
        np.testing.assert_allclose(errors, record["errors"], rtol=0, atol=1e-15)
    seeds = [r["seed"] for r in p["models"]]
    assert set(cases) == {(s, c) for s in seeds for c in range(3)}
    assert not any(
        (ROOT / name).exists()
        for name in [
            "SCIENTIFIC_AUTHORIZATION.json",
            "RUN_REQUESTED.json",
            "scientific_run",
            "RUN_FAILURE.json",
        ]
    )
    # Durable synthetic fixture only;neither callback reads models or VCTK arrays.
    synthetic_ids = [
        dict(speaker=s, event=f"{s}_{i // 2}", origin=i % 2)
        for s in ["syntheticA", "syntheticB"]
        for i in range(4)
    ]

    def predict(seed):
        index = seeds.index(seed)
        target = np.zeros((8, 3, 64), np.float32)
        mask = np.ones_like(target, dtype=bool)
        mask[:, 0] = False
        ridge = np.ones_like(target)
        gru = np.full_like(target, 0.2 + index * 0.01)
        gru[4:] = 0.4 + index * 0.01
        ridge[:, 0] = np.nan
        gru[:, 0] = np.inf
        return dict(ridge=ridge, gru=gru, targets=target, observed=mask)

    print("SYNTHETIC lock/scoring integration only;no scientific VCTK model inference.", flush=True)
    result = run_inference_only(
        ROOT / "synthetic_integration",
        seeds,
        synthetic_ids,
        predict,
        bootstrap_seed=p["uncertainty"]["seed"],
        replicates=199,
        context=dict(synthetic=True, VCTK_predictions=False),
    )
    for seed in seeds:
        index = seeds.index(seed)
        d = np.array(
            [
                1 - float(np.float32(0.2 + index * 0.01)) ** 2,
                1 - float(np.float32(0.4 + index * 0.01)) ** 2,
            ]
        )
        draws = np.random.default_rng(p["uncertainty"]["seed"]).integers(0, 2, size=(199, 2))
        ci = np.quantile(d[draws].mean(1), [0.025, 0.975])
        np.testing.assert_allclose(
            result["per_seed"][str(seed)]["80ms"]["difference"], d.mean(), rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            result["per_seed"][str(seed)]["80ms"]["interval95"], ci, rtol=0, atol=1e-12
        )
    # Prior source audit and scientific evidence remain unchanged.
    for manifest in [
        "outputs/rclat_rtpe_vctk_support_execution_v2/source_artifact_manifest.json",
        "outputs/rclat_rtpe_multiseed_v2_execution/scientific_artifact_manifest.json",
    ]:
        assert all(sha(q) == h for q, h in json.loads(Path(manifest).read_text()).items())
    entry.bindings()
    audit = dict(
        status="ZERO_SHOT_INFERENCE_RUNNER_PREPARATION_PASSED_NOT_AUTHORIZED",
        CPU_tests=130,
        lint_passed=True,
        protected_recordings_verified=3456,
        protected_origins_verified=27648,
        protected_speakers=108,
        all_selected_source_PCM_feature_mask_origin_hashes_verified=True,
        frozen_Ridge_scaler_states_identical=True,
        trained_checkpoint_numerical_cases=15,
        max_CPU_XPU_relative_error=maximum,
        tolerance=1e-4,
        independent_numerical_tensor_check=True,
        synthetic_all_five_prediction_lock_and_scoring_verified=True,
        synthetic_bootstrap_independently_recomputed=True,
        source_preparation_seconds=verification["elapsed_seconds"],
        protocol_sha256=sha(entry.SPEC / "protocol.json"),
        execution_sha256=sha(ROOT / "execution_manifest.json"),
        source_verification_sha256=sha(ROOT / "source_verification.json"),
        preflight_sha256=sha(ROOT / "preflight.json"),
        VCTK_model_predictions=0,
        VCTK_predictive_scores=0,
        optimizer_updates=0,
        model_fitting=False,
        scientific_execution_authorized=False,
        transfer_gate="UNEVALUATED",
    )
    write(ROOT / "readiness_closeout.json", audit)
    report = f"""# RTPE-ZS1 inference runner preparation — PASSED

A dedicated inference-only runner and separate execution manifest are pinned to the unchanged scientific protocol and protected inventory. **130 focused CPU tests and lint passed.** Scientific VCTK inference is not authorised or started.

## Completed preparation

- All3,456selected source recordings were regenerated using the frozen193tapFIR-v2. Source-byte,decoded-PCM,mel,mask andeligible-origin hashes matched the source audit. Saved unscaled arrays contain27,648origins from108speakers;allprotected IDs andequal256origins-per-speaker counts were independently rechecked.
- AllfivefrozenRidge/scaler numerical states matched exactly. Models restore only hash-bound selected checkpoints;there is no fit,adaptation,retuning or newselection path. Ridge uses its unchanged CPUfloat64design/coefficients and preserves the frozen class's output cast to the FP32 input dtype;scoring uses float64 residuals.
- All15CPU–XPUcases used the actual selected GRU checkpoints and the original frozen LibriSpeech training fixtures. The maximum relative error was **{maximum:.9g}**,below1e-4. Saved forward/loss/input-gradient/parameter-gradient/logit-gradient tensors were independently audited. No optimiser updates occurred and noVCTKprediction was generated by this preflight.
- Synthetic tests verify exact history reconstruction,hierarchical weighting,paired-speaker bootstrap,all-five prediction locking,disabled40ms isolation,invalidretained rejection,tamper detection andimmutable failures/no-retry. A durable synthetic five-model fixture also passed;its speaker-bootstrap calculations were independently recomputed. Synthetic scores are not VCTK scientific outcomes.

## Execution boundary

The runner exposes prepare,verify,preflight andexecute. Execute first requires explicit authorisation bound to protocol,execution-manifest,source-verification andpreflight hashes. Noauthorisation file orRUN_REQUESTEDmarker exists. The protected source arrays may be read only under the verified cache binding;allfiveprediction sets are saved andhash-locked before anyprimaryorsecondaryloss is computed. The40msoutput/target is excluded even when its values are nonfinite. Unsupportedselectedrecords cannotbe replaced.

Scientific inference remains sequential onXPU0withFP32GRU andunchangedCPURidge,bounded batches,twothreads,8GiBhostRSS and7,200seconds. Sourceverification took **{verification["elapsed_seconds"]:.1f}seconds** andcomputed no forecasting outputs. Preparation and preflight records are immutable;failures cannotbe overwritten orautomaticallyretried. Exact source/cache hashes and environment versions are recorded.

The original locallyfrozen protocol's runner-pending flags are historical;this separately pinned execution package records completed readiness without modifying that specification. No scientific VCTK predictions,losses,modelselection ormodelfitting occurred. RTPE-ZS1 remains UNEVALUATED. RTPE-1/R1 positive results,P100/P960 andMACDE–CAVEM blocks remain unchanged.
"""
    with (ROOT / "RESULTS.md").open("x") as f:
        f.write(report)
    write(
        ROOT / "preparation_artifact_manifest.json",
        {str(q): sha(q) for q in ROOT.rglob("*") if q.is_file()},
    )
    write(
        ROOT / "preparation_report_manifest.json",
        {
            str(Path(__file__)): sha(__file__),
            str(ROOT / "preparation_artifact_manifest.json"): sha(
                ROOT / "preparation_artifact_manifest.json"
            ),
        },
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
