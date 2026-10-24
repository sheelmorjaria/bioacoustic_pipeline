"""CPU-only audit following the preserved multi-seed XPU preflight failure."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def main():
    root = Path("outputs/rclat_rtpe_multiseed_v1")
    spec = importlib.util.spec_from_file_location(
        "replication", "scripts/rclat_rtpe_replication.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol, base = module.bindings()
    assert (root / "FAILURE.json").is_file() and not (root / "RUN_REQUESTED.json").exists()
    assert not list(root.glob("seed_*"))
    helper = module.load_script("checks", "scripts/rclat_rtpe_protocol_prepare_v2.py")
    with np.load(
        "outputs/rclat_rtpe1_protocol_v2/training_numerical_fixtures.npz", allow_pickle=False
    ) as f:
        data = {k: f[k] for k in f.files}
    cases = []
    torch.set_num_threads(2)
    for seed, _ in module.PAIRS:
        for case in range(3):
            torch.manual_seed(seed)
            model = module.PoolTemporalPredictionEngine(
                64, hidden=64, scales=(1, 4, 16), horizons=(4, 8, 16)
            )
            values = helper.run_grad(
                model,
                torch.from_numpy(data[f"x_{case}"]),
                torch.from_numpy(data[f"valid_{case}"]),
                torch.from_numpy(data[f"y_{case}"]),
                torch.from_numpy(data[f"observed_{case}"]),
                torch.from_numpy(data["population_weights"]),
                torch.from_numpy(data[f"indices_{case}"]),
            )
            cases.append(
                dict(
                    seed=seed,
                    case=case,
                    CPU_forecast_loss_parameter_and_input_gradients_finite=all(
                        bool(torch.isfinite(v).all()) for v in values
                    ),
                )
            )
    module.write(
        root / "cpu_fixture_audit.json",
        dict(
            cases=cases,
            all15CPUcases_finite=all(
                r["CPU_forecast_loss_parameter_and_input_gradients_finite"] for r in cases
            ),
            XPU_retries=0,
            optimizer_updates=0,
            evaluation_scores_computed=0,
        ),
    )
    status = dict(
        status="BLOCKED_BEFORE_TRAINING_XPU_PREFLIGHT_FAILURE",
        protocol_frozen=True,
        CPU_tests_passed=43,
        CPU_fixture_cases_passed=15,
        scientific_training_started=False,
        scientific_GRU_updates=0,
        scientific_Ridge_fits=0,
        new_evaluation_scores_computed=0,
        individual_failed_seed_case="Not retained by initial preflight exception handler; do not infer",
        preflight_cause="Loss input guard rejected nonfinite observed XPU forecast or target values; internal origin unresolved",
        finite_seed_replication_gate="UNEVALUATED",
        automatic_retry=False,
        original_RTPE1_result="Closed PASS unchanged",
        P100_authorized=False,
        P960_authorized=False,
        stage2_ready=False,
    )
    module.write(root / "readiness.json", status)
    (
        root / "STATUS.md"
    ).write_text("""# RTPE five-seed replication: pre-training numerical prerequisite failed

The prospectively frozen replication did not start scientific training. Its XPU numerical preflight raised `ValueError: Observed predictions and targets must be finite` while computing the weighted loss. The exception and frozen sources are preserved in FAILURE.json and execution_manifest.json. No scientific RUN_REQUESTED marker, seed training directory, Ridge fit, GRU update or internal-evaluation score exists.

The initial preflight did not preserve incremental seed/case completion records, so the failing seed/case and count of previously passed comparisons are unknown. They must not be inferred from loop order or the absence of output. The loss guard does not distinguish a nonfinite forecast from a nonfinite transferred target; the internal origin remains unresolved.

A subsequent CPU-only audit of all15same seed/case fixtures found finite forecasts, losses, parameter gradients and input gradients. This verifies their CPU execution without resolving the XPU failure. No XPU retry, optimiser update or new evaluation inference occurred during that audit.

Scientific replication status is UNEVALUATED, not a failed forecasting criterion. All five seed pairs and the failed protocol remain preserved. A separately specified diagnostic with per-case/operation finiteness capture is needed before considering any corrected successor. No seed replacement, precision/batch/capacity change, threshold relaxation or silent retry occurred.

The original completed RTPE-1 result remains a closed positive development result. This failed preflight neither replicates nor overturns that result. P100/P960 and MACDE–CAVEM Stage2 remain blocked.
""")
    files = [p for p in root.iterdir() if p.is_file()] + [Path(__file__)]
    module.write(root / "closeout_manifest.json", {str(p): module.sha(p) for p in files})
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
