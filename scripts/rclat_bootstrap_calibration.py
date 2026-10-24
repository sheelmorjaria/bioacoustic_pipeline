"""Execute a frozen synthetic coverage study; no speech or model fitting."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from bioacoustic_pipeline.diagnostics.rclat_bootstrap_calibration import (
    simulate,
    population_difference,
    fast_crossed_interval,
    wilson_interval,
)
from bioacoustic_pipeline.diagnostics.rclat_communication import crossed_interval


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    root = args.protocol.parent
    p = json.loads(args.protocol.read_text())
    for path, digest in p["dependencies"].items():
        if sha(path) != digest:
            raise ValueError("Changed dependency: " + path)
    with (root / "STARTED.json").open("x") as f:
        json.dump(
            {
                "protocol_sha256": sha(args.protocol),
                "real_data_used": False,
                "scientific_models_fitted": 0,
            },
            f,
            indent=2,
        )
    start = time.monotonic()
    all_results = {}
    parity = []
    with (root / "journal.jsonl").open("x", buffering=1) as journal:
        for index, (name, scenario) in enumerate(p["scenarios"].items()):
            truth = population_difference(scenario, nodes=80)
            assert abs(truth - population_difference(scenario, nodes=128)) <= 1e-9
            rows = []
            for replicate in range(p["datasets_per_scenario"]):
                if time.monotonic() - start > p["walltime_seconds"]:
                    raise TimeoutError("Frozen simulation walltime bound")
                ds = p["dataset_seed"] + 100000 * index + replicate
                bs = p["bootstrap_seed"] + 100000 * index + replicate
                data = simulate(
                    scenario, seed=ds, speakers=18, listeners=82, min_raters=12, max_raters=31
                )
                values = [data[k] for k in ["stimuli", "responses", "c1", "c2"]]
                if replicate == 0:
                    a = crossed_interval(*values, seed=bs, replicates=31)
                    b = fast_crossed_interval(*values, seed=bs, replicates=31)
                    error = max(
                        abs(a["difference"] - b["difference"]),
                        max(abs(x - y) for x, y in zip(a["interval95"], b["interval95"])),
                    )
                    assert error <= 1e-12
                    parity.append(
                        {
                            "scenario": name,
                            "maximum_absolute_difference": error,
                            "reference_draws": 31,
                        }
                    )
                estimate = fast_crossed_interval(
                    *values, seed=bs, replicates=p["bootstrap_replicates"]
                )
                lo, hi = estimate["interval95"]
                row = {
                    "replicate": replicate,
                    "dataset_seed": ds,
                    "bootstrap_seed": bs,
                    "estimate": estimate["difference"],
                    "interval95": [lo, hi],
                    "covered": lo <= truth <= hi,
                    "zero_excluded": not (lo <= 0 <= hi),
                    "stimuli": len(data["stimuli"]),
                    "responses": len(data["responses"]),
                }
                rows.append(row)
                journal.write(json.dumps({"scenario": name, **row}) + "\n")
            estimates = np.array([r["estimate"] for r in rows])
            coverage = sum(r["covered"] for r in rows) / len(rows)
            ci = wilson_interval(sum(r["covered"] for r in rows), len(rows))
            result = {
                "true_population_difference": truth,
                "coverage": coverage,
                "coverage_Wilson95": ci,
                "coverage_MCSE": float(np.sqrt(coverage * (1 - coverage) / len(rows))),
                "bias": float(estimates.mean() - truth),
                "bias_MCSE": float(estimates.std(ddof=1) / np.sqrt(len(rows))),
                "empirical_estimate_SD": float(estimates.std(ddof=1)),
                "mean_interval_width": float(
                    np.mean([r["interval95"][1] - r["interval95"][0] for r in rows])
                ),
                "zero_exclusion_rate": sum(r["zero_excluded"] for r in rows) / len(rows),
                "undercoverage_flag": ci[1] < 0.95,
                "overcoverage_flag": ci[0] > 0.95,
                "datasets": rows,
            }
            all_results[name] = result
            print(name, "coverage", coverage, "MC95", ci, "bias", result["bias"], flush=True)
    output = {
        "completed": True,
        "protocol_sha256": sha(args.protocol),
        "scenarios": all_results,
        "reference_kernel_parity": parity,
        "elapsed_seconds": time.monotonic() - start,
        "real_data_used": False,
        "scientific_models_fitted": 0,
        "scientific_bootstrap_certified": False,
        "P100_authorized": False,
        "P960_authorized": False,
        "stage2_ready": False,
    }
    with (root / "summary.json").open("x") as f:
        json.dump(output, f, indent=2)
    lines = [
        "# Crossed-bootstrap synthetic calibration pilot",
        "",
        "Six frozen scenarios; 200 independently generated datasets per scenario; 499 paired positive-weight bootstrap draws per dataset. Conditional predictions are fixed, not trained. Eighteen speakers and82listeners; balanced four-point covariate grid per speaker (four or eight stimuli in uneven scenario);12–31 raters perstimulus.",
        "",
        "| Scenario | Coverage | Monte Carlo Wilson95 | Bias (nats) | Mean interval width | Flag |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, r in all_results.items():
        flag = (
            "undercoverage"
            if r["undercoverage_flag"]
            else ("overcoverage" if r["overcoverage_flag"] else "no detectable nominal discrepancy")
        )
        lines.append(
            f"| {name} | {r['coverage']:.1%} | [{r['coverage_Wilson95'][0]:.1%}, {r['coverage_Wilson95'][1]:.1%}] | {r['bias']:.5f} | {r['mean_interval_width']:.5f} | {flag} |"
        )
    lines += [
        "",
        "Truth integrates logistic-normal speaker/listener effects by80-node Gauss–Hermite quadrature, checked against128nodes. Symmetric null model predictions differ by opposite logit offsets; they are not identical predictions yielding trivial zero-width intervals. Positive alternatives compare a constant0.5 predictor with a fixed acoustic-covariate predictor. Response assignment is independent of latent effects; informative missingness is not tested.",
        "",
        "Coverage flags are descriptive: undercoverage if the Monte Carlo Wilson95 upper bound is below nominal95%; overcoverage if its lower bound is above95%. No multiple-scenario adjustment or scientific certification rule is applied. Absence of a detectable discrepancy is not proof of calibrated coverage. Only200datasets and499bootstrapdraws limit precision. This pilot uses fewer stimuli per speaker than the real corpus and excludes model-fitting, feature-selection, lexical, trial-history and nonrandom-listener-assignment uncertainty.",
        "",
        "The vectorised implementation matched the original reference kernel within1e-12 on the first dataset of every scenario (31draws). Full per-dataset outcomes and seeds are retained. No audio, real listener data, scientific model fitting or running-experiment code was touched. Independent wording, source-use and phonological-reference blockers remain; P100/P960 and MACDE–CAVEM Stage2 remain blocked.",
        "",
    ]
    (root / "RESULTS.md").write_text("\n".join(lines))
    print("Calibration pilot completed.", flush=True)


if __name__ == "__main__":
    main()
