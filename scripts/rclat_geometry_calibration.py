"""Execute a frozen synthetic coverage study; no speech or model fitting."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from bioacoustic_pipeline.diagnostics.rclat_bootstrap_calibration import (
    population_difference,
    fast_crossed_interval,
    wilson_interval,
)
from bioacoustic_pipeline.diagnostics.rclat_communication import crossed_interval
from bioacoustic_pipeline.diagnostics.rclat_geometry_calibration import simulate_geometry


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
                "observed_assignment_used": True,
                "observed_judgments_used": False,
                "scientific_models_fitted": 0,
            },
            f,
            indent=2,
        )
    geometry = json.loads((root / "geometry.json").read_text())
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
                data = simulate_geometry(geometry, scenario, seed=ds)
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
        "observed_assignment_used": True,
        "observed_judgments_used": False,
        "scientific_models_fitted": 0,
        "scientific_bootstrap_certified": False,
        "P100_authorized": False,
        "P960_authorized": False,
        "stage2_ready": False,
    }
    with (root / "summary.json").open("x") as f:
        json.dump(output, f, indent=2)
    lines = [
        "# Corpus-geometry crossed-bootstrap calibration",
        "",
        "Fixed observed assignment; synthetic judgments only. Six frozen scenarios, 1,000 datasets each, 999 draws per dataset.",
        "",
        "| Scenario | Coverage | Monte Carlo Wilson 95% | Flag |",
        "|---|---:|---|---|",
    ]
    for name, r in all_results.items():
        flag = (
            "undercoverage"
            if r["undercoverage_flag"]
            else ("overcoverage" if r["overcoverage_flag"] else "no detectable discrepancy")
        )
        lines.append(f"| {name} | {r['coverage']:.1%} | {r['coverage_Wilson95']} | {flag} |")
    lines += [
        "",
        "The estimand averages equal stimuli within equal speakers and integrates independently generated speaker/listener effects on the fixed observed graph. Covariates are a balanced synthetic four-point grid, unrelated to wording or production intent. Truth uses 80-node quadrature checked against 128 nodes. Flags use Wilson Monte Carlo bounds relative to 95%; absence of a flag is not certification. No multiple-scenario correction applies.",
        "",
        "This tests the actual assignment geometry, not informative assignment, model fitting, checkpoint selection, feature selection, model seeds or new assignment graphs. Observed response outcome fields were neither accessed nor used; source JSON files containing them were parsed to whitelist identities. The unchanged bootstrap matched the reference kernel on each scenario's first dataset. Scientific uncertainty remains uncertified. Wording, source-use and final-methodology gates remain blocked; no scientific model fitting or final-test scoring occurred. P100/P960 and Stage 2 remain blocked.",
    ]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (root / "README.md").write_text(
        "See RESULTS.md and protocol.json. Completed methodological simulation; scientific certification remains withheld.\n"
    )
    manifest = {str(f): sha(f) for f in root.iterdir() if f.is_file()}
    (root / "report_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Geometry calibration completed.", flush=True)


if __name__ == "__main__":
    main()
