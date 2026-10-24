"""Independent fixed-artifact VCTK score/bootstrap recomputation;no inference."""

from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import numpy as np
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha, write

ROOT = Path("outputs/rclat_rtpe_vctk_zero_shot_execution_v1")


def independent_speakers(losses, ids):
    recordings = defaultdict(list)
    owners = {}
    for row, loss in zip(ids, losses):
        recordings[row["event"]].append(float(loss))
        owners[row["event"]] = row["speaker"]
    speakers = defaultdict(list)
    for event, values in recordings.items():
        speakers[owners[event]].append(float(np.mean(values)))
    return {s: float(np.mean(values)) for s, values in sorted(speakers.items())}


def main():
    spec = importlib.util.spec_from_file_location("runner", "scripts/rclat_rtpe_vctk_zero_shot.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    p, e = entry.bindings()
    entry.require_authorization()
    run = ROOT / "scientific_run"
    close = json.loads((ROOT / "scientific_closeout.json").read_text())
    assert close["completed"]
    assert not (ROOT / "RUN_FAILURE.json").exists() and not (run / "FAILURE.json").exists()
    for path, h in json.loads((run / "artifact_manifest.json").read_text()).items():
        assert sha(path) == h
    result = json.loads((run / "result.json").read_text())
    assert result["completed"]
    ids = json.loads((run / "ids.json").read_text())
    assert ids == json.loads((ROOT / "origin_ids.json").read_text()) and len(ids) == 27648
    assert (
        len({row["speaker"] for row in ids}) == 108 and len({row["event"] for row in ids}) == 3456
    )
    lock = json.loads((run / "prediction_lock.json").read_text())
    seeds = [info["seed"] for info in p["models"]]
    assert lock["seeds"] == seeds and lock["ids_sha256"] == sha(run / "ids.json")
    assert lock["context"]["protocol_sha256"] == sha(entry.SPEC / "protocol.json") and lock[
        "context"
    ]["execution_sha256"] == sha(ROOT / "execution_manifest.json")
    for seed in seeds:
        record = lock["predictions"][str(seed)]
        assert sha(run / f"seed_{seed}_predictions.npz") == record["sha256"]
        assert sha(run / f"seed_{seed}_complete.json") == record["completion_sha256"]
        assert (run / f"seed_{seed}_complete.json").stat().st_mtime_ns <= (
            run / "prediction_lock.json"
        ).stat().st_mtime_ns
        assert (run / f"seed_{seed}_losses.npz").stat().st_mtime_ns >= (
            run / "prediction_lock.json"
        ).stat().st_mtime_ns
    speakers = sorted({row["speaker"] for row in ids})
    draws = np.random.default_rng(p["uncertainty"]["seed"]).integers(0, 108, size=(10000, 108))
    summaries = []
    precision_checks = []
    reference = None
    for seed in seeds:
        with np.load(run / f"seed_{seed}_predictions.npz", allow_pickle=False) as f:
            data = {k: f[k] for k in f.files}
        assert all(data[k].shape == (27648, 3, 64) for k in ["ridge", "gru", "targets", "observed"])
        if reference is None:
            reference = {k: data[k][:, 1:].copy() for k in ["ridge", "targets", "observed"]}
        else:
            for key in reference:
                np.testing.assert_array_equal(data[key][:, 1:], reference[key])
        per_seed = {}
        for column, label in [(1, "80ms"), (2, "160ms")]:
            mask = data["observed"][:, column]
            counts = mask.sum(1)
            assert (counts > 0).all()
            loss = {}
            precise_loss = {}
            for name in ["ridge", "gru"]:
                a = np.asarray(data[name][:, column], dtype=np.float64)
                b = np.asarray(data["targets"][:, column], dtype=np.float64)
                assert data[name].dtype == np.float32 and data["targets"].dtype == np.float32
                residual64 = np.zeros_like(a)
                np.subtract(a, b, out=residual64, where=mask)
                precise_loss[name] = np.sum(residual64**2, axis=1) / counts
                # Independently reproduce the frozen scorer's actual ufunc
                # precision explicitly, not by reusing retained_losses.
                residual32 = np.subtract(
                    data[name][:, column], data["targets"][:, column], dtype=np.float32
                )
                residual = np.where(mask, residual32.astype(np.float64), 0)
                loss[name] = np.sum(residual**2, axis=1) / counts
            r = independent_speakers(loss["ridge"], ids)
            g = independent_speakers(loss["gru"], ids)
            delta = np.array([r[s] - g[s] for s in speakers])
            ci = np.quantile(delta[draws].mean(1), [0.025, 0.975], method="linear")
            r64 = independent_speakers(precise_loss["ridge"], ids)
            g64 = independent_speakers(precise_loss["gru"], ids)
            delta64 = np.array([r64[s] - g64[s] for s in speakers])
            ci64 = np.quantile(delta64[draws].mean(1), [0.025, 0.975], method="linear")
            precision_checks.append(
                dict(
                    seed=seed,
                    horizon=label,
                    canonical_difference=float(delta.mean()),
                    pure_FP64_difference=float(delta64.mean()),
                    pure_FP64_interval95=ci64.tolist(),
                    point_absolute_difference=float(abs(delta64.mean() - delta.mean())),
                    interval_max_absolute_difference=float(np.max(np.abs(ci64 - ci))),
                    max_origin_loss_difference=float(
                        max(np.max(np.abs(loss[k] - precise_loss[k])) for k in loss)
                    ),
                    canonical_decision=bool(delta.mean() > 0 and ci[0] > 0),
                    pure_FP64_decision=bool(delta64.mean() > 0 and ci64[0] > 0),
                )
            )
            saved = result["per_seed"][str(seed)][label]
            np.testing.assert_allclose(saved["difference"], delta.mean(), atol=1e-12, rtol=0)
            np.testing.assert_allclose(saved["interval95"], ci, atol=1e-12, rtol=0)
            np.testing.assert_allclose(
                saved["ridge_loss"], np.mean(list(r.values())), atol=1e-12, rtol=0
            )
            np.testing.assert_allclose(
                saved["gru_loss"], np.mean(list(g.values())), atol=1e-12, rtol=0
            )
            for s in speakers:
                np.testing.assert_allclose(
                    saved["speaker_differences"][s], r[s] - g[s], atol=1e-12, rtol=0
                )
            with np.load(run / f"seed_{seed}_losses.npz", allow_pickle=False) as f:
                np.testing.assert_allclose(
                    f["ridge" + label[:-2]], loss["ridge"], atol=1e-12, rtol=0
                )
                np.testing.assert_allclose(f["gru" + label[:-2]], loss["gru"], atol=1e-12, rtol=0)
            passed = bool(delta.mean() > 0 and ci[0] > 0)
            assert saved["transfer_criterion_passed"] == passed
            per_seed[label] = dict(
                ridge_loss=float(np.mean(list(r.values()))),
                gru_loss=float(np.mean(list(g.values()))),
                difference=float(delta.mean()),
                interval95=ci.tolist(),
                passed=passed,
                speakers_favouring_GRU=int((delta > 0).sum()),
            )
        assert set(result["per_seed"][str(seed)]) == {"80ms", "160ms"}
        summaries.append(dict(seed=seed, **per_seed))
    primary = np.array([s["80ms"]["difference"] for s in summaries])
    gate = "PASS" if all(s["80ms"]["passed"] for s in summaries) else "FAIL"
    assert result["all_five_transfer_gate"] == close["all_five_transfer_gate"] == gate
    np.testing.assert_allclose(
        result["mean_primary_difference"], primary.mean(), atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        result["between_seed_sample_SD"], primary.std(ddof=1), atol=1e-12, rtol=0
    )
    # The earlier source/model evidence is immutable.
    for name in [
        "outputs/rclat_rtpe_vctk_support_execution_v2/source_artifact_manifest.json",
        "outputs/rclat_rtpe_multiseed_v2_execution/scientific_artifact_manifest.json",
    ]:
        assert all(sha(q) == h for q, h in json.loads(Path(name).read_text()).items())
    entry.bindings()
    audit = dict(
        status="ZERO_SHOT_SCIENTIFIC_EXECUTION_COMPLETED_INDEPENDENTLY_VERIFIED",
        all_five_transfer_gate=gate,
        speakers=108,
        recordings=3456,
        origins=27648,
        seeds=summaries,
        mean_primary_difference=float(primary.mean()),
        between_seed_sample_SD=float(primary.std(ddof=1)),
        source_models_unchanged=True,
        all_five_lock_before_losses_verified=True,
        shared_retained_Ridge_predictions_bit_identical=True,
        independent_origin_recording_speaker_scores_verified=True,
        independent_bootstrap_recomputation_verified=True,
        excluded_40ms_not_scored=True,
        scoring_arithmetic="Frozen FP32 subtraction followed by FP64 square/reduction;matched-precision independent reference reproduced exactly",
        pure_FP64_sensitivity=dict(
            cases=precision_checks,
            max_point_absolute_difference=max(
                x["point_absolute_difference"] for x in precision_checks
            ),
            max_interval_absolute_difference=max(
                x["interval_max_absolute_difference"] for x in precision_checks
            ),
            all_decisions_agree=all(
                x["canonical_decision"] == x["pure_FP64_decision"] for x in precision_checks
            ),
        ),
        first_pure_FP64_audit_discrepancy_preserved=True,
        elapsed_seconds=close["elapsed_seconds"],
        model_fitting=False,
        optimizer_updates=0,
        additional_inference=False,
        uncertainty="Per-seed fixed-model/fixed-sample paired-speaker intervals;no joint seed/speaker or globalhumanidentity claim",
        P100_authorized=False,
        P960_authorized=False,
        stage2_ready=False,
    )
    write(ROOT / "independent_scientific_closeout.json", audit)
    lines = [
        f"# RTPE-ZS1 VCTK zero-shot scientific closeout — {gate}",
        "\nThe user-authorised frozen all-five evaluation completed on108speakers,3,456recordings and27,648protected origins. Allfiveprediction sets were saved andhash-locked beforeprimaryorsecondarylosses. No model/scaler fitting,adaptation,retuning,checkpointselection,seedreplacement orretry occurred.",
        "\n## Primary80ms results",
        "\n| Seed | Ridge MSE | GRU MSE | Ridge − GRU | Conditional95% speaker interval | Gate |",
        "|---|---:|---:|---:|---|---|",
    ]
    for s in summaries:
        a = s["80ms"]
        lines.append(
            f"| {s['seed']} | {a['ridge_loss']:.6f} | {a['gru_loss']:.6f} | {a['difference']:.6f} | [{a['interval95'][0]:.6f},{a['interval95'][1]:.6f}] | {'PASS' if a['passed'] else 'FAIL'} |"
        )
    lines += [
        f"\nThe registered all-five transfer criterion is **{gate}**. Equal-seed mean primary contrast:{primary.mean():.6f};descriptive between-seed sampleSD:{primary.std(ddof=1):.6f}. Neither aggregate nor secondary outcomes replace the primary criterion.",
        "\n## Descriptive160ms results",
        "\n| Seed | Ridge MSE | GRU MSE | Ridge − GRU | Conditional95% interval |",
        "|---|---:|---:|---:|---|",
    ]
    for s in summaries:
        a = s["160ms"]
        lines.append(
            f"| {s['seed']} | {a['ridge_loss']:.6f} | {a['gru_loss']:.6f} | {a['difference']:.6f} | [{a['interval95'][0]:.6f},{a['interval95'][1]:.6f}] |"
        )
    lines += [
        f"\nIndependent host recomputation verified everyper-origin loss,recording/speakermean,primary/secondarybootstrapinterval,all-fivegate andprediction/IDbinding. Savedfileordering corroborates completion of allprediction sets beforeanylossfile. Runtime:{close['elapsed_seconds']:.1f}seconds. No further inference orfitting was needed for this audit.",
        "\n## Scoring precision disclosure",
        "\nThe preparation report overstated residual arithmetic as fully FP64. The immutable frozen scorer actually subtracts FP32 prediction/target inputs using an FP32 ufunc loop, then squares and reduces the promoted residuals in FP64. The experiment used that exact hash-bound implementation. Its scientific protocol did not separately specify subtraction dtype. The first host audit compared against a pure FP64 reference at an audit-only 1e-12 absolute assertion and detected a 2.1922e-11 difference in the first primary contrast. That failed comparison and original audit source remain preserved; no prediction/scientific artifact was changed or rerun.",
        "\nA separate host-only closeout independently reproduced the frozen mixed-precision per-origin scores and bootstrap results at the original 1e-12 audit assertion. It also reported pure FP64 recomputations as post-hoc numerical sensitivity checks, not replacement endpoints or a relaxed scientific gate. Both arithmetic paths give the same pass/fail decisions for every primary and secondary comparison. Exact sensitivity values are in independent_scientific_closeout.json. The registered result remains the frozen implementation's result. The earlier preparation report is retained with this correction rather than silently edited.",
        "\n## Scientific scope",
        "\nThis tests fixed-model zero-shot acoustic forecasting on the protected VCTK mic2sample under the certified193tapcausal resampler.80ms isprimary,160msdescriptive and40msunscored. Model-head offsets are not source-supportgaps;the retained primary gap remains36.0417ms relative to the distributed48kHzwaveform. Speaker identity is corpus-local;globalcross-corpus humanidentity is unresolved. Intervals condition on models,selectedrecordings andorigins;seedspread is descriptive.",
        "\nThe outcome does not establish temporal-ordercausality,transcriptionorsemanticrecovery,animaltransfer orbehaviouralutility. Completed RTPE-1/R1 findings are unchanged. P100/P960 andMACDE–CAVEM Stage2 remain blocked. Any corrective,adaptation ornewsource study requires a separatelyprospective specification;no post-outcome changes are part of this completed run.",
    ]
    with (ROOT / "SCIENTIFIC_RESULTS.md").open("x") as f:
        f.write("\n".join(lines) + "\n")
    write(
        ROOT / "scientific_artifact_manifest.json",
        {str(q): sha(q) for q in ROOT.rglob("*") if q.is_file()},
    )
    write(
        ROOT / "scientific_report_manifest.json",
        {
            str(Path(__file__)): sha(__file__),
            str(ROOT / "scientific_artifact_manifest.json"): sha(
                ROOT / "scientific_artifact_manifest.json"
            ),
        },
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
