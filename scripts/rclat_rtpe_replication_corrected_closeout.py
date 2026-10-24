"""Independent host recomputation of completed five-seed outputs; no inference."""

import importlib.util
import json
from pathlib import Path
import numpy as np
import torch


def main():
    spec = importlib.util.spec_from_file_location(
        "runner", "scripts/rclat_rtpe_replication_corrected.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    protocol, base = entry.bindings()
    root = entry.ROOT
    entry.require_authorization()
    result = json.loads((root / "result.json").read_text())
    assert result["completed"]
    assert not (root / "FAILURE.json").exists()
    entry.assert_global_lock(root / "global_selection_lock.json")
    checked = 0
    maxrss = 0.0
    ridge_reference = None
    deltas = []
    seed_results = []
    for seed, sampler in entry.PAIRS:
        r = root / f"seed_{seed}"
        saved = json.loads((r / "result.json").read_text())
        assert (
            saved["completed_updates"] == 5000
            and saved["initialization_seed"] == seed
            and saved["sampler_seed"] == sampler
        )
        assert not any((r / n).exists() for n in ["FAILURE.json", "EVALUATION_FAILURE.json"])
        manifest = json.loads((r / "artifact_manifest.json").read_text())
        assert all(entry.sha(p) == h for p, h in manifest.items())
        selection = json.loads((r / "selection.json").read_text())
        assert selection["completed_updates"] == 5000 and selection["validation_steps"] == list(
            range(250, 5001, 250)
        )
        for p, h in selection["checkpoint_hashes"].items():
            assert entry.sha(p) == h
            checked += 1
        steps = [json.loads(x) for x in (r / "steps.jsonl").read_text().splitlines()]
        assert [x["step"] for x in steps] == list(range(1, 5001))
        rng = torch.Generator().manual_seed(sampler)
        for x in steps:
            assert x["indices"] == torch.randint(15840, (128,), generator=rng).tolist()
            assert x["learning_rate"] == 0.001
            assert np.isfinite(
                [x["loss"], x["gradient_norm"], x["elapsed_seconds"], x["host_peak_RSS_GiB"]]
            ).all()
            maxrss = max(maxrss, x["host_peak_RSS_GiB"])
            if x["step"] % 250 == 0:
                state = torch.load(
                    r / f"gru_{x['step']:06d}.pt", map_location="cpu", weights_only=True
                )
                assert torch.equal(state["sampler_rng"], rng.get_state())
                assert state["protocol_sha256"] == entry.sha(root / "protocol.json") and state[
                    "execution_sha256"
                ] == entry.sha(root / "execution_manifest.json")
        validation = [
            json.loads((r / f"gru_{n:06d}_validation.json").read_text())
            for n in range(250, 5001, 250)
        ]
        assert (
            selection["gru_step"]
            == min(validation, key=lambda x: (x["score"]["macro"][1], x["step"]))["step"]
        )
        candidates = [
            json.loads((r / f"ridge_{n:02d}_validation.json").read_text()) for n in range(6)
        ]
        assert [x["alpha"] for x in candidates] == base["ridge"]["alpha_candidates"]
        assert (
            selection["ridge_alpha"]
            == min(enumerate(candidates), key=lambda p: (p[1]["score"]["macro"][1], p[0]))[1][
                "alpha"
            ]
        )
        ids = json.loads((r / "evaluation_origin_ids.json").read_text())
        assert len(ids) == 3352 and len({x["speaker"] for x in ids}) == 38
        with np.load(r / "evaluation_forecasts.npz") as f:
            arrays = {k: f[k] for k in f.files}
        if ridge_reference is None:
            ridge_reference = arrays["ridge"]
        else:
            np.testing.assert_array_equal(ridge_reference, arrays["ridge"])
        mask = arrays["observed"]
        target = arrays["targets"].astype(float)

        def score(pred):
            residual = np.zeros_like(target)
            np.subtract(pred.astype(float), target, out=residual, where=mask)
            losses = (residual**2).sum(-1) / mask.sum(-1)
            recordings = {}
            owners = {}
            for row, loss in zip(ids, losses):
                recordings.setdefault(row["uid"], []).append(loss)
                owners[row["uid"]] = row["speaker"]
            speakers = {}
            for uid, values in recordings.items():
                speakers.setdefault(owners[uid], []).append(np.mean(values, axis=0))
            return {s: np.mean(v, axis=0) for s, v in speakers.items()}

        scores = {key: score(arrays[key]) for key in ["ridge", "gru_selected", "gru_final"]}
        for key, values in scores.items():
            for speaker, loss in values.items():
                np.testing.assert_allclose(
                    loss, saved[key]["speakers"][speaker], rtol=0, atol=1e-12
                )
        speakers = sorted(scores["ridge"])
        delta = np.array([scores["ridge"][s][1] - scores["gru_selected"][s][1] for s in speakers])
        draws = np.random.default_rng(base["uncertainty"]["seed"]).integers(0, 38, size=(10000, 38))
        ci = np.quantile(delta[draws].mean(1), [0.025, 0.975])
        np.testing.assert_allclose(delta.mean(), saved["primary"]["difference"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(ci, saved["primary"]["interval95"], rtol=0, atol=1e-12)
        passed = bool(delta.mean() > 0 and ci[0] > 0)
        assert saved["development_utility_gate"] == ("PASS" if passed else "FAIL")
        seed_results.append(
            dict(
                seed=seed,
                selected_step=selection["gru_step"],
                ridge_alpha=selection["ridge_alpha"],
                ridge_loss=float(np.mean([v[1] for v in scores["ridge"].values()])),
                gru_loss=float(np.mean([v[1] for v in scores["gru_selected"].values()])),
                difference=float(delta.mean()),
                interval95=ci.tolist(),
                passed=passed,
                speakers_favouring_GRU=int((delta > 0).sum()),
            )
        )
        deltas.append(delta)
        # Bind saved targets/masks to waveform-verified cached future frames.
        moments = torch.load(r / "scaler.pt", map_location="cpu", weights_only=True)
        mean = moments["mean"].numpy().astype(np.float32)
        scale = moments["scale"].numpy().astype(np.float32)
        expected = []
        support = []
        for record in json.loads(Path(base["inventory"]).read_text()):
            if record["split"] != "test":
                continue
            assert entry.sha(record["cache_path"]) == record["cache_sha256"]
            with np.load(record["cache_path"]) as f:
                mel = f["mel"]
                valid = f["mel_valid"]
            for origin in record["selected_origins"]:
                offsets = origin + np.array([4, 8, 16])
                expected.append((mel[offsets] - mean) / scale)
                support.append(valid[offsets])
        np.testing.assert_allclose(np.stack(expected), arrays["targets"], atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(np.stack(support), mask)
    matrix = np.stack(deltas)
    gate = "PASS" if all(s["passed"] for s in seed_results) else "FAIL"
    assert result["finite_seed_replication_gate"] == gate
    np.testing.assert_allclose(result["mean_difference"], matrix.mean(), atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        result["between_seed_macro_sample_SD"], matrix.mean(1).std(ddof=1), atol=1e-12, rtol=0
    )
    for name in ["preregistration.json", "execution_manifest.json"]:
        d = json.loads((Path("outputs/rclat_bilstm_dropout_v1") / name).read_text())
        assert all(entry.sha(p) == h for p, h in d["dependencies"].items())
    audit = dict(
        verified=True,
        scientific_updates=25000,
        Ridge_candidates=30,
        validation_events=100,
        checkpoint_hashes=checked,
        sampler_batches=25000,
        sampler_rng_states=100,
        evaluation_origins_per_seed=3352,
        evaluation_speakers=38,
        primary_gate=gate,
        seed_results=seed_results,
        mean_difference=float(matrix.mean()),
        between_seed_SD=float(matrix.mean(1).std(ddof=1)),
        peak_RSS_GiB=maxrss,
        elapsed_seconds=result["elapsed_seconds"],
        additional_fitting=False,
        additional_inference=False,
        uncertainty="Conditional speaker intervals per selected model;seed spread descriptive;no joint confidence interval",
        P100_authorized=False,
        P960_authorized=False,
        stage2_ready=False,
    )
    entry.write(root / "scientific_closeout.json", audit)
    lines = [
        "# RTPE-R1 v2 scientific closeout",
        f"\n**Frozen all-five replication criterion: {gate}.** All five models completed 5,000 updates before internal evaluation opened.",
        "\n| Seed | Selected update | Ridge MSE | GRU MSE | Difference | Conditional 95% interval | Gate |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for s in seed_results:
        lines.append(
            f"| {s['seed']} | {s['selected_step']} | {s['ridge_loss']:.6f} | {s['gru_loss']:.6f} | {s['difference']:.6f} | [{s['interval95'][0]:.6f}, {s['interval95'][1]:.6f}] | {'PASS' if s['passed'] else 'FAIL'} |"
        )
    lines += [
        f"\nEqual-seed mean difference: {audit['mean_difference']:.6f}; descriptive between-seed sample SD: {audit['between_seed_SD']:.6f}.",
        f"\nAudit verified all 25,000 sampler batches, 100 sampler checkpoints, {checked} model-checkpoint hashes, 100 validation events, frozen selection policies and saved evaluation forecasts/target-cache bindings. Shared Ridge forecasts were bit-identical across seeds. Elapsed execution: {result['elapsed_seconds']:.1f} seconds; maximum journalled host RSS: {maxrss:.3f} GiB.",
        "\nThis is finite-five-seed robustness evidence on the previously exposed, frozen human-speech development population. Per-seed intervals are conditional on the selected models and sampled origins; seed spread is descriptive. No independent source confirmation, temporal-order mechanism, linguistic or semantic recovery, or animal behavioural utility is established. Original RTPE-1 and the failed predecessor remain preserved. P100/P960 and MACDE–CAVEM Stage 2 remain blocked.",
    ]
    with (root / "SCIENTIFIC_RESULTS.md").open("x") as f:
        f.write("\n".join(lines) + "\n")
    entry.write(
        root / "scientific_artifact_manifest.json",
        {str(p): entry.sha(p) for p in root.rglob("*") if p.is_file()},
    )
    entry.write(
        root / "scientific_report_manifest.json",
        {
            str(Path(__file__)): entry.sha(__file__),
            str(root / "scientific_artifact_manifest.json"): entry.sha(
                root / "scientific_artifact_manifest.json"
            ),
        },
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
