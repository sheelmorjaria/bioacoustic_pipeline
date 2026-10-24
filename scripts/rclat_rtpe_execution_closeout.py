"""Post-completion audit of immutable RTPE results; no new model inference or fitting."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def main():
    spec = importlib.util.spec_from_file_location("entry", "scripts/rclat_rtpe_train.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    protocol, execution = entry.bindings()
    root = entry.ROOT
    run = root / "training"
    result = json.loads((run / "result.json").read_text())
    assert result["completed"] and result["completed_updates"] == 5000
    assert (root / "closeout.json").is_file()
    assert not (root / "RUN_FAILURE.json").exists() and not (run / "FAILURE.json").exists()
    for p, h in json.loads((run / "artifact_manifest.json").read_text()).items():
        assert entry.sha(p) == h
    selection = result["selection"]
    for p, h in selection["checkpoint_hashes"].items():
        assert entry.sha(p) == h
    steps = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
    assert [r["step"] for r in steps] == list(range(1, 5001))
    generator = torch.Generator().manual_seed(2026102301)
    for row in steps:
        indices = torch.randint(15840, (128,), generator=generator)
        assert row["indices"] == indices.tolist()
        assert row["learning_rate"] == 0.001
        assert np.isfinite(
            [row["loss"], row["gradient_norm"], row["elapsed_seconds"], row["host_peak_RSS_GiB"]]
        ).all()
        if row["step"] % 250 == 0:
            state = torch.load(
                run / f"gru_{row['step']:06d}.pt", map_location="cpu", weights_only=True
            )
            assert state["step"] == row["step"] and torch.equal(
                state["sampler_rng"], generator.get_state()
            )
            assert state["protocol_sha256"] == entry.sha(entry.PROTOCOL)
            assert state["execution_sha256"] == entry.sha(root / "execution_manifest.json")
    alphas = protocol["ridge"]["alpha_candidates"]
    ridge_scores = [
        json.loads((run / f"ridge_{i:02d}_validation.json").read_text()) for i in range(6)
    ]
    assert (
        selection["ridge_alpha"]
        == min(enumerate(ridge_scores), key=lambda p: (p[1]["score"]["macro"][1], p[0]))[1]["alpha"]
    )
    assert [r["alpha"] for r in ridge_scores] == alphas
    val = [
        json.loads((run / f"gru_{step:06d}_validation.json").read_text())
        for step in range(250, 5001, 250)
    ]
    assert (
        selection["gru_step"] == min(val, key=lambda r: (r["score"]["macro"][1], r["step"]))["step"]
    )
    ids = json.loads((run / "evaluation_origin_ids.json").read_text())
    assert len(ids) == 3352 and len({r["speaker"] for r in ids}) == 38
    with np.load(run / "evaluation_forecasts.npz", allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    mask = arrays["observed"]
    targets = arrays["targets"].astype(float)

    def independent_scores(pred):
        loss = np.where(mask, (pred.astype(float) - targets) ** 2, 0).sum(-1) / mask.sum(-1)
        records = {}
        owners = {}
        for r, origin_loss in zip(ids, loss):
            records.setdefault(r["uid"], []).append(origin_loss)
            owners[r["uid"]] = r["speaker"]
        speakers = {}
        for uid, v in records.items():
            speakers.setdefault(owners[uid], []).append(np.mean(v, axis=0))
        return {s: np.mean(v, axis=0) for s, v in speakers.items()}

    rs = independent_scores(arrays["ridge"])
    gs = independent_scores(arrays["gru_selected"])
    fs = independent_scores(arrays["gru_final"])
    speakers = sorted(rs)
    delta = np.array([rs[s][1] - gs[s][1] for s in speakers])
    rng = np.random.default_rng(2026102302)
    draws = delta[rng.integers(0, 38, size=(10000, 38))].mean(1)
    ci = np.quantile(draws, [0.025, 0.975], method="linear")
    np.testing.assert_allclose(delta.mean(), result["primary"]["difference"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(ci, result["primary"]["interval95"], rtol=0, atol=1e-12)
    for s in speakers:
        np.testing.assert_allclose(rs[s], result["ridge"]["speakers"][s], atol=1e-12, rtol=0)
        np.testing.assert_allclose(gs[s], result["gru_selected"]["speakers"][s], atol=1e-12, rtol=0)
        np.testing.assert_allclose(fs[s], result["gru_final"]["speakers"][s], atol=1e-12, rtol=0)
    passed = bool(delta.mean() > 0 and ci[0] > 0)
    assert result["development_utility_gate"] == ("PASS" if passed else "FAIL")
    assert selection["completed_updates"] == 5000 and len(selection["validation_steps"]) == 20
    # Independently bind saved evaluation targets to the actual cached future frames.
    scaler = torch.load(run / "scaler.pt", map_location="cpu", weights_only=True)
    mean = scaler["mean"].numpy().astype(np.float32)
    scale = scaler["scale"].numpy().astype(np.float32)
    inventory = {
        r["uid"]: r
        for r in json.loads(Path(protocol["inventory"]).read_text())
        if r["split"] == "test"
    }
    expected = []
    targetmask = []
    for uid, r in inventory.items():
        assert entry.sha(r["cache_path"]) == r["cache_sha256"]
        with np.load(r["cache_path"], allow_pickle=False) as f:
            mel = f["mel"]
            m = f["mel_valid"]
        for t in r["selected_origins"]:
            target = mel[t + np.array([4, 8, 16])]
            expected.append((target - mean) / scale)
            targetmask.append(m[t + np.array([4, 8, 16])])
    np.testing.assert_allclose(np.stack(expected), arrays["targets"], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.stack(targetmask), mask)
    for name in ["preregistration.json", "execution_manifest.json"]:
        d = json.loads((Path("outputs/rclat_bilstm_dropout_v1") / name).read_text())
        assert all(entry.sha(p) == h for p, h in d["dependencies"].items())
    audit = dict(
        verified=True,
        sequential_updates_verified=5000,
        sampled_index_batches_verified=5000,
        checkpoint_hashes_verified=len(selection["checkpoint_hashes"]),
        rng_checkpoints_verified=20,
        validation_points_verified=20,
        evaluation_origins_verified=3352,
        evaluation_speakers=38,
        future_target_cache_bindings_verified=419,
        independent_primary_difference=float(delta.mean()),
        independent_interval95=ci.tolist(),
        speakers_favouring_GRU=int((delta > 0).sum()),
        development_utility_gate=result["development_utility_gate"],
        peak_host_RSS_GiB=max(r["host_peak_RSS_GiB"] for r in steps),
        elapsed_seconds=result["elapsed_seconds"],
        additional_optimizer_updates=0,
        additional_model_inference=0,
        bootstrap_scope="Conditional fixed selected models and origin sample; exposed development speakers",
        dropout_dependencies_unchanged=True,
        P100_authorized=False,
        P960_authorized=False,
        stage2_ready=False,
    )
    entry.write(root / "scientific_closeout.json", audit)
    primary = result["primary"]
    relative = 100 * primary["difference"] / primary["ridge_loss"]
    lines = [
        "# RTPE-1 revision-2 scientific closeout",
        "",
        f"Computational execution completed; registered development predictive-utility criterion: **{result['development_utility_gate']}**. All5000updates, all six Ridge candidates and all20validation/checkpoint events completed without recorded runtime or numerical exception. No retry, resumption or budget change occurred.",
        "",
        "| Primary80ms endpoint | Selected Ridge | Selected GRU |",
        "|---|---:|---:|",
        f"| Speaker-macro standardised MSE | {primary['ridge_loss']:.9f} | {primary['gru_loss']:.9f} |",
        f"| Selected configuration | alpha={selection['ridge_alpha']} | update{selection['gru_step']} |",
        "",
        f"Ridge-minus-GRU contrast: **{primary['difference']:.9f}**, paired speaker-percentile95% interval **[{ci[0]:.9f}, {ci[1]:.9f}]** (10000draws,seed2026102302). Both the point contrast and lower bound are strictly positive. Relative loss reduction: {relative:.2f}%. GRU lower loss for {int((delta > 0).sum())}/38 evaluation speakers.",
        "",
        f"Final5000update GRU primary80ms loss: {result['gru_final']['macro'][1]:.9f}; descriptive only. Selected checkpoint remained validation-chosen after full completion; evaluation was opened only after the locked selection. Execution elapsed {result['elapsed_seconds']:.2f}s; peak logged host RSS {audit['peak_host_RSS_GiB']:.3f}GiB.",
        "",
        "Independent saved-forecast recomputation reproduced all speaker/horizon losses, the primary contrast and bootstrap bounds. All5000uniform sampled index batches,20sampler-RNG states,26checkpoint hashes and419evaluation future-target/cache bindings passed audit. No new model inference or optimiser update was used for this closeout.",
        "",
        "Interpretation: development evidence that the tested causal pooled-multiscale GRU improves80ms future-acoustic prediction over weighted linear Ridge using the same ordered history on this speaker-disjoint sampled-origin population. This is not a test of order versus unordered features, and not ASR, linguistic, semantic, behavioural or animal-transfer evidence.",
        "",
        "Limits: previously exposed corpus; one model seed; eight frozen origins per recording. The interval conditions on fixed fitted/selected models and origins and does not include model-fitting, selection, origin-sampling or source-selection uncertainty. Positive development utility does not unlock P100/P960 or MACDE–CAVEM Stage2. Previous study outcomes and dropout dependencies remain unchanged.",
        "",
        "The earlier RESULTS.md/readiness.json/preparation_manifest.json are preserved historical preparation artifacts. Scientific results are in training/result.json and this scientific closeout; the frozen protocol and execution sources were not edited.",
    ]
    (root / "SCIENTIFIC_RESULTS.md").write_text("\n".join(lines) + "\n")
    entry.write(
        root / "scientific_manifest.json",
        {
            str(p): entry.sha(p)
            for p in [
                root / "scientific_closeout.json",
                root / "SCIENTIFIC_RESULTS.md",
                root / "closeout.json",
                run / "result.json",
                run / "artifact_manifest.json",
                Path(__file__),
            ]
        },
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
