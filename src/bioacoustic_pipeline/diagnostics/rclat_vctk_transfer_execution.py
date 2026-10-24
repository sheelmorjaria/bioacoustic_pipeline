"""Inference-only persistence/scoring;all five sets must lock before loss access."""

from collections import defaultdict
import json
from pathlib import Path
import traceback
import numpy as np
from .rclat_rtpe_execution import sha, write
from .rclat_rtpe_protocol import paired_speaker_summary
from .rclat_vctk_horizons import retained_losses


def speaker_scores(losses, ids):
    if len(losses) != len(ids) or not len(ids):
        raise ValueError("Matching losses and origin IDs required")
    recordings = defaultdict(list)
    owners = {}
    for loss, row in zip(losses, ids):
        if row["event"] in owners and owners[row["event"]] != row["speaker"]:
            raise ValueError("Recording belongs to multiple speakers")
        owners[row["event"]] = row["speaker"]
        recordings[row["event"]].append(float(loss))
    speakers = defaultdict(list)
    for event, values in recordings.items():
        speakers[owners[event]].append(float(np.mean(values)))
    return {s: float(np.mean(v)) for s, v in sorted(speakers.items())}


def validate_predictions(data, n):
    for key in ["ridge", "gru", "targets", "observed"]:
        if np.asarray(data[key]).shape != (n, 3, 64):
            raise ValueError("Protected prediction/target shape changed")
    mask = data["observed"][:, 1:]
    if mask.dtype != bool or not mask.sum(-1).all():
        raise ValueError("Retained horizons need support for every origin")
    for key in ["ridge", "gru", "targets"]:
        if not np.isfinite(np.where(mask, data[key][:, 1:], 0)).all():
            raise ValueError("Observed retained values must be finite")
    # Do not subtract targets or compute any losses here;40ms is ignored.


def assert_prediction_lock(root, seeds, ids):
    root = Path(root)
    path = root / "prediction_lock.json"
    if not path.is_file():
        raise ValueError("Scoring locked until all five prediction sets are complete")
    lock = json.loads(path.read_text())
    if lock["seeds"] != list(seeds) or len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("All five registered models required")
    if (
        sha(root / "ids.json") != lock["ids_sha256"]
        or json.loads((root / "ids.json").read_text()) != ids
    ):
        raise ValueError("Protected IDs hash changed")
    for seed in seeds:
        path = root / f"seed_{seed}_predictions.npz"
        record = root / f"seed_{seed}_complete.json"
        if (
            sha(path) != lock["predictions"][str(seed)]["sha256"]
            or sha(record) != lock["predictions"][str(seed)]["completion_sha256"]
        ):
            raise ValueError("Prediction hash changed")
    return lock


def score_predictions(root, seeds, ids, *, bootstrap_seed, replicates=10000, check=lambda: None):
    root = Path(root)
    assert_prediction_lock(root, seeds, ids)
    scores = {}
    reference = None
    means = []
    for seed in seeds:
        check()
        with np.load(root / f"seed_{seed}_predictions.npz", allow_pickle=False) as f:
            data = {k: f[k] for k in f.files}
        validate_predictions(data, len(ids))
        if reference is None:
            reference = {k: data[k][:, 1:].copy() for k in ["ridge", "targets", "observed"]}
        elif not all(
            np.array_equal(reference[k], data[k][:, 1:], equal_nan=True) for k in reference
        ):
            raise ValueError(
                "Shared Ridge/scaled targets/support did not reproduce bit-identically"
            )
        r = retained_losses(data["ridge"], data["targets"], data["observed"])
        g = retained_losses(data["gru"], data["targets"], data["observed"])
        result = {}
        for h in [8, 16]:
            summary = paired_speaker_summary(
                speaker_scores(r[h], ids),
                speaker_scores(g[h], ids),
                seed=bootstrap_seed,
                replicates=replicates,
            )
            summary["transfer_criterion_passed"] = summary.pop("development_utility_passed")
            summary["scope"] = (
                "Conditional fixed models and fixed protected VCTK recordings/origins;paired speaker resampling only"
            )
            result[str(h * 10) + "ms"] = summary
        scores[str(seed)] = result
        means.append(result["80ms"]["difference"])
        with (root / f"seed_{seed}_losses.npz").open("xb") as f:
            np.savez_compressed(f, ridge80=r[8], gru80=g[8], ridge160=r[16], gru160=g[16])
    passed = all(v["80ms"]["transfer_criterion_passed"] for v in scores.values())
    return dict(
        completed=True,
        all_five_transfer_gate="PASS" if passed else "FAIL",
        per_seed=scores,
        mean_primary_difference=float(np.mean(means)),
        between_seed_sample_SD=float(np.std(means, ddof=1)),
        primary_difference_range=[float(min(means)), float(max(means))],
        shared_Ridge_predictions_bit_identical=True,
        model_adaptation=False,
        optimizer_updates=0,
        uncertainty_scope="Per-seed conditional speaker intervals;seed aggregate descriptive;no joint seed/speaker confidence claim",
        P100_authorized=False,
        P960_authorized=False,
        stage2_ready=False,
    )


def run_inference_only(
    directory,
    seeds,
    ids,
    predict,
    *,
    bootstrap_seed,
    replicates=10000,
    check=lambda: None,
    context=None,
):
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Exactly five registered models required")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    write(
        root / "STARTED.json", dict(context=context or {}, model_fitting=False, optimizer_updates=0)
    )
    write(root / "ids.json", ids)
    phase = "predictions"
    current = None
    try:
        complete = {}
        for seed in seeds:
            current = seed
            check()
            data = predict(seed)
            validate_predictions(data, len(ids))
            path = root / f"seed_{seed}_predictions.npz"
            with path.open("xb") as f:
                np.savez_compressed(f, **data)
            marker = root / f"seed_{seed}_complete.json"
            write(marker, dict(seed=seed, records=len(ids), prediction_sha256=sha(path)))
            complete[str(seed)] = dict(sha256=sha(path), completion_sha256=sha(marker))
            print(f"Predictions persisted for seed{seed};scoring still locked.", flush=True)
        check()
        write(
            root / "prediction_lock.json",
            dict(
                seeds=list(seeds),
                ids_sha256=sha(root / "ids.json"),
                predictions=complete,
                context=context or {},
            ),
        )
        phase = "scoring"
        current = None
        result = score_predictions(
            root, seeds, ids, bootstrap_seed=bootstrap_seed, replicates=replicates, check=check
        )
        check()
        write(root / "result.json", result)
        write(
            root / "artifact_manifest.json", {str(p): sha(p) for p in root.iterdir() if p.is_file()}
        )
        return result
    except Exception:
        write(
            root / "FAILURE.json",
            dict(
                phase=phase,
                seed=current,
                traceback=traceback.format_exc(),
                gate="UNEVALUATED_INCOMPLETE",
                automatic_retry=False,
            ),
        )
        raise
