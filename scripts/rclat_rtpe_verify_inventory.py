"""Frozen bounded-origin certification; reads existing speech sources, fits no model."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import soundfile as sf

from bioacoustic_pipeline.diagnostics.rclat_rtpe_inventory import (
    eligible_origins,
    select_origins,
    trailing_mel,
    recording_weights,
)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = Path("outputs/rclat_rtpe_source_support_v1/selected_metadata.json")
    cache_receipt = Path("outputs/rclat_p10/cache_integrity.json")
    dependencies = [
        metadata,
        cache_receipt,
        Path(__file__),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_rtpe_inventory.py"),
        Path("tests/test_rclat_rtpe_inventory.py"),
    ]
    policy = dict(
        status="LOCALLY_FROZEN_INVENTORY_POLICY_ONLY_NOT_TRAINING_PROTOCOL",
        source_metadata_sha256=sha(metadata),
        dependencies={str(p): sha(p) for p in dependencies},
        origins_per_recording=8,
        salt="RTPE-1-common-origins-v1-20261023",
        history=32,
        horizons=[4, 8, 16],
        primary_horizon=8,
        sampling="Rank unique eligible frame indices by SHA256(salt:uid:origin); select first8, store chronological order; no numeric target values used",
        support="All frames from history start through furthest target have some valid feature; observed cells finite; each target retains its mask",
        weighting="Each split: equal speakers, equal recordings within speaker, equal selected origins within recording. Shared by both arms.",
        failure_policy="Stop whole audit on any integrity/schema/regeneration failure or fewer than8 origins. No replacements, exclusions, retries or threshold changes.",
        source_checks="Compressed file SHA256, native mono16k PCM16 sample count and decoded little-endian PCM16 SHA256",
        cache_checks="Original NPZ SHA256; mel float32 and boolean masks; exact640+160j exclusive frame ends; finite valid cells; recompute all64mel values from waveform",
        regeneration_rtol=1e-6,
        regeneration_atol=1e-6,
        walltime_seconds=1800,
        no_model_fitting=True,
        no_old_scaler_reuse=True,
        scope="Previously exposed development sources; no official final-test source",
        model_budget_frozen=False,
        uncertainty_frozen=False,
    )
    write(args.output / "sampling_policy.json", policy)
    write(
        args.output / "STARTED.json",
        dict(policy_sha256=sha(args.output / "sampling_policy.json"), models_fitted=0),
    )
    started = time.monotonic()
    records = json.loads(metadata.read_text())
    cache_hashes = json.loads(cache_receipt.read_text())["files"]
    weights = {}
    for split in ("train", "validation", "test"):
        weights.update(recording_weights([r for r in records if r["split"] == split]))
    selected = []
    all_origins = {}
    max_error = 0.0
    try:
        with (args.output / "verification.jsonl").open("x", buffering=1) as journal:
            for index, row in enumerate(records):
                if time.monotonic() - started > policy["walltime_seconds"]:
                    raise TimeoutError("Frozen verification walltime")
                uid = row["uid"]
                cache = Path("cache/rclat_p10") / (uid + ".npz")
                if (
                    sha(row["path"]) != row["source_sha256"]
                    or sha(cache) != cache_hashes[str(cache)]
                ):
                    raise ValueError("Source/cache hash mismatch: " + uid)
                info = sf.info(row["path"])
                if (
                    info.samplerate != 16000
                    or info.channels != 1
                    or info.subtype != "PCM_16"
                    or info.frames != row["samples"]
                ):
                    raise ValueError("Native waveform schema mismatch: " + uid)
                pcm, rate = sf.read(row["path"], dtype="int16")
                if (
                    hashlib.sha256(pcm.astype("<i2", copy=False).tobytes()).hexdigest()
                    != row["decoded_pcm16_sha256"]
                ):
                    raise ValueError("Decoded PCM hash mismatch: " + uid)
                regenerated = trailing_mel(pcm.astype(float) / 32768.0)
                with np.load(cache, allow_pickle=False) as f:
                    mel = f["mel"]
                    valid = f["mel_valid"]
                    ends = f["frame_end_samples"]
                n = row["feature_frames"]
                if (
                    mel.shape != (n, 64)
                    or mel.dtype != np.float32
                    or valid.shape != mel.shape
                    or valid.dtype != np.dtype(bool)
                    or not valid.all()
                    or not np.isfinite(mel).all()
                    or not np.array_equal(ends, np.arange(n) * 160 + 640)
                ):
                    raise ValueError("Mel schema/validity/timing mismatch: " + uid)
                np.testing.assert_allclose(
                    regenerated,
                    mel,
                    rtol=policy["regeneration_rtol"],
                    atol=policy["regeneration_atol"],
                    err_msg=uid,
                )
                error = float(np.max(np.abs(regenerated - mel)))
                max_error = max(max_error, error)
                origins = eligible_origins(valid, history=32, horizons=(4, 8, 16))
                chosen = select_origins(uid, origins, count=8, salt=policy["salt"])
                all_origins[uid] = origins
                targets = np.array(chosen)[:, None] + np.array([4, 8, 16])[None]
                item = dict(
                    uid=uid,
                    speaker=row["speaker"],
                    split=row["split"],
                    source_path=row["path"],
                    cache_path=str(cache),
                    cache_sha256=cache_hashes[str(cache)],
                    source_sha256=row["source_sha256"],
                    decoded_pcm16_sha256=row["decoded_pcm16_sha256"],
                    eligible_origins=len(origins),
                    selected_origins=chosen,
                    recording_weight_within_split=weights[uid],
                    origin_weight_within_split=weights[uid] / 8,
                    selected_target_mask_sha256=hashlib.sha256(
                        valid[targets].tobytes()
                    ).hexdigest(),
                    selected_observed_target_cells=int(valid[targets].sum()),
                    eligible_origin_array_sha256=hashlib.sha256(
                        origins.astype("<i8").tobytes()
                    ).hexdigest(),
                    regeneration_max_absolute_error=error,
                )
                selected.append(item)
                journal.write(json.dumps(item) + "\n")
                if (index + 1) % 100 == 0:
                    print(f"Verified {index + 1}/{len(records)} sources/cache entries", flush=True)
        for path, digest in policy["dependencies"].items():
            if sha(path) != digest:
                raise ValueError("Dependency changed during verification: " + path)
        write(args.output / "selected_origins.json", selected)
        np.savez_compressed(args.output / "eligible_origins.npz", **all_origins)
        counts = {}
        for split in ("train", "validation", "test"):
            sub = [r for r in selected if r["split"] == split]
            counts[split] = dict(
                speakers=len({r["speaker"] for r in sub}),
                recordings=len(sub),
                eligible_origins=sum(r["eligible_origins"] for r in sub),
                selected_origins=sum(len(r["selected_origins"]) for r in sub),
                selected_observed_target_cells=sum(
                    r["selected_observed_target_cells"] for r in sub
                ),
            )
            if not np.isclose(sum(r["recording_weight_within_split"] for r in sub), 1):
                raise ValueError("Split weights do not sum to one")
        bytes_required = counts["train"]["selected_origins"] * (32 * 64 * 2 + 1) * 8
        result = dict(
            completed=True,
            common_inventory_certified=True,
            source_cache_entries_verified=len(selected),
            split_summary=counts,
            regeneration_max_absolute_error=max_error,
            elapsed_seconds=time.monotonic() - started,
            models_fitted=0,
            scientific_execution_ready=False,
            bounded_training_design_float64_GiB=bytes_required / 2**30,
            weighting_required_for_future_training=True,
            scientific_protocol_frozen=False,
            uncertainty_certified=False,
            blockers=[
                "Freeze weighted Ridge/GRU objectives, optimiser and selection budgets",
                "Test solver/resource preflight with bounded design",
                "Freeze paired speaker-level uncertainty and scientific decision rule",
                "Fit one new training-only scaler; do not reuse P10 scaler",
            ],
            P100_authorized=False,
            P960_authorized=False,
            stage2_ready=False,
        )
        write(args.output / "summary.json", result)
        lines = [
            "# RTPE-1 verified common-origin inventory",
            "",
            "Source/cache integrity, decoded waveform hashes, causal frame ends and full mel regeneration passed. Inventory sampling policy was frozen before waveform/cache verification. No model was fitted.",
            "",
            "| Partition | Speakers | Recordings | Eligible origins | Shared sampled origins |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, c in counts.items():
            lines.append(
                f"| {name} | {c['speakers']} | {c['recordings']} | {c['eligible_origins']:,} | {c['selected_origins']:,} |"
            )
        lines += [
            "",
            f"The bounded training design is {bytes_required / 2**30:.3f} GiB in float64, before solver/window/workspace allocations. Full observed mel regeneration maximum absolute error was {max_error:.8g}; frozen tolerance was rtol=atol=1e-6.",
            "",
            "Every recording contributes eight hash-ranked eligible origins. The salt, identities and validity support determine selection; acoustic target values and model scores do not. One inventory serves both arms. All valid target cells remain explicit in the hashed masks. Loss weights give speakers equal weight, recordings equal weight within speakers and sampled origins equal weight within recordings. Both training arms and the new training-only scaler must honour the frozen weighting policy; the original unweighted HistoryRidge interface does not implement this weighted fit yet.",
            "",
            "All-eligible origins are retained in eligible_origins.npz; chosen origins and exact source/cache bindings are retained in selected_origins.json. Overlapping windows are not independent replicates. The eventual endpoint is conditional on this fixed sample; eight origins approximate each recording’s full-origin loss and omit sampling uncertainty unless prospectively addressed. This is previously exposed development material, not untouched confirmation.",
            "",
            "Scientific execution remains unfrozen pending weighted training/solver verification, resource preflight, model budgets, selection rules and paired speaker-level uncertainty/decision rules. No evaluation scores, predictions, model optimisation, behavioural labels, official final-test access or changes to the dropout experiment occurred. Prior outcomes and P100/P960/Stage 2 blocks remain unchanged.",
        ]
        (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
        (args.output / "README.md").write_text(
            "Verified bounded common-origin inventory. See RESULTS.md and frozen sampling_policy.json. Scientific training remains unfrozen.\n"
        )
        files = [p for p in args.output.iterdir() if p.is_file()]
        write(args.output / "manifest.json", {str(p): sha(p) for p in files})
        print(json.dumps(result, indent=2), flush=True)
    except Exception:
        write(
            args.output / "FAILURE.json",
            dict(traceback=traceback.format_exc(), verified_records=len(selected), models_fitted=0),
        )
        raise


if __name__ == "__main__":
    main()
