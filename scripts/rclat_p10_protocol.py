"""Freeze acquisition, deterministic cohorts and P10 rules; never fit or score models.

Only checksum-verified official archives are extracted. No final test is opened.
"""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import tarfile
import time

import soundfile as sf

from bioacoustic_pipeline.representation.rclat_ctc import encode_text

ROOT = Path("data/rclat_scale")
OUT = Path("outputs/rclat_p10")
ARCHIVES = {
    "train-clean-100": "2a93770f6d5c6c964bc36631d331a522",
    "dev-other": "c8d0bcc9cca99d4f8b62fcc847357931",
}
SALT = "RCLAT-P10-v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def software_versions():
    return {
        "python": platform.python_version(),
        **{
            name: importlib.metadata.version(name)
            for name in ["torch", "torchaudio", "numpy", "scipy", "soundfile"]
        },
    }


def freeze(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def select_ten_hours(rows, budget_samples=10 * 3600 * 16000):
    """Hash-ranked whole utterances; skip those exceeding remaining sample budget."""
    ranked = sorted(rows, key=lambda r: hashlib.sha256(f"{SALT}:{r['uid']}".encode()).hexdigest())
    chosen, total = [], 0
    for row in ranked:
        if total + row["samples"] <= budget_samples:
            chosen.append(row)
            total += row["samples"]
    return chosen


def ctc_support(samples, transcript):
    tokens = encode_text(transcript)
    frames = max(0, 1 + (samples - 640) // 160)
    output_frames = (frames + 3) // 4
    minimum = len(tokens) + int((tokens[1:] == tokens[:-1]).sum())
    return output_frames >= minimum and len(tokens) > 0, frames, output_frames, minimum


def acquisition(subsets=None):
    receipts = {}
    for subset, md5 in (ARCHIVES if subsets is None else subsets).items():
        archive = ROOT / f"{subset}.tar.gz"
        with archive.open("rb") as stream:
            actual = hashlib.file_digest(stream, "md5").hexdigest()
        if actual != md5:
            raise ValueError(f"Official MD5 mismatch: {subset}; extraction prohibited")
        receipts[subset] = {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "md5": actual,
            "sha256": sha(archive),
            "url": f"https://openslr.trmal.net/resources/12/{subset}.tar.gz",
        }
        dest = ROOT / "source"
        marker = dest / f"{subset}.verified.json"
        if marker.exists():
            if json.loads(marker.read_text())["sha256"] != receipts[subset]["sha256"]:
                raise ValueError("Extraction marker conflicts with archive")
            continue
        # Write regular source FLAC/transcript files only; no archive-supplied paths/links.
        count = 0
        with tarfile.open(archive, "r:gz") as source:
            for member in source:
                parts = Path(member.name).parts
                if not member.isfile() or len(parts) != 5 or parts[:2] != ("LibriSpeech", subset):
                    continue
                if not parts[2].isdigit() or not parts[3].isdigit():
                    raise ValueError("Unexpected source path")
                if not (parts[4].endswith(".flac") or parts[4].endswith(".trans.txt")):
                    continue
                target = dest.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                count += 1
        freeze(marker, {**receipts[subset], "regular_files_extracted": count})
    return receipts


def inventory(subset):
    base = ROOT / "source" / "LibriSpeech" / subset
    words = {}
    for path in sorted(base.rglob("*.trans.txt")):
        for line in path.read_text().splitlines():
            uid, text = line.split(" ", 1)
            if uid in words:
                raise ValueError("Duplicate transcript identifier")
            words[uid] = text
    rows, exclusions = [], []
    for path in sorted(base.rglob("*.flac")):
        uid = path.stem
        if uid.split("-")[:2] != [path.parent.parent.name, path.parent.name]:
            raise ValueError("Recording identifier conflicts with source directory")
        info = sf.info(path)
        if (
            uid not in words
            or info.samplerate != 16000
            or info.channels != 1
            or info.subtype != "PCM_16"
        ):
            raise ValueError(f"Invalid native source or missing transcript: {uid}")
        feasible, frames, output_frames, minimum = ctc_support(info.frames, words[uid])
        row = {
            "uid": uid,
            "speaker": uid.split("-")[0],
            "path": str(path),
            "transcript": words.pop(uid),
            "samples": info.frames,
            "duration_seconds": info.frames / 16000,
            "feature_frames": frames,
            "ctc_frames": output_frames,
            "ctc_minimum_frames": minimum,
        }
        (rows if feasible else exclusions).append(row)
    if words or not rows:
        raise ValueError("Unpaired transcripts or empty inventory")
    return rows, exclusions


def verify_selected(rows):
    hashes = {}
    for row in rows:
        audio, rate = sf.read(row["path"], dtype="int16")
        if rate != 16000 or audio.ndim != 1 or len(audio) != row["samples"]:
            raise ValueError("Decoded waveform disagrees with inventory")
        digest = hashlib.sha256(audio.astype("<i2", copy=False).tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f"Decoded duplicate: {row['uid']} and {hashes[digest]}")
        hashes[digest] = row["uid"]
        row["decoded_pcm16_sha256"] = digest
        row["source_sha256"] = sha(row["path"])


def summarize(rows):
    return {
        "utterances": len(rows),
        "speakers": len({r["speaker"] for r in rows}),
        "hours": sum(r["samples"] for r in rows) / 16000 / 3600,
        "speaker_utterances": dict(sorted(Counter(r["speaker"] for r in rows).items())),
    }


def build():
    if (OUT / "preregistration.json").exists():
        raise RuntimeError("P10 already frozen; do not replace")
    receipts = acquisition()
    train_all, train_bad = inventory("train-clean-100")
    validation, validation_bad = inventory("dev-other")
    train = select_ten_hours(train_all)
    if sum(r["samples"] for r in train) / 16000 < 9.99 * 3600:
        raise ValueError("Insufficient 10-hour cohort")
    if {r["speaker"] for r in train_all} & {r["speaker"] for r in validation}:
        raise ValueError("Official training/validation speaker overlap")
    # No adaptive replacement of invalid validation examples; abstain instead.
    if validation_bad:
        raise ValueError("CTC-invalid validation sources; revised protocol needed before fitting")
    verify_selected(train + validation)
    old_path = Path("outputs/rclat_study_b_source_audit.json")
    if not old_path.exists():
        import rclat_study_b as closed

        old_path = closed.AUDIT
    old = json.loads(old_path.read_text())
    selected_ids = {r["uid"] for r in train}
    reused = [r for r in old["records"] if r["uid"] in selected_ids]
    exposure = {
        "prior_study_b_audit": str(old_path),
        "prior_study_b_reused_utterances": [
            {"uid": r["uid"], "prior_split": r["split"]} for r in reused
        ],
        "notice": "Known prior Mini LibriSpeech/engineering exposure; this is development evidence. dev-clean/test-clean are not protected here. No final test acquired or scored.",
    }
    audit = {
        "archives": receipts,
        "training_pool": summarize(train_all),
        "training": summarize(train),
        "validation": summarize(validation),
        "training_exclusions": train_bad,
        "validation_exclusions": validation_bad,
        "speaker_overlap": [],
        "decoded_duplicates": 0,
        "exposure": exposure,
        "selected_training": train,
        "selected_validation": validation,
    }
    freeze(OUT / "cohort.json", audit)
    deps = [
        Path(__file__),
        OUT / "cohort.json",
        Path("scripts/rclat_p10_train.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_scale_v3.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_ctc.py"),
        Path("src/bioacoustic_pipeline/diagnostics/frozen_acoustic_transfer.py"),
        Path("src/bioacoustic_pipeline/diagnostics/macde_data.py"),
        Path("src/bioacoustic_pipeline/diagnostics/acoustic_derivatives.py"),
        Path("src/bioacoustic_pipeline/diagnostics/acoustic_fingerprint/_dsp.py"),
        Path("src/bioacoustic_pipeline/diagnostics/acoustic_fingerprint/physics.py"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat.py"),
        Path("tests/test_rclat_p10_protocol.py"),
        Path("tests/test_rclat_p10_training_policy.py"),
        ROOT / "archive_integrity.json",
        ROOT / "source/train-clean-100.verified.json",
        ROOT / "source/dev-other.verified.json",
    ]
    spec = {
        "status": "LOCALLY_FROZEN_P10_LEARNING_VALIDATION_NOT_EXTERNAL_REGISTRATION",
        "software": software_versions(),
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Development learning gate; no representation-superiority or semantic claim.",
        "selection": {
            "salt": SALT,
            "rule": "SHA256(salt:uid) ascending; whole utterances greedily included while cumulative samples<=576000000. Native16k mono; nonempty representable transcript; 4x CTC feasibility. Full dev-other validation, no adaptive validation omissions.",
        },
        "frontends": {
            "mel": {"dimensions": 64, "device": "xpu:0"},
            "combined": {"dimensions": 76, "device": "xpu:1"},
        },
        "features": "Original acoustic_features:40ms/10ms,64mel vs64spectral+12physical;validity masks;no deletion of unsupported frames. Fit each scaler on selected training sources only. No denoising/augmentation/external LM/pretrained weights/word boundaries.",
        "architecture": {
            "module": "rclat_scale_v3.ScaleCTC",
            "width": 256,
            "layers": 12,
            "heads": 4,
            "expansion": 1024,
            "kernel": 15,
            "dropout": 0.0,
            "attention": "Explicit FP32 qkv/matmul/softmax, finite padding bias;offline bidirectional,not streaming",
            "subsampling": 4,
        },
        "initialization": "Seed0 for Python/NumPy/torch/XPU;common encoder/head initialized identically as ScaleCTC;arm-specific input projection dimension differs. No parameter reuse from closed studies.",
        "training": {
            "epochs": 30,
            "batch_utterances": 4,
            "order": "Each epoch sort training UIDs by SHA256('RCLAT-P10-v1:epoch:<epoch>:<uid>');partition consecutive batches of4,retain final short batch;identical across arms.",
            "optimizer": "AdamW betas(0.9,0.999),eps1e-8,weight_decay0.01",
            "learning_rate": "linear warmup from1e-6 to1e-3 over first500 optimizer steps,then cosine to1e-5 at final step;no restarts",
            "gradient_clip_norm": 5.0,
            "precision": "FP32;no autocast/scaling",
            "ctc": "Strict deterministic differentiable CPU FP32 CTC;blank0;mean loss;no zero_infinity masking;neural graph on assigned XPU",
            "early_stopping": False,
            "resume": "Only exact checkpoint+optimizer+RNG+step+source/code hash recovery;never restart seed or change settings based on outcome",
        },
        "validation": {
            "when": "After each completed epoch,eval/no_grad,greedy CTC,no LM",
            "selection": "Per arm minimum speaker-macro CER over epochs1..30,ties earliest;save every epoch checkpoint and report final epoch too",
            "metrics": "CER without spaces,WER,nonempty utterance fraction;each speaker equally weighted for macro CER/WER;report speaker vector and pooled scores",
        },
        "learning_gate": {
            "each_arm_nonempty_fraction_at_selected_checkpoint_minimum": 0.95,
            "each_arm_speaker_macro_CER_strict_maximum": 0.5,
            "both_arms_required": True,
            "interpretation": "Validation-selected developmental learning evidence only;not confirmatory generalisation or superiority. No automatic100h/960h progression.",
        },
        "abstention": "Missing/corrupt sources,changed source/code hashes,nonfinite features/logits/loss/gradients,CTC infeasibility,OOM,or unsupported device:halt affected arm and preserve failure;no silent dropping,batch shrink,precision switch,epoch extension,seed retry or test access. If either arm incomplete or fails thresholds,the paired gate is not passed.",
        "final_test": "None in P10;test-other remains unopened. Historical exposure elsewhere unresolved;no global untouched claim.",
        "runtime_record": "Before execution record Python/torch/torchaudio/NumPy/runtime/device versions,determinism flags,actual source+cache hashes and device bindings;strict deterministic algorithms. Cache extraction/I/O included in end-to-end timings separately from neural steps.",
        "dependencies": {str(p): sha(p) for p in deps},
        "training_started": False,
        "stage2_ready": False,
    }
    freeze(OUT / "preregistration.json", spec)
    print(
        json.dumps(
            {
                "training": summarize(train),
                "validation": summarize(validation),
                "training_started": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["build"])
    parser.parse_args()
    build()
