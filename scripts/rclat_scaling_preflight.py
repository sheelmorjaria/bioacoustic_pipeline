"""New-programme source/exposure audit and staged plan; no scientific training."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import time

import soundfile as sf
from threadpoolctl import threadpool_limits

import rclat_study_b as closed
from bioacoustic_pipeline.representation.rclat_ctc import acoustic_features

OUT = Path("outputs/rclat_scaling_preflight.json")
DATA = Path("data/rclat_scale")
MD5 = "2a93770f6d5c6c964bc36631d331a522"
URL = "https://openslr.trmal.net/resources/12/train-clean-100.tar.gz"


def build():
    if OUT.exists():
        raise RuntimeError("Scaling preflight already exists")
    closed.check()
    population = json.loads(closed.AUDIT.read_text())["records"]
    train = [r for r in population if r["split"] == "train"]
    selected = sorted(train, key=lambda r: (-r["duration_seconds"], r["uid"]))[:2]
    timings = []
    with threadpool_limits(limits=2):
        for r in selected:
            start = time.perf_counter()
            audio, sr = sf.read(r["path"])
            features = acoustic_features(audio, sr)
            elapsed = time.perf_counter() - start
            timings.append(
                {
                    "uid": r["uid"],
                    "audio_seconds": len(audio) / sr,
                    "wall_seconds_read_and_features": elapsed,
                    "audio_seconds_per_wall_second": len(audio) / sr / elapsed,
                    "feature_frames": len(features["mel"]),
                }
            )
    speaker_meta = {}
    for line in Path("data/LibriSpeech/SPEAKERS.TXT").read_text().splitlines():
        if line.strip() and not line.startswith(";"):
            fields = [v.strip() for v in line.split("|")]
            speaker_meta[fields[0]] = fields[2]
    temporal = json.loads(Path("outputs/rclat_temporal_control_preregistration.json").read_text())
    temporal_speakers = sorted({str(r["speaker_id"]) for r in temporal["sources"]})
    test_speakers = sorted({r["speaker"] for r in population if r["split"] == "evaluation"})
    known = {
        "dev-clean": {
            "status": "Previously used for boundary/retrieval and RCLAT studies;not untouched.",
            "documented_temporal_speaker_ids": temporal_speakers,
        },
        "test-clean": {
            "status": "Partially exposed by completed Study B. Do not describe the complete test-clean partition as untouched.",
            "documented_exposed_speakers": test_speakers,
            "scored_utterances": 32,
        },
        "dev-other": {
            "status": "No usage reference found in the inspected current scripts/src/outputs search. Candidate training-side validation only;must acquire and verify before assignment. Not a guarantee of no historical exposure."
        },
        "test-other": {
            "status": "No usage reference found in the inspected current scripts/src/outputs search. Candidate protected final evaluation;no download or scoring in this preflight. Not a global untouched-data certification."
        },
        "train-clean-100": {
            "status": "Training use permitted;previous Mini LibriSpeech training/validation exposure must be disclosed.",
            "known_training_utterances": 136,
            "known_validation_utterances": 16,
            "known_training_speaker_subsets": dict(
                Counter(speaker_meta[r["speaker"]] for r in train)
            ),
        },
        "train-clean-360": {
            "status": "Historical HECB script names an external source path that is absent now;execution history outside this workspace remains unresolved."
        },
    }
    closed.prior.freeze_json(
        OUT,
        {
            "scope": "New programme planning,source exposure and training-only feature throughput. No scientific model fitting.",
            "official_source": "https://www.openslr.org/12",
            "closed_studies_unchanged": True,
            "known_exposure": known,
            "training_only_feature_benchmark": timings,
            "feature_timing_limit": "Two known training files,warm page cache/two CPU threads;not an end-to-end training or storage benchmark.",
            "storage": {
                "available_bytes": shutil.disk_usage(".").free,
                "official_train100_download_size_description": "6.3GB on OpenSLR",
                "dense_float32_100h_cache_estimate_bytes": int(100 * 3600 * 100 * 140 * 5),
                "estimate_basis": "64mel+64spectral+12physical dimensions at100Hz,4-byte values+1-byte masks;uncompressed estimate excludes index/frame timestamps and model storage.",
            },
            "acquisition": {
                "url": URL,
                "expected_official_md5": MD5,
                "download_started": False,
                "rule": "Resume only the official source archive; verify checksum before extraction or cohort selection. No960h or larger acquisition automatically triggered.",
            },
            "study_status": "PROTOCOL_DRAFT_NOT_PREREGISTERED",
            "phases": [
                {
                    "name": "P10",
                    "training_hours": 10,
                    "source": "Deterministic whole-utterance subset of train-clean-100,selected after integrity/native-format/CTC feasibility audit,before fitting.",
                    "purpose": "Non-degenerate learning on speaker-disjoint training-side validation,not a competitive-ASR declaration.",
                    "proposed_gate": "For both matched arms,>=95%nonempty transcripts and speaker-macro CER<50% on prespecified dev-other validation;no final-test access. Exact selection and stopping policy must be frozen before any scientific training.",
                },
                {
                    "name": "P100",
                    "training_hours": 100,
                    "purpose": "Separately preregistered logmel-versus-continuous representation comparison after P10 learning gate.",
                    "primary": "Held-out WER with all speaker outcomes and paired speaker-clustered uncertainty;strong standard acoustic baseline and equal training access.",
                    "candidate_test": "test-other after source/exposure audit;test-clean reported only with prior-exposure disclosure.",
                },
                {
                    "name": "P960",
                    "training_hours": 960,
                    "purpose": "New scaling experiment only after P100 review;no acquisition/training currently authorized by an automatic gate.",
                },
            ],
            "model_plan": {
                "reference": "Matched offline Conformer-style encoders;two XPUs used for independent arms initially.",
                "current_engineering_model": "1.65M parameters,4blocks,width128,FP32;not yet a validated competitive-ASR recipe.",
                "scientific_training_policy": "Exact depth,width,dropout,learning-rate schedule,dynamic batching,epoch cap,validation selection and seeds remain unfrozen until corpus support and engineering review. No final-test-driven tuning.",
                "auxiliary_losses": "None initially. Temporal objectives require a separately specified ablation.",
                "numerical_policy": "Keep verified differentiable CPU CTC,neural computation on XPU;use attention execution path that passes training gradient checks. No mixed-precision or distributed-scaling claim yet.",
            },
            "remaining_requirements": [
                "Resolve matched engineering baseline numerical failure and archive every attempted variant.",
                "Acquire/checksum train-clean-100 and audit complete waveform/transcript support.",
                "Verify dev-other/test-other speaker identities and historical exposure before protected assignment.",
                "Freeze exact P10 cohort,architecture,optimization,validation/stopping and learning gate before scientific training.",
                "Benchmark realistic dynamic batches and end-to-end extraction/cache/I/O before projecting long-run wall time.",
            ],
            "semantic_reversibility_demonstrated": False,
            "request_interpretation_wording_ready": False,
            "stage2_ready": False,
        },
    )
    print("Scaling preflight saved;no large archive download or scientific training performed.")


def acquire():
    if not OUT.exists():
        raise RuntimeError("Source/exposure preflight required first")
    for arm in ["mel", "combined"]:
        p = Path(f"outputs/rclat_scale_engineering_v3_{arm}.json")
        if not p.exists() or not json.loads(p.read_text())["memorisation_gate_passed"]:
            raise RuntimeError("Both revised engineering gates required before acquisition")
    DATA.mkdir(exist_ok=True)
    path = DATA / "train-clean-100.tar.gz"
    subprocess.run(
        ["curl", "-fL", "-C", "-", "--max-time", "600", URL, "-o", str(path)], check=True
    )
    import hashlib

    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "md5").hexdigest()
    if digest != MD5:
        raise ValueError("Official archive integrity failed;no extraction permitted")
    receipt = DATA / "archive_integrity.json"
    closed.prior.freeze_json(
        receipt, {"path": str(path), "md5": digest, "sha256": closed.prior.sha(path), "source": URL}
    )
    print("Official100h archive acquired and verified;no extraction,partitioning or model fitting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["build", "acquire100"])
    args = parser.parse_args()
    build() if args.action == "build" else acquire()
