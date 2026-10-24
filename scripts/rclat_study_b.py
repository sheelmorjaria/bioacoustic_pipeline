#!/usr/bin/env python3
"""Bounded trainable English CTC pilot;fresh training and evaluation sources."""

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from threadpoolctl import threadpool_limits

import rclat_study_a as prior
from bioacoustic_pipeline.representation.rclat_ctc import (
    CausalCTC,
    LABELS,
    acoustic_features,
    ctc_objective,
    encode_text,
    decode_logits,
    fit_scaler,
)
from bioacoustic_pipeline.diagnostics.frozen_acoustic_transfer import frozen_scale
from bioacoustic_pipeline.diagnostics.rclat import normalize_text, text_errors
from bioacoustic_pipeline.utils.devices import resolve_device

ROOT = prior.ROOT
FOLDER = ROOT / "data/rclat_study_b"
PREFIX = ROOT / "outputs/rclat_study_b"
AUDIT = Path(str(PREFIX) + "_source_audit.json")
FEATURES = Path(str(PREFIX) + "_feature_audit.json")
SPEC = Path(str(PREFIX) + "_preregistration.json")
CACHE = ROOT / "cache/rclat_study_b"
ARMS = {
    "B0": ("mel", 64),
    "B1": ("physical", 12),
    "B2": ("combined", 76),
    "B3": ("pretrained", 768),
}


def rank(value):
    return hashlib.sha256(("rclat-study-b-v1:0:" + value).encode()).hexdigest()


def audit():
    if AUDIT.exists():
        raise RuntimeError("Source audit exists")
    inventory, integrity = [], []
    subsets = {}
    for line in (ROOT / "data/LibriSpeech/SPEAKERS.TXT").read_text().splitlines():
        if line.strip() and not line.startswith(";"):
            fields = [f.strip() for f in line.split("|")]
            subsets[fields[0]] = fields[2]
    for name, checksum in [
        ("train-clean-5", "mini_md5sum.txt"),
        ("test-clean", "librispeech_md5sum.txt"),
    ]:
        path = FOLDER / (name + ".tar.gz")
        expected = {
            Path(v.split()[-1].lstrip("*")).name: v.split()[0]
            for v in (FOLDER / checksum).read_text().splitlines()
            if v.strip()
        }
        with path.open("rb") as f:
            md5 = hashlib.file_digest(f, "md5").hexdigest()
        if md5 != expected[path.name]:
            raise ValueError("Official archive checksum mismatch")
        integrity.append(
            {"path": str(path.relative_to(ROOT)), "md5": md5, "sha256": prior.sha(path)}
        )
        transcripts = {}
        with tarfile.open(path) as archive:
            for member in archive:
                if member.isfile() and member.name.endswith(".trans.txt"):
                    for line in archive.extractfile(member).read().decode().splitlines():
                        uid, text = line.split(" ", 1)
                        if uid in transcripts:
                            raise ValueError("Repeated transcript identifier")
                        transcripts[uid] = normalize_text(text)
        with tarfile.open(path) as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".flac"):
                    continue
                raw = archive.extractfile(member).read()
                info = sf.info(io.BytesIO(raw))
                uid = Path(member.name).stem
                speaker = uid.split("-")[0]
                if name == "train-clean-5" and not subsets.get(speaker, "").startswith("train-"):
                    raise ValueError(
                        "Mini training source is not assigned to original training partition"
                    )
                if name == "test-clean" and subsets.get(speaker) != "test-clean":
                    raise ValueError("Test speaker provenance conflict")
                if (
                    info.samplerate == 16000
                    and info.channels == 1
                    and info.subtype == "PCM_16"
                    and 2 <= info.duration <= 12
                ):
                    encode_text(transcripts[uid])
                    inventory.append(
                        {
                            "uid": uid,
                            "speaker": speaker,
                            "transcript": transcripts[uid],
                            "archive": str(path.relative_to(ROOT)),
                            "member": member.name,
                            "flac_sha256": hashlib.sha256(raw).hexdigest(),
                            "samples": info.frames,
                            "sample_rate": 16000,
                            "duration_seconds": info.duration,
                            "source": name,
                        }
                    )
    groups = defaultdict(list)
    for item in inventory:
        groups[(item["source"], item["speaker"])].append(item)
    qualified = sorted(
        [s for source, s in groups if source == "train-clean-5" and len(groups[(source, s)]) >= 8],
        key=rank,
    )
    if len(qualified) < 12:
        raise ValueError("Insufficient support for training and validation speakers")
    training = qualified[: min(24, len(qualified) - 4)]
    validation = qualified[len(training) : len(training) + 4]
    testing = sorted(
        [s for source, s in groups if source == "test-clean" and len(groups[(source, s)]) >= 4],
        key=rank,
    )[:8]
    if len(testing) != 8:
        raise ValueError("Insufficient fresh evaluation speakers")
    if set(training) & set(validation) or (set(training) | set(validation)) & set(testing):
        raise ValueError("Speaker leakage")
    old_speakers = {s["speaker_id"] for s in json.loads(prior.SPEC.read_text())["sources"]}
    if (set(training) | set(validation) | set(testing)) & old_speakers:
        raise ValueError("Prior pilot speaker reuse")
    selected = []
    for split, speakers, source, n in [
        ("train", training, "train-clean-5", 8),
        ("validation", validation, "train-clean-5", 4),
        ("evaluation", testing, "test-clean", 4),
    ]:
        for speaker in speakers:
            for item in sorted(groups[(source, speaker)], key=lambda r: rank(r["uid"]))[:n]:
                selected.append(
                    {**item, "split": split, "path": f"data/rclat_study_b/audio/{item['uid']}.flac"}
                )
    if len({i["uid"] for i in selected}) != len(selected):
        raise ValueError("Repeated source recording")
    audio_dir = FOLDER / "audio"
    if audio_dir.exists():
        raise RuntimeError("Selected audio directory already exists")
    audio_dir.mkdir()
    selected_map = {i["member"]: i for i in selected}
    for source in integrity:
        with tarfile.open(ROOT / source["path"]) as archive:
            for member in archive:
                if member.name in selected_map:
                    record = selected_map[member.name]
                    raw = archive.extractfile(member).read()
                    if hashlib.sha256(raw).hexdigest() != record["flac_sha256"]:
                        raise ValueError("Source member changed")
                    (ROOT / record["path"]).write_bytes(raw)
    prior.freeze_json(
        AUDIT,
        {
            "study": "RCLAT Study B bounded English CTC source population",
            "records": selected,
            "archive_integrity": integrity,
            "selection": "Metadata-only fixedhash ranking of speakers and utterances2to12s. Up to24training speakers8clips each,four separate validation speakers4clips each,eight fresh test-clean speakers4clips each. Training count bounded by source support,minimum8.",
            "counts": dict(Counter(i["split"] for i in selected)),
            "speaker_sets": {"train": training, "validation": validation, "evaluation": testing},
            "B3_exposure": "Pretrained ASR encoder saw original train960including mini training/validation audio and labels. Test-clean speakers not in its documented training set. Validation is diagnostic only,no epoch/model selection.",
            "source_references": ["https://www.openslr.org/31", "https://www.openslr.org/12"],
            "official_partition_claim": "Custom bounded regression pilot,not a full official benchmark;fresh evaluation drawn from test-clean,no prior temporal-study sources used.",
            "training_performed": False,
            "stage2_ready": False,
        },
    )
    print("Source audit", dict(Counter(i["split"] for i in selected)), flush=True)


def prepare(device):
    if FEATURES.exists() or CACHE.exists():
        raise RuntimeError("Feature population exists")
    population = json.loads(AUDIT.read_text())
    for source in population["archive_integrity"]:
        if prior.sha(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("Source archive changed")
    resolved = resolve_device(device)
    checkpoint = prior.CHECKPOINT
    speech = (
        prior.BUNDLE.get_model(dl_kwargs={"model_dir": str(checkpoint.parent), "progress": False})
        .eval()
        .to(resolved)
    )
    for p in speech.parameters():
        p.requires_grad_(False)
    CACHE.mkdir(parents=True)
    records = []
    for i, item in enumerate(population["records"], 1):
        if prior.sha(ROOT / item["path"]) != item["flac_sha256"]:
            raise ValueError("Source changed")
        audio, sr = sf.read(ROOT / item["path"])
        feature = acoustic_features(audio, sr)
        with torch.inference_mode():
            hidden, _ = speech.extract_features(
                torch.from_numpy(audio.astype(np.float32))[None].to(resolved), num_layers=12
            )
            embedding = hidden[-1][0].float().cpu().numpy()
        if not np.isfinite(embedding).all():
            raise ValueError("Invalid pretrained features")
        feature["pretrained"] = embedding
        feature["pretrained_valid"] = np.ones_like(embedding, bool)
        target = encode_text(item["transcript"])
        minimum = len(target) + int((target[1:] == target[:-1]).sum())
        if min(len(feature["mel"]), len(embedding)) < minimum:
            raise ValueError(
                "An arm cannot align the same target;stop rather than excluding adaptively"
            )
        path = CACHE / (item["uid"] + ".npz")
        np.savez_compressed(path, **feature)
        records.append(
            {"uid": item["uid"], "path": str(path.relative_to(ROOT)), "sha256": prior.sha(path)}
        )
        if i % 20 == 0:
            print("Prepared", i, "utterances", flush=True)
    prior.freeze_json(
        FEATURES,
        {
            "source_audit_sha256": prior.sha(AUDIT),
            "records": records,
            "pretrained_checkpoint_sha256": prior.sha(checkpoint),
            "pretrained_feature_device": resolved,
            "frontends": "B0logmel64,40ms/10ms causal;B1frozen physical12withsupportmask;B2spectral64+physical12withmask,causal;B3frozen supervised Wav2Vec2 final hidden768,offline20msstride.",
            "silence": "Frames retained;unsupported physical/spectral features masked,never delete time to concatenate speech.",
            "training_performed": False,
        },
    )
    print("Prepared", len(records), "sources;no CTC training yet", flush=True)


def dependencies():
    paths = [
        Path(__file__).resolve(),
        AUDIT,
        FEATURES,
        ROOT / "src/bioacoustic_pipeline/representation/rclat_ctc.py",
        ROOT / "src/bioacoustic_pipeline/diagnostics/frozen_acoustic_transfer.py",
        ROOT / "tests/test_rclat_ctc.py",
        ROOT / "src/bioacoustic_pipeline/diagnostics/macde_data.py",
        ROOT / "src/bioacoustic_pipeline/diagnostics/rclat.py",
        ROOT / "src/bioacoustic_pipeline/utils/devices.py",
        prior.CHECKPOINT,
        ROOT / "scripts/rclat_study_a.py",
        ROOT / "scripts/rclat_transcribe.py",
        ROOT / "scripts/rclat_ctc_numerical_check.py",
        ROOT / "outputs/rclat_ctc_numerical_check.json",
        ROOT / "outputs/rclat_study_b_pretraining_verification.json",
        ROOT / "outputs/rclat_ctc_xpu_smoke.json",
        prior.RESULT,
        ROOT / "outputs/rclat_temporal_control_results.json",
        ROOT / "outputs/bfsongrepo_multibird_results.json",
    ]
    return {str(p.relative_to(ROOT)): prior.sha(p) for p in paths}


def freeze():
    if SPEC.exists():
        raise RuntimeError("Study already frozen")
    population = json.loads(AUDIT.read_text())
    features = json.loads(FEATURES.read_text())
    if features["source_audit_sha256"] != prior.sha(AUDIT):
        raise ValueError("Population mismatch")
    for record in features["records"]:
        if prior.sha(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("Feature cache changed")
    prior.freeze_json(
        SPEC,
        {
            "study": "RCLAT Study B bounded trainable English CTC v1",
            "dependencies": dependencies(),
            "arms": ARMS,
            "population": population["counts"],
            "speaker_sets": population["speaker_sets"],
            "architecture": "64-D token projection,identical initialized causalGRU64and29-characterCTChead across arms. Frontend trainable parameter counts differ and are reported. No manual word or phoneme boundaries.",
            "B3_limit": "Only B0/B1/B2 are causal waveform-to-logit systems. B3frozen pretrained encoder is full-context and supervised-pretrained with vastly more data;offline reference,not matched-pretraining or streaming control.",
            "training": {
                "epochs": 24,
                "batch": 8,
                "seed": 0,
                "optimizer": "Adam lr0.001 clip5,FP32",
                "policy": "All training utterances once/epoch in common deterministic order. Fixed24epochs,no validation/tuning/early stopping. Freshfromscratch acoustic models B0/B1/B2;B3frozen pretrainedencoder plus newly initialized/trainableCTCnetwork.",
                "compute": "Two Intel Arc Pro B60XPUs,one independent arm perdevice. Strict deterministic neural ops;differentiable CPUCTC loss because nativeXPU CTCbackward lacks deterministic implementation.",
                "preprocessing": "Per-arm input mean/std/support fit on training speakers only. Validation/evaluation neverfit normalizers or weights.",
                "state": "Hidden state reset for every utterance;packed lengths protect padding. Unsupported frames retained as masked observations.",
            },
            "primary": "Evaluation-speaker-macro WER(B0)-WER(B2),positive favors combined representation. Report all8speaker effects and pooled count WER/CER plus exact utterance counts.",
            "secondary": "B1andB3descriptive;validation diagnostics only;training CTC loss traces and mechanicalcausality tests. No word-level latency or information-availability claims until independenttimingreviewready.",
            "primary_success_rule": "Primary superiority only if the lower endpoint of the prespecified paired speaker-bootstrap 95% interval for B0-minus-B2 macro WER is strictly positive. Report all effects regardless. No absolute competitive-ASR claim from this bounded pilot.",
            "device_assignment": {"B0": "xpu:0", "B1": "xpu:1", "B2": "xpu:0", "B3": "xpu:1"},
            "uncertainty": "2000pairedspeakerbootstrap draws seed0,conditional selected8speakers;no broad-language performance claim.",
            "recognition_gate": "Engineering pilot can demonstrate successful training while remaining noncompetitive. No attribution of previous frozenASR2.11%WER to these newly trained networks.",
            "stop": "One fixedtrainingrun perarm,no hyperparameter/model changes or reruns after evaluation;stop on nonfinite losses,provenance,CTCalignment or checkpointfailures.",
            "software": {
                "torch": torch.__version__,
                "torchaudio": torchaudio.__version__,
                "numpy": np.__version__,
            },
            "stage2_ready": False,
            "semantic_reversibility_tested": False,
        },
    )
    print("Preregistered", prior.sha(SPEC), flush=True)


def check():
    spec = json.loads(SPEC.read_text())
    actual_software = {
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "numpy": np.__version__,
    }
    if actual_software != spec["software"]:
        raise ValueError("Frozen software version mismatch")
    for path, digest in spec["dependencies"].items():
        if prior.sha(ROOT / path) != digest:
            raise ValueError("Frozen dependency changed: " + path)
    for record in json.loads(FEATURES.read_text())["records"]:
        if prior.sha(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("Frozen feature cache changed")
    return spec


def sequence(cache, arm):
    with np.load(cache) as z:
        if arm == "B2":
            return np.concatenate([z["spectral"], z["physical"]], 1), np.concatenate(
                [z["spectral_valid"], z["physical_valid"]], 1
            )
        key = {"B0": "mel", "B1": "physical", "B3": "pretrained"}[arm]
        return z[key], z[key + "_valid"]


def batch(data, indices, device):
    lengths = torch.tensor([len(data[i][0]) for i in indices])
    values = pad_sequence([torch.from_numpy(data[i][0]) for i in indices], batch_first=True).to(
        device
    )
    masks = pad_sequence([torch.from_numpy(data[i][1]) for i in indices], batch_first=True).to(
        device
    )
    return values, masks, lengths


def train(arm, device):
    check()
    marker = Path(str(PREFIX) + "_" + arm + "_RUN_STARTED.json")
    result_path = Path(str(PREFIX) + "_" + arm + "_results.json")
    if marker.exists() or result_path.exists():
        raise RuntimeError("One-run arm stop rule")
    resolved = resolve_device(device)
    if resolved != json.loads(SPEC.read_text())["device_assignment"][arm]:
        raise ValueError("Frozen device assignment mismatch")
    if not resolved.startswith("xpu:"):
        raise ValueError("This preregisteredpilot requires anexplicitXPU")
    prior.freeze_json(marker, {"spec_sha256": prior.sha(SPEC), "device": resolved})
    population = json.loads(AUDIT.read_text())["records"]
    caches = {r["uid"]: ROOT / r["path"] for r in json.loads(FEATURES.read_text())["records"]}
    fit = [i for i, r in enumerate(population) if r["split"] == "train"]
    arrays = [sequence(caches[r["uid"]], arm) for r in population]
    scaler = fit_scaler(arrays[i] for i in fit)
    data = [frozen_scale(x, mask, scaler) for x, mask in arrays]
    del arrays
    targets = [encode_text(r["transcript"]) for r in population]
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)
    kind, width = ARMS[arm]
    model = CausalCTC(kind, width).to(resolved)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    rng = np.random.default_rng(0)
    trace = []
    for epoch in range(1, 25):
        model.train()
        losses = []
        order = rng.permutation(fit)
        for start in range(0, len(order), 8):
            indices = order[start : start + 8]
            values, masks, lengths = batch(data, indices, resolved)
            optimizer.zero_grad()
            logits, _ = model(values, masks, lengths)
            loss = ctc_objective(logits, lengths, [targets[i] for i in indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        trace.append(float(np.mean(losses)))
        if epoch % 4 == 0:
            print(arm, resolved, "epoch", epoch, "loss", trace[-1], flush=True)
    model.eval()
    checkpoint = ROOT / f"checkpoints/rclat_study_b_{arm}.pt"
    if checkpoint.exists():
        raise RuntimeError("Checkpoint already exists")
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "arm": arm,
        "model_arm": kind,
        "input_dim": width,
        "width": 64,
        "labels": list(LABELS),
        "scaler": scaler,
        "preregistration_sha256": prior.sha(SPEC),
        "causal_audio_encoder": arm != "B3",
        "pretrained_source_sha256": prior.sha(prior.CHECKPOINT) if arm == "B3" else None,
        "stage2_ready": False,
    }
    torch.save(payload, checkpoint)
    reload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    verifier = CausalCTC(kind, width).to(resolved)
    verifier.load_state_dict(reload["state_dict"])
    verifier.eval()
    evaluations = {}
    for split in ["validation", "evaluation"]:
        rows = []
        indices = [i for i, r in enumerate(population) if r["split"] == split]
        with torch.inference_mode():
            for i in indices:
                values, masks, lengths = batch(data, [i], resolved)
                logits, _ = model(values, masks, lengths)
                repeated, _ = verifier(values, masks, lengths)
                torch.testing.assert_close(logits, repeated, atol=1e-5, rtol=1e-5)
                decoded = decode_logits(logits[0].cpu(), lengths[0])
                rows.append(
                    {
                        "uid": population[i]["uid"],
                        "speaker": population[i]["speaker"],
                        "reference": population[i]["transcript"],
                        "prediction": decoded,
                        "metrics": text_errors(population[i]["transcript"], decoded),
                    }
                )
        per_speaker = {}
        for speaker in sorted({r["speaker"] for r in rows}):
            subset = [r for r in rows if r["speaker"] == speaker]
            per_speaker[speaker] = {
                "wer": sum(r["metrics"]["word_errors"] for r in subset)
                / sum(r["metrics"]["reference_words"] for r in subset),
                "cer": sum(r["metrics"]["character_errors"] for r in subset)
                / sum(r["metrics"]["reference_characters"] for r in subset),
            }
        evaluations[split] = {
            "rows": rows,
            "speaker_macro_wer": float(np.mean([v["wer"] for v in per_speaker.values()])),
            "pooled_wer": sum(r["metrics"]["word_errors"] for r in rows)
            / sum(r["metrics"]["reference_words"] for r in rows),
            "pooled_cer": sum(r["metrics"]["character_errors"] for r in rows)
            / sum(r["metrics"]["reference_characters"] for r in rows),
            "per_speaker": per_speaker,
            "exact_utterances": sum(r["metrics"]["utterance_exact"] for r in rows),
        }
    check()
    prior.freeze_json(
        result_path,
        {
            "arm": arm,
            "spec_sha256": prior.sha(SPEC),
            "device": resolved,
            "ctc_loss_device": "cpu_deterministic",
            "training_loss_trace": trace,
            "evaluations": evaluations,
            "parameters": sum(p.numel() for p in model.parameters()),
            "shared_decoder_parameters": sum(
                p.numel() for module in [model.temporal, model.head] for p in module.parameters()
            ),
            "checkpoint": {
                "path": str(checkpoint.relative_to(ROOT)),
                "sha256": prior.sha(checkpoint),
            },
            "checkpoint_prediction_reload_passed": True,
            "causal_audio_encoder": arm != "B3",
            "stage2_ready": False,
        },
    )
    print(arm, "complete,evaluationWER", evaluations["evaluation"]["pooled_wer"], flush=True)


def report():
    spec = check()
    results = {
        a: json.loads(Path(str(PREFIX) + "_" + a + "_results.json").read_text()) for a in ARMS
    }
    for value in results.values():
        if value["spec_sha256"] != prior.sha(SPEC):
            raise ValueError("Arm protocol mismatch")
    speakers = sorted(results["B0"]["evaluations"]["evaluation"]["per_speaker"])
    effects = np.array(
        [
            results["B0"]["evaluations"]["evaluation"]["per_speaker"][s]["wer"]
            - results["B2"]["evaluations"]["evaluation"]["per_speaker"][s]["wer"]
            for s in speakers
        ]
    )
    rng = np.random.default_rng(0)
    draws = effects[rng.integers(0, len(effects), (2000, len(effects)))].mean(1)
    value = {
        "study": spec["study"],
        "spec_sha256": prior.sha(SPEC),
        "arms": results,
        "primary_B0_minus_B2_macro_wer": {
            "point": float(effects.mean()),
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "per_speaker": dict(zip(speakers, effects.tolist())),
        },
        "stage2_ready": False,
        "semantic_recovery_tested": False,
        "timing_claims_validated": False,
        "scope": "Small English CTCengineeringpilot;not a competitiveASR benchmark,multilingual system or semantic decoder.",
    }
    prior.freeze_json(Path(str(PREFIX) + "_results.json"), value)
    print(
        json.dumps(
            {a: r["evaluations"]["evaluation"]["pooled_wer"] for a, r in results.items()}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["audit", "prepare", "freeze", "train", "report"])
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--arm", choices=list(ARMS))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        if args.action == "prepare":
            prepare(args.device)
        elif args.action == "train":
            if not args.arm:
                parser.error("--arm required")
            train(args.arm, args.device)
        else:
            {"audit": audit, "freeze": freeze, "report": report}[args.action]()
