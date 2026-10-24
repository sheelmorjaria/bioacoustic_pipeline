"""Prepare a proposed request study and blinded wording review; never fit models."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import soundfile as sf

from bioacoustic_pipeline.ingestion.request_protocol import proposed_speaker_folds, wording_gate

OUT = Path("outputs/rclat_request_protocol")
SOURCES = Path("outputs/rclat_study_c_source_audit_stimuli.json")
JUDGMENTS = Path("outputs/rclat_study_c_source_audit_human_judgments.json")
FIELDS = ["review_id", "status", "heard_text", "notes"]
FLAGS = [
    "human_review",
    "independent_review",
    "source_scripts_hidden",
    "production_intentions_hidden",
    "listener_judgments_hidden",
    "model_transcripts_hidden",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    if path.exists():
        raise RuntimeError("Refusing to overwrite artifact")
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def csv_write(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def csv_read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def build():
    if OUT.exists():
        raise RuntimeError("Protocol/review workspace already exists; preserve human work")
    stimuli = json.loads(SOURCES.read_text())["stimuli"]
    judgments = json.loads(JUDGMENTS.read_text())["responses"]
    folds = proposed_speaker_folds(stimuli)
    by_name = {r["source_filename"]: r for r in stimuli}
    listeners = defaultdict(set)
    for row in judgments:
        listeners[row["listener_id"]].add(by_name[row["source_filename"]]["speaker_id"])
    adjacency = defaultdict(set)
    for person, speakers in listeners.items():
        for speaker in speakers:
            adjacency["l:" + person].add("s:" + speaker)
            adjacency["s:" + speaker].add("l:" + person)
    unseen = set(adjacency)
    components = []
    while unseen:
        stack = [unseen.pop()]
        component = set(stack)
        while stack:
            for neighbor in adjacency[stack.pop()] - component:
                component.add(neighbor)
                unseen.discard(neighbor)
                stack.append(neighbor)
        components.append(component)
    private = OUT / "custodian_only"
    private.mkdir(parents=True)
    master = private / "left_channel_audio"
    master.mkdir()
    review = []
    channel_rows = []
    ordered = sorted(
        stimuli,
        key=lambda r: hashlib.sha256(
            ("wording-review-v1:" + r["stimulus_id"]).encode()
        ).hexdigest(),
    )
    for index, item in enumerate(ordered, 1):
        path = Path(item["source_path"])
        if sha(path) != item["sha256"]:
            raise ValueError("Audited audio changed")
        audio, sr = sf.read(path, dtype="int16", always_2d=True)
        if sr != 44100 or audio.shape[1] != 2:
            raise ValueError("Unexpected source format")
        if not np.any(audio[:, 0]):
            raise ValueError("Left channel unsupported; no channel switching")
        left = audio[:, 0]
        target = f"W{index:03d}"
        destination = master / (target + ".wav")
        sf.write(destination, left, sr, subtype="PCM_16")
        reloaded, rate = sf.read(destination, dtype="int16")
        if rate != sr or not np.array_equal(reloaded, left):
            raise ValueError("Export changed left-channel samples")
        scaled = audio.astype(float) / 32768
        rms = np.sqrt((scaled * scaled).mean(0))
        channel_rows.append(
            {
                "review_id": target,
                "left_rms": float(rms[0]),
                "right_rms": float(rms[1]),
                "right_left_rms_ratio": float(rms[1] / rms[0]),
                "channel_correlation": float(np.corrcoef(scaled.T)[0, 1]),
                "left_fullscale_fraction": float(np.mean(np.abs(left.astype(np.int32)) >= 32767)),
                "export_sha256": sha(destination),
                "source_stereo_sha256": item["sha256"],
                "sample_exact_fixed_left_export": True,
            }
        )
        review.append({**item, "review_id": target, "review_audio_sha256": sha(destination)})
    for fold in folds:
        train = set(fold["training_speakers"])
        held = {fold["held_out_speaker"]}
        train_listeners = {person for person, speakers in listeners.items() if speakers & train}
        test_listeners = {person for person, speakers in listeners.items() if speakers & held}
        fold["listeners_shared_train_heldout"] = len(train_listeners & test_listeners)
        fold["heldout_listener_ids"] = len(test_listeners)
    write_json(private / "wording_crosswalk.json", {"stimuli": review, "not_for_reviewers": True})
    csv_write(
        private / "adjudication.csv",
        [
            dict(
                review_id=r["review_id"],
                status="pending",
                heard_text="",
                adjudicator_id="",
                evidence="",
                human_review="false",
                source_scripts_hidden="false",
                production_intentions_hidden="false",
            )
            for r in review
        ],
        [
            "review_id",
            "status",
            "heard_text",
            "adjudicator_id",
            "evidence",
            "human_review",
            "source_scripts_hidden",
            "production_intentions_hidden",
        ],
    )
    for name in ["reviewer_A", "reviewer_B"]:
        folder = OUT / name
        folder.mkdir()
        (folder / "audio").mkdir()
        packet_items = []
        for item in sorted(
            review, key=lambda r: hashlib.sha256((name + ":" + r["review_id"]).encode()).hexdigest()
        ):
            filename = item["review_id"] + ".wav"
            shutil.copyfile(master / filename, folder / "audio" / filename)
            packet_items.append({"review_id": item["review_id"], "audio": "audio/" + filename})
        csv_write(
            folder / "review.csv",
            [
                dict(review_id=r["review_id"], status="pending", heard_text="", notes="")
                for r in packet_items
            ],
            FIELDS,
        )
        write_json(folder / "metadata.json", {"reviewer_id": "", **{f: False for f in FLAGS}})
        packet = {"packet_id": name + ":request-wording-v1", "items": packet_items}
        template = Path("scripts/assets/rclat_request_word_review.html").read_text()
        (folder / "review.html").write_text(template.replace("__PACKET_JSON__", json.dumps(packet)))
        (folder / "START_HERE.txt").write_text(
            "Open review.html offline. Listen to the full audio and transcribe only the words heard. Do not use ASR, source scripts, production labels, listener judgments, or another reviewer's work. All fields start pending. Export review.csv and metadata.json regularly. Return both files to the custodian; preserve your reviewer pseudonym. Do not mark completed without hearing the whole recording. Contractions remain distinct from their expanded word forms. Mark uncertain/unusable rather than guessing. Your browser may retain work locally; exports are the returnable record.\n"
        )
        with zipfile.ZipFile(OUT / (name + ".zip"), "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(folder))
    protocol = {
        "study": "Proposed RCLAT Request-Interpretation Study v1",
        "status": "DRAFT_NOT_PREREGISTERED_NOT_CERTIFIED_NO_FITTING",
        "source_audit_sha256": sha(Path("outputs/rclat_study_c_source_audit.json")),
        "stimuli_sha256": sha(SOURCES),
        "judgments_sha256": sha(JUDGMENTS),
        "target": "Observed human Request/Non-Request choices; target distribution marginalized over the recorded listener panel and presentation histories, not production intentions.",
        "population": "All432 source-audited stimuli,18speakers,12scripts,82listeners,8856critical choices. No majority filter or automatic exclusion.",
        "primary_generalisation": "New speaker, familiar verified scripts, observed listener panel. No new-listener, new-script, session-independence or causal-manipulation claim.",
        "proposed_evaluation": "18leave-one-speaker-out outer folds. Each fold14training speakers/336stimuli,3validation speakers/72stimuli,1heldout speaker/24stimuli. Validation membership fixed by metadata-only hash. These are conditional support proposals, not activated partitions.",
        "speaker_folds": folds,
        "stereo_contract": {
            "native_rate": 44100,
            "channel": 0,
            "rule": "Fixed left channel for every stimulus; no per-file switching, averaging, resampling, denoising, trimming, amplitude fitting or silence deletion. Native stereo originals preserved. Mono exports must match native left PCM16 samples exactly. Human wording gate still required.",
            "decision_basis": "Source-only channel diagnostics, never a condition-specific channel choice or model performance comparison.",
        },
        "wording_rule": "Two independent blinded human transcriptions for all432stimuli. Punctuation/case folded; contractions not expanded. Disagreement requires a distinct third human adjudicator with explicit evidence. Every actual transcript must match its source script for this strict full-population proposal. Any mismatch/unresolved item blocks certification; no automatic repair or dropping pairs.",
        "models": {
            "C0": "Training-only verified script baseline; fixed smoothing of the mean stimulus-level training response. No public norming outcomes or heldout listener responses for fitting.",
            "C1": "C0 lexical access plus whole-utterance acoustic summaries.",
            "C2": "Same lexical access and whole-utterance summaries plus continuous acoustic trajectories.",
            "capacity_control": "Proposed identical C1/C2 parameterised architecture, common global-summary branch, equal-length temporal branch, paired initialization and training schedule. C1 temporal branch receives only constant summary-derived features and duration; C2 receives time-resolved values/masks. Do not expose fine time-varying masks to C1. Exact layer dimensions, acoustic measurements, scaler fitting, optimization and selection code remain to be frozen after reference readiness. More information remains an intended difference; no abstract continuous-semantic-code claim.",
        },
        "primary_endpoint": "Speaker-macro C1-minus-C2 binary cross-entropy in nats. Within each speaker average over its24stimuli; within each stimulus average over actual human responses (equivalently empirical soft-label cross-entropy). Equal stimulus weights, not raw response-count weighting. Probability clipping fixed at1e-6 for loss evaluation. All18speaker effects reported.",
        "secondary_endpoints": "C0 comparisons and speaker-level Brier loss descriptive. No question/statement relabeling, transcript WER proxy, hard-majority accuracy success gate, prefix timing or semantic reversibility claim.",
        "listener_dependency": {
            "bipartite_connected_components": len(components),
            "listeners_rating_multiple_speakers": sum(len(v) > 1 for v in listeners.values()),
            "interpretation": "Speaker-disjoint folds retain repeated listeners. To hold out listeners and speakers while retaining all observed edges would require splitting disconnected components; available structure is reported rather than assumed independent. Listener IDs are not model inputs.",
        },
        "uncertainty_proposal": "Crossed speaker/listener resampling or a prespecified crossed positive-weight bootstrap, paired across arms, preserving all responses for the same listener under one weight. Conditional on fitted models; not an8856-independent-trial interval. Final resampling algorithm, number of draws and decision rule require methodological freeze before fitting. Small18speaker sample limits population inference.",
        "uncertainty_primary_sources": [
            "https://arxiv.org/abs/0712.1111",
            "https://arxiv.org/abs/1106.2125",
        ],
        "forbidden_model_inputs": [
            "production_intent",
            "source_filename",
            "listener_id",
            "heldout_response_distribution",
            "unverified_reference_transcript",
        ],
        "history_scope": "Judgments include trial/presentation history. Source already-heard flags remain recorded, not silently pooled into a claim of context-free meaning. No inference about novel listener adaptation.",
        "certification_requirements": [
            "Independent wording gate ready for all432.",
            "Source reuse/redistribution terms resolved.",
            "Exact feature, architecture, fitting and uncertainty policies hash-frozen.",
            "Candidate-fold support rechecked against approved population; no fitting when support changes.",
        ],
        "training_performed": False,
        "official_partitions_assigned": False,
        "stage2_ready": False,
        "question_statement_study_ready": False,
    }
    write_json(OUT / "protocol_draft.json", protocol)
    write_json(
        OUT / "stereo_and_dependency_audit.json",
        {
            "channels": channel_rows,
            "connected_component_sizes": [len(c) for c in components],
            "wording_listening_performed": False,
            "all_fixed_left_exports_sample_exact": True,
            "models_fitted": 0,
        },
    )
    write_json(
        OUT / "build_manifest.json",
        {
            "source_files": {
                str(p): sha(p)
                for p in [SOURCES, JUDGMENTS, Path("outputs/rclat_study_c_source_audit.json")]
            },
            "code": {
                str(p): sha(p)
                for p in [
                    Path(__file__),
                    Path("scripts/assets/rclat_request_word_review.html"),
                    Path("src/bioacoustic_pipeline/ingestion/request_protocol.py"),
                ]
            },
            "review_audio": {r["review_id"]: r["review_audio_sha256"] for r in review},
        },
    )
    validate(initial=True)
    print("Prepared18conditional folds and two432-item blinded wording packets; no model fitting.")


def validate(initial=False):
    crosswalk = json.loads((OUT / "custodian_only/wording_crosswalk.json").read_text())["stimuli"]
    manifest = json.loads((OUT / "build_manifest.json").read_text())
    for path, digest in {**manifest["source_files"], **manifest["code"]}.items():
        if sha(Path(path)) != digest:
            raise ValueError("Audited source changed")
    reviews = []
    metadata = []
    for name in ["reviewer_A", "reviewer_B"]:
        folder = OUT / name
        for item in crosswalk:
            if sha(folder / "audio" / (item["review_id"] + ".wav")) != item["review_audio_sha256"]:
                raise ValueError("Review stimulus changed")
        reviews.append(csv_read(folder / "review.csv"))
        metadata.append(json.loads((folder / "metadata.json").read_text()))
    result = wording_gate(
        crosswalk, *reviews, *metadata, csv_read(OUT / "custodian_only/adjudication.csv")
    )
    result.update(
        {
            "study_ready": False,
            "reason": "Human wording and final methodological/source-use certification required; no automatic training activation.",
            "models_fitted": 0,
            "stage2_ready": False,
        }
    )
    if initial:
        write_json(OUT / "initial_status.json", result)
    else:
        print(json.dumps(result, indent=2))
    return result["ready"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["build", "validate"])
    args = parser.parse_args()
    if args.action == "build":
        build()
    else:
        raise SystemExit(0 if validate() else 2)
