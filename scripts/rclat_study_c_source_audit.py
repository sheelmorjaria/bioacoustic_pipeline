"""Reference-validity and source-eligibility audit only; no partitions or models."""

import csv
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import re

import numpy as np
import soundfile as sf

from bioacoustic_pipeline.ingestion.prosody_reference import (
    extract_request_judgments,
    request_distributions,
)

ROOT = Path("data/rclat_study_c_audit")
OUT = Path("outputs/rclat_study_c_source_audit")


def read_csv(path, delimiter=","):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    if path.exists():
        raise RuntimeError("Refusing to overwrite completed audit")
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def canonical_topic(topic):
    # Author behavioral.Rmd explicitly applies tolower(topic), then temp -> temperature.
    value = topic.lower()
    return "temperature" if value == "temp" else value


def main():
    if Path(str(OUT) + ".json").exists():
        raise RuntimeError("Audit already completed")
    tree = json.loads((ROOT / "pros_scaled_tree.json").read_text())
    inventory = {r["path"]: r for r in tree["tree"] if r["type"] == "blob"}
    behavioral = ROOT / "pros_scaled/data/processed/behavioral/pros_scaled_anonymized.csv"
    raw = read_csv(behavioral)
    judgments, row_accounting = extract_request_judgments(raw)
    references = request_distributions(judgments)
    production = {
        (r["speaker"], r["intent"], canonical_topic(r["topic"])): r
        for r in read_csv(
            ROOT / "pros_scaled/data/processed/audio/audio_features_scaled_stimuli.csv"
        )
    }
    texts = defaultdict(set)
    for row in read_csv(ROOT / "pros_scaled/data/processed/norming/norming_data.csv"):
        if canonical_topic(row["topic"]) in {canonical_topic(r["text_id"]) for r in judgments}:
            texts[canonical_topic(row["topic"])].add(
                html.unescape(re.sub("<[^>]*>", "", row["stimulus"])).strip()
            )
    if len(texts) != 12 or any(len(v) != 1 for v in texts.values()):
        raise ValueError("Ambiguous or missing lexical reference")
    experiment = (ROOT / "pros_scaled/experiment/prosody_exp_full.html").read_text()
    records = []
    decoded = defaultdict(list)
    native = defaultdict(set)
    person_sources = defaultdict(set)
    for row in judgments:
        person_sources[row["listener_id"]].add(row["speaker_id"])
        if row["native_speaker_self_report"] == "Yes":
            native[row["source_filename"]].add(row["listener_id"])
    for filename in sorted(references):
        relative = "data/raw/scaled_stimuli_normed/" + filename
        item = inventory[relative]
        path = ROOT / "pros_scaled" / relative
        blob = path.read_bytes()
        if (
            len(blob) != item["size"]
            or hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest()
            != item["sha"]
        ):
            raise ValueError("Source no longer matches pinned Git blob")
        if relative not in experiment:
            raise ValueError("No exact stimulus path in released listening experiment")
        info = sf.info(path)
        audio, sr = sf.read(path, dtype="int16", always_2d=True)
        if info.subtype != "PCM_16" or info.channels not in (1, 2) or sr < 16000 or not len(audio):
            raise ValueError("Unexpected waveform format; stop instead of adaptive exclusion")
        pcm_sha = hashlib.sha256(str(sr).encode() + b":" + audio.tobytes()).hexdigest()
        decoded[pcm_sha].append(filename)
        speaker, intent, topic = filename.removesuffix(".wav").split("_")
        source = production[(speaker, intent, canonical_topic(topic))]
        records.append(
            {
                "stimulus_id": hashlib.sha256(filename.encode()).hexdigest()[:16],
                "source_filename": filename,
                "speaker_id": source["speaker"],
                "text_id": canonical_topic(source["topic"]),
                "script_text": next(iter(texts[canonical_topic(source["topic"])])),
                "production_intent_private_not_prediction_target": source["intent"],
                "source_path": str(path),
                "source_git_blob_sha1": item["sha"],
                "sha256": sha(path),
                "decoded_pcm_sha256": pcm_sha,
                "sample_rate": sr,
                "samples": len(audio),
                "channels": info.channels,
                "channels_sample_identical": bool(
                    info.channels == 2 and np.array_equal(audio[:, 0], audio[:, 1])
                ),
                "sample_format": info.subtype,
                "duration_seconds": len(audio) / sr,
                "full_scale_sample_fraction": float(
                    np.mean(np.abs(audio.astype(np.int32)) >= 32767)
                ),
                **references[filename],
            }
        )
    by_pair = defaultdict(set)
    for row in records:
        by_pair[(row["speaker_id"], row["text_id"])].add(
            row["production_intent_private_not_prediction_target"]
        )
    if any("request" not in v or len(v) != 2 for v in by_pair.values()):
        raise ValueError("Matched scripted-pair support incomplete")
    xie = read_csv(ROOT / "xie_data/QS_Comprehension_MaleAndFemale_Data.csv")
    xp = read_csv(ROOT / "xie_data/QS_ProductionData.csv")
    gate_data = read_csv(ROOT / "villani_data/Gate_natural_sentences.csv", delimiter=";")
    cross_data = read_csv(ROOT / "villani_data/Gate_crossSpliced_sentences.csv", delimiter=";")
    guide = {
        r["Video"]: r
        for r in read_csv(ROOT / "sinagra_data/Stimuli-guide.csv")
        if r["Video"].strip()
    }
    sinagra = read_csv(ROOT / "sinagra_data/prosody-task.csv")
    matched = [r for r in sinagra if r["video"].removesuffix(".mp4") in guide]
    intended = {"sentence": "Statement", "question": "Question", "happy": "Happy", "sad": "Sad"}
    label_matches = sum(
        r["answer"] == intended[guide[r["video"].removesuffix(".mp4")]["Prosody"]] for r in matched
    )
    output = {
        "scope": "Existing-dataset audit, not a preregistered prediction experiment.",
        "preferred_question_statement_task": {
            "status": "ABSTAIN: no complete cross-speaker benchmark certified among inspected question/statement sources.",
            "reason": "Perception talkers too few or unverified, source-response fields require care, and no protected prediction design exists. This is not a claim that no suitable corpus exists.",
            "candidates": [
                {
                    "name": "Xie, Buxo-Lugo, Kurumada 2021",
                    "repository": "https://osf.io/kr7y6/",
                    "production_table_speakers": len({r["Subject"] for r in xp}),
                    "perception_rows": len(xie),
                    "perception_listener_ids": len({r["Subject"] for r in xie}),
                    "perception_talker_ids": sorted({r["Talker"] for r in xie}),
                    "blocks": sorted({r["Block"] for r in xie}),
                    "exposure_contexts": sorted({r["Condition"] for r in xie}),
                    "status": "Genuine binary perception responses; only two perception talkers. Production speakers cannot substitute for judged speakers. Do not pool pre/post exposure contexts. Complete row-to-waveform identity not certified.",
                },
                {
                    "name": "Villani et al. 2025",
                    "repository": "https://osf.io/c83ep/",
                    "natural_gate_rows": len(gate_data),
                    "natural_gate_stimuli": len({r["stimulus"] for r in gate_data}),
                    "cross_spliced_gate_rows": len(cross_data),
                    "listeners": len({r["Subject"] for r in gate_data}),
                    "perceived_answer_mapping": {"1": "statement", "2": "unclear", "3": "question"},
                    "speaker_identity_column_present": any(
                        "speaker" in k.lower() or "talker" in k.lower() for k in gate_data[0]
                    ),
                    "status": "Individual raw choices and prefix stimuli released. Speaker-disjoint evaluation support not certified; prefixes from the same source utterance are dependent. Raw Answer must be distinguished from correctAnswer and confidence-recoded columns.",
                },
                {
                    "name": "Sinagra and Wiener 2022",
                    "repository": "https://osf.io/gs79t/",
                    "released_rows": len(sinagra),
                    "listener_ids": len({r["Participant.Public.ID"] for r in sinagra}),
                    "exact_guide_row_joins": len(matched),
                    "guide_label_matches_answer_field": label_matches,
                    "guide_join_failures": len(sinagra) - len(matched),
                    "status": "Single-speaker audiovisual experiment; no speaker-disjoint support. answer matches intended prosody on every exact guide join, including incorrect trials, so it is not accepted as a listener choice. Accuracy-derived reconstruction and unmatched stimulus IDs remain unresolved; no response labels fabricated.",
                },
            ],
        },
        "adjacent_indirect_request_candidate": {
            "name": "Trott et al., pros_scaled",
            "repository": "https://github.com/seantrott/pros_scaled",
            "commit": tree["sha"],
            "task": "Listener judgment: Is this person making a request? Request versus non-request.",
            "status": "REFERENCE_AND_WAVEFORM_SUPPORT_VERIFIED; not interchangeable with question-versus-statement.",
            "released_behavioral_rows": len(raw),
            "row_accounting": row_accounting,
            "audio_inventory_raw_files": sum(
                p.startswith("data/raw/scaled_stimuli/") and p.endswith(".wav") for p in inventory
            ),
            "listener_linked_verified_wavs": len(records),
            "speaker_ids": len({r["speaker_id"] for r in records}),
            "scripted_texts": len(texts),
            "complete_speaker_text_intent_pairs": len(by_pair),
            "listener_ids": len(person_sources),
            "actual_human_choices": len(judgments),
            "unique_listener_votes_per_stimulus_min_max": [
                min(r["independent_listeners"] for r in records),
                max(r["independent_listeners"] for r in records),
            ],
            "author_native_language_filter_descriptive_only": {
                "listeners": len(
                    {
                        r["listener_id"]
                        for r in judgments
                        if r["native_speaker_self_report"] == "Yes"
                    }
                ),
                "choices": sum(r["native_speaker_self_report"] == "Yes" for r in judgments),
                "no_audit_rows_removed_by_this_filter": True,
            },
            "native_sample_rates": sorted({r["sample_rate"] for r in records}),
            "duration_seconds_min_median_max": np.quantile(
                [r["duration_seconds"] for r in records], [0, 0.5, 1]
            ).tolist(),
            "sample_format": "Native stereo PCM16 WAV; both channels preserved, no downmix or resampling.",
            "identical_stereo_channels_count": sum(r["channels_sample_identical"] for r in records),
            "exact_decoded_duplicate_groups": [v for v in decoded.values() if len(v) > 1],
            "audio_source": "Normalized WAVs referenced by released experiment, not raw counterparts; each Git blob and SHA256 verified.",
            "reference": "Empirical human Request/Non-Request distribution retained. No majority filter, no agreement threshold, no missing-answer imputation, no production-intent target fallback.",
            "source_id_crosswalk": "Author behavioral.Rmd: topic lowercased, then temp recoded to temperature. Raw filenames and response rows retained; canonical text IDs and source numeric item IDs crosschecked.",
            "wording": "12 source-script texts mapped from independent text-norming table; actual spoken-word equality has not been independently transcribed or listened to in this audit.",
            "context": "Decontextualized binary intention judgment, not animal receiver behavior or universal semantics.",
            "remaining_design_requirements": [
                "Prospectively specify indirect-request estimand rather than relabel as question/statement.",
                "Define speaker/text evaluation splits and treatment of listeners crossing speakers; no random trial split.",
                "Independently verify spoken lexical identity if making an identical-words claim; no such check is fabricated.",
                "Freeze C0/C1/C2 frontends, training-side selection, distribution loss, crossed uncertainty and source exclusions before fitting.",
                "Freeze a channel and resampling policy using source metadata only; this audit preserves native stereo.",
                "Confirm applicable source-data reuse terms before redistribution; repository-level data license not found in inspected Git tree.",
            ],
            "listener_cross_speaker_dependence": {
                "listeners_rating_more_than_one_speaker": sum(
                    len(v) > 1 for v in person_sources.values()
                ),
                "max_speakers_per_listener": max(map(len, person_sources.values())),
            },
        },
        "model_training_performed": False,
        "evaluation_partitions_assigned": False,
        "semantic_reversibility_demonstrated": False,
        "stage2_ready": False,
        "sampling_notes": "No majority-agreement filtering; failed joins remain explicit. MACDE 128-frame gate is not silently imposed on this different utterance-interpretation task.",
        "source_file_sha256": {
            str(p): sha(p)
            for p in sorted(ROOT.rglob("*"))
            if p.is_file() and p.name != "acquisition.jsonl"
        },
        "code_sha256": {
            str(p): sha(p)
            for p in [
                Path(__file__),
                Path("src/bioacoustic_pipeline/ingestion/prosody_reference.py"),
            ]
        },
    }
    write_json(Path(str(OUT) + ".json"), output)
    write_json(
        Path(str(OUT) + "_stimuli.json"),
        {
            "stimuli": records,
            "identities_and_intents_private_do_not_use_as_model_features": True,
            "partitions_assigned": False,
        },
    )
    write_json(
        Path(str(OUT) + "_human_judgments.json"),
        {"responses": judgments, "model_generated_labels": False, "partitions_assigned": False},
    )
    print(
        "Audit complete:",
        len(records),
        "verified played waveforms,",
        len(judgments),
        "independent human choices; no models fitted.",
    )


if __name__ == "__main__":
    main()
