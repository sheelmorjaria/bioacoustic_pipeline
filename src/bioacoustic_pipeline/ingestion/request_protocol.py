"""Review and support gates for a proposed request-interpretation study, not model fitting."""

from collections import defaultdict
import hashlib
import re
import unicodedata


def lexical_tokens(value):
    value = unicodedata.normalize("NFKC", value).casefold().replace("’", "'")
    return tuple(re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)*", value))


def proposed_speaker_folds(stimuli):
    speakers = sorted({r["speaker_id"] for r in stimuli})
    texts = {r["text_id"] for r in stimuli}
    if len(speakers) != 18 or len(texts) != 12:
        raise ValueError("Source support differs from the audited 18-speaker/12-script population")
    pairs = defaultdict(list)
    for row in stimuli:
        pairs[(row["speaker_id"], row["text_id"])].append(row["stimulus_id"])
    if any(len(pairs[(s, t)]) != 2 for s in speakers for t in texts):
        raise ValueError("Complete paired speaker/script support missing")
    folds = []
    for held in speakers:
        remaining = [s for s in speakers if s != held]
        ranked = sorted(
            remaining,
            key=lambda s: hashlib.sha256(
                ("request-design-v1:" + held + ":" + s).encode()
            ).hexdigest(),
        )
        validation = ranked[:3]
        training = ranked[3:]
        folds.append(
            {
                "held_out_speaker": held,
                "training_speakers": training,
                "validation_speakers": validation,
                "training_stimuli": sum(r["speaker_id"] in training for r in stimuli),
                "validation_stimuli": sum(r["speaker_id"] in validation for r in stimuli),
                "held_out_stimuli": sum(r["speaker_id"] == held for r in stimuli),
                "scripts": "All12 familiar scripts; no held-out-wording claim.",
            }
        )
    return folds


def wording_gate(stimuli, rows_a, rows_b, meta_a, meta_b, adjudications):
    inventory = {r["review_id"]: r for r in stimuli}
    if len(inventory) != len(stimuli):
        raise ValueError("Duplicate review target")
    reviewed = []
    for rows in [rows_a, rows_b]:
        mapping = {}
        for row in rows:
            uid = row["review_id"]
            if uid not in inventory or uid in mapping:
                raise ValueError("Unknown or duplicate review target")
            status = row.get("status", "")
            text = str(row.get("heard_text", "")).strip()
            if status in ("", "pending"):
                if text:
                    raise ValueError("Pending review cannot carry a transcript")
            elif status == "completed":
                if not lexical_tokens(text):
                    raise ValueError("Completed review requires explicit heard words")
            elif status not in ("uncertain", "unusable"):
                raise ValueError("Unknown review status")
            mapping[uid] = row
        reviewed.append(mapping)
    ids = [str(m.get("reviewer_id", "")).strip() for m in [meta_a, meta_b]]
    if all(ids) and ids[0] == ids[1]:
        raise ValueError("Two distinct independent reviewers required")
    attestations = [
        "human_review",
        "independent_review",
        "source_scripts_hidden",
        "production_intentions_hidden",
        "listener_judgments_hidden",
        "model_transcripts_hidden",
    ]
    attested = all(ids) and all(m.get(k) is True for m in [meta_a, meta_b] for k in attestations)
    decisions = {}
    for row in adjudications:
        uid = row["review_id"]
        if uid not in inventory or uid in decisions:
            raise ValueError("Unknown or duplicate adjudication")
        decisions[uid] = row
    supported, mismatched, unresolved = [], [], []
    disagreements = []
    for uid, item in inventory.items():
        a, b = [m.get(uid, {}) for m in reviewed]
        if not all(r.get("status") == "completed" for r in [a, b]):
            unresolved.append(uid)
            continue
        a_tokens, b_tokens = [lexical_tokens(r["heard_text"]) for r in [a, b]]
        chosen = a_tokens
        if a_tokens != b_tokens:
            disagreements.append(uid)
            decision = decisions.get(uid, {})
            adjudicator = str(decision.get("adjudicator_id", "")).strip()
            if (
                decision.get("status") != "resolved"
                or not adjudicator
                or adjudicator in ids
                or not str(decision.get("evidence", "")).strip()
                or not lexical_tokens(decision.get("heard_text", ""))
                or not all(
                    decision.get(flag) in (True, "true")
                    for flag in (
                        "human_review",
                        "source_scripts_hidden",
                        "production_intentions_hidden",
                    )
                )
            ):
                unresolved.append(uid)
                continue
            chosen = lexical_tokens(decision["heard_text"])
        if chosen == lexical_tokens(item["script_text"]):
            supported.append(uid)
        else:
            mismatched.append(uid)
    return {
        "ready": bool(inventory) and bool(attested) and len(supported) == len(inventory),
        "human_independence_attested": bool(attested),
        "total_stimuli": len(inventory),
        "script_matched_stimuli": len(supported),
        "verified_wording_mismatches": mismatched,
        "unresolved_stimuli": unresolved,
        "reviewer_disagreements": disagreements,
        "automatic_exclusions_or_transcript_imputation": False,
        "scope": "Strict full-population script-match gate. Mismatches block this proposal; no automatic data deletion or model-driven repair.",
    }
