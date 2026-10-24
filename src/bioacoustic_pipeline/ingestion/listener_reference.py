"""Independent human interpretation references, separate from production intentions."""

from collections import Counter, defaultdict

CATEGORIES = ("statement", "question", "uncertain")


def listener_reference_gate(stimuli, responses, metadata, minimum_listeners=10):
    """Validate provenance and retain empirical response distributions, including ambiguity."""
    if minimum_listeners < 2:
        raise ValueError("Independent reference needs multiple listeners")
    inventory = {}
    people = {}
    texts = {}
    for item in stimuli:
        uid = item["stimulus_id"]
        if uid in inventory or item["split"] not in ("train", "validation", "evaluation"):
            raise ValueError("Duplicate stimulus or unknown partition")
        for key, mapping in [("speaker_id", people), ("text_id", texts)]:
            identity = item[key]
            if identity in mapping and mapping[identity] != item["split"]:
                raise ValueError("Speaker/text leakage between partitions")
            mapping[identity] = item["split"]
        inventory[uid] = item
    counts = defaultdict(Counter)
    seen = set()
    listener_splits = {}
    listener_pairs = set()
    for row in responses:
        uid = row["stimulus_id"]
        if uid not in inventory:
            raise ValueError("Unknown stimulus")
        status = row.get("status", "")
        if status in ("", "pending"):
            if any(
                str(row.get(k, "")).strip() for k in ("listener_id", "interpretation", "heard_text")
            ):
                raise ValueError("Pending row cannot carry listener evidence")
            continue
        if status != "completed":
            raise ValueError("Unknown response status")
        listener = str(row.get("listener_id", "")).strip()
        answer = row.get("interpretation")
        if not listener or answer not in CATEGORIES:
            raise ValueError("Completed response requires listener identity and allowed response")
        if not str(row.get("heard_text", "")).strip():
            raise ValueError("Independent intelligibility response is required")
        if (uid, listener) in seen:
            raise ValueError("Duplicate listener/stimulus response")
        seen.add((uid, listener))
        item = inventory[uid]
        split = item["split"]
        if listener in listener_splits and listener_splits[listener] != split:
            raise ValueError("Listener panels must be disjoint across partitions")
        listener_splits[listener] = split
        pair = (listener, item["speaker_id"], item["text_id"])
        if pair in listener_pairs:
            raise ValueError("Listener heard multiple versions of a speaker/text pair")
        listener_pairs.add(pair)
        counts[uid][answer] += 1
    attestations = (
        "human_judgments",
        "independent_of_models",
        "production_intentions_hidden",
        "reference_transcripts_hidden",
        "production_participants_excluded",
        "collection_authorization_recorded",
    )
    attested = bool(str(metadata.get("reference_custodian_id", "")).strip()) and all(
        metadata.get(key) is True for key in attestations
    )
    references = {}
    supported = 0
    for uid in inventory:
        total = sum(counts[uid].values())
        supported += total >= minimum_listeners
        references[uid] = {
            "responses": total,
            "counts": {k: counts[uid][k] for k in CATEGORIES},
            "distribution": {k: counts[uid][k] / total for k in CATEGORIES} if total else None,
        }
    return {
        "ready": bool(inventory) and attested and supported == len(inventory),
        "independence_attested": attested,
        "total_stimuli": len(inventory),
        "supported_stimuli": supported,
        "completed_judgments": len(seen),
        "unique_listeners": len(listener_splits),
        "references": references,
        "reference_basis": "Observed human choices only; production intention is never a target fallback.",
        "majority_agreement_filter_applied": False,
    }
