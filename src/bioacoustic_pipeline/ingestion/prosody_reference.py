"""Released indirect-request judgments: preserve perceived versus intended function."""

from collections import Counter, defaultdict


CONDITIONS = ("request", "question", "statement")
RESPONSE_BUTTONS = {"Request": "0", "Non-Request": "1"}


def extract_request_judgments(rows):
    counts = Counter()
    output = []
    seen = set()
    for index, row in enumerate(rows, 2):
        answer = row.get("response", "").strip()
        if not row.get("speaker", "").strip() or row.get("condition") not in CONDITIONS:
            counts["noncritical_or_practice_rows"] += 1
            continue
        if not answer:
            counts["critical_rows_without_response"] += 1
            continue
        if answer not in RESPONSE_BUTTONS or row.get("button_pressed") != RESPONSE_BUTTONS[answer]:
            raise ValueError("Actual response and documented button mapping disagree")
        listener = row.get("subject", "").strip()
        topic = row.get("topic", "").strip()
        speaker = row["speaker"].strip()
        if not listener or not topic:
            raise ValueError("Missing listener or item identity")
        filename = f"{speaker}_{row['condition']}_{topic}.wav"
        if (listener, filename) in seen:
            raise ValueError("Duplicate listener/source judgment")
        seen.add((listener, filename))
        output.append(
            {
                "source_row": index,
                "listener_id": listener,
                "source_filename": filename,
                "speaker_id": speaker,
                "text_id": topic,
                "perceived_request": int(answer == "Request"),
                "already_heard_by_speaker": row.get("already_heard_by_speaker", ""),
                "native_speaker_self_report": row.get("Native_Speaker", ""),
            }
        )
        counts["actual_critical_responses"] += 1
    return output, dict(counts)


def request_distributions(judgments):
    grouped = defaultdict(list)
    for row in judgments:
        grouped[row["source_filename"]].append(row)
    result = {}
    for filename, rows in grouped.items():
        listeners = {r["listener_id"] for r in rows}
        if len(listeners) != len(rows):
            raise ValueError("Repeated listener cannot increase independent vote support")
        yes = sum(r["perceived_request"] for r in rows)
        if any(r["perceived_request"] not in (0, 1) for r in rows):
            raise ValueError("Unknown perception category")
        result[filename] = {
            "independent_listeners": len(listeners),
            "request_votes": yes,
            "nonrequest_votes": len(rows) - yes,
            "perceived_distribution": {
                "request": yes / len(rows),
                "nonrequest": 1 - yes / len(rows),
            },
        }
    return result
