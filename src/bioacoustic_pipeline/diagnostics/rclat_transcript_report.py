"""Read-only ASR word alignment and self-contained report;no model imports."""

from pathlib import Path
import json
import re


def normalized_words(text):
    return re.sub("[^A-Z' ]", " ", text.upper()).split()


def align_words(reference, hypothesis):
    ref = normalized_words(reference)
    hyp = normalized_words(hypothesis)
    n = len(ref)
    m = len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(
                d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1
            )
    out = []
    i = n
    j = m
    while i or j:
        if i and j and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            op = "="
            a = ref[i - 1]
            b = hyp[j - 1]
            i -= 1
            j -= 1
        elif i and j and d[i][j] == d[i - 1][j - 1] + 1:
            op = "S"
            a = ref[i - 1]
            b = hyp[j - 1]
            i -= 1
            j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            op = "D"
            a = ref[i - 1]
            b = None
            i -= 1
        else:
            op = "I"
            a = None
            b = hyp[j - 1]
            j -= 1
        out.append(dict(op=op, reference=a, hypothesis=b))
    return list(reversed(out))


def word_metrics(alignment):
    counts = {op: sum(row["op"] == op for row in alignment) for op in ["S", "D", "I"]}
    n = sum(row["reference"] is not None for row in alignment)
    h = sum(row["hypothesis"] is not None for row in alignment)
    errors = sum(counts.values())
    return dict(
        substitutions=counts["S"],
        deletions=counts["D"],
        insertions=counts["I"],
        errors=errors,
        reference_words=n,
        hypothesis_words=h,
        wer=errors / n if n else None,
        exact=errors == 0,
    )


def render_html(records, metadata):
    template = Path(__file__).resolve().parents[3] / "scripts/assets/rclat_transcript_report.html"
    payload = (
        json.dumps(
            dict(records=records, metadata=metadata), ensure_ascii=True, separators=(",", ":")
        )
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    return template.read_text().replace("__RCLAT_DATA__", payload, 1)
