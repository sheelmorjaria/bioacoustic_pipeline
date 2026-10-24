import pytest
from bioacoustic_pipeline.diagnostics.rclat_transcript_report import (
    align_words,
    word_metrics,
    render_html,
)


def test_minimum_word_alignment_and_empty_predictions():
    a = align_words(
        "The little bird flew across the garden.", "The little bird flew around the garden"
    )
    assert sum(op["op"] == "S" for op in a) == 1
    assert word_metrics(a)["wer"] == pytest.approx(1 / 7)
    empty = word_metrics(align_words("HELLO WORLD", ""))
    assert empty["deletions"] == 2 and empty["wer"] == 1
    ins = word_metrics(align_words("A B", "A NEW B"))
    assert ins["insertions"] == 1 and ins["wer"] == 0.5
    assert word_metrics(align_words("", "NEW"))["wer"] is None


def test_alignment_reconstructs_both_sequences_and_has_minimum_cost():
    rows = align_words("A B A", "B A B")
    assert [r["reference"] for r in rows if r["reference"] is not None] == ["A", "B", "A"]
    assert [r["hypothesis"] for r in rows if r["hypothesis"] is not None] == ["B", "A", "B"]
    assert word_metrics(rows)["errors"] == 2


def test_json_script_payload_cannot_escape_into_html():
    html = render_html(
        [dict(reference="</script><script>alert(1)</script>", hypothesis="x")], {"records": 1}
    )
    assert "</script><script>alert(1)" not in html
    assert "\\u003c/script" in html
    assert "RTPE" in html and "application/json" in html
