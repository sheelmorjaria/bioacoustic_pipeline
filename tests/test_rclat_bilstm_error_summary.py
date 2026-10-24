import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rclat_bilstm_p10_error_summary import describe_predictions  # noqa: E402
from rclat_bilstm_p10_train import score  # noqa: E402


def test_speaker_macro_decomposition_uses_reference_weights_within_speaker():
    rows = [
        score({"uid": "a", "speaker": "s1", "transcript": "AB"}, "A", 12),
        score({"uid": "b", "speaker": "s2", "transcript": "ABCDEFGH"}, "ABCDEFGH", 24),
    ]
    r = describe_predictions(rows)
    assert r["metrics"]["speaker_macro_cer"] == 0.25
    assert r["metrics"]["pooled_cer"] == 0.1
    assert r["speaker_macro_decomposition"]["character"]["deletions"] == 0.25
    assert r["exact_utterances"] == 1
    assert r["distinct_hypotheses"] == 2
