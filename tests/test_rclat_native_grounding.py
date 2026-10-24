import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from rclat_bilstm_native_grounding import pairing_scores  # noqa: E402


def test_native_swap_preserves_donor_hypothesis_and_length():
    rows=[{"uid":"a","speaker":"s","reference":"AB","hypothesis":"AB","native_length":12}, {"uid":"b","speaker":"s","reference":"CD","hypothesis":"CD","native_length":24}]
    scores=pairing_scores(rows,{"a":"b","b":"a"})
    assert all(r["cer"]==0 for r in scores["matched"])
    assert all(r["cer"]==1 for r in scores["native_swapped"])
    assert scores["native_swapped"][0]["input_uid"]=="b"
    assert scores["native_swapped"][0]["input_native_length"]==24
    assert scores["native_swapped"][0]["reference"]=="AB"
