"""Contract checks for event ordering, caller mapping and partition identity."""
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'scripts/grijseels_dataset_audit.py'
spec = importlib.util.spec_from_file_location('grijseels_dataset_audit', path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_tied_onsets_do_not_create_arbitrary_next_caller_labels():
    events = [(0, 5, 'a'), (10, 15, 'a'), (10, 15, 'b'), (20, 25, 'b'), (30, 35, 'a')]
    stats = audit.timeline_stats(events, fps=10)
    assert stats['tied_onset_groups'] == 1
    assert stats['adjacent_pairs_with_singleton_onsets'] == 1
    assert stats['different_caller_singleton_pairs_within_5s'] == 1


def test_horizon_boundary_overlap_and_empty_recording():
    stats = audit.timeline_stats([(0, 70, 'a'), (50, 60, 'b'), (101, 110, 'c')], fps=10)
    assert stats['different_caller_singleton_pairs_within_5s'] == 1
    assert stats['next_call_overlaps_current_call'] == 1
    assert stats['chains_with_exactly_two_callers'] == 1
    assert stats['chains_with_exactly_three_callers'] == 0
    empty = audit.timeline_stats([], fps=10)
    assert empty['first_onset_s'] is None
    assert empty['adjacent_pairs_with_singleton_onsets'] == 0


def test_versions_share_recording_identity_and_preserve_caller_order():
    first = audit.recording_identity('two_monkey/Cain_Hela_20220223/call_data_v1.mat')
    second = audit.recording_identity('two_monkey/Cain_Hela_20220223/call_data_v2.mat')
    assert first == second == ('two_monkey/cain_hela_20220223', '20220223', ['cain', 'hela'])
    trio = audit.recording_identity('three_monkey/cassi_loretta_leo_20230227ttl_call_data_v2cleaned.mat')
    assert trio[2] == ['cassi', 'loretta', 'leo']
    assert audit.partition_for_group(first[1]) == audit.partition_for_group(second[1])
