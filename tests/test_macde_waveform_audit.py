import numpy as np
from bioacoustic_pipeline.diagnostics.macde_waveform_audit import window_decision


def test_length_and_support_are_separate_and_gaps_are_not_concatenated():
    assert not window_decision(np.ones(127, bool))['passes_length']
    rejected = window_decision(np.zeros(128, bool))
    assert rejected['passes_length'] and not rejected['passes_acoustic_support']
    valid = np.ones(128, bool)
    result = window_decision(valid)
    assert result['passes_acoustic_support'] and result['eligible_positions'] == 57
    valid[::60] = False
    assert not window_decision(valid)['passes_acoustic_support']


def test_window_selection_matches_closed_protocol_without_future_reordering():
    valid = np.zeros(256, bool)
    valid[100:220] = True
    result = window_decision(valid)
    assert result['selected_start_frame'] == 100
    assert result['eligible_positions'] == 49
