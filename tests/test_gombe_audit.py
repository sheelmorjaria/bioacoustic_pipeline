import numpy as np
from bioacoustic_pipeline.ingestion.gombe_audit import support_summary, catalogue_rows


def test_support_counts_do_not_certify_cut_or_session_boundaries():
    valid = np.r_[np.zeros(4, bool), np.ones(128, bool), np.zeros(8, bool)]
    summary = support_summary(valid, 96000)
    assert summary['longest_valid_run_frames'] == 128
    assert summary['technical_eligible_positions'] == 57
    assert summary['certified_cut_local_positions'] == 0
    assert summary['certified_independent_sessions'] == 0


def test_catalogue_selection_keeps_numeric_ids_and_does_not_invent_focal_names():
    rows = catalogue_rows([['Macaulay', 'focal'], [195800., 'A'], ['', ''], [195801., 'A']])
    assert len(rows) == 2
    assert rows[0][0] == 195800.
