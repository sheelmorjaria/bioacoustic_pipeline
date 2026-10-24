import numpy as np
import pytest
from bioacoustic_pipeline.contextual.bfsongrepo_development import partition_sources, selected_positions


def test_whole_day_holdout_and_recording_validation_are_disjoint():
    records = [{'recording_id': f'or60yw70/092812/{i}', 'bird_id': 'or60yw70', 'archive_day_code': '092812'} for i in range(8)]
    records += [{'recording_id': 'validation_day', 'bird_id': 'or60yw70', 'archive_day_code': '093012'},
                {'recording_id': 'day2', 'bird_id': 'or60yw70', 'archive_day_code': '100112'},
                {'recording_id': 'otherbird', 'bird_id': 'gy6or6', 'archive_day_code': '032612'}]
    split = partition_sources(records)
    assert split['day2'] == 'heldout_day' and split['otherbird'] == 'cross_bird'
    assert split['validation_day'] == 'validation'
    assert all(split[str(r['recording_id'])] == 'fit' for r in records[:8])
    assert partition_sources(list(reversed(records))) == split
    with pytest.raises(ValueError):
        partition_sources(records + [records[0]])


def test_selected_targets_have_full_support_and_do_not_bridge_gaps():
    valid = np.r_[np.ones(180, bool), False, np.ones(180, bool)]
    positions = selected_positions(valid)
    assert len(positions) == 16
    for t in positions:
        assert valid[t-51:t+21].all()
    assert selected_positions(np.ones(100, bool)).size == 0
    assert np.array_equal(positions, selected_positions(valid))
