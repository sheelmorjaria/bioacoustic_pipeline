import numpy as np
import pytest
from bioacoustic_pipeline.ingestion.bfsongrepo_audit import rec_metadata, decode_cbin_channel, continuity_summary


def test_rec_parsing_and_cbin_channel_alias():
    rec = rec_metadata('File created: Mon, Mar 26, 2012, 08:10:19\nADFREQ = 32000\nChans = 2\nSamples = 3\n')
    payload = np.array([1, 11, 2, 12, 3, 13], dtype='>i2').tobytes()
    np.testing.assert_array_equal(decode_cbin_channel(payload, rec), [1, 2, 3])
    assert rec['file_created'] == '2012-03-26T08:10:19'
    with pytest.raises(ValueError):
        decode_cbin_channel(payload[:-1], rec)


def test_128_run_gate_is_reported_separately_from_forecast_support():
    summary = continuity_summary(np.r_[np.zeros(14, bool), np.ones(100, bool), np.zeros(14, bool)], 32000)
    assert summary['runs_at_least128_frames'] == 0
    assert summary['technical_forecast_positions'] > 0
    assert summary['positions_in_runs_at_least128_frames'] == 0
    result = continuity_summary(np.r_[np.ones(128, bool), False, np.ones(128, bool)], 32000)
    assert result['runs_at_least128_frames'] == 2
    assert result['positions_in_runs_at_least128_frames'] == 114
