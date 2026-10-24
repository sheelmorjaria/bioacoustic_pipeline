import numpy as np
import torch
import pytest
from bioacoustic_pipeline.diagnostics.rclat import (
    prefix_audio, transform_audio, greedy_text, message_score, text_errors,
)


def test_future_audio_cannot_enter_prefix_or_transform():
    x = np.arange(16000,dtype=np.int16)
    y = x.copy()
    y[8000:] = -999
    for condition in ['intact','lossless','block_shuffle','silence']:
        a = transform_audio(prefix_audio(x,.5),16000,condition,seed=2)
        b = transform_audio(prefix_audio(y,.5),16000,condition,seed=2)
        np.testing.assert_array_equal(a,b)
        assert len(a) == 8000


def test_shuffle_preserves_sample_inventory_and_sham_is_exact():
    x = np.arange(16000,dtype=np.int16)
    shuffled = transform_audio(x,16000,'block_shuffle',seed=5)
    np.testing.assert_array_equal(np.sort(shuffled),np.sort(x))
    assert not np.array_equal(x,shuffled)
    np.testing.assert_array_equal(transform_audio(x,16000,'lossless',seed=0),x)


def test_ctc_greedy_collapses_repeats_but_preserves_blank_separation():
    labels = ('-','|','A','B')
    ids = [2,2,0,2,1,3,3]
    logits = torch.full((len(ids),4),-10.)
    logits[torch.arange(len(ids)),ids] = 10
    assert greedy_text(logits,labels) == 'AA B'


def test_impossible_full_message_is_missing_not_zero_likelihood():
    labels = ('-','|','A','B')
    out = message_score(torch.zeros(2,4),'AA',labels)
    assert not out['structurally_feasible'] and out['ctc_log_score'] is None
    out = message_score(torch.zeros(4,4),'AA',labels)
    assert out['structurally_feasible'] and np.isfinite(out['ctc_log_score'])


def test_full_reference_prefix_errors_are_distinct_from_word_inventory():
    out = text_errors('A B','B A')
    assert out['word_errors'] == 2 and out['bag_word_overlap'] == 2
    assert text_errors('A B','A')['word_errors'] == 1


def test_invalid_prefix_is_rejected():
    with pytest.raises(ValueError):
        prefix_audio(np.ones(10),0)
