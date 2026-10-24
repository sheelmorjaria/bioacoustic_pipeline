"""Receiver-grounding contracts, written before the implementation."""
import hashlib
import zipfile

import numpy as np
import pytest

from bioacoustic_pipeline.ingestion.receiver_grounding import (
    parse_birdpark_events, covers_window, response_label, join_stimulus_features,
    independent_groups, verified_extract, endpoint_gate,
)


def segment(**changes):
    row = dict(onset=1, offset=10, segmentType='consolidated', bird='a',
               transmitter='a', nOverlapSyl=1)
    row.update(changes)
    return row


def test_birdpark_sample_origin_and_inclusive_offset():
    events, excluded = parse_birdpark_events([segment()], 10, ['a', 'b'])
    assert not excluded
    assert events[0]['onset_s'] == 0
    assert events[0]['offset_s'] == 1
    assert events[0]['caller'] == 'a'
    assert events[0]['recipient'] is None
    assert events[0]['overlapping_annotation']


def test_birdpark_does_not_pool_annotation_layers_or_guess_callers():
    rows = [segment(), segment(segmentType='transmitter-based'),
            segment(transmitter='None'), segment(bird='z', transmitter='z'),
            segment(onset=0), segment(offset=0)]
    events, excluded = parse_birdpark_events(rows, 10, ['a', 'b'])
    assert len(events) == 1
    assert sum(excluded.values()) == 5


def test_coverage_never_bridges_gaps_and_cannot_be_inferred_from_calls():
    assert covers_window(0, 2, [(0, 1), (1, 2)])
    assert not covers_window(0, 2, [(0, .9), (1, 2)])
    assert not covers_window(0, 2, [])
    assert not covers_window(-.1, 1, [(0, 2)])
    assert not covers_window(0, 2.1, [(0, 2)])


def test_missing_and_censored_response_labels_are_distinct():
    assert response_label(3., True, 10., 5.) == 1
    assert response_label(None, False, 10., 5.) == 0
    assert response_label(None, None, 10., 5.) is None
    assert response_label(None, False, 4., 5.) is None
    assert response_label(7., True, 10., 5.) == 0
    assert response_label(5., True, 10., 5.) == 1
    assert response_label(np.nan, None, 10., 5.) is None
    with pytest.raises(ValueError):
        response_label(-1, True, 10, 5)


def test_stimulus_join_uses_exact_unique_identifiers():
    trials = [{'stimulus_id': 'x', 'subject': 'a'}, {'stimulus_id': 'x', 'subject': 'b'}]
    assert len(join_stimulus_features(trials, [{'call_id': 'x', 'acoustic': [1, 2]}])) == 2
    with pytest.raises(ValueError, match='Missing'):
        join_stimulus_features(trials, [{'call_id': 'prefix_x', 'acoustic': [1, 2]}])
    with pytest.raises(ValueError, match='Duplicate'):
        join_stimulus_features(trials, [{'call_id': 'x'}, {'call_id': 'x'}])


def test_subjects_and_reused_stimuli_cannot_cross_evaluation_groups():
    rows = [{'subject': 'a', 'stimulus_id': 'x'}, {'subject': 'a', 'stimulus_id': 'y'},
            {'subject': 'b', 'stimulus_id': 'y'}, {'subject': 'c', 'stimulus_id': 'z'}]
    groups = independent_groups(rows)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_download_checksum_precedes_extraction_and_zip_slip_is_rejected(tmp_path):
    path = tmp_path / 'data.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('../escape', 'bad')
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='checksum'):
        verified_extract(path, tmp_path / 'out', 'sha256:' + '0' * 64)
    with pytest.raises(ValueError, match='Unsafe'):
        verified_extract(path, tmp_path / 'out', 'sha256:' + checksum)
    assert not (tmp_path / 'escape').exists()


def test_contact_microphone_is_not_independently_validated_behaviour():
    state = endpoint_gate(coverage=True, attributable=True, independent=False,
                          validated=False, groups=2, valid_events=1000)
    assert not state['ready']
    assert 'independent_outcome' in state['blocked_by']
    assert 'validated_behaviour' in state['blocked_by']
    assert 'independent_groups' in state['blocked_by']
    assert endpoint_gate(coverage=True, attributable=True, independent=True,
                         validated=True, groups=4, valid_events=20)['ready']


def test_elephant_loader_uses_corrected_file_and_never_treats_video_time_as_stimulus(tmp_path):
    from bioacoustic_pipeline.ingestion.receiver_grounding import load_elephant_playback
    name = '20250704_playback_data_for_upload_corrected.csv'
    (tmp_path/name).write_text('Date,Time,Subject,Caller,Treatment,Playback,DurVigPre,DurVigPost,DurVigDiff,LatVoc,VocCen\n20200101,900,a,b,T,12.5,0,5,5,600,0\n')
    trials = load_elephant_playback(tmp_path)
    assert trials[0]['stimulus_id'] is None
    assert trials[0]['video_playback_offset_s'] == 12.5
    assert trials[0]['vocal_response_600s'] == 0
    assert trials[0]['vigilance_post_s'] == 5
    assert trials[0]['addressed_condition'] == 1


def test_elephant_loader_does_not_fall_back_to_uncorrected_file(tmp_path):
    from bioacoustic_pipeline.ingestion.receiver_grounding import load_elephant_playback
    (tmp_path/'20230409_playback_data_for_upload.csv').write_text('Subject\na\n')
    with pytest.raises(FileNotFoundError):
        load_elephant_playback(tmp_path)


def test_birdpark_h5_metadata_controls_timeline_and_candidate_availability(tmp_path):
    import h5py
    from bioacoustic_pipeline.ingestion.receiver_grounding import audit_birdpark_pair
    data = tmp_path/'test.h5'
    with h5py.File(data, 'w') as f:
        rec = f.create_group('recInfo')
        rec.attrs['sampleRate_Hz'] = 10.
        rec.attrs['frameRate_Hz'] = 2.
        rec.attrs['nSamplesPerFrame'] = 5
        rec.attrs['radioChNames'] = ['a','b']
        rec.attrs['recId'] = 'r'
        info = f.create_group('fileInfo')
        info.attrs['nSamplesInFile'] = 100
        info.attrs['nFramesInFile'] = 20
        f.create_dataset('radioSignals',data=np.zeros((2,100)))
        f.create_dataset('daqSignals',data=np.zeros((1,100)))
    csv_path=tmp_path/'test_annotations.csv'
    csv_path.write_text('onset,offset,segmentType,bird,transmitter,nOverlapSyl\n11,20,consolidated,a,a,0\n')
    info, candidates = audit_birdpark_pair(data,csv_path,pre_seconds=1,horizon_seconds=5)
    assert info['duration_s'] == 10
    assert info['available_birds'] == ['a','b']
    assert info['temporally_bounded_candidate_windows'] == 1
    assert info['validated_behaviour_windows'] == 0
    assert candidates[0]['candidate_receiver'] == 'b'
    assert candidates[0]['recipient'] is None
    assert candidates[0]['prediction_time_s'] == 2
    assert not candidates[0]['behaviour_validated']
