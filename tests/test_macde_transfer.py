import numpy as np
import pytest
from bioacoustic_pipeline.diagnostics.macde_transfer import recording_group, history_order, partition


def test_recording_groups_keep_related_clips_together():
    assert recording_group('pipit', 'pipit2017fg_train_0212_0037.wav') == recording_group('pipit', 'pipit2017fg_train_0212_0042.wav')
    assert recording_group('chiffchaff', 'cutted_day1_PC1103_0115.wav') == 'day1'
    assert recording_group('marmoset', '20160930_Twin5_marmoset2.wav') == '20160930'
    with pytest.raises(ValueError):
        recording_group('pipit', 'unparseable.wav')
    assert partition('pipit', 'group1') == partition('pipit', 'group1')


def test_matched_neural_disruption_preserves_values_masks_and_recent_states():
    order = history_order(123, disrupted=True)
    np.testing.assert_array_equal(np.sort(order), np.arange(52))
    np.testing.assert_array_equal(order[-2:], [50, 51])
    assert not np.array_equal(order[:50], np.arange(50))
    np.testing.assert_array_equal(order, history_order(123, disrupted=True))
    np.testing.assert_array_equal(history_order(123, disrupted=False), np.arange(52))


def test_release_partitions_do_not_split_same_bird_proxy():
    assert recording_group('littleowl', 'littleowl2017fg_train_CER_0004.wav') == recording_group('littleowl', 'littleowl2017fg_test_CER_0012.wav')
    assert recording_group('pipit', 'pipit2017fg_train_0212_0037.wav') == recording_group('pipit', 'pipit2017fg_test_0212_0011.wav')


def test_neural_batches_shuffle_both_streams_and_masks_without_mutating_targets():
    import importlib.util
    import sys
    from pathlib import Path
    import torch
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('macde_transfer_test', scripts / 'macde_transfer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = {}
    for name, width in [('spectral', 64), ('physical', 12)]:
        data[name] = torch.arange(128, dtype=torch.float32)[None, :, None].expand(1, 128, width).clone()
        data[name+'_valid'] = (data[name].long() % 2 == 0)
    positions = np.array([[0, 51]])
    output = module.batches(data, positions, 'combined_disrupted', [0])
    np.testing.assert_array_equal(output['spectral'][0, :, 0], output['physical'][0, :, 0])
    np.testing.assert_array_equal(output['physical_valid'], output['physical'].long() % 2 == 0)
    np.testing.assert_array_equal(output['physical'][0, -2:, 0], [50, 51])
    assert data['physical'][0, 55, 0] == 55
    physical = module.batches(data, positions, 'physical', [0])
    assert not physical['spectral'].any() and not physical['spectral_valid'].any()
    spectral = module.batches(data, positions, 'spectral', [0])
    assert not spectral['physical'].any() and not spectral['physical_valid'].any()


def test_insufficient_support_blocks_fitting_before_run_marker(tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys
    from pathlib import Path
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('macde_transfer_gate_test', scripts / 'macde_transfer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = {'frozen': True}
    registration = tmp_path / 'preregistration.json'
    registration.write_text(module.pilot.text(plan))
    cache = tmp_path / 'cache.npz'
    cache.write_bytes(b'no model data needed')
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps({'preregistration_sha256': module.pilot.sha(registration),
                                'cache_sha256': module.pilot.sha(cache), 'support_gate_passed': False}))
    monkeypatch.setattr(module, 'SPEC', registration)
    monkeypatch.setattr(module, 'AUDIT', audit)
    monkeypatch.setattr(module, 'CACHE', cache)
    monkeypatch.setattr(module, 'protocol', lambda: plan)
    monkeypatch.setattr(module, 'PREFIX', tmp_path / 'study')
    with pytest.raises(RuntimeError, match='fitting is blocked'):
        module.run()
    assert not list(tmp_path.glob('*RUN_STARTED*'))
