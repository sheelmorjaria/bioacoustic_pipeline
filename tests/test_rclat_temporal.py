from collections import Counter
import numpy as np
from bioacoustic_pipeline.diagnostics.rclat_temporal import tapered_cells, rearrange, match_distortion


def cells(x):
    return [row.tobytes() for row in x[:len(x)//320*320].reshape(-1,320)]


def test_all_temporal_arms_have_exact_same_cell_inventory_and_energy():
    x = np.random.default_rng(0).integers(-20000,20000,16000,dtype=np.int16)
    sham = tapered_cells(x,16000)
    for policy in ['global','local']:
        y = rearrange(sham,16000,policy,7)
        assert Counter(cells(y)) == Counter(cells(sham))
        assert np.sum(y.astype(np.int64)**2) == np.sum(sham.astype(np.int64)**2)
        assert (y.reshape(-1,320)[:,[0,-1]] == 0).all()


def test_global_moves_macroblocks_local_preserves_each_macro_inventory():
    x = np.arange(16000,dtype=np.int16)
    sham = tapered_cells(x,16000)
    local = rearrange(sham,16000,'local',3)
    for start in [0,8000]:
        assert Counter(cells(local[start:start+8000])) == Counter(cells(sham[start:start+8000]))
    global_audio = rearrange(sham,16000,'global',3)
    np.testing.assert_array_equal(global_audio[:8000],sham[8000:16000])


def test_tail_is_unchanged_and_pair_matching_is_deterministic():
    x = np.random.default_rng(2).integers(-12000,12000,32321,dtype=np.int16)
    sham = tapered_cells(x,16000)
    a,b,info = match_distortion(sham,16000,seed=4,candidates=8)
    c,d,other = match_distortion(sham,16000,seed=4,candidates=8)
    np.testing.assert_array_equal(a,c)
    np.testing.assert_array_equal(b,d)
    assert info == other
    np.testing.assert_array_equal(a[32000:],sham[32000:])
    np.testing.assert_array_equal(b[32000:],sham[32000:])
