"""Exact native PCM reconstruction, including wraparound and corruption."""
import numpy as np
import pytest
from bioacoustic_pipeline.representation.lossless_acoustic import encode, decode


@pytest.mark.parametrize('codec', ['flac', 'delta_zlib'])
@pytest.mark.parametrize('samples', [np.array([-32768, 32767, -1, 0, 1], dtype=np.int16),
                                    np.zeros(3200, dtype=np.int16),
                                    np.random.default_rng(0).integers(-32768,32768,4000,dtype=np.int16)])
def test_exact_roundtrip(codec, samples):
    restored, sr = decode(encode(samples, 32000, codec), codec)
    assert sr == 32000
    assert restored.dtype == np.dtype('int16')
    assert restored.tobytes() == samples.tobytes()


def test_native_format_is_enforced():
    for x in [np.array([.1]), np.zeros((3,2), np.int16), np.array([], np.int16)]:
        with pytest.raises(ValueError):
            encode(x, 32000, 'delta_zlib')


def test_corrupt_predictive_stream_rejected():
    blob = bytearray(encode(np.arange(30, dtype=np.int16), 32000, 'delta_zlib'))
    blob[20] ^= 1  # original PCM digest
    with pytest.raises(ValueError, match='integrity'):
        decode(bytes(blob), 'delta_zlib')
    with pytest.raises(ValueError):
        decode(bytes(blob[:10]), 'delta_zlib')


def test_byte_order_does_not_change_native_sample_values():
    x = np.array([-32768, 15, 32767], dtype='>i2')
    y, _ = decode(encode(x, 44100, 'delta_zlib'), 'delta_zlib')
    np.testing.assert_array_equal(x, y)
