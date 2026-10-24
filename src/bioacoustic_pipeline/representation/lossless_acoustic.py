"""Reference codecs for mono signed PCM16. No learned or semantic representation.

Predictive format v1: little-endian header (magic,rate,count,PCM SHA256), then
zlib level9 of little-endian uint16 zigzag-coded modular first differences.
The first sample is predicted as zero. Sizes include all codec headers.
"""
import hashlib
import io
import struct
import zlib

import numpy as np
import soundfile as sf

HEADER = struct.Struct('<4sIQ32s')


def encode(samples, sample_rate, codec):
    x = np.asarray(samples)
    if x.ndim != 1 or not len(x) or x.dtype.kind != 'i' or x.dtype.itemsize != 2:
        raise ValueError('Nonempty mono native signed PCM16 required')
    if isinstance(sample_rate, bool) or int(sample_rate) != sample_rate or not 0 < sample_rate < 2**32:
        raise ValueError('Invalid integer sample rate')
    pcm = np.ascontiguousarray(x, dtype='<i2')
    if codec == 'flac':
        output = io.BytesIO()
        sf.write(output, pcm, int(sample_rate), format='FLAC', subtype='PCM_16', compression_level=1.)
        return output.getvalue()
    if codec != 'delta_zlib':
        raise ValueError('Unknown codec')
    differences = np.diff(pcm.astype(np.int32), prepend=0)
    residual = ((differences + 32768) % 65536 - 32768).astype(np.int32)
    zigzag = ((residual << 1) ^ (residual >> 31)).astype('<u2')
    return HEADER.pack(b'ADZ1', int(sample_rate), len(pcm), hashlib.sha256(pcm.tobytes()).digest()) + zlib.compress(zigzag.tobytes(), level=9)


def decode(blob, codec):
    if codec == 'flac':
        with sf.SoundFile(io.BytesIO(blob)) as source:
            if source.channels != 1 or source.subtype != 'PCM_16':
                raise ValueError('Expected mono PCM16 FLAC')
            return source.read(dtype='int16'), source.samplerate
    if codec != 'delta_zlib':
        raise ValueError('Unknown codec')
    if len(blob) <= HEADER.size:
        raise ValueError('Truncated predictive stream')
    magic, rate, count, digest = HEADER.unpack(blob[:HEADER.size])
    if magic != b'ADZ1' or rate == 0 or not 0 < count <= 100_000_000:
        raise ValueError('Invalid predictive header')
    inflater = zlib.decompressobj()
    try:
        raw = inflater.decompress(blob[HEADER.size:], 2*count+1)
    except zlib.error as exc:
        raise ValueError('Corrupt predictive stream') from exc
    if len(raw) != 2*count or not inflater.eof or inflater.unused_data:
        raise ValueError('Invalid predictive payload length')
    zigzag = np.frombuffer(raw, '<u2').astype(np.int64)
    residual = (zigzag >> 1) ^ -(zigzag & 1)
    samples = (np.cumsum(residual) % 65536).astype('<u2').view('<i2')
    if hashlib.sha256(samples.tobytes()).digest() != digest:
        raise ValueError('Predictive PCM integrity failure')
    return samples, rate
