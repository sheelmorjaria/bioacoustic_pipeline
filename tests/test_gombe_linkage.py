import io
import struct
import pytest
from bioacoustic_pipeline.ingestion.gombe_linkage import riff_chunks, explicit_tape_marker, validate_link


def test_riff_inspection_finds_chunks_and_respects_padding():
    payload = b'fmt ' + struct.pack('<I', 1) + b'x\0' + b'cue ' + struct.pack('<I', 4) + b'\0'*4
    stream = io.BytesIO(b'RIFF'+struct.pack('<I',len(payload)+4)+b'WAVE'+payload)
    chunks = riff_chunks(stream)
    assert [r['id'] for r in chunks] == ['fmt ', 'cue ']
    assert chunks[1]['payload_offset'] == 30


def test_tape_markers_do_not_supply_sample_coordinates():
    assert explicit_tape_marker('Pallas grunts (ADULT TAPE III STARTS HERE)') == 3
    assert explicit_tape_marker('Adult chimpanzee vocalizes') is None
    row = {'sample_start': None, 'sample_end': None, 'interval_verified': False, 'group_id': None}
    validate_link(row)
    row['interval_verified'] = True
    with pytest.raises(ValueError):
        validate_link(row)


def test_source_identity_alone_does_not_authorize_group_assignment():
    row = {'sample_start': None, 'sample_end': None, 'interval_verified': False, 'group_id': 'session1'}
    with pytest.raises(ValueError):
        validate_link(row)
