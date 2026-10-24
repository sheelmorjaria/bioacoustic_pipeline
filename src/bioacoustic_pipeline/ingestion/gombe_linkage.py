"""Evidence-preserving Gombe linkage checks; no cut-boundary inference."""
import re
import struct


def riff_chunks(stream):
    header = stream.read(12)
    if len(header) != 12 or header[:4] != b'RIFF' or header[8:] != b'WAVE':
        raise ValueError('Expected RIFF WAVE')
    chunks = []
    while True:
        header = stream.read(8)
        if not header:
            return chunks
        if len(header) != 8:
            raise ValueError('Truncated RIFF chunk header')
        name, size = struct.unpack('<4sI', header)
        offset = stream.tell()
        chunks.append({'id': name.decode('ascii', errors='replace'), 'bytes': size, 'payload_offset': offset})
        stream.seek(offset + size + size % 2)


def explicit_tape_marker(text):
    match = re.search(r'ADULT\s+TAPE\s+(VII|VI|IV|V|III|II|I|[1-7])\s+STARTS\s+HERE', text, re.I)
    if match is None:
        return None
    romans = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7}
    value = match[1].upper()
    return int(value) if value.isdigit() else romans[value]


def validate_link(row):
    if row['interval_verified'] and (row['sample_start'] is None or row['sample_end'] is None or
                                     row['sample_end'] <= row['sample_start']):
        raise ValueError('Verified intervals need independently established start/end samples')
    if row['group_id'] is not None and not (row.get('session_relationship_verified') and row.get('grouping_contract')):
        raise ValueError('Group assignment requires verified relationships and a prospective contract')
