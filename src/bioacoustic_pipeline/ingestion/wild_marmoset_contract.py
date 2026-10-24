"""Non-fitting contract audit for Abreu et al. wild-marmoset annotations."""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from zipfile import ZipFile

MISSING = {'', 'NA', 'NaN', 'nan', 'None', 'Unk', 'UNK'}


def read_workbook(path):
    """Read first worksheet without coercing original cell text or row numbers."""
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(n.itertext()) for n in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', ns)]
        matrix = []
        for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//m:row', ns):
            values = {}
            for cell in row.findall('m:c', ns):
                key = ''.join(c for c in cell.attrib['r'] if c.isalpha())
                v = cell.find('m:v', ns)
                if cell.attrib.get('t') == 'inlineStr':
                    values[key] = ''.join(cell.find('m:is', ns).itertext())
                elif v is not None:
                    values[key] = strings[int(v.text)] if cell.attrib.get('t') == 's' else v.text
            if values:
                matrix.append((int(row.attrib['r']), values))
    headers = matrix[0][1]
    if len(set(headers.values())) != len(headers):
        raise ValueError('Duplicate workbook headers')
    return [dict({headers[k]: v.get(k, '') for k in headers}, excel_row=i) for i, v in matrix[1:]]


def identity_state(value):
    value = str(value).strip()
    return None if value in MISSING else value


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def classify_timing(rows, tolerance_ms=1.):
    by_audio = defaultdict(list)
    for row in rows:
        by_audio[row['ID_Audio']].append(row)
    output = []
    for row in rows:
        start, end, duration = [number(row[k]) for k in ('Begin_Time', 'End_Time', 'Duration')]
        valid = start is not None and end is not None and 0 <= start < end
        missing = any(v is None for v in (start, end, duration))
        delta = end-start-duration if not missing else None
        overlaps = []
        simultaneous = False
        if valid:
            for other in by_audio[row['ID_Audio']]:
                if other['excel_row'] == row['excel_row']:
                    continue
                a, b = number(other['Begin_Time']), number(other['End_Time'])
                if a is None or b is None or b <= a:
                    continue
                simultaneous |= a == start
                overlap = min(end, b) - max(start, a)
                if overlap > 0:
                    overlaps.append({'excel_row': other['excel_row'], 'overlap_ms': overlap,
                                     'other_caller': identity_state(other['ID'])})
        if missing:
            category = 'incomplete_timing'
        elif not valid or duration <= 0:
            category = 'invalid_timing_span_or_duration'
        elif abs(delta) <= tolerance_ms:
            category = 'consistent'
        elif delta > 0 and any(abs(delta-o['overlap_ms']) <= tolerance_ms for o in overlaps):
            category = 'duration_shortfall_matches_overlap'
        elif overlaps:
            category = 'other_overlap_associated_duration_disagreement'
        else:
            category = 'unexplained_duration_disagreement'
        output.append({'excel_row': row['excel_row'], 'audio_id': row['ID_Audio'],
                       'span_valid': valid, 'category': category,
                       'span_minus_stored_duration_ms': delta,
                       'simultaneous_onset': simultaneous, 'overlaps': overlaps,
                       'timing_quarantine': category != 'consistent',
                       'action': 'retain raw timestamps; no correction; quarantine conflicting recording in strict cohort'
                       if category != 'consistent' else 'retain, including valid overlap cases'})
    return output


def parse_recording_date(filename):
    candidates = set()
    for value in re.findall(r'(?<!\d)(?:\d{8}|\d{6})(?!\d)', filename):
        formats = ['%y%m%d'] if len(value) == 6 else ['%d%m%Y', '%Y%m%d']
        for fmt in formats:
            try:
                date = datetime.strptime(value, fmt).date()
            except ValueError:
                continue
            candidates.add(date.isoformat())
    return next(iter(candidates)) if len(candidates) == 1 else None


def role_pair_audit(rows, *, quarantined_audio):
    """One initial S/first R pair per explicitly annotated interaction.

    Roles are required; no arbitrary adjacent caller is promoted to responder.
    These are operational annotation-linked candidates, not verified addressees.
    Unknown events remain in the full row ledger and in earlier history.
    """
    interactions, by_audio = defaultdict(list), defaultdict(list)
    for row in rows:
        interactions[(row['ID_Audio'], row['ID_interaction'])].append(row)
        by_audio[row['ID_Audio']].append(row)
    output = []
    for (audio, interaction), events in sorted(interactions.items()):
        events = sorted(events, key=lambda r: (number(r['Begin_Time']) if number(r['Begin_Time']) is not None else math.inf,
                                               r['excel_row']))
        signal = [r for r in events if r['S_or_R'].strip() == 'S']
        response = [r for r in events if r['S_or_R'].strip() == 'R']
        reasons = []
        if not signal:
            reasons.append('no_annotated_initiator')
        if not response:
            reasons.append('no_annotated_responder')
        s, r = (signal[0] if signal else None), (response[0] if response else None)
        gap = None
        unknown_history, history = [], []
        if s is not None:
            if identity_state(s['ID']) is None:
                reasons.append('unknown_initiator')
            if identity_state(s['Group']) is None:
                reasons.append('unknown_initiator_group')
        if r is not None:
            if r.get('Reply', '').strip() != 'Yes' or r.get('Turns', '').strip() != 'TT':
                reasons.append('responder_reply_marker_not_positive')
            if identity_state(r['ID']) is None:
                reasons.append('unknown_responder')
            if identity_state(r['Group']) is None:
                reasons.append('unknown_responder_group')
        if s is not None and r is not None:
            ss, se = number(s['Begin_Time']), number(s['End_Time'])
            rs, re = number(r['Begin_Time']), number(r['End_Time'])
            if any(t is None for t in (ss, se, rs, re)) or se <= ss or re <= rs:
                reasons.append('invalid_pair_timing')
            else:
                gap = rs-se
                if rs == ss:
                    reasons.append('simultaneous_initiator_responder')
                if rs < ss:
                    reasons.append('responder_precedes_initiator')
                if len([v for v in signal if number(v['Begin_Time']) == ss]) > 1:
                    reasons.append('ambiguous_initial_signaller_onset')
                if len([v for v in response if number(v['Begin_Time']) == rs]) > 1:
                    reasons.append('ambiguous_initial_responder_onset')
                if gap > 10000:
                    reasons.append('first_response_exceeds_initial_signal_10s_window')
                if (identity_state(s['ID']) is not None and identity_state(r['ID']) is not None
                        and (s['Group'].strip(), s['ID'].strip()) == (r['Group'].strip(), r['ID'].strip())):
                    reasons.append('same_known_individual_in_both_roles')
            if ss is not None:
                prior = [x for x in by_audio[audio] if number(x['Begin_Time']) is not None and number(x['Begin_Time']) < ss]
                prior.sort(key=lambda x: (float(x['Begin_Time']), x['excel_row']))
                history = [x['excel_row'] for x in prior]
                unknown_history = [x['excel_row'] for x in prior if identity_state(x['ID']) is None]
        if audio in quarantined_audio:
            reasons.append('recording_timing_quarantine')
        output.append({'audio_id': audio, 'interaction_id': interaction,
                       'initiator_excel_row': s['excel_row'] if s else None,
                       'responder_excel_row': r['excel_row'] if r else None,
                       'initiator_id': identity_state(s['ID']) if s else None,
                       'responder_id': identity_state(r['ID']) if r else None,
                       'initiator_group': s['Group'].strip() if s else None,
                       'responder_group': r['Group'].strip() if r else None,
                       'gap_ms': gap, 'overlapping': gap is not None and gap < 0,
                       'prior_history_rows': history, 'unknown_prior_history_rows': unknown_history,
                       'eligible': not reasons, 'exclusion_reasons': reasons})
    return output
