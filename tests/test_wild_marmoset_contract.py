"""Wild-marmoset audit mechanics, not fitted scientific outcome tests."""
import copy

from bioacoustic_pipeline.ingestion.wild_marmoset_contract import (
    classify_timing, identity_state, role_pair_audit, parse_recording_date,
)


def event(i, start, end, duration, caller='a', role='S', interaction='1', audio='1'):
    return {'excel_row': i, 'Begin_Time': str(start), 'End_Time': str(end),
            'Duration': str(duration), 'ID': caller, 'Group': 'POM', 'S_or_R': role,
            'ID_interaction': interaction, 'ID_Audio': audio, 'File': 'x_221017_08_30_00.eaf',
            'Reply': 'Yes' if role == 'R' else 'No', 'Turns': 'TT' if role == 'R' else 'No_TT'}


def test_overlapping_calls_are_not_negative_duration_errors_or_repaired():
    rows = [event(2, 0, 100, 80), event(3, 80, 200, 100, caller='b', role='R')]
    original = copy.deepcopy(rows)
    records = classify_timing(rows)
    assert all(r['span_valid'] for r in records)
    assert all(r['category'] == 'duration_shortfall_matches_overlap' for r in records)
    assert rows == original


def test_duration_mismatch_and_simultaneity_have_separate_states():
    rows = [event(2,0,100,100), event(3,0,50,50,caller='b',role='R'),
            event(4,200,300,12,interaction='2')]
    flags = classify_timing(rows)
    assert flags[0]['category'] == 'consistent'
    assert flags[0]['simultaneous_onset']
    assert flags[2]['category'] == 'unexplained_duration_disagreement'
    pairs = role_pair_audit(rows, quarantined_audio=set())
    assert not any(p['eligible'] for p in pairs)
    assert 'simultaneous_initiator_responder' in pairs[0]['exclusion_reasons']


def test_unknown_target_is_not_scored_but_unknown_history_is_preserved():
    rows = [event(2,0,10,10,caller=' Unk',interaction='0'),
            event(3,20,30,10,caller='a',interaction='1'),
            event(4,40,50,10,caller='b',role='R',interaction='1'),
            event(5,60,70,10,caller='a',interaction='2'),
            event(6,80,90,10,caller='Unk',role='R',interaction='2')]
    pairs = role_pair_audit(rows, quarantined_audio=set())
    pair = next(p for p in pairs if p['interaction_id']=='1')
    assert pair['eligible']
    assert pair['unknown_prior_history_rows'] == [2]
    unknown = next(p for p in pairs if p['interaction_id']=='2')
    assert 'unknown_responder' in unknown['exclusion_reasons']
    assert identity_state(' Unk') is None
    assert identity_state('a') == 'a'


def test_roles_are_required_instead_of_labelling_any_next_caller_a_responder():
    rows = [event(2,0,10,10), event(3,20,30,10,caller='b')]
    pairs = role_pair_audit(rows, quarantined_audio=set())
    assert not pairs[0]['eligible']
    assert 'no_annotated_responder' in pairs[0]['exclusion_reasons']


def test_quarantine_is_recording_scoped_and_date_typos_are_not_silently_fixed():
    rows = [event(2,0,10,10), event(3,20,30,10,caller='b',role='R')]
    assert 'recording_timing_quarantine' in role_pair_audit(rows, quarantined_audio={'1'})[0]['exclusion_reasons']
    assert parse_recording_date('bhy_pom_221017_06;22;00_SIM.eaf') == '2022-10-17'
    assert parse_recording_date('bhy_sws_22922_06;51;00_LEI.eaf') is None


def test_unambiguous_eight_digit_filename_dates_are_parsed_without_editing():
    assert parse_recording_date('bhy_coq_Ccm_30072022_06;35;00_GRA.eaf') == '2022-07-30'
    assert parse_recording_date('bhy_coq_Ccm_20220730_06;35;00_GRA.eaf') == '2022-07-30'


def test_workbook_loader_retains_raw_text_and_physical_excel_rows(tmp_path):
    from zipfile import ZipFile
    from bioacoustic_pipeline.ingestion.wild_marmoset_contract import read_workbook
    path = tmp_path/'test.xlsx'
    with ZipFile(path,'w') as z:
        z.writestr('xl/sharedStrings.xml','<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><t>ID</t></si><si><t>Begin_Time</t></si><si><t xml:space="preserve"> Unk</t></si></sst>')
        z.writestr('xl/worksheets/sheet1.xml','<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row><row r="5"><c r="A5" t="s"><v>2</v></c><c r="B5"><v>123.5</v></c></row></sheetData></worksheet>')
    rows = read_workbook(path)
    assert rows == [{'ID':' Unk','Begin_Time':'123.5','excel_row':5}]


def test_a_receiver_role_without_a_positive_reply_marker_is_not_a_scored_reply():
    rows = [event(2,0,10,10), event(3,20,30,10,caller='b',role='R')]
    rows[1]['Reply']='No'
    rows[1]['Turns']='No_TT'
    pair=role_pair_audit(rows, quarantined_audio=set())[0]
    assert not pair['eligible']
    assert 'responder_reply_marker_not_positive' in pair['exclusion_reasons']


def test_unknown_first_responder_is_not_skipped_to_find_a_known_target():
    rows = [event(2,0,10,10), event(3,20,30,10,caller='Unk',role='R'),
            event(4,40,50,10,caller='b',role='R')]
    pair=role_pair_audit(rows, quarantined_audio=set())[0]
    assert pair['responder_excel_row']==3
    assert 'unknown_responder' in pair['exclusion_reasons']


def test_missing_reply_markers_are_not_assumed_positive():
    rows = [event(2,0,10,10), event(3,20,30,10,caller='b',role='R')]
    rows[1].pop('Reply',None)
    rows[1].pop('Turns',None)
    pair=role_pair_audit(rows, quarantined_audio=set())[0]
    assert 'responder_reply_marker_not_positive' in pair['exclusion_reasons']
