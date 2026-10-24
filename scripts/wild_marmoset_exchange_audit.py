#!/usr/bin/env python3
"""Freeze timing, identity, role and partition eligibility without fitting models.

Original workbook values are preserved. Duration disagreements quarantine the
whole recording in the strict cohort; overlap alone is not an exclusion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from bioacoustic_pipeline.ingestion.wild_marmoset_contract import (
    classify_timing, identity_state, parse_recording_date, read_workbook, role_pair_audit,
)

DATA = ROOT/'data/wild_marmoset'
PUBLICATION = 'https://www.nature.com/articles/s41598-026-60403-2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'


def freeze(path, value):
    payload = canonical(value)
    if path.exists() and path.read_text() != payload:
        raise RuntimeError(f'Existing frozen artifact differs; choose a new --output: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def audit():
    manifest_path = ROOT/'outputs/wild_marmoset_download_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for f in manifest['files']:
        path = DATA/f['file']
        if sha(path) != f['sha256'] or hashlib.md5(path.read_bytes()).hexdigest() != f['md5']:
            raise ValueError(f'Input changed from verified download: {path}')
    rows = read_workbook(DATA/'Data_Hallmarks.xlsx')
    by_row = {r['excel_row']: r for r in rows}
    timing = classify_timing(rows)
    flagged = {r['excel_row'] for r in timing if r['timing_quarantine']}
    unknown = {r['excel_row'] for r in rows if identity_state(r['ID']) is None}
    quarantined = {by_row[i]['ID_Audio'] for i in flagged}
    by_audio, by_interaction = defaultdict(list), defaultdict(list)
    for r in rows:
        by_audio[r['ID_Audio']].append(r)
        by_interaction[r['ID_interaction']].append(r)
    pairs = role_pair_audit(rows, quarantined_audio=quarantined)
    span_only_pairs = role_pair_audit(rows, quarantined_audio=set())
    row_ledger = []
    for r,t in zip(rows,timing):
        row_ledger.append({'excel_row': r['excel_row'], 'raw': r,
                           'normalized_caller': identity_state(r['ID']),
                           'normalized_group': identity_state(r['Group']),
                           'identity_whitespace_normalized': r['ID'] != r['ID'].strip(),
                           'timing': t, 'strict_recording_quarantine': r['ID_Audio'] in quarantined})
    # Components of known social groups co-occurring in the same coded recording.
    group_names = sorted({g for r in rows if (g := identity_state(r['Group'])) is not None})
    parent = {g:g for g in group_names}

    def find(g):
        while parent[g] != g:
            g = parent[g]
        return g

    for rec_rows in by_audio.values():
        gs = sorted({g for r in rec_rows if (g := identity_state(r['Group'])) is not None})
        for g in gs[1:]:
            a,b = sorted((find(gs[0]),find(g)))
            parent[b] = a
    records = []
    filename_to_audio = defaultdict(set)
    for audio,events in sorted(by_audio.items(), key=lambda x:int(x[0])):
        files = sorted({r['File'] for r in events})
        dates = sorted({d for f in files if (d := parse_recording_date(f)) is not None})
        date = dates[0] if len(dates)==1 and all(parse_recording_date(f) is not None for f in files) else None
        # Cohort metadata only; no model or scored-target partition tuning.
        partition = None
        if date is not None:
            h = int(hashlib.sha256(('wild-marmoset-audit-v1:'+date).encode()).hexdigest(),16)%100
            partition = 'fit' if h<70 else 'heldout'
        for f in files:
            filename_to_audio[f].add(audio)
        eligible = [p for p in pairs if p['audio_id']==audio and p['eligible']]
        gs = sorted({g for r in events if (g := identity_state(r['Group'])) is not None})
        records.append({'audio_id':audio, 'filenames':files, 'date':date,
                        'date_partition_candidate':partition,
                        'date_unresolved':date is None, 'rows':len(events),
                        'unknown_caller_rows':sum(r['excel_row'] in unknown for r in events),
                        'timing_quarantined':audio in quarantined,
                        'strict_eligible_role_pairs':len(eligible), 'known_social_groups':gs,
                        'known_social_component':find(gs[0]) if gs else None})
    eligible = [p for p in pairs if p['eligible']]
    strict_rows = [r for r in rows if r['ID_Audio'] not in quarantined]
    clean_known_ids = {(r['Group'].strip(), r['ID'].strip()) for r in strict_rows if identity_state(r['ID']) is not None}
    partition_eligible = Counter()
    partition_callers = defaultdict(set)
    for rec in records:
        partition_eligible[str(rec['date_partition_candidate'])] += rec['strict_eligible_role_pairs']
        if rec['date_partition_candidate'] is not None:
            for r in by_audio[rec['audio_id']]:
                if identity_state(r['ID']) is not None:
                    partition_callers[rec['date_partition_candidate']].add((r['Group'].strip(),r['ID'].strip()))
    adjacency = Counter()
    for events in by_audio.values():
        events = sorted(events,key=lambda r:(float(r['Begin_Time']),float(r['End_Time'])))
        for i,event in enumerate(events):
            if i==0:
                adjacency[f'first_recording_event_Reply={event["Reply"]}']+=1
                continue
            prev=events[i-1]
            gap=float(event['Begin_Time'])-float(prev['End_Time'])
            different = identity_state(event['ID']) != identity_state(prev['ID'])
            adjacency[f'Reply={event["Reply"]};raw_normalized_labels_different={different};gap_le_10s={gap<=10000}']+=1
    flags_unknown = len(flagged & unknown)
    joint = {'unknown_and_timing_flag':flags_unknown,
             'unknown_without_timing_flag':len(unknown-flagged),
             'known_with_timing_flag':len(flagged-unknown),
             'known_without_timing_flag':len(rows)-len(unknown|flagged)}
    unknown_rate = flags_unknown/len(unknown)
    known_rate = len(flagged-unknown)/(len(rows)-len(unknown))
    timing_by_group = {}
    for group in sorted({r['Group'].strip() for r in rows}):
        subset = [r for r in rows if r['Group'].strip()==group]
        timing_by_group[group] = {'rows':len(subset),
                                 'unknown':sum(r['excel_row'] in unknown for r in subset),
                                 'timing_flags':sum(r['excel_row'] in flagged for r in subset),
                                 'both':sum(r['excel_row'] in flagged & unknown for r in subset)}
    group_components = defaultdict(list)
    for g in group_names:
        group_components[find(g)].append(g)
    result = {
        'schema_version':1, 'models_fitted':False,
        'study':'Abreu et al. 2026 wild-marmoset exchange eligibility audit',
        'source_doi':manifest['doi'],
        'provenance':{'download_manifest_sha256':sha(manifest_path),
                       'input_sha256':{f['file']:f['sha256'] for f in manifest['files']},
                       'audit_script_sha256':sha(Path(__file__)),
                       'contract_module_sha256':sha(ROOT/'src/bioacoustic_pipeline/ingestion/wild_marmoset_contract.py'),
                       'publication_methods':PUBLICATION},
        'source_selection_contract':{
            'publication_evidence':'Methods: Recording techniques, Data extraction and coding, and Flexibility of turn-taking organization',
            'summary':'Field notes identified initiators and replying animals when possible. Only 703 of about 5820 audio files were selected for quality, audibility and identity. Exchange coding used a ten-second timing rule. These labels do not prove independently assigned intended recipients or complete nonresponse monitoring.',
            'raw_audio_or_eaf_in_release':False,
            'nonresponse_prediction':'blocked: no complete recording observation bounds or selection-denominator risk set'},
        'identity':{
            'rows':len(rows), 'raw_exact_Unk':sum(r['ID']=='Unk' for r in rows),
            'normalized_unknown':len(unknown),'unknown_fraction':len(unknown)/len(rows),
            'normalization_ledger':[{'excel_row':r['excel_row'],'raw':r['ID'],'normalized_state':'unknown' if identity_state(r['ID']) is None else identity_state(r['ID'])}
                                    for r in rows if r['ID'] != r['ID'].strip()],
            'policy':'unknowns remain explicit observations in row/history ledgers; do not impute identity from neighbouring calls; unknown target or initiator excluded only from confidently identified role-pair cohort',
            'known_identity_labels':sorted({r['ID'].strip() for r in rows if identity_state(r['ID']) is not None}),
            'global_identity_key':'(social group, ID); no aliases merged; individual labels alone are not assumed globally unique',
            'known_identity_keys_in_unquarantined_recordings':len(clean_known_ids)},
        'timing':{
            'units':'milliseconds per README', 'tolerance_ms':1,
            'category_counts':dict(Counter(t['category'] for t in timing)),
            'original_99_flags_reproduced':len(flagged)==99,
            'nonpositive_or_missing_endpoint_spans':sum(not t['span_valid'] for t in timing),
            'events_with_raw_endpoint_overlap':sum(bool(t['overlaps']) for t in timing),
            'unflagged_overlap_events':sum(bool(t['overlaps']) and not t['timing_quarantine'] for t in timing),
            'simultaneous_onset_events':sum(t['simultaneous_onset'] for t in timing),
            'quarantined_recordings':len(quarantined),
            'corrections_applied':0,
            'classification_limit':'Overlap matching explains a numerical pattern, not provenance of the stored Duration field. No raw audio/EAF or authoritative correction key is supplied; all duration conflicts remain unresolved.',
            'strict_policy':'quarantine entire recording for any missing/invalid time or stored-duration disagreement >1 ms; prevent contaminated intervals entering history. Overlap alone is not an exclusion.',
            'sensitivity_only':'raw-endpoint-only candidates are reported for support assessment, not approved as corrected data',
            'distinct_time_reference_status':'timestamps group by original audio; no evidence justifies rescaling isolated discrepant rows or stitching recordings'},
        'identity_timing_overlap':{
            'event_contingency':joint, 'unknown_timing_flag_rate':unknown_rate,
            'known_timing_flag_rate':known_rate, 'descriptive_rate_ratio':unknown_rate/known_rate,
            'interpretation':'descriptive composition diagnostic, not a significance test or proof of missingness mechanism',
            'either_event_issue':len(unknown|flagged), 'by_social_group':timing_by_group},
        'role_label_contract':{
            'role_counts':dict(Counter(r['S_or_R'] for r in rows)),
            'reply_counts':dict(Counter(r['Reply'] for r in rows)),
            'reply_direction':'README describes whether a call is a reply; Yes is a backward-looking current-call label, not a future response-to-current outcome.',
            'adjacency_diagnostic':dict(adjacency),
            'initiator_responder_roles':'S_or_R and interaction IDs explicitly supplied; field notes support individual attribution, but exchange membership is partly defined by temporal proximity.',
            'pair_definition':'one earliest annotated S and earliest annotated R in the same interaction/audio; require positive Reply=Yes/Turns=TT on R, singleton initial onsets, known distinct identities/groups, positive onset order, first R onset within 10 seconds of initial S offset; no invented responder for S-only interactions',
            'repeat_policy':'initial signal is not replaced post hoc by the last repeated S to make the ten-second criterion pass',
            'causal_inputs':'Reply, Turns, Beginning, S_or_R, response-derived intervals, and interaction membership cannot automatically be used as predictors; labels and eligibility only',
            'history':'all strictly earlier annotated events in same recording, including unknowns; equal-onset events are unordered; no concatenation across audio'},
        'exchange_eligibility':{
            'annotated_interactions':len(by_interaction), 'operational_pair_records':len(pairs),
            'interactions_with_both_roles':sum(p['initiator_excel_row'] is not None and p['responder_excel_row'] is not None for p in pairs),
            'strict_known_identity_temporal_role_pairs':len(eligible),
            'strict_pairs_recordings':len({p['audio_id'] for p in eligible}),
            'strict_pairs_dated_recordings':sum(r['strict_eligible_role_pairs']>0 and r['date'] is not None for r in records),
            'strict_pairs_distinct_parsed_dates':len({r['date'] for r in records if r['strict_eligible_role_pairs']>0 and r['date'] is not None}),
            'strict_pairs_with_unknown_prior_history':sum(bool(p['unknown_prior_history_rows']) for p in eligible),
            'strict_overlapping_role_pairs':sum(p['overlapping'] for p in eligible),
            'strict_offset_time_eligible_nonoverlapping_pairs':sum(not p['overlapping'] for p in eligible),
            'raw_endpoints_only_support_pairs_NOT_approved':sum(p['eligible'] for p in span_only_pairs),
            'raw_endpoints_only_overlapping_support_pairs_NOT_approved':sum(p['eligible'] and p['overlapping'] for p in span_only_pairs),
            'exclusion_reason_counts_nonexclusive':dict(Counter(reason for p in pairs for reason in p['exclusion_reasons'])),
            'pair_issue_overlap':dict(Counter(('identity_and_timing' if any(x.startswith('unknown_') for x in p['exclusion_reasons']) and 'recording_timing_quarantine' in p['exclusion_reasons']
                                             else 'identity_only' if any(x.startswith('unknown_') for x in p['exclusion_reasons'])
                                             else 'timing_recording_only' if 'recording_timing_quarantine' in p['exclusion_reasons'] else 'neither_identity_nor_recording_timing_issue') for p in pairs))},
        'independence':{
            'coded_recordings':len(by_audio),
            'recordings_without_any_timing_conflict':len(by_audio)-len(quarantined),
            'audio_ids_mapping_to_multiple_files':[a for a,rr in by_audio.items() if len({r['File'] for r in rr})>1],
            'filenames_mapping_to_multiple_audio_ids':{f:sorted(aa) for f,aa in filename_to_audio.items() if len(aa)>1},
            'interaction_ids_crossing_audio':[i for i,rr in by_interaction.items() if len({r['ID_Audio'] for r in rr})>1],
            'parsed_dates':len({r['date'] for r in records if r['date']}),
            'unresolved_date_recordings':[r['audio_id'] for r in records if r['date_unresolved']],
            'date_parser':'exact six-digit YYMMDD and eight-digit DDMMYYYY/YYYYMMDD tokens; require unique valid date; no repair of five-digit dates',
            'candidate_date_split':'SHA256(wild-marmoset-audit-v1:YYYY-MM-DD) modulo100 <70 => fit, else heldout; all same-day recordings together; unresolved dates unassigned',
            'candidate_split_strict_pair_counts':dict(partition_eligible),
            'known_callers_shared_between_date_partitions':sorted([list(x) for x in partition_callers['fit'] & partition_callers['heldout']]),
            'social_group_components':dict(group_components),
            'limits':'Audio is the minimum block, not proof of independent session. Five-minute clips may share a 30-minute focal session. Date blocks support within-population temporal holdout; shared individuals/groups prevent unseen-animal or independent-population claims. Unknown-group events prevent fully verified social independence.'},
        'endpoint_decision':{
            'annotated_responder_identity':'conditionally supported by strict role-pair cohort; candidate availability roster and label-provenance restrictions require a separate modelling specification',
            'response_type':'feasible candidate for an initial baseline in the strict labelled-exchange cohort; no fit authorized in this audit',
            'intended_recipient':'not established independently of observed replies',
            'nonresponse':'blocked', 'independent_nonvocal_receiver_behaviour':'not in release',
            'prediction_time':'must be frozen before fitting: initiating onset permits only onset-available inputs; offset-time prediction must exclude already-started overlapping responders',
            'protected_evaluation':'conditional labelled-exchange study possible with date/recording grouping and recurring-social-group limitations; no claim of a population-complete or independent-addressee benchmark'},
        'records':records, 'row_ledger':row_ledger, 'role_pair_ledger':pairs,
    }
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/wild_marmoset_exchange_eligibility_audit.json')
    args=ap.parse_args()
    result=audit()
    freeze(args.output,result)
    print(canonical({'output':str(args.output),'sha256':sha(args.output),
                     'identity':{k:result['identity'][k] for k in ('rows','raw_exact_Unk','normalized_unknown')},
                     'timing_categories':result['timing']['category_counts'],
                     'quarantined_recordings':result['timing']['quarantined_recordings'],
                     'joint_issues':result['identity_timing_overlap']['event_contingency'],
                     'eligibility':result['exchange_eligibility'],
                     'date_split_counts':result['independence']['candidate_split_strict_pair_counts']}))


if __name__=='__main__':
    main()
