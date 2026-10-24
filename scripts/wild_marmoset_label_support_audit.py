#!/usr/bin/env python3
"""Audit reply call-type support in the frozen 178-pair population; never fit.

No label pooling, class selection, date reassignment, timestamp corrections,
or dropping rare/unseen heldout targets is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bioacoustic_pipeline.contextual.response_type_support import label_support

AUDIT=ROOT/'outputs/wild_marmoset_exchange_eligibility_audit.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    source=json.loads(AUDIT.read_text())
    for name,expected in source['provenance']['input_sha256'].items():
        if sha(ROOT/'data/wild_marmoset'/name)!=expected:
            raise ValueError(f'Frozen source changed: {name}')
    contract=ROOT/'src/bioacoustic_pipeline/ingestion/wild_marmoset_contract.py'
    if sha(contract)!=source['provenance']['contract_module_sha256']:
        raise ValueError('Frozen exchange contract changed')
    raw={r['excel_row']:r['raw'] for r in source['row_ledger']}
    recs={r['audio_id']:r for r in source['records']}
    samples=[]
    for pair in source['role_pair_ledger']:
        if not pair['eligible']:
            continue
        rec=recs[pair['audio_id']]
        initiator=raw[pair['initiator_excel_row']]
        responder=raw[pair['responder_excel_row']]
        if responder['S_or_R']!='R' or responder['Reply']!='Yes' or responder['Turns']!='TT':
            raise ValueError('Exchange eligibility changed')
        if float(responder['Begin_Time'])<=float(initiator['End_Time']):
            raise ValueError('Unexpected already-started reply in strict cohort')
        sample={'event_id':f'{pair["audio_id"]}:{pair["interaction_id"]}',
                'audio_id':pair['audio_id'],'interaction_id':pair['interaction_id'],
                'initiator_excel_row':pair['initiator_excel_row'],
                'target_excel_row':pair['responder_excel_row'],
                'target_field':'Vocalization','target':responder['Vocalization'],
                'target_role':responder['S_or_R'],
                'partition':rec['date_partition_candidate'] or 'unassigned', 'date':rec['date'],
                'unknown_prior_history_rows':pair['unknown_prior_history_rows']}
        samples.append(sample)
    if len(samples)!=178:
        raise ValueError('Frozen eligible population changed')
    screening=label_support(samples)
    return {
        'schema_version':1,'models_fitted':False,'cohort_modified':False,
        'source':'Abreu et al. 2026; 10.6084/m9.figshare.30061930.v1',
        'provenance':{'exchange_audit_sha256':sha(AUDIT),
                       'input_sha256':source['provenance']['input_sha256'],
                       'script_sha256':sha(Path(__file__)),
                       'support_module_sha256':sha(ROOT/'src/bioacoustic_pipeline/contextual/response_type_support.py'),
                       'source_readme':'data/wild_marmoset/README_Hallmarks.txt; dictionary entry Vocalization'},
        'target_contract':{
            'target':'annotated call type (Vocalization) on the explicitly labelled first responder row',
            'target_role_counts':dict(Counter(s['target_role'] for s in samples)),
            'distinct_call_type_categories':len({s['target'] for s in samples}),
            'not_role_tautology':'All candidate targets have role R, but their call types vary; the call-type annotation is a distinct field, not a transformation of S/R.',
            'annotation_independence':'Call types and roles are distinct coded properties; separate blinded annotators for these labels are not established by this audit.',
            'target_is_post_event':'The reply label is determined post hoc for supervised evaluation. It is not available at prediction time and must never enter predictor construction.',
            'no_semantic_mapping':'Original call-type labels retained literally; no invented reply intentions or semantic categories.',
            'no_acoustic_measurements':'Release contains annotations, not source audio. No claim of testing a learned acoustic encoder.',
        },
        'cohort':{'eligible_pairs':len(samples),'assigned_pairs':sum(s['partition']!='unassigned' for s in samples),
                   'partition_counts':dict(Counter(s['partition'] for s in samples)),
                   'date_counts':screening['date_counts'],
                   'recording_counts':{p:len({s['audio_id'] for s in samples if s['partition']==p}) for p in ('fit','heldout','unassigned')},
                   'unknown_history_pairs':sum(bool(s['unknown_prior_history_rows']) for s in samples),
                   'rules':'same 178 strict pairs and original date assignments; four unresolved-date pairs remain unassigned; 51 timing-quarantined recordings remain excluded'},
        'support_screen':screening,
        'decision':{
            'run_preregistered_baseline_comparison':screening['fit_allowed'],
            'status':'ABSTAIN' if not screening['fit_allowed'] else 'eligible_for_separate_preregistration',
            'reason':'Full original response-type target fails support screening: heldout Chirp is absent from fit; Submissive Cry and Phee_String have two fit examples each and no heldout examples.',
            'floors_status':'These are conservative label-support screening floors, not a preregistered experiment or a power guarantee. The unseen heldout class blocks the proposed full-label comparison independently of the numerical floors.',
            'no_adaptation':'No dropping heldout Chirp, collapsing Other classes, restricting to common calls, switching to binary targets or reassigning dates after label inspection.',
            'identity_fallback':'Not attempted: call types exist but are poorly supported; switching estimands is not an automatic remedy and candidate-availability metadata remain unresolved.',
            'no_study_preregistration':'No fitting protocol registered because the required support gate failed.',
        },
        'future_prediction_time_contract':{
            'onset':'Only initiating identity/context and strictly prior observations can enter; whole initiating call type/duration is not assumed available at onset.',
            'offset':'A future baseline using the completed initiating call type must declare prediction at initiating offset and require the responder has not started; all 178 strict candidates satisfy that temporal check.',
            'response_fields':'Responder type, identity, start/end, duration, and response-derived columns are targets/eligibility only, never predictors.',
            'history':'Keep unknown identity events explicitly in preceding history; do not interpolate identity or concatenate recordings.',
            'forbidden_exchange_inputs':['Reply','S_or_R','Turns','Beginning','ID_interaction','Vocalization_Pair','Reply_to_Rep'],
        },
        'blocked_endpoints':['intended recipient','response/no-response','semantic meaning','independent nonvocal receiver behaviour'],
        'independence':'Date groups are protected, but animals and social groups recur; only a conditional within-population evaluation could be supported.',
        'sample_ledger':samples,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/wild_marmoset_response_type_label_support_audit.json')
    args=parser.parse_args()
    result=build()
    payload=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if args.output.exists() and args.output.read_text()!=payload:
        raise RuntimeError('Refusing to overwrite a differing frozen support audit')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(payload)
    print(json.dumps({'output':str(args.output),'sha256':sha(args.output),
                      'cohort':result['cohort'],'support':result['support_screen'],
                      'decision':result['decision']['status']},indent=2))


if __name__=='__main__':
    main()
