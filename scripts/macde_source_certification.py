#!/usr/bin/env python3
"""Apply explicit S0-S3 evidence stages without certifying session independence."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bioacoustic_pipeline.ingestion.macde_source_ids import certification_level


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    paths = [ROOT / 'outputs/macde_source_resolution.json', ROOT / 'outputs/macde_additional_sources_audit.json']
    prior, additional = [json.loads(p.read_text()) for p in paths]
    rows = []
    for source, records in [('MarmAudio', prior['marmaudio']['records']),
                            ('CEB_tree_pipit', prior['ceb_tree_pipit']['records']),
                            ('Petruskova', additional['petruskova']['records'])]:
        for r in records:
            acoustic = bool(r['acoustic_audit']['passes_acoustic_support'])
            rows.append({'source': source, 'id': r.get('id', str(r.get('test_id'))),
                         'entity': 'waveform', 'level': certification_level(integrity=True, acoustic=acoustic, provenance=False, independent=False),
                         'acoustic_support': acoustic, 'parent_session_resolved': False,
                         'independent_group_certified': False, 'partition': None})
    for r in additional['zhao']['records']:
        rows.append({'source': 'Zhao', 'id': r['annotation_member'], 'entity': 'annotation_artifact',
                     'level': 'S0', 'audio_reference': r['recording_reference'],
                     'acoustic_support': None, 'parent_session_resolved': False,
                     'independent_group_certified': False, 'partition': None,
                     'note': 'Verified annotation archive; original audio not acquired or certified'})
    result = {'convention': 'New audit layer S0 acquired verified artifact; S1 waveform eligible; S2 parent/session resolved; S3 contract-certified independent group. This is not a preregistration or a weaker grouping protocol.',
              'code_sha256': sha(Path(__file__)),
              'inputs': {str(p.relative_to(ROOT)): sha(p) for p in paths}, 'records': rows,
              'certified_independent_groups': 0, 'partitions_assigned': False, 'models_fitted': False, 'stage2_ready': False,
              'remaining_requirements': {'MarmAudio': 'Original clip-to-parent ID/offset and session crosswalk',
                                        'CEB_tree_pipit': 'Authoritative original recording dates and parent/session relationships; partial-overlap audit',
                                        'Petruskova': 'Blind test ID to original recording-code crosswalk and reconciliation of date/year conflicts; Stowell source relationship',
                                        'Zhao': 'Original waveform acquisition/integrity/support, validated recording/session/subject-condition crosswalk'},
              'claims': 'Negative exact deduplication is not independence. Date, recordist and subject diversity are not certified sessions.'}
    out = ROOT / 'outputs/macde_source_certification_ledger.json'
    if out.exists():
        raise RuntimeError('Refusing to overwrite ledger')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print('Saved', out)


if __name__ == '__main__':
    main()
