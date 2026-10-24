#!/usr/bin/env python3
"""Close the acquisition phase without changing frozen studies or grouping claims."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bioacoustic_pipeline.ingestion.macde_closeout import validate_decision


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    names = ['macde_provenance_recovery_audit', 'macde_additional_sources_audit',
             'macde_source_resolution', 'macde_source_certification_ledger']
    paths = {n: ROOT / f'outputs/{n}.json' for n in names}
    data = {n: json.loads(p.read_text()) for n, p in paths.items()}
    additional = data['macde_additional_sources_audit']
    missing = additional['infant_marmosets']['summary']['missing_label_referenced_waveforms']
    rows = [
        {'source': 'MarmAudio', 'verdict': 'Blocked', 'waveform_population': '6 selected examples;6 acoustic passes;5 dates',
         'highest_waveform_stage': 'S1', 'source_lineage': False, 'acoustic_support': True,
         'remaining_blocker': 'Clip-to-parent ID/offset and session crosswalk not located in acquired annotations, upstream code index or deposited code bundle',
         'authoritative_artifact': None,
         'reopen_when': 'Obtain an authoritative mapping for these six extracts to parent recordings and offsets, plus a supported session relationship; a hypothetical sidecar is not an identified artifact'},
        {'source': 'CEB tree pipit', 'verdict': 'Resolvable', 'waveform_population': '38 focal files;35 acoustic passes',
         'highest_waveform_stage': 'S1 for35;S0 for3', 'source_lineage': False, 'acoustic_support': '35/38',
         'remaining_blocker': 'Original recording metadata access failed HTTP401; dates, parent sessions and partial/transformed overlap unresolved',
         'authoritative_artifact': 'Original Xeno-canto records listed by immutable xc_id in outputs/macde_ceb_tree_pipit_provenance_manifest.json',
         'reopen_when': 'Obtain original recording metadata for the fixed XC IDs via authorized access or an authoritative deposited export; then assess whether it actually establishes session relationships'},
        {'source': 'Petruskova tree pipit', 'verdict': 'Blocked', 'waveform_population': '30 test WAVs;30 acoustic passes;10 solution-key male identities',
         'highest_waveform_stage': 'S1', 'source_lineage': False, 'acoustic_support': True,
         'remaining_blocker': 'No blind-test-to-recording-code crosswalk;248 spreadsheet date/year conflicts preserved; partial overlap and fieldwork relationship to Stowell unresolved',
         'authoritative_artifact': None,
         'reopen_when': 'Obtain a documented test-ID to original recording-code mapping and authoritative date reconciliation, or a separately preregistered individual-held-out estimand supported by verified identity evidence'},
        {'source': 'Zhao marmosets', 'verdict': 'Resolvable', 'waveform_population': '45 annotation/audio references;39 filename-encoded dates;waveforms not acquired',
         'highest_waveform_stage': 'unverified audio;S0 annotation artifacts', 'source_lineage': False, 'acoustic_support': None,
         'remaining_blocker': 'No waveform integrity/support check;filename timestamps do not certify sessions;subject roles and experimental conditions unresolved',
         'authoritative_artifact': 'https://zenodo.org/records/20317625/files/wave.zip',
         'reopen_when': 'Under a new prospective source plan, match the45 annotation references to verified original audio, check acoustic support, and obtain authoritative subject/session/condition crosswalks'},
        {'source': 'InfantMarmosetsVox', 'verdict': 'Blocked', 'waveform_population': '201 verified native long recordings;203 annotation-referenced paths;2 missing',
         'highest_waveform_stage': 'No new population-wide acoustic certification;closed-v1 subset results retained separately',
         'source_lineage': 'Native recording references and annotation callers identified;session relationships unresolved',
         'acoustic_support': None,
         'remaining_blocker': 'Same previously used corpus;parent/session independence unresolved;full-recording acoustic window search not prospectively specified;two referenced source files absent',
         'authoritative_artifact': None,
         'missing_source_records': missing,
         'reopen_when': 'Obtain authoritative session/grouping documentation and a prospective full-recording sampling rule; retain or recover the two missing originals and explicitly distinguish native/resampled aliases'},
    ]
    for row in rows:
        row.update(independence=False, contract_reference=None, certified_independent_groups=0,
                   partitions_assigned=False, models_fitted=False)
        validate_decision(row)
    output = {
        'phase': 'MACDE source acquisition closeout', 'status': 'closed; no successor fitting population certified',
        'scope': 'Session-separated transfer remains the target. No date-, recordist- or individual-based substitute grouping was adopted.',
        'verdict_definitions': {'Certified': 'All lineage, waveform and independence gates pass an explicit prospective contract',
                               'Resolvable': 'A specific authoritative artifact/reference offers a concrete recovery route; its contents are not assumed sufficient to certify a group',
                               'Blocked': 'A required crosswalk, reconciliation or grouping artifact has not been located; new evidence or a new estimand is needed'},
        'code_sha256': sha(Path(__file__)), 'evidence_contract_sha256': sha(ROOT / 'src/bioacoustic_pipeline/ingestion/macde_closeout.py'),
        'inputs': {str(p.relative_to(ROOT)): sha(p) for p in paths.values()},
        'decisions': rows, 'missing_source_records': missing,
        'no_silent_corrections': 'Petruskova date cells and year columns retained verbatim in source audit;no heuristic repair;Infant missing paths retained explicitly',
        'reopening_rule': 'Revisit a source only when its named evidence trigger changes. Resolvable is not permission to partition or fit.',
        'scientific_limits': 'Eligibility and exact nonduplication do not establish session independence, unseen animals, acquisition-domain transfer, biological causality or semantics',
        'certified_sources': 0, 'certified_independent_groups': 0, 'partitions_assigned': False,
        'models_fitted': False, 'stage2_ready': False,
    }
    out = ROOT / 'outputs/macde_acquisition_closeout.json'
    matrix = ROOT / 'outputs/macde_acquisition_decision_matrix.csv'
    if out.exists() or matrix.exists():
        raise RuntimeError('Refusing to replace acquisition closeout')
    out.write_text(json.dumps(output, indent=2)+'\n')
    columns = ['source', 'verdict', 'waveform_population', 'highest_waveform_stage', 'remaining_blocker', 'authoritative_artifact', 'reopen_when']
    with matrix.open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in columns} for r in rows)
    print(json.dumps({'decisions': {r['source']: r['verdict'] for r in rows}, 'certified_groups': 0,
                      'json': str(out), 'matrix': str(matrix)}, indent=2))


if __name__ == '__main__':
    main()
