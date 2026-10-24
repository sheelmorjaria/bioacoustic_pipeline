#!/usr/bin/env python3
"""Recompute preserved single-day support and distinguish reported from saved results."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bioacoustic_pipeline.diagnostics.macde_data import frame_observations
from bioacoustic_pipeline.ingestion.bfsongrepo_audit import continuity_summary


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    torch.set_num_threads(2)
    final_path = ROOT / 'outputs/bfsongrepo_tier_a_audit.json'
    final = json.loads(final_path.read_text())
    frozen_path = ROOT / 'outputs/macde_stage1_source_audit.json'
    frozen = json.loads(frozen_path.read_text())
    code_checks = {p: {'frozen_sha256': digest, 'current_sha256': sha(ROOT/p), 'matches': digest == sha(ROOT/p)}
                   for p, digest in frozen['code_provenance'].items() if p.startswith('src/')}
    if not all(v['matches'] for v in code_checks.values()):
        raise ValueError('Frozen dependency drift')
    if sha(ROOT/'scripts/bfsongrepo_tier_a_audit.py') != final['code_sha256'] or sha(ROOT/'src/bioacoustic_pipeline/ingestion/bfsongrepo_audit.py') != final['helper_sha256']:
        raise ValueError('Final audit implementation changed')
    mask_path = ROOT/final['support_mask']['path']
    if sha(mask_path) != final['support_mask']['sha256']:
        raise ValueError('Preserved support-mask integrity mismatch')
    rows = {r['wav_member']: r for r in final['records']}
    comparisons = []
    with np.load(mask_path) as stored, tarfile.open(ROOT/'data/bfsongrepo/wav_csv.tar.gz') as archive:
        for member in archive:
            if not member.isfile() or member.name not in rows:
                continue
            audio, sr = sf.read(io.BytesIO(archive.extractfile(member).read()))
            valid = frame_observations(audio, sr)['time_valid']
            summary = continuity_summary(valid, sr)
            equal = np.array_equal(valid, stored[Path(member.name).name])
            summary_equal = summary == rows[member.name]['full_recording_support']
            if not equal or not summary_equal:
                raise ValueError('Recomputed source support differs: '+member.name)
            comparisons.append({'recording': member.name, 'mask_identical': equal, 'summary_identical': summary_equal})
    result = {'purpose': 'Resolve claimed preliminary BFSongRepo discrepancy without rewriting original audit',
              'code_sha256': sha(Path(__file__)), 'final_audit_sha256': sha(final_path),
              'frozen_stage1_audit_sha256': sha(frozen_path), 'dependency_checks': code_checks,
              'recomputed_recordings': len(comparisons), 'comparisons': comparisons,
              'verified_result': final['summary'],
              'preliminary_0_of39': {'status': 'unverified external/user-reported claim',
                                     'local_artifact_located': False,
                                     'cause': 'unknown;no original source checksum,code,configuration or mask located',
                                     'interpretation': 'Do not manufacture a correction from a local zero-of39 audit that was not found. The earlier synthetic test failure concerned a100-frame fixture,not a39-recording dataset result.'},
              'contract_distinction': 'Frozen feature extraction and eligibility dependencies match;the single-day audit intentionally reports whole-recording support and first256-frame support separately. This does not reproduce or reopen Stage1 fixed128-frame sampling or establish an unchanged model-evaluation population.',
              'ledger_conclusion': '39-of39 is reproduced with identical masks;the claimed earlier zero-of39 result and its cause remain unverified. No evidence of frontend/physics/eligibility code drift was detected.',
              'models_fitted': False, 'partitions_assigned': False, 'certified_independent_sessions': 0}
    out = ROOT/'outputs/bfsongrepo_reconciliation.json'
    if out.exists():
        raise RuntimeError('Refusing to overwrite reconciliation')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print('Recomputed identical masks and summaries for',len(comparisons),'recordings',flush=True)


if __name__ == '__main__':
    main()
