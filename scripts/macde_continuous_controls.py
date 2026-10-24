"""Separately frozen past-only summary and temporal-order pilot controls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import macde_stage1 as study
from bioacoustic_pipeline.diagnostics.macde_controls import history_controls

ROOT = study.ROOT
SPEC = ROOT / 'outputs/macde_continuous_controls_preregistration.json'
HELPER = ROOT / 'src/bioacoustic_pipeline/diagnostics/macde_controls.py'


def protocol():
    return {
        'study': 'MACDE continuous trajectory controls, prospective companion',
        'primary_preregistration_sha256': study.sha(study.PREREG),
        'source_audit_sha256': study.sha(study.MANIFEST),
        'code': {str(p.relative_to(ROOT)): study.sha(p) for p in (Path(__file__), HELPER)},
        'arms': ['summary: mean and standard deviation of same 52 past/current frames',
                 'ordered: flattened same 52 frames',
                 'scrambled: independently permute oldest50 frames per source/frame; retain newest2'],
        'seed': 'sha256(source index:frame), first8 bytes little endian',
        'endpoint': '40ms physical forecasting; identical primary heldout source/frame rows',
        'fitting': 'same Ridge alpha grid; fit-only selection on validation; refit fit+validation',
        'contrasts': ['summary minus frozen primary R2', 'scrambled minus ordered', 'ordered minus R2'],
        'uncertainty': '2000 paired recording-date bootstrap, descriptive secondary analysis',
        'scope': 'no whole-future-call summaries, learned units, receiver outcomes or semantic claims; original event boundaries remain source metadata, not inferred internal units',
        'stop_rule': 'one run; no changes to primary eligibility, architecture or success criteria',
    }


def design(arrays, items, indices):
    _, target, groups, rows = study.designs(arrays, {}, items, indices, 4)
    controls = []
    for i, t in rows:
        seed = int.from_bytes(hashlib.sha256(f'{i}:{t}'.encode()).digest()[:8], 'little')
        controls.append(history_controls(arrays['physical'][i, t-51:t+1], seed))
    return {k: np.asarray([v[j] for v in controls])
            for j, k in enumerate(('summary', 'ordered', 'scrambled'))}, target, groups, rows


def run():
    if SPEC.read_text() != study.text(protocol()):
        raise RuntimeError('Control specification changed')
    result_path = ROOT / 'outputs/macde_stage1_results.json'
    primary = json.loads(result_path.read_text())
    if primary['preregistration_sha256'] != study.sha(study.PREREG):
        raise RuntimeError('Primary provenance mismatch')
    residual_path = ROOT / primary['residual_artifact']['path']
    if study.sha(residual_path) != primary['residual_artifact']['sha256']:
        raise RuntimeError('Primary residual checksum mismatch')
    manifest, arrays = study.load()
    study.freeze(ROOT / 'outputs/macde_continuous_controls_RUN_STARTED.json', {'spec_sha256': study.sha(SPEC)})
    items = manifest['items']
    parts = {s: np.array([i for i, v in enumerate(items) if v['split'] == s], int)
             for s in ('fit', 'validation', 'heldout')}
    xt, yt, _, _ = design(arrays, items, parts['fit'])
    xv, yv, _, _ = design(arrays, items, parts['validation'])
    choices = {k: study.choose_ridge(xt[k], yt, xv[k], yv) for k in xt}
    xf, yf, _, _ = design(arrays, items, np.sort(np.r_[parts['fit'], parts['validation']]))
    xe, ye, groups, rows = design(arrays, items, parts['heldout'])
    with np.load(residual_path) as stored:
        np.testing.assert_array_equal(stored['h4_source_frame'], rows)
        res = {'R2': stored['h4_R2'].copy()}
    for k in choices:
        res[k] = study.residual(study.fit_ridge(xf[k], yf, choices[k]['alpha']), xe[k], ye)
    output = {
        'spec_sha256': study.sha(SPEC), 'positions': len(rows), 'ridge_selection': choices,
        'nrmse': {k: study.loss(v) for k, v in res.items()},
        'summary_minus_R2': study.boot(res['summary'], res['R2'], groups),
        'scrambled_minus_ordered': study.boot(res['scrambled'], res['ordered'], groups),
        'ordered_minus_R2': study.boot(res['ordered'], res['R2'], groups),
        'interpretation': 'Secondary acoustic prediction controls; no evidence about receiver behaviour or meaning',
    }
    study.freeze(ROOT / 'outputs/macde_continuous_controls_results.json', output)
    print(study.text(output), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preregister', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.preregister:
            if (ROOT / 'outputs/macde_stage1_RUN_STARTED.json').exists():
                raise RuntimeError('Must freeze before primary training')
            study.freeze(SPEC, protocol())
        else:
            run()
