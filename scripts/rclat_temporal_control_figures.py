#!/usr/bin/env python3
"""Descriptive speaker-level figures and verification for temporal controls."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/rclat_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import rclat_study_a as pilot

ROOT = pilot.ROOT


def main():
    prefix = ROOT/'outputs/rclat_temporal_control'
    result_path = Path(str(prefix)+'_results.json')
    spec_path = Path(str(prefix)+'_preregistration.json')
    result = json.loads(result_path.read_text())
    spec = json.loads(spec_path.read_text())
    verification_path = Path(str(prefix)+'_verification.json')
    if verification_path.exists():
        raise RuntimeError('Verification already exists')
    assert result['preregistration_sha256'] == pilot.sha(spec_path)
    assert len(result['rows']) == 400 and result['lossless_output_checks'] == 80
    for path,digest in spec['dependencies'].items():
        assert pilot.sha(ROOT/path) == digest
    assert all(m['cell_inventory_exact'] and m['energy_exact'] and m['relative_rmse_mismatch'] <= .05 for m in spec['matching'])
    conditions = spec['conditions']
    rows = result['cohorts']['prospective_extension']['summary']
    assert rows['intact']['speakers'] == 32 and rows['intact']['utterances'] == 64
    assert result['cohorts']['pilot_replication']['summary']['intact']['speakers'] == 8
    fig,axes = plt.subplots(1,2,figsize=(12,5))
    names = ['Intact','Lossless','Taper sham','Global 500 ms','Local 20 ms']
    for ax,metric,ylabel in zip(axes,['bag_word_recall','wer'],['Word-inventory recall','Word error rate']):
        for i,c in enumerate(conditions):
            values = np.array([row[metric] for _,row in sorted(rows[c]['per_speaker'].items())])
            ax.scatter(i+np.linspace(-.13,.13,len(values)),values,alpha=.45,s=20,color=f'C{i}')
            ax.plot([i-.23,i+.23],[values.mean()]*2,color='black',lw=2)
        ax.set_xticks(range(len(conditions)),names,rotation=20,ha='right')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Full-utterance recovery: 64 utterances from 32 extension speakers')
    fig.text(.5,.015,'Points: speakers; black lines: means. Common tapered cells and matched waveform RMSE do not establish perceptual equivalence.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94))
    artifacts = []
    for suffix in ['png','svg']:
        path = Path(str(prefix)+'_speaker_results.'+suffix)
        if path.exists():
            raise RuntimeError('Figure already exists')
        fig.savefig(path,dpi=180)
        artifacts.append(path)
    plt.close(fig)
    path = Path(str(prefix)+'_condition_summary.csv')
    if path.exists():
        raise RuntimeError('Summary already exists')
    with path.open('w',newline='') as output:
        writer = csv.DictWriter(output,fieldnames=['cohort','condition','speakers','utterances',
            'macro_wer','macro_cer','macro_bag_word_recall'])
        writer.writeheader()
        for cohort,values in result['cohorts'].items():
            for condition,summary in values['summary'].items():
                writer.writerow({'cohort':cohort,'condition':condition,**{k:summary[k] for k in
                    ['speakers','utterances','macro_wer','macro_cer','macro_bag_word_recall']}})
    artifacts.append(path)
    for source in spec['sources']:
        original = next(r for r in result['rows'] if r['utterance_id'] == source['utterance_id'] and r['condition'] == 'intact')
        lossless = next(r for r in result['rows'] if r['utterance_id'] == source['utterance_id'] and r['condition'] == 'lossless')
        assert original['decoded_text'] == lossless['decoded_text'] and original['metrics'] == lossless['metrics']
    pilot.freeze_json(verification_path,{'results_sha256':pilot.sha(result_path),
        'preregistration_sha256':pilot.sha(spec_path),'frozen_dependencies_unchanged':True,
        'verified_condition_evaluations':400,'lossless_output_checks':80,
        'all80_acoustic_matches_passed':True,
        'maximum_relative_rmse_mismatch':max(m['relative_rmse_mismatch'] for m in spec['matching']),
        'taper_control_passed':result['taper_preservation_control_passed'],
        'prefix_relative_scoring_executed':False,'stage2_ready':False,
        'artifacts':[{'path':str(p.relative_to(ROOT)),'sha256':pilot.sha(p)} for p in artifacts]})
    print('Verified400 evaluations,80exactshams,all acoustic matches and unchanged prior studies.')


if __name__ == '__main__':
    main()
