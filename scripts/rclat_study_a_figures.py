#!/usr/bin/env python3
"""Export descriptive RCLAT pilot curves without changing frozen endpoints."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT/'outputs/rclat_study_a_results.json'
    result = json.loads(source.read_text())
    spec = json.loads((ROOT/'outputs/rclat_study_a_preregistration.json').read_text())
    prefix = ROOT/'outputs/rclat_study_a'
    expected = [Path(str(prefix)+suffix) for suffix in
                ['_prefix_recovery.png','_prefix_recovery.svg','_ctc_curves.png','_ctc_curves.svg',
                 '_transcript_progression.csv','_artifact_verification.json']]
    if any(p.exists() for p in expected):
        raise RuntimeError('Figure artifacts already exist')
    fig,ax = plt.subplots(figsize=(7,4.5))
    styles = {'intact':('o','-'),'lossless':('s',':'),
              'block_shuffle':('^','--'),'silence':('x','-.')}
    names = {'intact':'Intact','lossless':'Lossless sham','block_shuffle':'500 ms block shuffle','silence':'Silence'}
    for condition,(marker,line) in styles.items():
        x = spec['prefix_fractions']
        y = [result['summary'][condition][str(f)]['speaker_macro_wer'] for f in x]
        ax.plot(x,y,marker=marker,linestyle=line,label=names[condition],
                markerfacecolor='none' if condition == 'lossless' else None)
    ax.set(xlabel='Fraction of utterance available',ylabel='Speaker-macro error against FULL transcript',
           title='Frozen decoder prefix recovery: 16 utterances, 8 speakers',xticks=spec['prefix_fractions'])
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    fig.text(.5,.015,'Prefix errors include not-yet-spoken words. This is not ordinary prefix ASR WER.',
             ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1))
    for suffix in ['png','svg']:
        fig.savefig(Path(str(prefix)+'_prefix_recovery.'+suffix),dpi=180)
    plt.close(fig)
    fig,axes = plt.subplots(4,4,figsize=(14,10))
    for ax,item in zip(axes.flat,spec['sources']):
        rows = sorted([r for r in result['rows'] if r['utterance_id'] == item['utterance_id'] and r['condition'] == 'intact'],key=lambda r:r['fraction'])
        times = [r['available_seconds'] for r in rows]
        scores = [r['message']['ctc_log_score'] if r['message']['ctc_log_score'] is not None else np.nan for r in rows]
        ax.plot(times,scores,'o-',color='tab:blue',markersize=4)
        ax.set_title(item['utterance_id'],fontsize=9)
        ax.set_xlabel('Available seconds',fontsize=8)
        ax.set_ylabel('CTC log score',fontsize=8)
        missing = sum(r['message']['ctc_log_score'] is None for r in rows)
        if missing:
            ax.text(.02,.96,f'{missing} infeasible prefix(es)',transform=ax.transAxes,va='top',fontsize=7)
        ax.grid(alpha=.2)
        ax.tick_params(labelsize=7)
    fig.suptitle('Intact-speech full-reference CTC scores across independently decoded prefixes',fontsize=13)
    fig.text(.5,.01,'Missing scores are structurally infeasible, not zeros. Scores are decoder-relative; not semantic entropy or local word timing.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.96))
    for suffix in ['png','svg']:
        fig.savefig(Path(str(prefix)+'_ctc_curves.'+suffix),dpi=180)
    plt.close(fig)
    references = {s['utterance_id']:s['normalized_transcript'] for s in spec['sources']}
    path = Path(str(prefix)+'_transcript_progression.csv')
    with path.open('w',newline='') as output:
        writer = csv.DictWriter(output,fieldnames=['utterance_id','speaker_id','condition','fraction',
            'available_seconds','reference','decoded_text','full_reference_wer','ctc_log_score',
            'ctc_log_score_accrual','waveform_rmse'])
        writer.writeheader()
        for row in result['rows']:
            writer.writerow({'utterance_id':row['utterance_id'],'speaker_id':row['speaker_id'],
                'condition':row['condition'],'fraction':row['fraction'],'available_seconds':row['available_seconds'],
                'reference':references[row['utterance_id']],'decoded_text':row['decoded_text'],
                'full_reference_wer':row['metrics']['wer'],'ctc_log_score':row['message']['ctc_log_score'],
                'ctc_log_score_accrual':row['ctc_log_score_accrual'],
                'waveform_rmse':row['distortion']['waveform_rmse']})
    assert len(result['rows']) == 256
    assert result['sham_exact_checks'] == 64
    assert result['real_model_future_exclusion_checks'] == 16
    for row in result['rows']:
        if row['condition'] == 'lossless':
            assert row['distortion']['waveform_rmse'] == 0
        if row['condition'] == 'block_shuffle':
            assert row['ctc_log_score_accrual'] is None
    verification = {'result_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'verified_rows':256,'sample_exact_sham_checks':64,'real_model_future_exclusion_checks':16,
        'no_accrual_assigned_to_non_nested_shuffled_prefixes':True,
        'figures_descriptive_only':True,'stage2_ready':False,
        'artifacts':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in expected[:-1]]}
    expected[-1].write_text(json.dumps(verification,indent=2)+'\n')
    print('Exported prefix curves,16 CTC score panels,transcript progression and artifact verification.')


if __name__ == '__main__':
    main()
