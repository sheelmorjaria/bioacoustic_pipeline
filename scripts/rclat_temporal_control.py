#!/usr/bin/env python3
"""Two-scale speech perturbation study;full-utterance linguistic references only."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from threadpoolctl import threadpool_limits

import rclat_study_a as pilot
from mfa_alignment_qa import parse_textgrid
from bioacoustic_pipeline.diagnostics.rclat import greedy_text,text_errors,normalize_text,transform_audio
from bioacoustic_pipeline.diagnostics.rclat_temporal import tapered_cells,match_distortion,rmse

ROOT = pilot.ROOT
PREFIX = ROOT/'outputs/rclat_temporal_control'
PLAN = Path(str(PREFIX)+'_plan.json')
SPEC = Path(str(PREFIX)+'_preregistration.json')
RESULT = Path(str(PREFIX)+'_results.json')
CACHE = ROOT/'cache/rclat_temporal_control.npz'
CONDITIONS = ['intact','lossless','taper_sham','global','local']


def source_population():
    prior = json.loads(pilot.SPEC.read_text())
    old_speakers = {s['speaker_id'] for s in prior['sources']}
    transcripts = {}
    for path in sorted(pilot.CORPUS.rglob('*.trans.txt')):
        for line in path.read_text().splitlines():
            uid,text = line.split(' ',1)
            if uid in transcripts:
                raise ValueError('Repeated transcript ID')
            transcripts[uid] = (text,path)
    groups = defaultdict(list)
    for path in sorted(pilot.CORPUS.rglob('*.flac')):
        info = sf.info(path)
        if info.samplerate == 16000 and info.channels == 1 and info.subtype == 'PCM_16' and 5 <= info.duration <= 10:
            groups[path.stem.split('-')[0]].append(path)
    subsets = {}
    for line in (ROOT/'data/LibriSpeech/SPEAKERS.TXT').read_text().splitlines():
        if line.strip() and not line.startswith(';'):
            fields = [f.strip() for f in line.split('|')]
            subsets[fields[0]] = fields[2]
    sources = []
    for speaker in sorted(groups,key=pilot.rank):
        if subsets.get(speaker) != 'dev-clean':
            raise ValueError('Speaker metadata conflict')
        if len(groups[speaker]) < 2:
            continue
        for path in sorted(groups[speaker],key=lambda p:pilot.rank(p.stem))[:2]:
            text,transcript = transcripts[path.stem]
            sources.append({'utterance_id':path.stem,'speaker_id':speaker,
                'path':str(path.relative_to(ROOT)),'sha256':pilot.sha(path),
                'transcript':normalize_text(text),'transcript_path':str(transcript.relative_to(ROOT)),
                'transcript_sha256':pilot.sha(transcript),
                'cohort':'pilot_replication' if speaker in old_speakers else 'prospective_extension'})
    if len({s['speaker_id'] for s in sources}) != 40 or len(sources) != 80:
        raise ValueError('Expected40speakers/two utterances each;do not change selection')
    if {s['utterance_id'] for s in prior['sources']} != {s['utterance_id'] for s in sources if s['cohort'] == 'pilot_replication'}:
        raise ValueError('Pilot population not preserved')
    return sources


def dependencies():
    paths = [Path(__file__).resolve(),ROOT/'scripts/rclat_study_a.py',
        ROOT/'scripts/mfa_alignment_qa.py',ROOT/'src/bioacoustic_pipeline/diagnostics/rclat.py',
        ROOT/'src/bioacoustic_pipeline/diagnostics/rclat_temporal.py',
        ROOT/'src/bioacoustic_pipeline/representation/lossless_acoustic.py',
        ROOT/'tests/test_rclat_temporal.py',pilot.CHECKPOINT,pilot.SPEC,pilot.RESULT,
        ROOT/'data/LibriSpeech/SPEAKERS.TXT',
        ROOT/'outputs/mfa_alignment_qa.json',ROOT/'outputs/mfa_alignment_review/review.csv',
        ROOT/'outputs/bfsongrepo_multibird_results.json',ROOT/'outputs/bfsongrepo_frozen_transfer_results.json']
    return {str(p.relative_to(ROOT)):pilot.sha(p) for p in paths}


def timing_audit(sources):
    rows = []
    for source in sources:
        native = ROOT/source['path']
        relative = native.relative_to(pilot.CORPUS).with_suffix('.TextGrid')
        path = ROOT/'data/mfa_alignments/dev-clean'/relative
        row = {'utterance_id':source['utterance_id'],'alignment_path':str(path.relative_to(ROOT)),
               'exists':path.exists(),'human_validated':False}
        if path.exists():
            intervals = [item for item in parse_textgrid(path).get('words',[]) if item[2].strip()]
            row.update({'alignment_sha256':pilot.sha(path),'word_intervals':len(intervals),
                'transcript_matches':normalize_text(' '.join(w for a,b,w in intervals)) == source['transcript']})
        rows.append(row)
    review_path = ROOT/'outputs/mfa_alignment_review/review.csv'
    review = list(csv.DictReader(review_path.open()))
    count = sum(bool(r.get('judgement','').strip()) for r in review)
    return {'records':rows,'existing_review_items':len(review),'completed_existing_judgements':count,
            'alignment_type':'Machine forced alignment,not independent human reference',
            'prefix_relative_scoring_ready':False,'prefix_relative_scoring_executed':False,
            'blocker':'No independently validated source-specific word availability reference;existing review does not certify these selected words.'}


def freeze():
    if PLAN.exists() or SPEC.exists() or RESULT.exists() or CACHE.exists():
        raise RuntimeError('Study already started/frozen')
    sources = source_population()
    plan = {'study':'RCLAT two-scale temporal control v1','sources':sources,
        'dependencies':dependencies(),
        'selection':'All40 local dev-clean speakers with2metadata-selected native16kHz monoPCM16 clips5to10s,using samefixedhash rule as pilot;retain original16clips and64new clips. No ASR outcomes used to match/select audio.',
        'primary_cohort':'32prospective-extension speakers;original8reported as replication and all40descriptive.',
        'conditions':CONDITIONS,
        'acoustic_rule':'Split into20ms/320sample cells;every complete cell gets identical1ms raised-cosine endpoint fades,rounded toPCM16. Same tapered-cell inventory in sham/global/local;terminal remainder unchanged.',
        'global':'Permute complete500ms groups of25contiguous20ms cells;preserve within-group temporal structure.',
        'local':'Keep500ms group sequence,but permute25cells within each complete group. Unfinished500ms region unchanged in both arms.',
        'matching':'32deterministic candidate seeds per policy;select pair minimizing relative absolute difference in normalized waveform RMSE against taper-only sham. No decoder calls. Stop before ASR unless every pair mismatch<=5%.',
        'matching_limits':'Common cell boundaries,taper,sample inventory and energy;waveformRMSE matched. Does not establish perceptual equivalence,all-feature matching or absence of unnatural context/OOD effects.',
        'primary':'Speaker-macro bag-word recall(global)-bag-word recall(local) on full reference transcript in32extension speakers. Positive favors retaining500ms contiguous local structure despite global rearrangement. This is decoder-relative lexical recoverability,not message-order or semantic recovery.',
        'secondary':'Full-transcript WER,CER,exactness,per-speaker outcomes,distortion,and separate original8/all40cohorts. No prefix or CTC information-accrual endpoint.',
        'uncertainty':'2000paired speaker-bootstrap draws,seed0,within declared cohort;selected corpus individuals,not human perception or broad-language-population inference.',
        'control_gate':'Extension speaker-macro taper-sham WER minus intact WER<=0.05. If exceeded,report all outcomes but label mechanistic interpretation unsupported by the preservation control. No taper retuning or sample exclusions.',
        'decoder':'Same frozen Wav2Vec2_ASR_BASE_960H,same checkpoint and greedy decoding as pilot. No prompts,word boundaries,training,parameter selection or human semantic labels.',
        'stop':'One complete80utterance x5condition evaluation;no outcome-driven substitutions,matching changes,ASR tuning or reruns.',
        'scope':'Timescale-sensitive recovery of known lexical inventory from perturbed speech. Does not isolate semantic units,causal human perception or communication in animals.',
        'stage2_ready':False}
    pilot.freeze_json(PLAN,plan)
    cache,matching = {},[]
    for source in sources:
        pcm,sr = sf.read(ROOT/source['path'],dtype='int16')
        sham = tapered_cells(pcm,sr)
        seed = int(pilot.rank(source['utterance_id'])[:12],16)
        global_audio,local_audio,match = match_distortion(sham,sr,seed=seed)
        if match['relative_rmse_mismatch'] > .05:
            raise ValueError('Distortion matching gate failed;no ASR permitted')
        for value in [global_audio,local_audio]:
            raw = value[:len(value)//320*320].tobytes()
            expected = sham[:len(sham)//320*320].tobytes()
            if Counter(raw[i:i+640] for i in range(0,len(raw),640)) != Counter(expected[i:i+640] for i in range(0,len(expected),640)):
                raise ValueError('Tapered-cell inventory mismatch')
            if np.sum(value.astype(np.int64)**2) != np.sum(sham.astype(np.int64)**2):
                raise ValueError('Sample energy mismatch')
        for condition,value in [('intact',pcm),('taper_sham',sham),('global',global_audio),('local',local_audio)]:
            cache[source['utterance_id']+'__'+condition] = value
        matching.append({'utterance_id':source['utterance_id'],**match,
            'cell_inventory_exact':True,'energy_exact':True,
            'sham_rmse_vs_intact':rmse(sham,pcm),
            'global_rmse_vs_intact':rmse(global_audio,pcm),'local_rmse_vs_intact':rmse(local_audio,pcm)})
        if len(matching)%10 == 0:
            print('Matched without ASR',len(matching),flush=True)
    np.savez_compressed(CACHE,**cache)
    audit = timing_audit(sources)
    pilot.freeze_json(Path(str(PREFIX)+'_timing_support_audit.json'),audit)
    pilot.freeze_json(SPEC,{**plan,'plan_sha256':pilot.sha(PLAN),'cache_sha256':pilot.sha(CACHE),
        'matching':matching,'matching_gate_passed':True,
        'timing_audit_sha256':pilot.sha(Path(str(PREFIX)+'_timing_support_audit.json')),
        'ASR_calls_before_preregistration':0})
    prior_result = json.loads(pilot.RESULT.read_text())
    closeout = ROOT/'outputs/rclat_study_a_closeout.json'
    if not closeout.exists():
        pilot.freeze_json(closeout,{'status':'completed_feasibility_pilot_closed',
            'results_sha256':pilot.sha(pilot.RESULT),'intact_full_wer':prior_result['summary']['intact']['1.0']['wer'],
            'interpretation':'Continuous-speech transcript recovery and technical invariants supported;temporal information localisation and semantic reversibility not established.',
            'prefix_limitation':'Complete-transcript error penalises unspoken suffix;prefix-relative accuracy remains unvalidated.',
            'semantic_reversibility_tested':False,'stage2_ready':False})
    print('Preregistered80utterances/40speakers;all acoustic matches pass;no ASR yet',flush=True)


def check():
    spec = json.loads(SPEC.read_text())
    for path,digest in spec['dependencies'].items():
        if pilot.sha(ROOT/path) != digest:
            raise ValueError('Frozen dependency changed')
    if pilot.sha(PLAN) != spec['plan_sha256'] or pilot.sha(CACHE) != spec['cache_sha256']:
        raise ValueError('Study population changed')
    for source in spec['sources']:
        if pilot.sha(ROOT/source['path']) != source['sha256'] or pilot.sha(ROOT/source['transcript_path']) != source['transcript_sha256']:
            raise ValueError('Source changed')
    return spec


def summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['speaker_id']].append(row)
    per_speaker = {}
    for speaker,values in grouped.items():
        words = sum(r['metrics']['reference_words'] for r in values)
        chars = sum(r['metrics']['reference_characters'] for r in values)
        per_speaker[speaker] = {'wer':sum(r['metrics']['word_errors'] for r in values)/words,
            'cer':sum(r['metrics']['character_errors'] for r in values)/chars,
            'bag_word_recall':sum(r['metrics']['bag_word_overlap'] for r in values)/words,
            'exact_utterances':sum(r['metrics']['utterance_exact'] for r in values)}
    return {'speakers':len(per_speaker),'utterances':len(rows),'per_speaker':per_speaker,
            'macro_wer':float(np.mean([v['wer'] for v in per_speaker.values()])),
            'macro_cer':float(np.mean([v['cer'] for v in per_speaker.values()])),
            'macro_bag_word_recall':float(np.mean([v['bag_word_recall'] for v in per_speaker.values()]))}


def run():
    spec = check()
    marker = Path(str(PREFIX)+'_RUN_STARTED.json')
    if marker.exists() or RESULT.exists():
        raise RuntimeError('One-run stop rule')
    pilot.freeze_json(marker,{'preregistration_sha256':pilot.sha(SPEC)})
    model = pilot.BUNDLE.get_model(dl_kwargs={'model_dir':str(pilot.CHECKPOINT.parent),'progress':False}).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    labels = pilot.BUNDLE.get_labels()
    rows,sham_checks = [],0
    with np.load(CACHE) as cache:
        for i,source in enumerate(spec['sources'],1):
            uid = source['utterance_id']
            intact = cache[uid+'__intact']
            intact_logits = None
            for condition in CONDITIONS:
                audio = transform_audio(intact,16000,'lossless',seed=0) if condition == 'lossless' else cache[uid+'__'+condition]
                logits = pilot.infer(model,audio)
                if condition == 'intact':
                    intact_logits = logits
                if condition == 'lossless':
                    np.testing.assert_array_equal(logits.numpy(),intact_logits.numpy())
                    sham_checks += 1
                text = greedy_text(logits,labels)
                rows.append({'utterance_id':uid,'speaker_id':source['speaker_id'],'cohort':source['cohort'],
                    'condition':condition,'decoded_text':text,'metrics':text_errors(source['transcript'],text)})
            if i%10 == 0:
                print('Decoded',i,'of80utterances',flush=True)
    cohorts = {}
    for cohort in ['prospective_extension','pilot_replication','all']:
        selected = [r for r in rows if cohort == 'all' or r['cohort'] == cohort]
        summaries = {c:summary([r for r in selected if r['condition'] == c]) for c in CONDITIONS}
        speakers = sorted(summaries['global']['per_speaker'])
        effects = np.array([summaries['global']['per_speaker'][s]['bag_word_recall']-
                            summaries['local']['per_speaker'][s]['bag_word_recall'] for s in speakers])
        rng = np.random.default_rng(0)
        draws = effects[rng.integers(0,len(effects),(2000,len(effects)))].mean(1)
        cohorts[cohort] = {'summary':summaries,
            'global_minus_local_bag_recall':{'point':float(effects.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),
                                           'speakers':len(effects)},
            'taper_sham_minus_intact_macro_wer':summaries['taper_sham']['macro_wer']-summaries['intact']['macro_wer']}
    check()
    gate = cohorts['prospective_extension']['taper_sham_minus_intact_macro_wer'] <= .05
    pilot.freeze_json(RESULT,{'study':spec['study'],'preregistration_sha256':pilot.sha(SPEC),
        'cohorts':cohorts,'rows':rows,'lossless_output_checks':sham_checks,
        'distortion_matching_gate_passed':True,'taper_preservation_control_passed':gate,
        'prefix_relative_scoring_executed':False,'stage2_ready':False,'semantic_reversibility_tested':False,
        'prior_results_unchanged':True,
        'limits':[spec['matching_limits'],spec['scope'],
                  'Bag-word recall measures lexical inventory,not sentence-order correctness or independently validated meaning.',
                  'WaveformRMSE matching is not perceptual matching. Tapered splice boundaries may still cause model-domain shifts.']})
    print(json.dumps({'primary':cohorts['prospective_extension']['global_minus_local_bag_recall'],
                      'taper_control_passed':gate,'stage2_ready':False},indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=['freeze','run'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        {'freeze':freeze,'run':run}[args.action]()
