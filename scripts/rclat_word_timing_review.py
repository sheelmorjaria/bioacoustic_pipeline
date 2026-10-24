#!/usr/bin/env python3
"""Freeze a small blinded two-reviewer reference pilot. No annotation is inferred."""
import argparse
import base64
from collections import Counter,defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

os.environ.setdefault('MPLCONFIGDIR','/tmp/rclat_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
import soundfile as sf

import rclat_study_a as pilot
from mfa_alignment_qa import parse_textgrid
from bioacoustic_pipeline.diagnostics.rclat import normalize_text
from bioacoustic_pipeline.ingestion.word_timing_review import review_gate

ROOT = pilot.ROOT
OUT = ROOT/'outputs/rclat_word_timing_review'
TEMPLATE = ROOT/'scripts/assets/rclat_word_review.html'
FIELDS = ['target_id','status','onset_sample','offset_sample','onset_uncertainty_samples',
          'offset_uncertainty_samples','evidence','notes']


def rank(value):
    return hashlib.sha256(('rclat-word-timing-v1:0:'+value).encode()).hexdigest()


def write_csv(path, rows, fields):
    with path.open('w',newline='') as output:
        writer = csv.DictWriter(output,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build():
    if OUT.exists():
        raise RuntimeError('Review pilot already exists;never overwrite human work')
    parent_spec_path = ROOT/'outputs/rclat_temporal_control_preregistration.json'
    parent_result_path = ROOT/'outputs/rclat_temporal_control_results.json'
    parent = json.loads(parent_spec_path.read_text())
    result = json.loads(parent_result_path.read_text())
    if result['preregistration_sha256'] != pilot.sha(parent_spec_path):
        raise ValueError('Parent provenance mismatch')
    sources = parent['sources']
    audit_path = ROOT/'outputs/rclat_temporal_control_timing_support_audit.json'
    audit = json.loads(audit_path.read_text())
    if pilot.sha(audit_path) != parent['timing_audit_sha256']:
        raise ValueError('Timing audit changed')
    audited = {r['utterance_id']:r for r in audit['records']}
    candidates,quarantine = defaultdict(list),[]
    for source in sources:
        uid = source['utterance_id']
        timing = audited[uid]
        if not timing.get('transcript_matches',False):
            quarantine.append({'utterance_id':uid,'reason':'Original timing audit transcript mismatch'})
            continue
        alignment = ROOT/timing['alignment_path']
        if pilot.sha(alignment) != timing['alignment_sha256']:
            raise ValueError('Source alignment changed')
        intervals = [(a,b,normalize_text(w)) for a,b,w in parse_textgrid(alignment).get('words',[]) if w.strip()]
        words = source['transcript'].split()
        if [w for a,b,w in intervals] != words:
            quarantine.append({'utterance_id':uid,'reason':'One-to-one interval/token mapping unavailable'})
            continue
        counts = Counter(words)
        for index,word in enumerate(words):
            if 0 < index < len(words)-1 and len(word) >= 4 and counts[word] == 1:
                candidates[source['speaker_id']].append((source,index,intervals))
    speakers = sorted(candidates,key=rank)[:20]
    if len(speakers) != 20:
        raise ValueError('Insufficient eligible speaker-distinct review targets')
    selected,proposals = [],[]
    for i,speaker in enumerate(speakers,1):
        source,index,intervals = min(candidates[speaker],key=lambda item:rank(item[0]['utterance_id']+':'+str(item[1])))
        audio,sr = sf.read(ROOT/source['path'],dtype='int16')
        if pilot.sha(ROOT/source['path']) != source['sha256'] or sr != 16000:
            raise ValueError('Selected source changed')
        uid = source['utterance_id']
        target = {'target_id':f'T{i:02d}','utterance_id':uid,'speaker_id':speaker,
            'words':source['transcript'].split(),'word_index':index,'target_word':source['transcript'].split()[index],
            'sample_rate':sr,'samples':len(audio),'sample_origin':0,
            'coordinate_reference':uid+':native-utterance-samples',
            'source_path':source['path'],'source_sha256':source['sha256'],
            'pcm_sha256':hashlib.sha256(audio.astype('<i2').tobytes()).hexdigest()}
        selected.append(target)
        a,b,w = intervals[index]
        proposals.append({'target_id':target['target_id'],'source_alignment_path':audited[uid]['alignment_path'],
            'source_alignment_sha256':audited[uid]['alignment_sha256'],
            'model_proposed_onset_seconds':a,'model_proposed_offset_seconds':b,
            'model_proposal_is_validated_reference':False})
    OUT.mkdir(parents=True)
    (OUT/'adjudication_only').mkdir()
    protocol = {
        'study':'RCLAT independently reviewed word-interval feasibility reference v1',
        'status':'prepared_pending_two_independent_human_reviews',
        'selection':'20speakers ranked by fixedhash among source-compatible candidates;one unique word>=4letters per speaker,excluding first/last word,ranked by fixedhash(uid:index). No ASR scores,word timings or boundary confidence used in selection.',
        'population':'Targets drawn from unchanged80utterance temporal-control population. Timing mismatches quarantined only for this new reference task;no completed result is rescored or filtered.',
        'review':'Two different identified humans independently annotate complete utterance audio/waveform/spectrogram,with transcript and target occurrence but no machine or other-reviewer intervals. Independent orders are fixed by hash.',
        'coordinates':'Native16kHz monoPCM16 utterance samples,origin0;onset inclusive/offset exclusive. Give explicit uncertainty in samples;do not treat integer coordinates as perceptual certainty.',
        'status_options':['unreviewed','supported','uncertain','unusable'],
        'agreement_reporting':'Absolute onset/offset differences in samples;320samples(20ms) is a descriptive pilot agreement tolerance,not a biological unit or automatic certification.',
        'adjudication':'After both independent passes are locked,an identified human resolves intervals and uncertainty with evidence;retain original reviews. No automatic averaging,copying model proposals or promoting agreement alone.',
        'readiness':'Every target must have supported intervals from both independently attested reviewers plus an explicitly approved manual adjudication;otherwise reference pilot remains blocked. No automatic target replacement.',
        'independence_limits':'Distinct IDs and attestations are operational checks,not independent proof of human identity or conduct.',
        'availability_limits':'Word intervals annotate acoustic realization,not exact information arrival. Coarticulation/context may carry target information outside intervals. No human-listener recoverability or semantic labels collected.',
        'analysis':'No ASR scoring,word-availability curves,training or Stage2 activation in this task. Any future prefix study requires a separate prospectively frozen design after reference review.',
        'targets':selected,'quarantined_sources':quarantine,
        'provenance':{'parent_spec_sha256':pilot.sha(parent_spec_path),'parent_result_sha256':pilot.sha(parent_result_path),
            'timing_audit_sha256':pilot.sha(audit_path),'code_sha256':pilot.sha(Path(__file__).resolve()),
            'validator_sha256':pilot.sha(ROOT/'src/bioacoustic_pipeline/ingestion/word_timing_review.py'),
            'template_sha256':pilot.sha(TEMPLATE)},'stage2_ready':False}
    pilot.freeze_json(OUT/'protocol.json',protocol)
    pilot.freeze_json(OUT/'adjudication_only/mfa_proposals.json',{'share_with_independent_reviewers':False,'proposals':proposals})
    write_csv(OUT/'adjudication_only/adjudication.csv',
              [{'target_id':t['target_id']} for t in selected],FIELDS[:1]+['status','adjudicator_id']+FIELDS[2:])
    media = OUT/'media'
    (media/'audio').mkdir(parents=True)
    (media/'images').mkdir()
    public = []
    for target in selected:
        uid = target['utterance_id']
        audio,sr = sf.read(ROOT/target['source_path'],dtype='int16')
        wave = media/'audio'/f'{uid}.wav'
        sf.write(wave,audio,sr,subtype='PCM_16')
        reread,rate = sf.read(wave,dtype='int16')
        if rate != sr or not np.array_equal(audio,reread):
            raise ValueError('Review audio not sample-exact')
        fig,axes = plt.subplots(2,1,figsize=(12,4),sharex=True)
        axes[0].plot(np.arange(len(audio))/sr,audio.astype(float)/32768.,lw=.35)
        axes[0].set_ylabel('Amplitude')
        f,t,power = spectrogram(audio.astype(float)/32768.,sr,nperseg=400,noverlap=240,nfft=512)
        axes[1].pcolormesh(t,f/1000,10*np.log10(power+1e-14),vmin=-100,vmax=-30,cmap='magma',shading='auto')
        axes[1].set(xlabel='Native utterance time (seconds)',ylabel='Frequency (kHz)',ylim=(0,8))
        fig.suptitle('Complete utterance; no proposed word boundaries')
        fig.tight_layout()
        fig.savefig(media/'images'/f'{uid}.png',dpi=140)
        plt.close(fig)
        public.append({k:target[k] for k in ['target_id','utterance_id','speaker_id','words','word_index','target_word','sample_rate','samples','sample_origin','coordinate_reference']} |
            {'pcm_base64':base64.b64encode(audio.astype('<i2').tobytes()).decode()})
    packets = []
    for role in ('A','B'):
        folder = OUT/f'reviewer_{role}'
        folder.mkdir()
        for kind in ('audio','images'):
            shutil.copytree(media/kind,folder/kind)
        ordered = sorted(public,key=lambda t:rank(role+':'+t['target_id']))
        html = TEMPLATE.read_text().replace('__ROLE__',role).replace('__DATA__',json.dumps({'targets':ordered}).replace('</','<\\/'))
        if 'model_proposed' in html or '.TextGrid' in html:
            raise ValueError('Machine boundaries leaked to reviewer page')
        (folder/'review.html').write_text(html)
        write_csv(folder/f'review_{role}.csv',[{'target_id':t['target_id']} for t in ordered],FIELDS)
        pilot.freeze_json(folder/f'reviewer_{role}.json',{'role':role,'annotator_id':'','independent_review':False,'model_boundaries_hidden':False})
        pilot.freeze_json(folder/'instructions.json',{
            'role':role,'instructions':protocol['review'],'coordinates':protocol['coordinates'],
            'statuses':protocol['status_options'],'uncertainty':protocol['availability_limits'],
            'export':'Download both CSV and metadata JSON from review.html. Never use the other review or adjudication proposals during this independent pass.',
            'target_order':[t['target_id'] for t in ordered]})
        archive = OUT/f'reviewer_{role}_packet.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(folder.rglob('*')):
                if path.is_file():
                    z.write(path,path.relative_to(folder))
        with zipfile.ZipFile(archive) as z:
            if any('adjudication' in n or 'mfa' in n.lower() or n.endswith('.TextGrid') for n in z.namelist()):
                raise ValueError('Blinding violation in packet')
        packets.append({'role':role,'path':str(archive.relative_to(ROOT)),'sha256':pilot.sha(archive),
                        'self_contained':True,'model_boundaries_included':False})
    lengths = {t['target_id']:t['samples'] for t in selected}
    blank = [{'target_id':t['target_id']} for t in selected]
    meta = {'annotator_id':'','independent_review':False,'model_boundaries_hidden':False}
    status = review_gate(blank,blank,meta,meta,{},lengths)
    pilot.freeze_json(OUT/'status.json',{**status,'protocol_sha256':pilot.sha(OUT/'protocol.json'),
        'review_packets':packets,'human_reviews_completed':0,'model_boundaries_blinded_in_packets':True,
        'prefix_relative_scoring_executed':False,'stage2_ready':False})
    closeout_path = ROOT/'outputs/rclat_temporal_control_closeout.json'
    if not closeout_path.exists():
        pilot.freeze_json(closeout_path,{'study':result['study'],'status':'completed_closed',
            'results_sha256':pilot.sha(parent_result_path),'preregistration_sha256':pilot.sha(parent_spec_path),
            'primary_extension_effect':result['cohorts']['prospective_extension']['global_minus_local_bag_recall'],
            'taper_preservation_control_passed':result['taper_preservation_control_passed'],
            'inventory_definition':'Multiset overlap: repeated words count up to reference multiplicity.',
            'interpretation':'Timescale-sensitive lexical recoverability in a frozen ASR under specified rearrangements;not human perception,semantic encoding,information-onset timing or natural unit-duration discovery.',
            'prior_results_modified':False,'stage2_ready':False})
    print('Prepared20targets/20speakers,two blinded packets;0human reviews;reference readiness=False.')


def validate_completed_reviews():
    protocol = json.loads((OUT/'protocol.json').read_text())
    original = json.loads((OUT/'status.json').read_text())
    if pilot.sha(OUT/'protocol.json') != original['protocol_sha256']:
        raise ValueError('Frozen review protocol changed')
    rows,metadata = [],[]
    for role in ('A','B'):
        folder = OUT/f'reviewer_{role}'
        with (folder/f'review_{role}.csv').open(newline='') as source:
            rows.append(list(csv.DictReader(source)))
        metadata.append(json.loads((folder/f'reviewer_{role}.json').read_text()))
        for target in protocol['targets']:
            audio,sr = sf.read(folder/'audio'/f"{target['utterance_id']}.wav",dtype='int16')
            if sr != target['sample_rate'] or hashlib.sha256(audio.astype('<i2').tobytes()).hexdigest() != target['pcm_sha256']:
                raise ValueError('Reviewer source audio changed')
    with (OUT/'adjudication_only/adjudication.csv').open(newline='') as source:
        adjudications = {}
        for row in csv.DictReader(source):
            if row['target_id'] in adjudications:
                raise ValueError('Duplicated adjudication target')
            adjudications[row['target_id']] = row
    result = review_gate(*rows,*metadata,adjudications,
                         {t['target_id']:t['samples'] for t in protocol['targets']})
    print(json.dumps(result,indent=2))
    return result['ready']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--validate',action='store_true',help='Check completed independent reviews and adjudications;exit2 while blocked')
    args = parser.parse_args()
    if args.validate:
        raise SystemExit(0 if validate_completed_reviews() else 2)
    build()
