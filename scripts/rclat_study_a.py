#!/usr/bin/env python3
"""Bounded RCLAT prefix-recovery positive control; no semantic or animal claims."""
import argparse
from collections import defaultdict
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from threadpoolctl import threadpool_limits

from bioacoustic_pipeline.diagnostics.rclat import (
    prefix_audio,transform_audio,greedy_text,message_score,text_errors,normalize_text,block_order,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT/'outputs/rclat_study_a_preregistration.json'
RESULT = ROOT/'outputs/rclat_study_a_results.json'
CHECKPOINT = ROOT/'data/rclat/checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth'
CORPUS = ROOT/'data/LibriSpeech/dev-clean'
CONDITIONS = ['intact','lossless','block_shuffle','silence']
FRACTIONS = [.25,.5,.75,1.]
BUNDLE = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source,'sha256').hexdigest()


def freeze_json(path,value):
    if path.exists():
        raise RuntimeError('Refusing to overwrite research artifact: '+str(path))
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def rank(value):
    return hashlib.sha256(('rclat-a-v1:0:'+value).encode()).hexdigest()


def dependencies():
    paths = [Path(__file__).resolve(),
        ROOT/'src/bioacoustic_pipeline/diagnostics/rclat.py',
        ROOT/'src/bioacoustic_pipeline/representation/lossless_acoustic.py',
        ROOT/'tests/test_rclat.py',CHECKPOINT,ROOT/'data/LibriSpeech/SPEAKERS.TXT',
        Path(inspect.getfile(type(BUNDLE))),
        ROOT/'outputs/bfsongrepo_development_results.json',
        ROOT/'outputs/bfsongrepo_frozen_transfer_results.json',
        ROOT/'outputs/bfsongrepo_multibird_results.json',
        ROOT/'outputs/bfsongrepo_lossless_baseline_results.json']
    return [{'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
             'sha256':sha(p)} for p in paths]


def select_sources():
    speaker_subsets = {}
    for line in (ROOT/'data/LibriSpeech/SPEAKERS.TXT').read_text().splitlines():
        if line.strip() and not line.startswith(';'):
            parts = [p.strip() for p in line.split('|')]
            if parts[0] in speaker_subsets:
                raise ValueError('Repeated speaker metadata')
            speaker_subsets[parts[0]] = parts[2]
    references = {}
    for path in sorted(CORPUS.rglob('*.trans.txt')):
        for line in path.read_text().splitlines():
            uid,text = line.split(' ',1)
            if uid in references:
                raise ValueError('Duplicate transcript identifier')
            references[uid] = (text,path)
    groups = defaultdict(list)
    for path in sorted(CORPUS.rglob('*.flac')):
        info = sf.info(path)
        if info.samplerate != 16000 or info.channels != 1 or info.subtype != 'PCM_16':
            continue
        if 5 <= info.frames/info.samplerate <= 10:
            speaker = path.stem.split('-')[0]
            if speaker_subsets.get(speaker) != 'dev-clean':
                raise ValueError('Speaker subset conflict')
            if path.stem not in references:
                raise ValueError('Missing transcript')
            groups[speaker].append(path)
    eligible = [s for s in groups if len(groups[s]) >= 2]
    speakers = sorted(eligible,key=rank)[:8]
    if len(speakers) != 8:
        raise ValueError('Fewer8 speakers with two eligible sources')
    sources = []
    for speaker in speakers:
        for path in sorted(groups[speaker],key=lambda p:rank(p.stem))[:2]:
            text,transcript = references[path.stem]
            normalized = normalize_text(text)
            if any(c not in BUNDLE.get_labels() for c in normalized.replace(' ','|')):
                raise ValueError('Unsupported transcript;do not replace selected source')
            audio,sr = sf.read(path,dtype='int16')
            sources.append({'utterance_id':path.stem,'speaker_id':speaker,
                'path':str(path.relative_to(ROOT)),'sha256':sha(path),
                'pcm_sha256':hashlib.sha256(audio.astype('<i2').tobytes()).hexdigest(),
                'sample_rate':sr,'samples':len(audio),'duration_seconds':len(audio)/sr,
                'transcript':text,'normalized_transcript':normalized,
                'transcript_path':str(transcript.relative_to(ROOT)),
                'transcript_file_sha256':sha(transcript),'speaker_subset':'dev-clean'})
    return sources


def freeze():
    if SPEC.exists() or RESULT.exists():
        raise RuntimeError('Pilot already frozen')
    if not CHECKPOINT.exists():
        raise ValueError('Official model checkpoint not acquired')
    sources = select_sources()
    freeze_json(SPEC,{
        'study':'RCLAT Study A prefix-recovery feasibility pilot v1',
        'status':'Prospectively fixed local analysis using previously explored dev-clean corpus',
        'sources':sources,'selection_rule':'Eight speakers SHA256-ranked with fixedsalt,then two independently hash-ranked utterances per speaker from native16kHz monoPCM16 durations5to10s. Metadata-only selection before ASR;no substitutions after outcomes.',
        'decoder':{'bundle':'WAV2VEC2_ASR_BASE_960H','checkpoint_path':str(CHECKPOINT.relative_to(ROOT)),
            'url':'https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth',
            'labels':list(BUNDLE.get_labels()),'parameters':BUNDLE._params,
            'reference':'https://docs.pytorch.org/audio/2.6.0/generated/torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.html',
            'training_scope':'Documented LibriSpeech train-clean-100/train-clean-360/train-other-500,not dev-clean. Local speaker table confirms selected speakers belong to dev-clean.',
            'policy':'Frozen pretrained model,eval,no gradient,no language model,beam search,prompt,reference text or boundary labels at inference. Greedy CTC readout. No training or model selection.'},
        'conditions':CONDITIONS,'prefix_fractions':FRACTIONS,
        'prefix_policy':'Crop native samples first using floor(N*fraction),then transform and infer independently with batch1. Prefix-only offline recomputation of bidirectional Wav2Vec2;not a causal streaming encoder. No full-utterance feature slicing or padding with future audio.',
        'transform_policy':{'lossless':'FLACPCM16 compression1,assert sample-exact prefix roundtrip',
            'block_shuffle':'Permute complete500ms sample blocks within current prefix;tail unchanged. Stable utterance-ID seed. Identity permutation rotated byone when at least2blocks. Samples/duration preserved,splice artifacts not controlled.',
            'silence':'Same-length all-zero waveform,not a fitted acoustic model.',
            'nesting':'Shuffled prefixes are not nested evidence because the block permutation changes with prefix length;do not assign an accrual curve to this condition.'},
        'metrics':{'full_utterance':'WER,CER excluding spaces,exact transcript match,and multiset word overlap;report pooled count ratios AND speaker-macro rates.',
            'prefix':'Greedy decoded text,error against FULL reference,CTC full-reference sequence log score and structural feasibility. Prefix WER penalizes not-yet-spoken words;not ordinary ASR error.',
            'accrual':'Intact/lossless/silence difference in raw full-reference CTC log score between adjacent feasible prefixes only. Undefined transitions retained missing,not zero. No semantic entropy,mutual information,calibration or word-onset inference.',
            'distortion':'Waveform RMSE in normalized full-scale units,RMS ratio and500ms boundary jump RMS. No perceptual-equivalence or speech-source purity claim.'},
        'primary_feasibility':'Full utterance recovery by the frozen decoder and exact FLAC sham agreement;no novel-ASR-superiority or semantic success threshold.',
        'secondary':'Paired speaker-conditional bootstrap of shuffled-minus-intact FULL-utterance WER,2000draws seed0. Descriptive eight-speaker pilot,not population-level or human-listener evidence.',
        'controls_not_implemented':'H0/H1/H2 capacity-controlled training,learned codec,synthesis,independent prosodic labels and listeners. This pilot does not claim the full RCLAT programme is validated.',
        'stop_rule':'One bounded run16utterances x4prefixes x4conditions=256inferences. Abort on missing source,checksum,nonfinite output,sham or reproducibility failure. No outcome-driven sample replacement,decoder changes or tuning.',
        'dependencies':dependencies(),
        'software':{'torch':torch.__version__,'torchaudio':torchaudio.__version__,'numpy':np.__version__,
                    'soundfile':sf.__version__,'libsndfile':sf.__libsndfile_version__},
        'scientific_scope':'Known transcript recovery and decoder-relative information availability in read speech excerpts. Not semantic reversibility,physical analog transmission,continuous semantic coding or animal communication.',
        'source_documentation':'https://www.openslr.org/12',
        'stage2_ready':False,'MACDE_studies_modified':False})
    print('Frozen16 utterances,eight speakers;protocol',sha(SPEC),flush=True)


def check():
    spec = json.loads(SPEC.read_text())
    for item in spec['dependencies']:
        path = Path(item['path'])
        path = path if path.is_absolute() else ROOT/path
        if sha(path) != item['sha256']:
            raise ValueError('Frozen dependency changed: '+str(path))
    for source in spec['sources']:
        for path,digest in [(source['path'],source['sha256']),
                            (source['transcript_path'],source['transcript_file_sha256'])]:
            if sha(ROOT/path) != digest:
                raise ValueError('Frozen source changed')
    return spec


def infer(model, pcm):
    """Acoustic-only inference: this function never receives the transcript."""
    tensor = torch.from_numpy(pcm.astype(np.float32)/32768.)[None]
    with torch.inference_mode():
        emissions,_ = model(tensor)
    logits = emissions[0].cpu()
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite decoder output')
    return logits


def summarise(rows):
    words = sum(r['metrics']['reference_words'] for r in rows)
    chars = sum(r['metrics']['reference_characters'] for r in rows)
    return {'utterances':len(rows),
        'wer':sum(r['metrics']['word_errors'] for r in rows)/words,
        'cer':sum(r['metrics']['character_errors'] for r in rows)/chars,
        'exact_utterances':sum(r['metrics']['utterance_exact'] for r in rows),
        'bag_word_recall':sum(r['metrics']['bag_word_overlap'] for r in rows)/words,
        'structurally_feasible_ctc_messages':sum(r['message']['structurally_feasible'] for r in rows)}


def run():
    spec = check()
    marker = ROOT/'outputs/rclat_study_a_RUN_STARTED.json'
    if marker.exists() or RESULT.exists():
        raise RuntimeError('One-run stop rule')
    freeze_json(marker,{'preregistration_sha256':sha(SPEC)})
    model = BUNDLE.get_model(dl_kwargs={'model_dir':str(CHECKPOINT.parent),'progress':False}).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    labels = BUNDLE.get_labels()
    if list(labels) != spec['decoder']['labels']:
        raise ValueError('Decoder alphabet changed')
    rows,sham_checks,causality_checks = [],[],[]
    for source in spec['sources']:
        pcm,sr = sf.read(ROOT/source['path'],dtype='int16')
        if sr != 16000 or hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest() != source['pcm_sha256']:
            raise ValueError('Source PCM changed')
        seed = int(rank(source['utterance_id'])[:16],16)
        previous = {}
        prefix_logits = None
        for fraction in FRACTIONS:
            observed = prefix_audio(pcm,fraction)
            intact_logits = None
            for condition in CONDITIONS:
                acoustic = transform_audio(observed,sr,condition,seed=seed)
                logits = infer(model,acoustic)
                decoded = greedy_text(logits,labels)
                message = message_score(logits,source['normalized_transcript'],labels)
                if condition == 'intact':
                    intact_logits = logits
                    if fraction == .5:
                        prefix_logits = logits
                if condition == 'lossless':
                    np.testing.assert_array_equal(logits.numpy(),intact_logits.numpy())
                    sham_checks.append(True)
                score = message['ctc_log_score']
                old = previous.get(condition)
                accrual = score-old if condition != 'block_shuffle' and score is not None and old is not None else None
                previous[condition] = score
                normalized = acoustic.astype(float)/32768.
                original = observed.astype(float)/32768.
                block = round(.5*sr)
                boundaries = np.arange(block,len(acoustic),block)
                jump = np.sqrt(np.mean((normalized[boundaries]-normalized[boundaries-1])**2)) if len(boundaries) else 0.
                row = {'utterance_id':source['utterance_id'],'speaker_id':source['speaker_id'],
                    'fraction':fraction,'prefix_end_sample':len(observed),'available_seconds':len(observed)/sr,
                    'condition':condition,'decoded_text':decoded,
                    'metrics':text_errors(source['normalized_transcript'],decoded),
                    'message':message,'ctc_log_score_accrual':accrual,
                    'distortion':{'waveform_rmse':float(np.sqrt(np.mean((normalized-original)**2))),
                        'rms_ratio':float(np.sqrt(np.mean(normalized**2))/max(np.sqrt(np.mean(original**2)),1e-20)),
                        'block_join_jump_rms':float(jump)},
                    'transformed_pcm_sha256':hashlib.sha256(acoustic.astype('<i2').tobytes()).hexdigest()}
                if condition == 'block_shuffle':
                    row['complete_block_order'] = block_order(len(observed)//block,seed).tolist()
                rows.append(row)
        # Real-model check: changing unseen future samples cannot change half-prefix inference.
        altered = pcm.copy()
        altered[len(prefix_audio(pcm,.5)):] = 0
        repeated = infer(model,prefix_audio(altered,.5))
        np.testing.assert_array_equal(prefix_logits.numpy(),repeated.numpy())
        causality_checks.append(True)
        print('Completed acoustic-only inference',source['utterance_id'],flush=True)
    summaries = {}
    for condition in CONDITIONS:
        summaries[condition] = {}
        for fraction in FRACTIONS:
            chosen = [r for r in rows if r['condition'] == condition and r['fraction'] == fraction]
            summary = summarise(chosen)
            per_speaker = {s:summarise([r for r in chosen if r['speaker_id'] == s]) for s in sorted({r['speaker_id'] for r in chosen})}
            summary['per_speaker'] = per_speaker
            summary['speaker_macro_wer'] = float(np.mean([v['wer'] for v in per_speaker.values()]))
            summaries[condition][str(fraction)] = summary
    speakers = sorted({s['speaker_id'] for s in spec['sources']})
    effect = np.array([summaries['block_shuffle']['1.0']['per_speaker'][s]['wer']-
                       summaries['intact']['1.0']['per_speaker'][s]['wer'] for s in speakers])
    rng = np.random.default_rng(0)
    bootstrap = effect[rng.integers(0,len(effect),(2000,len(effect)))].mean(1)
    check()
    result = {'study':spec['study'],'preregistration_sha256':sha(SPEC),
        'summary':summaries,'rows':rows,'sham_exact_checks':sum(sham_checks),
        'real_model_future_exclusion_checks':sum(causality_checks),
        'shuffled_minus_intact_speaker_macro_wer':{'point':float(effect.mean()),
            'ci95':np.quantile(bootstrap,[.025,.975]).tolist(),'speakers':len(speakers),
            'scope':'Descriptive speaker bootstrap in selected eight-speaker pilot;not human perception or broad population validation.'},
        'decoder_fitted':False,'stage2_ready':False,'semantic_recovery_tested':False,
        'causal_streaming_encoder':False,'MACDE_studies_unchanged':True,
        'limits':[spec['metrics']['prefix'],spec['metrics']['accrual'],spec['transform_policy']['nesting'],
                  spec['controls_not_implemented'],spec['scientific_scope'],
                  'Block shuffling changes word order and creates splice artifacts;error differences do not isolate an abstract temporal mechanism.']}
    freeze_json(RESULT,result)
    print(json.dumps({c:summaries[c]['1.0'] for c in CONDITIONS},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=['freeze','run'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        {'freeze':freeze,'run':run}[args.action]()
