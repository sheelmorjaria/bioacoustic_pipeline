"""Prefix-only speech measurements. CTC scores are not semantic entropy."""
from collections import Counter
import re

import numpy as np
import torch
from bioacoustic_pipeline.representation.lossless_acoustic import encode, decode


def normalize_text(text):
    return ' '.join(re.sub("[^A-Z' ]",' ',text.upper()).split())


def prefix_audio(audio, fraction):
    audio = np.asarray(audio)
    if audio.ndim != 1 or not 0 < fraction <= 1:
        raise ValueError('Mono audio and a fraction in (0,1] required')
    stop = int(np.floor(len(audio)*fraction))
    if stop < 400:
        raise ValueError('Prefix shorter than decoder frontend support')
    return audio[:stop].copy()


def block_order(count, seed):
    order = np.random.default_rng(seed).permutation(count)
    if count > 1 and np.array_equal(order,np.arange(count)):
        order = np.roll(order,1)
    return order


def transform_audio(audio, sr, condition, *, seed):
    audio = np.asarray(audio)
    if condition == 'intact':
        return audio.copy()
    if condition == 'lossless':
        result,rate = decode(encode(audio,sr,'flac'),'flac')
        if rate != sr or not np.array_equal(audio,result):
            raise ValueError('Lossless sham mismatch')
        return result
    if condition == 'silence':
        return np.zeros_like(audio)
    if condition != 'block_shuffle':
        raise ValueError('Unknown acoustic condition')
    block = round(.5*sr)
    count = len(audio)//block
    complete = audio[:count*block].reshape(count,block)
    return np.concatenate([complete[block_order(count,seed)].reshape(-1),audio[count*block:]])


def greedy_text(logits, labels):
    ids = logits.argmax(-1).detach().cpu().tolist()
    tokens = [token for i,token in enumerate(ids) if token != 0 and (i == 0 or token != ids[i-1])]
    return normalize_text(''.join(labels[token] for token in tokens).replace('|',' '))


def message_score(logits, reference, labels):
    text = normalize_text(reference).replace(' ','|')
    lookup = {v:i for i,v in enumerate(labels)}
    if not text or any(c not in lookup or lookup[c] == 0 for c in text):
        raise ValueError('Reference not supported by decoder alphabet')
    target = torch.tensor([lookup[c] for c in text],dtype=torch.long)
    minimum = len(target)+sum(a == b for a,b in zip(text,text[1:]))
    feasible = len(logits) >= minimum
    score = None
    if feasible:
        loss = torch.nn.functional.ctc_loss(logits.detach().cpu().double().log_softmax(-1)[:,None],
            target,torch.tensor([len(logits)]),torch.tensor([len(target)]),
            blank=0,reduction='sum',zero_infinity=False)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite feasible CTC score')
        score = -float(loss)
    return {'ctc_log_score':score,'structurally_feasible':feasible,
            'ctc_frames':len(logits),'minimum_ctc_frames':minimum}


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis)+1))
    for i,a in enumerate(reference,1):
        current = [i]
        for j,b in enumerate(hypothesis,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(a != b)))
        previous = current
    return previous[-1]


def text_errors(reference, hypothesis):
    reference,hypothesis = normalize_text(reference),normalize_text(hypothesis)
    ref_words,hyp_words = reference.split(),hypothesis.split()
    ref_chars,hyp_chars = reference.replace(' ',''),hypothesis.replace(' ','')
    word_errors,char_errors = edit_distance(ref_words,hyp_words),edit_distance(ref_chars,hyp_chars)
    return {'word_errors':word_errors,'reference_words':len(ref_words),
            'wer':word_errors/max(1,len(ref_words)),
            'character_errors':char_errors,'reference_characters':len(ref_chars),
            'cer':char_errors/max(1,len(ref_chars)),
            'utterance_exact':reference == hypothesis,
            'bag_word_overlap':sum((Counter(ref_words)&Counter(hyp_words)).values()),
            'hypothesis_words':len(hyp_words)}
