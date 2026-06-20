"""Track A yardsticks: linear probes + unsupervised consistency (Semantic Ceiling).

These are *measurement instruments*, not pipeline models — they never feed a
verdict and are not registered in the ethological firewall. A linear probe proves
the word signal EXISTS in a representation (just entangled); unsupervised
token→label consistency measures whether a tokenizer recovers label identity
*without supervision* (it should NOT — the Semantic Ceiling: unsupervised VQ-VAE
tokens are acoustic-similarity classes, not word identity). wav2vec2 forced
alignment supplies word-level ground truth for *human* speech only.

Migrated + generalized from the prototype's ``analysis/semantic_ceiling_validation.py``
and ``analysis/human_word_level_validation.py`` / ``pipeline/extract_word_features.py``.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Linear probe — does the label signal exist in a representation?
# --------------------------------------------------------------------------- #

@dataclass
class ProbeResult:
    accuracy: float
    chance_baseline: float
    n_train: int
    n_test: int
    n_classes: int
    lift_over_chance: float = 0.0
    signal_exists: bool = False
    skipped: str | None = None

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "chance_baseline": self.chance_baseline,
            "lift_over_chance": self.lift_over_chance,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_classes": self.n_classes,
            "signal_exists": self.signal_exists,
            **({"skipped": self.skipped} if self.skipped else {}),
        }


def linear_probe(
    X, y, *, min_per_class: int = 5, test_size: float = 0.3, random_state: int = 42
) -> ProbeResult:
    """Multi-class logistic-regression probe: can a linear readout recover labels?

    Filters to labels with >= ``min_per_class`` instances, stratified train/test
    split, standardized features. Reports accuracy vs ``chance = 1/n_classes``;
    ``signal_exists`` when accuracy clears 3x chance (the signal is present, just
    not surfaced by unsupervised clustering). Returns a skipped result on too-few
    data rather than raising.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X)
    y = np.asarray(y)
    counts = Counter(y.tolist())
    keep = {c for c, n in counts.items() if n >= min_per_class}
    mask = np.array([c in keep for c in y.tolist()])
    X, y = X[mask], y[mask]

    if len(X) < 100 or len(keep) < 5:
        return ProbeResult(0.0, 0.0, 0, 0, len(keep), skipped="insufficient data")

    classes = sorted(keep)
    idx = {c: i for i, c in enumerate(classes)}
    yc = np.array([idx[c] for c in y.tolist()])

    Xtr, Xte, ytr, yte = train_test_split(
        X, yc, test_size=test_size, random_state=random_state, stratify=yc
    )
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=random_state)
    clf.fit(Xtr, ytr)
    acc = float(clf.score(Xte, yte))
    chance = 1.0 / len(classes)
    return ProbeResult(
        accuracy=round(acc, 4),
        chance_baseline=round(chance, 4),
        n_train=len(Xtr),
        n_test=len(Xte),
        n_classes=len(classes),
        lift_over_chance=round(acc / chance, 2),
        signal_exists=acc > 3 * chance,
    )


# --------------------------------------------------------------------------- #
# Unsupervised token -> label consistency (the Semantic Ceiling)
# --------------------------------------------------------------------------- #

@dataclass
class ConsistencyResult:
    consistency: float
    chance_baseline: float
    n_labeled: int
    n_unique_tokens: int
    lift: float = 0.0
    skipped: str | None = None

    def as_dict(self) -> dict:
        return {
            "consistency": self.consistency,
            "chance_baseline": self.chance_baseline,
            "lift_over_chance": self.lift,
            "n_labeled": self.n_labeled,
            "n_unique_tokens": self.n_unique_tokens,
            **({"skipped": self.skipped} if self.skipped else {}),
        }


def token_label_consistency(
    tokens, labels, *, purity: float = 0.5, ignore: str = "<unknown>"
) -> ConsistencyResult:
    """Fraction of labeled segments whose token maps consistently.

    A token is "consistent" if its dominant label covers >= ``purity`` of its
    instances; consistency is the dominant-label mass over all labeled segments.
    High consistency would mean the tokenizer discovered label identity
    unsupervised — for the Semantic Ceiling we expect it to sit near chance.
    ``chance_baseline = 1 / n_distinct_labels``.
    """
    tok_label: dict[int, Counter] = defaultdict(Counter)
    n_labeled = 0
    for t, w in zip(tokens, labels):
        if w == ignore or w is None:
            continue
        tok_label[int(t)][w] += 1
        n_labeled += 1

    if n_labeled == 0:
        return ConsistencyResult(0.0, 0.0, 0, 0, skipped="no labeled segments")

    consistent = 0
    for wc in tok_label.values():
        total = sum(wc.values())
        dom = wc.most_common(1)[0][1]
        if dom / total >= purity:
            consistent += dom

    consistency = consistent / n_labeled
    n_labels = len({w for wc in tok_label.values() for w in wc})
    chance = 1.0 / max(n_labels, 1)
    return ConsistencyResult(
        consistency=round(consistency, 4),
        chance_baseline=round(chance, 4),
        n_labeled=n_labeled,
        n_unique_tokens=len(tok_label),
        lift=round(consistency / chance, 2) if chance > 0 else 0.0,
    )


# --------------------------------------------------------------------------- #
# Word ground truth: wav2vec2 forced alignment (human speech only)
# --------------------------------------------------------------------------- #

def load_word_features(path) -> tuple[np.ndarray, np.ndarray]:
    """Load a precomputed (cpc_latents, word_labels) npz in the prototype format."""
    d = np.load(str(path), allow_pickle=True)
    return d["cpc_latents"], d["word_labels"]


def load_librispeech_transcript(flac_path) -> str:
    """Return the uppercase transcript line for a LibriSpeech utterance flac."""
    flac_path = Path(flac_path)
    stem = flac_path.stem  # e.g. 188-136483-0000
    parts = stem.split("-")
    trans_name = f"{parts[0]}-{parts[1]}.trans.txt"
    for parent in (flac_path.parent, flac_path.parent.parent):
        tp = parent / trans_name
        if tp.exists():
            with open(tp) as f:
                for line in f:
                    if line.startswith(stem):
                        return line.strip().split(" ", 1)[1] if " " in line else ""
    return ""


@dataclass
class AlignmentResult:
    """wav2vec2 CTC alignment of a transcript: word spans + boundary onsets."""

    word_spans: list       # [(start_ms, end_ms, word), ...]  — for label matching
    word_starts: list      # [start_ms, ...]                   — word-boundary GT
    phoneme_starts: list   # [start_ms, ...]                   — sub-word-boundary GT


def forced_align(transcript, audio, sr, *, model=None, labels=None) -> AlignmentResult:
    """wav2vec2 CTC forced-align a transcript.

    Returns word spans (for label matching) plus word-start and phoneme-start
    times (boundary GT for multi-resolution F1) from a single alignment pass.
    Pass a preloaded ``model`` + ``labels`` to avoid reloading per file. Needs
    ``torchaudio``; imported lazily. Human speech only — measurement instrument.
    """
    if not transcript:
        return AlignmentResult([], [], [])
    import torch  # noqa: PLC0415
    import torchaudio  # noqa: PLC0415

    if model is None:
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        model = bundle.get_model()
        labels = bundle.get_labels()

    wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).float().unsqueeze(0)
    with torch.no_grad():
        emission, _ = model(wav)
    log_probs = torch.log_softmax(emission, dim=-1)

    tokens_str = transcript.upper().replace(" ", "|")
    tokens = [labels.index(c) for c in tokens_str if c in labels]
    if len(tokens) < 2:
        return AlignmentResult([], [], [])
    aligned, _ = torchaudio.functional.forced_align(
        log_probs,
        torch.tensor(tokens, dtype=torch.int32).unsqueeze(0),
        input_lengths=torch.tensor([log_probs.shape[1]]),
        target_lengths=torch.tensor([len(tokens)]),
        blank=0,
    )
    aligned = aligned.squeeze(0)
    frame_ms = (len(audio) / sr * 1000.0) / aligned.shape[0]

    space_idx = labels.index("|")
    words = transcript.upper().split()
    spans, phoneme_starts = [], []
    word_start = None
    prev_blank = True
    prev_emit = 0
    for i, tok in enumerate(aligned.tolist()):
        if tok == space_idx:
            if not prev_blank and word_start is not None:
                spans.append((word_start * frame_ms, i * frame_ms))
                word_start = None
            prev_blank = True
            prev_emit = 0
            continue
        if tok != 0:
            if word_start is None:
                word_start = i
            if tok != prev_emit:  # char-token onset = a sub-word boundary
                phoneme_starts.append(i * frame_ms)
            prev_emit = tok
            prev_blank = False
    if word_start is not None:
        spans.append((word_start * frame_ms, len(aligned) * frame_ms))

    word_spans = [(s, e, words[idx]) for idx, (s, e) in enumerate(spans) if idx < len(words)]
    word_starts = [s for s, _e, _w in word_spans]
    return AlignmentResult(word_spans, word_starts, phoneme_starts)


def forced_align_words(transcript, audio, sr, *, model=None, labels=None):
    """Backward-compat wrapper: word spans only (see :func:`forced_align`)."""
    return forced_align(transcript, audio, sr, model=model, labels=labels).word_spans


def match_span_to_label(start_ms, end_ms, spans, *, tol: float = 80.0):
    """Assign a segment to the closest word span (by midpoint), within ``tol`` ms."""
    seg_mid = (start_ms + end_ms) / 2.0
    best_word, best_dist = None, float("inf")
    for ws, we, word in spans:
        dist = abs(seg_mid - ((ws + we) / 2.0))
        if dist < best_dist:
            best_dist, best_word = dist, word
    return best_word if best_dist < tol else None
