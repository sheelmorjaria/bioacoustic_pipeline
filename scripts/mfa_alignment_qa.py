#!/usr/bin/env python3
"""MFA LibriSpeech alignment QA + deterministic human-validation sample.

PROVENANCE: word/phoneme alignments are the public Montreal Forced Aligner
release for LibriSpeech (Zenodo 2619474, CC-BY-4.0; md5 verified at
download). They are MACHINE alignments — reference-grade only after human
validation. Until the packaged review is completed by a human, every
downstream result carries the caveat "MFA-reference, human-validation
pending".

AUTOMATED QA (this script, no labels beyond the alignment itself):
  coverage          TextGrid exists per dev-clean flac (2703 expected)
  transcript match  words-tier token sequence == .trans.txt transcript
  monotonicity      word intervals ordered, non-overlapping, inside [0,xmax]
  duration sanity   word durations within [0.02, 2.0] s
  phones tier       present and populated

HUMAN-VALIDATION SAMPLE (deterministic, fixed before any benchmark run):
  100 words across >= 10 speakers — every 27th word of the sorted corpus
  word list (seed-free), max 2 words per speaker per drawn slot. Rendered
  as one HTML page per word (waveform + spectrogram with the MFA word
  boundary and its neighbours marked; the QUERY word is highlighted).
  outputs/mfa_alignment_review/page_NNN.html + review.csv (judgement
  columns EMPTY — never auto-filled) + manifest.json.

Usage:
    python scripts/mfa_alignment_qa.py [--sample 100]

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

DATA = ROOT / "data"
TG_DIR = DATA / "mfa_alignments" / "dev-clean"
LS_DIR = DATA / "LibriSpeech" / "dev-clean"
OUT = ROOT / "outputs" / "mfa_alignment_review"
SR = 16000
MIN_DUR, MAX_DUR = 0.02, 2.0


def parse_textgrid(path: Path):
    """Minimal long/short ooTextFile IntervalTier parser -> {tier: [(t0,t1,text)]}."""
    txt = path.read_text()
    tiers = {}
    # item blocks
    for m in re.finditer(r"item \[\d+\]:.*?(?=item \[\d+\]:|\Z)", txt, re.S):
        block = m.group(0)
        name = re.search(r'name = "(.*?)"', block)
        if not name:
            continue
        intervals = []
        for im in re.finditer(r'xmin = ([\d.e+-]+)\s*xmax = ([\d.e+-]+)\s*text = "(.*?)"', block):
            intervals.append((float(im.group(1)), float(im.group(2)), im.group(3)))
        if intervals:
            tiers[name.group(1)] = intervals
    return tiers


def _words(tiers):
    return [(a, b, w) for a, b, w in tiers.get("words", []) if w.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", type=int, default=100)
    args = ap.parse_args()

    flacs = sorted(LS_DIR.rglob("*.flac"))
    trans = {}
    for tf in LS_DIR.rglob("*.trans.txt"):
        for line in tf.read_text().splitlines():
            uid, text = line.strip().split(" ", 1)
            trans[uid] = text.split()

    qa = {
        "n_flac": len(flacs),
        "n_textgrid": 0,
        "n_missing_textgrid": [],
        "n_transcript_mismatch": 0,
        "n_monotonicity_violation": 0,
        "n_duration_out_of_range": 0,
        "n_phones_tier_missing": 0,
        "n_words_total": 0,
    }
    all_words = []  # (uid, spk, idx, t0, t1, word, prev, next)
    for f in flacs:
        uid = f.stem
        tg = TG_DIR / f.relative_to(LS_DIR).parent / f"{uid}.TextGrid"
        if not tg.exists():
            qa["n_missing_textgrid"].append(uid)
            continue
        qa["n_textgrid"] += 1
        tiers = parse_textgrid(tg)
        words = _words(tiers)
        qa["n_words_total"] += len(words)
        if "phones" not in tiers or not tiers["phones"]:
            qa["n_phones_tier_missing"] += 1
        toks = trans.get(uid)
        # LibriSpeech transcripts are UPPERCASE; the TextGrid words tier is
        # lowercase — compare case-insensitively
        if toks is not None and [w.lower() for _, _, w in words] != [t.lower() for t in toks]:
            qa["n_transcript_mismatch"] += 1
        prev_t1, mono_ok = -1e-9, True
        for a, b, _w in words:
            if a < prev_t1 - 1e-6 or b <= a:
                mono_ok = False
                break
            prev_t1 = b
        if not mono_ok:
            qa["n_monotonicity_violation"] += 1
            continue
        if any(b - a < MIN_DUR or b - a > MAX_DUR for a, b, _ in words):
            qa["n_duration_out_of_range"] += 1
        spk = uid.split("-")[0]
        for i, (a, b, w) in enumerate(words):
            pw = words[i - 1][2] if i > 0 else ""
            nw = words[i + 1][2] if i + 1 < len(words) else ""
            all_words.append((uid, spk, i, a, b, w, pw, nw))

    # deterministic sample: every 27th word of the sorted list, cap 2 per speaker
    step = max(1, len(all_words) // args.sample)
    sample, per_spk = [], {}
    for w in all_words[::step]:
        if len(sample) >= args.sample:
            break
        if per_spk.get(w[1], 0) >= 2:
            continue
        per_spk[w[1]] = per_spk.get(w[1], 0) + 1
        sample.append(w)

    OUT.mkdir(parents=True, exist_ok=True)
    import numpy as np

    import soundfile as sf

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = []
    for k, (uid, spk, idx, t0, t1, w, pw, nw) in enumerate(sample):
        flac = next(LS_DIR.rglob(f"{uid}.flac"))
        audio, sr = sf.read(str(flac))
        dur = len(audio) / sr
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        tt = np.linspace(0, dur, len(audio))
        axes[0].plot(tt, audio, lw=0.3)
        axes[0].axvspan(t0, t1, color="tab:red", alpha=0.25, label=f"QUERY: '{w}'")
        for x in (t0, t1):
            axes[0].axvline(x, color="tab:red", ls="--", lw=0.8)
        axes[0].legend(fontsize=8)
        axes[0].set_ylabel("waveform")
        from scipy.signal import spectrogram

        f_, t_, S = spectrogram(audio, sr, nperseg=512, noverlap=384)
        axes[1].pcolormesh(t_, f_, 10 * np.log10(S + 1e-12), cmap="magma", shading="auto")
        axes[1].axvspan(t0, t1, color="white", alpha=0.0)
        for x in (t0, t1):
            axes[1].axvline(x, color="white", ls="--", lw=0.8)
        axes[1].set_ylabel("Hz")
        axes[1].set_xlabel(f"{uid}  prev='{pw}'  next='{nw}'")
        png = OUT / f"page_{k:03d}.png"
        fig.tight_layout()
        fig.savefig(png, dpi=90)
        plt.close(fig)
        html = OUT / f"page_{k:03d}.html"
        html.write_text(
            "<html><head><meta charset='utf-8'><title>"
            f"{uid} word {idx}</title></head><body style='font-family:monospace'>"
            f"<h3>{uid} — word [{idx}] '{w}' ({t0:.3f}–{t1:.3f} s)</h3>"
            f"<p>context: prev='{pw}' next='{nw}'</p>"
            f"<img src='page_{k:03d}.png' width='900'>"
            "<p>JUDGEMENT (human): does the highlighted interval cover the spoken "
            "word and only that word? [ good / shifted / wrong / unclear ]</p>"
            "</body></html>"
        )
        manifest.append(
            {
                "page": html.name,
                "uid": uid,
                "speaker": spk,
                "word_index": idx,
                "word": w,
                "t0": t0,
                "t1": t1,
                "prev": pw,
                "next": nw,
            }
        )

    with open(OUT / "review.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["page", "uid", "word", "judgement", "notes"])
        for m in manifest:
            wr.writerow([m["page"], m["uid"], m["word"], "", ""])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))

    qa["n_sample"] = len(sample)
    qa["n_sample_speakers"] = len(per_spk)
    qa["sample_rule"] = f"every {step}th word of sorted corpus word list, cap 2/speaker"
    qa["human_validation"] = "PENDING — review.csv judgement column is empty; "
    "downstream results carry 'MFA-reference, human-validation pending'"
    print(json.dumps({k: v for k, v in qa.items() if k != "n_missing_textgrid"}, indent=1))
    if qa["n_missing_textgrid"]:
        print(f"missing textgrids: {qa['n_missing_textgrid'][:5]} ...")
    (ROOT / "outputs" / "mfa_alignment_qa.json").write_text(json.dumps(qa, indent=1))
    print(f"wrote {OUT} ({len(sample)} pages) and outputs/mfa_alignment_qa.json")


if __name__ == "__main__":
    main()
