#!/usr/bin/env python3
"""DUSTED recurrent-acoustic-term discovery on Bengalese-finch song (GT-validated).

Bengalese finch song is the canonical rich-syllable-sequence corpus — the right dataset
class for diverse recurrent acoustic terms (unlike marmoset calls, which collapsed to one
dominant repeat). Each `Annotation.xml` (xwaves) gives, per song bout: the source wav,
the bout window (sample offset), and per-syllable `<Note>` boundaries **plus ground-truth
syllable `<Label>`**. So we run DUSTED *blind* on per-syllable acoustic fingerprints and
**validate** the discovered terms against the known syllable labels — a genuine positive
control: `label_purity` (fraction of a motif's members sharing one GT label-sequence)
measures whether DUSTED recovers the bird's true recurring syllable motifs.

Each bout = one CallUnits (ordered syllable sequence); syllable = unit (acoustic
fingerprint + GT label as the cluster id). Oracle syllable boundaries isolate discovery
from segmentation quality (HECB segmentation is orthogonal, already validated).

Per-bird 3-way comparison (exact n-grams / fixed-length cosine / DTW with indels) +
order-shuffle null + bootstrap, then aggregated across birds.

Discovery target = RECURRENT ACOUSTIC TERMS / MOTIFS, NOT words. `semantic_validation`
stays `none` (syllable labels are acoustic categories, not meanings).

Run:  python scripts/dusted_bengalese.py --birds 0 1 2

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import sys
import wave
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.diagnostics.motif_miner import subsequence_motifs as sm  # noqa: E402

DATA = Path("/mnt/c/Users/sheel/Desktop/data/bengalifinch")
SR = 32000
SPECIES = "bengalese_finch"
_WAV_CACHE: dict[str, np.ndarray | None] = {}


def _load_wav(path: Path) -> np.ndarray | None:
    key = str(path)
    if key not in _WAV_CACHE:
        try:
            with wave.open(str(path), "rb") as wf:
                n = wf.getnframes()
                raw = wf.readframes(n)
                a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            _WAV_CACHE[key] = a
        except Exception:  # noqa: BLE001
            _WAV_CACHE[key] = None
    return _WAV_CACHE[key]


def parse_annotation(xml_path: Path) -> list[tuple[str, list[tuple[int, int, int]]]]:
    """xwaves Annotation.xml → [(wav_name, [(abs_start_sample, length, label), ...]), ...].

    Note positions are relative to the bout (Sequence) window; absolute sample =
    Sequence.Position + Note.Position.
    """
    tree = ET.parse(xml_path)
    bouts = []
    label_map: dict[str, int] = {}  # xwaves labels can be numeric or letters ('a','i','*')

    def _label_id(text: str | None) -> int:
        key = (text or "").strip()
        if key not in label_map:
            label_map[key] = len(label_map)
        return label_map[key]

    for seq in tree.getroot().findall("Sequence"):
        wav = seq.findtext("WaveFileName") or ""
        bout_pos = int(seq.findtext("Position") or "0")
        notes = []
        for nt in seq.findall("Note"):
            rel = int(nt.findtext("Position") or "0")
            length = int(nt.findtext("Length") or "0")
            label = _label_id(nt.findtext("Label"))
            notes.append((bout_pos + rel, length, label))
        if notes:
            bouts.append((wav, notes))
    return bouts


def build_bird_call_units(bird: int, n_bouts: int) -> list[sm.CallUnits]:
    """One bird's bouts → CallUnits (syllable fingerprints + GT labels)."""
    bdir = DATA / f"Bird{bird}" / f"Bird{bird}"
    xml = bdir / "Annotation.xml"
    if not xml.exists():
        print(f"  [Bird{bird}] missing {xml}", flush=True)
        return []
    bouts = parse_annotation(xml)[:n_bouts]
    fp_fn = lambda seg, sr: afp.extract(seg, sr, species=SPECIES).combined  # noqa: E731
    calls: list[sm.CallUnits] = []
    for bi, (wav, notes) in enumerate(bouts):
        wav_path = bdir / "Wave" / wav
        audio = _load_wav(wav_path)
        if audio is None:
            continue
        fps, labels = [], []
        for abs_start, length, label in notes:
            seg = audio[abs_start : abs_start + length]
            if len(seg) < int(0.012 * SR):
                continue
            fp = np.asarray(fp_fn(seg, SR), dtype=np.float32)
            if np.any(fp):
                fps.append(fp)
                labels.append(label)
        if len(fps) >= 2:
            calls.append(
                sm.CallUnits(
                    fingerprints=fps,
                    clusters=labels,
                    calltype=f"Bird{bird}",
                    caller=f"Bird{bird}",
                    file_id=wav,
                    call_idx=len(calls),
                )
            )
    return calls


def run_bird(bird: int, args) -> dict | None:
    calls = build_bird_call_units(bird, args.n_bouts)
    if not calls:
        return None
    n_syl = sum(c.n_units for c in calls)
    labels = sorted({lab for c in calls for lab in c.clusters})
    print(
        f"\n[Bird{bird}] {len(calls)} bouts, {n_syl} syllables "
        f"({n_syl / len(calls):.1f}/bout), {len(labels)} GT labels {labels}",
        flush=True,
    )
    common = dict(min_n=args.min_n, max_n=args.max_n, min_support=3, min_calls=2)

    def _sum(name, motifs):
        s = sm.summarize_discovery(motifs, len(calls))
        print(
            f"  {name:20} motifs={s['n_motifs']:4d}  support={s['total_support']:5d}  "
            f"cov={s['coverage']:.3f}  purity(m/w)={s['mean_label_purity']:.2f}/"
            f"{s['wmean_label_purity']:.2f}"
        )
        return s

    exact = sm.exact_ngram_clusters(calls, **common)
    fixed = sm.fixed_length_cosine_clusters(
        calls, **common, dist_threshold=args.dist_threshold, centroid_radius=args.centroid_radius
    )
    dtw = sm.discover_recurrent_subsequences(
        calls, **common, dist_threshold=args.dist_threshold, centroid_radius=args.centroid_radius
    )
    _sum("exact n-grams", exact)
    _sum("fixed-length cosine", fixed)
    _sum("DTW with indels", dtw)

    sub = calls[: args.n_null] if len(calls) >= args.n_null else calls
    null = sm.discovery_null(
        sub,
        dist_threshold=args.dist_threshold,
        centroid_radius=args.centroid_radius,
        min_calls=2,
        n_perm=args.n_perm,
        seed=args.seed,
        min_n=args.min_n,
        max_n=args.max_n,
    )
    jac = sm.bootstrap_stability(
        sub,
        dist_threshold=args.dist_threshold,
        centroid_radius=args.centroid_radius,
        min_calls=2,
        n_boot=args.n_boot,
        top_k=10,
        seed=args.seed,
        min_n=args.min_n,
        max_n=args.max_n,
    )
    print(
        f"  DTW null: obs_support={null['observed_support']} null={null['null_mean']:.0f}"
        f"±{null['null_std']:.0f} p={null['p']:.3f}  |  bootstrap Jaccard={jac:.3f}",
        flush=True,
    )
    print(sm.format_motif_table(dtw, top=8), flush=True)
    return {
        "bird": bird,
        "n_bouts": len(calls),
        "exact": sm.summarize_discovery(exact, len(calls)),
        "fixed": sm.summarize_discovery(fixed, len(calls)),
        "dtw": sm.summarize_discovery(dtw, len(calls)),
        "null_p": null["p"],
        "null_obs": null["observed_support"],
        "null_mean": null["null_mean"],
        "bootstrap_jac": jac,
    }


def main():
    ap = argparse.ArgumentParser(description="DUSTED on Bengalese-finch song (GT-validated)")
    ap.add_argument("--birds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-bouts", type=int, default=100000, help="cap bouts per bird (default all)")
    ap.add_argument("--n-null", type=int, default=400, help="bouts for the null/bootstrap subset")
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--max-n", type=int, default=4)
    ap.add_argument(
        "--dist-threshold",
        type=float,
        default=0.15,
        help="DTW/cosine threshold (0.15 = Bengalese syllables; 0.30 collapses)",
    )
    ap.add_argument("--centroid-radius", type=float, default=0.20)
    ap.add_argument("--n-perm", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(
        f"DUSTED on Bengalese finch — birds {args.birds} (thr={args.dist_threshold}, "
        f"L={args.min_n}-{args.max_n}, k=GT labels)",
        flush=True,
    )
    results = [r for r in (run_bird(b, args) for b in args.birds) if r]

    if not results:
        print("no birds produced data")
        return
    print(f"\n{'=' * 78}\nAGGREGATE across {len(results)} birds (mean)\n{'=' * 78}")
    print(f"  {'method':20} {'#mot':>5} {'cov':>6} {'purity(w)':>9} {'H(norm)':>8} {'top-frac':>8}")
    for key in ("exact", "fixed", "dtw"):
        rs = [r[key] for r in results]
        print(
            f"  {key:20} {np.mean([s['n_motifs'] for s in rs]):5.0f} "
            f"{np.mean([s['coverage'] for s in rs]):6.3f} "
            f"{np.mean([s['wmean_label_purity'] for s in rs]):9.3f} "
            f"{np.mean([s['family_entropy_norm'] for s in rs]):8.2f} "
            f"{np.mean([s['top_family_support_fraction'] for s in rs]):8.2f}"
        )
    null_ratio = np.mean([r["null_obs"] / max(r["null_mean"], 1) for r in results])
    print(
        f"  DTW null-ratio (obs/null, mean)={null_ratio:.2f}x  "
        f"null p (mean)={np.mean([r['null_p'] for r in results]):.3f}  "
        f"bootstrap Jaccard (mean)={np.mean([r['bootstrap_jac'] for r in results]):.3f}"
    )
    print("\nDUSTED BENGALESE DONE  (discoveries = recurrent acoustic terms, NOT words)")


if __name__ == "__main__":
    main()
