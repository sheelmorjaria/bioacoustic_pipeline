#!/usr/bin/env python3
"""Token-matched human-vs-bat functional-aliasing control.

The falsifiability test for the central claim ("functional aliasing, not recurrence
alone, separates human from bat"): downsample HUMAN LibriSpeech to the bat corpus's
token count and check whether human functional aliasing still survives its nulls
while bat (corrected, `outputs/track_b_bat_corrected.json`) stays absent.

If human aliasing survives at bat-matched token count and bat remains absent, the
contrast is robust to corpus size (not a power artifact). If human aliasing vanishes
at the matched count, bat's absence could be under-powering and the contrast collapses.

Human tokens come from the human VQ-VAE (firewall-legal for human). LibriSpeech
speaker IDs (first path component) feed a within-speaker null where the speaker
distribution allows it.

Output: outputs/control_token_matched.json

Usage:
    python scripts/control_token_matched.py --n-files 500

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LIBRISPEECH = Path("/home/sheel/birdsong_analysis/data/LibriSpeech/train-clean-360")
BAT_TARGET_TOKENS = 15685  # bat corrected run token count (the matching size)
OUT = ROOT / "outputs" / "control_token_matched.json"


def _speaker_of(path: Path) -> str:
    """LibriSpeech speaker id = first dash component of the filename."""
    return path.stem.split("-")[0]


def main():
    ap = argparse.ArgumentParser(description="Token-matched human-vs-bat aliasing control")
    ap.add_argument("--audio-dir", type=Path, default=LIBRISPEECH)
    ap.add_argument("--n-files", type=int, default=500)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--targets", type=int, nargs="*", default=[4000, 8000, 12000, 15685, 20000, 30000])
    args = ap.parse_args()

    from bioacoustic_pipeline.diagnostics.alias_detector import (  # noqa: PLC0415
        controlled_alias_tests,
        detect_functional_aliases,
    )
    from bioacoustic_pipeline.diagnostics.lempel_ziv import lempel_ziv_complexity  # noqa: PLC0415
    from bioacoustic_pipeline.diagnostics.sequence_sufficiency import evaluate_sufficiency  # noqa: PLC0415
    from bioacoustic_pipeline.embeddings.firewall import Use, default_firewall  # noqa: PLC0415
    from bioacoustic_pipeline.ingestion.manifest import load_species_profiles  # noqa: PLC0415
    from bioacoustic_pipeline.orchestration import _default_calibrator, segment_audio  # noqa: PLC0415
    from bioacoustic_pipeline.tokenization.multiview import AcousticTokenizer  # noqa: PLC0415
    from bioacoustic_pipeline.utils.audio import load_audio  # noqa: PLC0415

    fw = default_firewall()
    fw.assert_allowed("human_vqvae_large", Use.TOKENIZE)
    fw.assert_cross_species("human_vqvae_large", "human")
    print("firewall: human_vqvae_large legal for human tokenize")

    profile = load_species_profiles()["human"]
    cal = _default_calibrator("human")
    cpc = None
    if cal is not None and getattr(cal, "fitted", False) and getattr(cal, "_use_context", False):
        from bioacoustic_pipeline.embeddings.backbones import CPCBackbone  # noqa: PLC0415
        cpc = CPCBackbone()
    tk = AcousticTokenizer()
    if not tk.available:
        raise SystemExit(f"human tokenizer unavailable: {tk.load_error}")

    files = sorted(args.audio_dir.rglob("*.flac"))[: args.n_files]
    print(f"capturing human sequences from {len(files)} LibriSpeech files "
          f"(calibrator={'fitted' if cal and cal.fitted else 'threshold'})...")

    # Capture per-file (sequence, speaker).
    captured: list[tuple[list[int], str]] = []
    n_tok = 0
    for i, f in enumerate(files):
        try:
            audio, sr = load_audio(str(f), target_sr=profile["sample_rate"])
        except Exception:  # noqa: BLE001
            continue
        segs = segment_audio(audio, sr, profile, calibrator=cal, cpc=cpc)
        seq = []
        for seg in segs:
            tok = tk.tokenize(seg.audio, seg.sr)
            if tok != -1:
                seq.append(tok)
        if seq:
            captured.append((seq, _speaker_of(f)))
            n_tok += len(seq)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)} files, {n_tok} tokens")
    print(f"captured {len(captured)} utterances, {n_tok} tokens\n")

    def _alias_at(sequences, speakers):
        rep = detect_functional_aliases(sequences, similarity_threshold=0.80, min_instances=10)
        elig = len(rep.singleton_tokens) + rep.n_merged
        density = rep.n_merged / max(elig, 1)
        # within-speaker null only meaningful with enough multi-file speakers.
        multi = sum(1 for _sp, cnt in __import__("collections").Counter(speakers).items() if cnt >= 2)
        kw = {"n_permutations": args.n_perm, "seed": 0}
        if multi >= 10:
            kw["emitter"] = speakers
        res = controlled_alias_tests(sequences, **kw)
        p = {k: round(v.p_value, 4) for k, v in res.items()}
        eff_p = max(p.values())  # alias accepted only if it survives every null run
        return density, rep.n_groups, p, eff_p, multi

    # Power curve: file-subset to ~target tokens, multiple seeds.
    targets = sorted(set(args.targets + [BAT_TARGET_TOKENS, n_tok]))
    curve = {}
    print(f"{'target':>8} {'seeds':>5} {'mean_tok':>9} {'density':>8} {'groups':>6} "
          f"{'detect':>7} {'nulls_p':>34}")
    for tgt in targets:
        if tgt > n_tok:
            continue
        seed_rows = []
        for seed in range(3):
            rng = np.random.RandomState(seed)
            order = rng.permutation(len(captured))
            acc, tok_count = [], 0
            for idx in order:
                seq, sp = captured[idx]
                acc.append((seq, sp))
                tok_count += len(seq)
                if tok_count >= tgt:
                    break
            seqs = [s for s, _ in acc]
            sps = [sp for _, sp in acc]
            suff = evaluate_sufficiency(seqs)
            if not suff.evaluable:
                seed_rows.append({"tokens": tok_count, "not_evaluable": True,
                                  "reason": suff.reason})
                continue
            density, groups, p, eff_p, multi = _alias_at(seqs, sps)
            lz = lempel_ziv_complexity([t for s in seqs for t in s][:5000]).normalized
            seed_rows.append({"tokens": tok_count, "density": round(density, 4),
                              "n_groups": groups, "per_null_p": p, "effective_p": eff_p,
                              "lz": round(lz, 4), "multi_speaker_bouts": multi})
        # summarize across seeds
        ev = [r for r in seed_rows if not r.get("not_evaluable")]
        if ev:
            mean_tok = int(np.mean([r["tokens"] for r in ev]))
            mean_dens = round(float(np.mean([r["density"] for r in ev])), 4)
            eff_ps = [r["effective_p"] for r in ev]
            det = sum(1 for p in eff_ps if p is not None and p < 0.05)
            nulls = ev[0]["per_null_p"]
            print(f"{tgt:>8} {len(ev):>5} {mean_tok:>9} {mean_dens:>8} "
                  f"{int(np.mean([r['n_groups'] for r in ev])):>6} "
                  f"{(det/len(ev)):>7.2f} {str(nulls):>34}")
        else:
            print(f"{tgt:>8} {0:>5} {'-':>9}  (not_evaluable: {seed_rows[0].get('reason')})")
        curve[tgt] = seed_rows

    # Headline: human at the bat-matched token count.
    matched = curve.get(BAT_TARGET_TOKENS, [])
    ev = [r for r in matched if not r.get("not_evaluable")]
    if ev:
        det = sum(1 for r in ev if r["effective_p"] < 0.05)
        headline = {
            "bat_matched_token_count": BAT_TARGET_TOKENS,
            "human_alias_detected_seeds": f"{det}/{len(ev)}",
            "human_mean_density": round(float(np.mean([r["density"] for r in ev])), 4),
            "human_effective_p_per_seed": [r["effective_p"] for r in ev],
            "bat_corrected_alias": "absent (genuinely tested; survives all nulls)",
        }
    else:
        headline = {"bat_matched_token_count": BAT_TARGET_TOKENS,
                    "human": "not_evaluable at matched size"}
    print("\n=== HEADLINE (token-matched) ===")
    print(json.dumps(headline, indent=2))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "species": "human",
        "n_captured_utterances": len(captured),
        "n_captured_tokens": n_tok,
        "bat_matched_token_count": BAT_TARGET_TOKENS,
        "power_curve": {str(k): v for k, v in curve.items()},
        "headline": headline,
        "note": "effective_p = max across nulls; alias detected iff effective_p < 0.05. "
                "Bat reference: outputs/track_b_bat_corrected.json (aliasing absent).",
    }, indent=2))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
