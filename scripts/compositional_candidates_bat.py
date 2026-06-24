#!/usr/bin/env python3
"""Compositional-candidate mining on Egyptian fruit bat (Semantic-Ceiling-safe).

Compositionality = "a sequence AB carries information/distinct behaviour that A or B alone
do not" (e.g. Campbell's monkey "-oo" suffixation). The pipeline's Semantic Ceiling forbids
claiming meaning from acoustics. So this script does NOT detect compositionality — it detects
the **structural prerequisites** for it (T0) and hands the wet lab the exact counterfactual
stimuli (T2) that would be needed to test it biologically.

It reuses ONLY frozen modules — no new "compositionality detector":
  * segmentation: raw NBD (bat_call_granular) — the validated, species-appropriate
    sub-call segmenter for graded bat bouts (HECB's CRNN is discrete-call-only; see script
    header discussion). NOT HECB.
  * tokenization: global k-means over sub-call acoustic fingerprints (cluster IDs).
  * structural_alternatives.probabilistic_suffix_tree (variable-order Markov syntax)
  * structural_alternatives.transition_network (graph topology + top weighted edges)
  * context_binding.association_test (cross-bout pattern↔Context Cramér's V)
  * experiments.playback_design.audio_splicer.generate_compositional_counterfactuals

STRUCTURAL PREREQUISITES detected (T0), all firewall-clean / meaning-free:
  1. Non-random transitions: bigrams A→B with P(B|A) >> marginal P(B) (PST-style).
  2. Positional specialization: clusters that are onset- or offset-biased (suffix/prefix
     candidates) — the acoustic signature of a positional morpheme.
  3. Cross-bout distributional test (honest): does *bout-level occurrence* of the AB bigram
     associate with a specific Context more strongly than occurrence of A or B alone? This is
     a structural prerequisite (the combination carries distinct distributional info), NOT
     meaning. CAVEAT: bat Context is per-bout, so this is a cross-bout co-occurrence test,
     not a within-sequence meaning test.

COUNTERFACTUALS generated (T2) for the top AB candidate bouts, via the DSP Audio Splicer:
  natural_AB · isolated_A · isolated_B · reversed_BA · amplitude_matched_AB.
Each stimulus carries full provenance; the manifest states the H0-CMC hypothesis:
"if AB is compositional, receiver response to natural_AB > max(isolated_A, isolated_B) and
differs from reversed_BA." The wet lab decides. The pipeline asserts NOTHING about meaning.

Run:  python scripts/compositional_candidates_bat.py --n-bouts 2000 --k 24

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = Path("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats")
SPECIES = "egyptian_fruit_bat"
_PROPOSAL = dict(floor_energy=0.05, floor_flux=0.05, debounce_ms=25.0)  # bat_call_granular
HOP_MS = 10.0


def parse_annotations(csv_path: Path) -> dict[str, tuple[str, str, str]]:
    out: dict[str, tuple[str, str, str]] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out[row["File Name"]] = (row["Emitter"], row["Addressee"], row["Context"])
    return out


def segment_bout(audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Raw NBD (bat_call_granular) → sub-call (start_ms, end_ms) segments."""
    from bioacoustic_pipeline.segmentation.nbd_proposal import (  # noqa: PLC0415
        compute_signals,
        generate_proposals,
    )

    sig = compute_signals(audio, sr, hop_ms=HOP_MS)
    props = generate_proposals(sig, **_PROPOSAL)
    times = [0.0] + [p.time_ms for p in props] + [len(audio) / sr * 1000.0]
    return [(times[i], times[i + 1]) for i in range(len(times) - 1)]


def build_bouts(ann, n_bouts, seed):
    """Load bouts; retain per-unit audio slices (for splicing) + fingerprints + labels."""
    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: PLC0415
    from bioacoustic_pipeline.utils.audio import load_audio  # noqa: PLC0415

    files = sorted(ann.keys())
    random.Random(seed).shuffle(files)
    bouts = []
    min_n = int(0.012 * 250000)  # 12 ms at a generous sr floor
    for fn in files:
        if len(bouts) >= n_bouts:
            break
        try:
            audio, sr = load_audio(str(DATA / "audio" / fn), target_sr=None)
        except Exception:  # noqa: BLE001
            continue
        if audio is None or len(audio) < int(0.03 * sr):
            continue
        units, fps = [], []
        for s_ms, e_ms in segment_bout(audio, sr):
            si = int(s_ms / 1000 * sr)
            sj = max(int(e_ms / 1000 * sr), si + int(0.012 * sr))
            seg = audio[si:sj]
            if len(seg) < min_n:
                continue
            fp = np.asarray(afp.extract(seg, sr, species=SPECIES).combined, dtype=np.float32)
            if np.any(fp):
                units.append(np.asarray(seg, dtype=np.float32))
                fps.append(fp)
        if len(fps) >= 2:
            emitter, _, context = ann[fn]
            bouts.append(
                {
                    "file_id": fn,
                    "sr": int(sr),
                    "units": units,  # per-unit audio (for splicing)
                    "fps": fps,  # per-unit fingerprints (for k-means)
                    "context": str(context),
                    "emitter": str(emitter),
                }
            )
    return bouts


def tokenize(bouts, k):
    """Global k-means over all sub-call fingerprints → cluster seq per bout (in place)."""
    from sklearn.cluster import KMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    all_fp = np.asarray([fp for b in bouts for fp in b["fps"]], dtype=np.float32)
    if len(all_fp) < k * 5:
        return False
    scaler = StandardScaler().fit(all_fp)
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(scaler.transform(all_fp))
    cents = km.cluster_centers_
    for b in bouts:
        sv = scaler.transform(np.asarray(b["fps"], dtype=np.float32))
        b["seq"] = np.argmin(((sv[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2), axis=1)
    return True


def bigram_transitions(sequences):
    """Return list of (a, b, count, p_b_given_a, marginal_p_b) sorted by P(B|A)/P(B)."""
    pair = Counter()
    follower = Counter()
    total_transitions = 0
    for s in sequences:
        for a, b in zip(s[:-1], s[1:]):
            pair[(int(a), int(b))] += 1
            follower[int(b)] += 1
            total_transitions += 1
    if total_transitions == 0:
        return []
    out = []
    for (a, b), c in pair.items():
        a_first = sum(v for (aa, _bb), v in pair.items() if aa == a)  # count of a-as-first-token
        p_b_given_a = c / a_first
        marginal_p_b = follower[b] / total_transitions
        out.append((a, b, c, p_b_given_a, marginal_p_b))
    out.sort(key=lambda t: (t[3] / max(t[4], 1e-9)), reverse=True)
    return out


def positional_specialization(sequences):
    """Per cluster: onset fraction (pos==0) and offset fraction (pos==last)."""
    onset = Counter()
    offset = Counter()
    total = Counter()
    for s in sequences:
        s = [int(x) for x in s]
        if not s:
            continue
        onset[s[0]] += 1
        offset[s[-1]] += 1
        for t in s:
            total[t] += 1
    specs = []
    for cl, n in total.items():
        if n < 5:
            continue
        specs.append(
            {
                "cluster": cl,
                "n": n,
                "onset_frac": onset[cl] / max(total[cl], 1),
                "offset_frac": offset[cl] / max(total[cl], 1),
            }
        )
    specs.sort(key=lambda d: max(d["onset_frac"], d["offset_frac"]), reverse=True)
    return specs


def pattern_context_v(bouts, predicate, contexts, seed=0):
    """Cramér's V between a per-bout binary pattern-presence indicator and Context."""
    from bioacoustic_pipeline.diagnostics.context_binding import association_test  # noqa: PLC0415

    tok = [[1 if predicate(b) else 0] for b in bouts]
    if len(set(t[0] for t in tok)) < 2:
        return {"cramers_v": 0.0, "p_value": 1.0, "n_present": int(sum(t[0] for t in tok))}
    res = association_test(tok, contexts, statistic="nmi", n_permutations=200, seed=seed).as_dict()
    res["n_present"] = int(sum(t[0] for t in tok))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bouts", type=int, default=2000)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--tp-ratio", type=float, default=3.0, help="min P(B|A)/P(B) for a candidate")
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--n-stim-bouts", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from bioacoustic_pipeline.diagnostics.structural_alternatives import (  # noqa: PLC0415
        probabilistic_suffix_tree as pst,
        transition_network as tn,
    )
    from bioacoustic_pipeline.embeddings.firewall import Use, default_firewall  # noqa: PLC0415
    from bioacoustic_pipeline.experiments.playback_design import audio_splicer  # noqa: PLC0415

    fw = default_firewall()
    fw.assert_allowed("acoustic_fingerprint", Use.TOKENIZE)
    fw.assert_cross_species("acoustic_fingerprint", SPECIES)
    print("firewall: acoustic_fingerprint tokenize legal for bat (raw-NBD sub-call units)")

    ann = parse_annotations(DATA / "annotations.csv")
    print(f"building sub-call units on {args.n_bouts} bouts (raw NBD, bat_call_granular)...")
    bouts = build_bouts(ann, args.n_bouts, args.seed)
    n_units = sum(len(b["seq"]) if "seq" in b else len(b["fps"]) for b in bouts)
    print(
        f"  {len(bouts)} bouts, {n_units} sub-call units ({n_units / max(len(bouts), 1):.1f}/bout)"
    )
    if not tokenize(bouts, args.k):
        print("ERROR: too few sub-call units for k-means")
        return

    sequences = [list(map(int, b["seq"])) for b in bouts]
    contexts = [b["context"] for b in bouts]

    # ---- T0 detection ----
    trans = bigram_transitions(sequences)
    candidates = [
        {
            "a": a,
            "b": b,
            "count": c,
            "p_b_given_a": round(pba, 3),
            "marginal_p_b": round(pb, 3),
            "tp_ratio": round(pba / max(pb, 1e-9), 2),
        }
        for a, b, c, pba, pb in trans
        if a != b and c >= args.min_count and (pba / max(pb, 1e-9)) >= args.tp_ratio
    ]
    specs = positional_specialization(sequences)
    net = tn.transition_network_metrics(sequences)
    pst_sum = pst.fit_pst(sequences)

    # ---- cross-bout compositional-prerequisite test (honest, per-bout Context caveat) ----
    comp_test = None
    if candidates:
        top = candidates[0]
        a_c, b_c = top["a"], top["b"]
        v_ab = pattern_context_v(
            bouts, lambda b, a=a_c, c=b_c: _has_bigram(b["seq"], a, c), contexts, args.seed
        )
        v_a = pattern_context_v(
            bouts, lambda b, a=a_c: _has_token(b["seq"], a), contexts, args.seed
        )
        v_b = pattern_context_v(
            bouts, lambda b, c=b_c: _has_token(b["seq"], c), contexts, args.seed
        )
        comp_test = {
            "candidate": [a_c, b_c],
            "v_AB_context": v_ab["cramers_v"],
            "p_AB": v_ab["p_value"],
            "n_AB": v_ab["n_present"],
            "v_A_context": v_a["cramers_v"],
            "n_A": v_a["n_present"],
            "v_B_context": v_b["cramers_v"],
            "n_B": v_b["n_present"],
            "AB_greater_than_parts": v_ab["cramers_v"] > max(v_a["cramers_v"], v_b["cramers_v"]),
        }

    # ---- report ----
    print("\n" + "=" * 92)
    print("COMPOSITIONAL-CANDIDATE MINING — Egyptian fruit bat (Semantic-Ceiling-safe, T0 only)")
    print("=" * 92)
    print(f"  n_bouts={len(bouts)}  k={args.k}  units/bout={n_units / max(len(bouts), 1):.1f}")
    print(
        f"  transition network: nodes={net.get('n_nodes')} edges={net.get('n_edges')} "
        f"communities={net.get('modularity_communities')} density={net.get('density', 0):.3f}"
    )
    print(
        f"  PST: significant_nodes={pst_sum['n_significant_nodes']} "
        f"max_depth={pst_sum['max_depth_reached']} entropy_rate="
        f"{pst_sum['entropy_rate'] if pst_sum['entropy_rate'] is not None else 'n/a'}"
    )

    print(
        f"\n  Top high-TP bigram candidates (P(B|A)/P(B) >= {args.tp_ratio}, count >= {args.min_count}):"
    )
    print(f"    {'A':>3} {'B':>3} {'count':>6} {'P(B|A)':>8} {'P(B)':>7} {'ratio':>6}")
    for c in candidates[:12]:
        print(
            f"    {c['a']:>3} {c['b']:>3} {c['count']:>6} {c['p_b_given_a']:>8.3f} "
            f"{c['marginal_p_b']:>7.3f} {c['tp_ratio']:>6.2f}"
        )

    print("\n  Positional specialists (onset/offset-biased clusters — suffix/prefix candidates):")
    print(f"    {'cluster':>7} {'n':>6} {'onset_frac':>11} {'offset_frac':>12}")
    for s in specs[:8]:
        print(
            f"    {s['cluster']:>7} {s['n']:>6} {s['onset_frac']:>11.2f} {s['offset_frac']:>12.2f}"
        )

    if comp_test:
        ct = comp_test
        print(f"\n  Cross-bout compositional-prerequisite test (top candidate {ct['candidate']}):")
        print(
            f"    V(AB-present, Context) = {ct['v_AB_context']:.3f}  (p={ct['p_AB']:.3f}, n={ct['n_AB']})"
        )
        print(f"    V(A-present,  Context) = {ct['v_A_context']:.3f}  (n={ct['n_A']})")
        print(f"    V(B-present,  Context) = {ct['v_B_context']:.3f}  (n={ct['n_B']})")
        print(
            f"    => AB association {'EXCEEDS' if ct['AB_greater_than_parts'] else 'does NOT exceed'} "
            f"both parts (structural prerequisite {'present' if ct['AB_greater_than_parts'] else 'absent'})."
        )
        print(
            "    CAVEAT: bat Context is per-bout; this is a cross-bout distributional test, "
            "NOT within-sequence meaning."
        )

    print("\n  Framing: recurrent acoustic sequences + positional specialists = STRUCTURAL")
    print("  prerequisites for compositionality (T0). Whether AB is compositional requires")
    print("  Track-C playback (T2-T4): natural_AB vs isolated_A/B vs reversed_BA.")

    # ---- T2 counterfactual stimuli for top candidate bouts ----
    stim_dir = ROOT / "outputs" / "compositional_stimuli"
    manifest_entries = []
    if candidates:
        a_c, b_c = candidates[0]["a"], candidates[0]["b"]
        donor_bouts = [b for b in bouts if _has_bigram(b["seq"], a_c, b_c)][: args.n_stim_bouts]
        stim_dir.mkdir(parents=True, exist_ok=True)
        for bi, b in enumerate(donor_bouts):
            # find first AB occurrence positions in this bout
            pos = next(
                (
                    j
                    for j in range(len(b["seq"]) - 1)
                    if b["seq"][j] == a_c and b["seq"][j + 1] == b_c
                ),
                None,
            )
            if pos is None:
                continue
            units = b["units"]
            sr = b["sr"]
            res = audio_splicer.generate_compositional_counterfactuals(units, sr, pos, pos + 1)
            names = [
                "natural_AB",
                "isolated_A",
                "isolated_B",
                "reversed_BA",
                "amplitude_matched_AB",
            ]
            for name, r in zip(names, res):
                wav = stim_dir / f"bout{bi:02d}_{name}.wav"
                try:
                    import soundfile as sf  # noqa: PLC0415

                    sf.write(str(wav), r.audio, sr)
                    manifest_entries.append(
                        {
                            "stimulus_id": f"bout{bi:02d}_{name}",
                            "condition": name,
                            "source_file": b["file_id"],
                            "context": b["context"],
                            "emitter": b["emitter"],
                            "candidate_AB": [int(a_c), int(b_c)],
                            "ab_positions": [int(pos), int(pos + 1)],
                            "n_units_in_bout": len(units),
                            "provenance": r.provenance,
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"    (write skip {name}: {e})")
        print(f"\n  Wrote {len(manifest_entries)} compositional stimuli -> {stim_dir}")

    # ---- save ----
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    payload = {
        "n_bouts": len(bouts),
        "k": args.k,
        "units_per_bout": round(n_units / max(len(bouts), 1), 2),
        "transition_network": net,
        "pst": pst_sum,
        "positional_specialists": specs[:12],
        "bigram_candidates": candidates[:20],
        "compositional_prerequisite_test": comp_test,
        "n_stimuli": len(manifest_entries),
        "framing": "Structural prerequisites (T0) + counterfactual stimuli (T2). NOT compositionality. "
        "Whether AB is compositional is decided by Track-C playback.",
        "h0_cmc_hypothesis": "If AB is compositional, receiver response to natural_AB exceeds "
        "max(isolated_A, isolated_B) and differs from reversed_BA.",
    }
    with open(out / "compositional_candidates_bat.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    with open(out / "compositional_candidates_bat.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["a", "b", "count", "p_b_given_a", "marginal_p_b", "tp_ratio"]
        )
        w.writeheader()
        w.writerows(candidates[:50])
    if manifest_entries:
        with open(stim_dir / "manifest.json", "w") as fh:
            json.dump(
                {
                    "experiment": "compositional_playback",
                    "h0_cmc_hypothesis": payload["h0_cmc_hypothesis"],
                    "conditions": [
                        "natural_AB",
                        "isolated_A",
                        "isolated_B",
                        "reversed_BA",
                        "amplitude_matched_AB",
                    ],
                    "caveats": [
                        "Exemplar-based DSP (slice + crossfade), NOT neural synthesis.",
                        "Pipeline asserts NOTHING about meaning (semantic ceiling intact).",
                        "Whether AB is compositional is decided by receiver response (T2-T4).",
                    ],
                    "stimuli": manifest_entries,
                },
                fh,
                indent=2,
            )
    print(
        "saved outputs/compositional_candidates_bat.{json,csv}"
        + (f" + {stim_dir}/manifest.json" if manifest_entries else "")
    )
    print("COMPOSITIONAL CANDIDATES DONE")


def _has_bigram(seq, a, b):
    s = [int(x) for x in seq]
    return any(s[j] == a and s[j + 1] == b for j in range(len(s) - 1))


def _has_token(seq, a):
    return int(a) in {int(x) for x in seq}


if __name__ == "__main__":
    main()
