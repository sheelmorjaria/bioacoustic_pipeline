#!/usr/bin/env python3
"""Bat representation-dependence cross-check.

Runs the SAME bat context_binding diagnostic (token<->Context, within-Emitter null)
under THREE independent representations:
  1. acoustic fingerprint (deterministic, handcrafted physics)
  2. spectrogram (deterministic log-mel freq-profile, species-aware bat settings)
  3. AVES (learned multi-taxa SSL, GPU, 16kHz resampled <8kHz caveat)

If context_binding is significant under all three -> representation-ROBUST (the strongest
possible claim). If only under some -> representation-dependent (a weaker signal).

Also runs the supervised call-type... er, context probe (latent->Context linear probe)
for each representation as the supervised ceiling.

Output: outputs/bat_representation_crosscheck.json

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BAT_AUDIO = Path("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio")
BAT_ANNOT = Path("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/annotations.csv")
OUT = ROOT / "outputs" / "bat_representation_crosscheck.json"
CALL_GRANULAR = {"sample_rate": 250000, "hop_ms": 10.0,
                 "energy_threshold": 0.05, "spectral_threshold": 0.05, "debounce_ms": 25.0}


def _load_annotations():
    out = {}
    with BAT_ANNOT.open() as fh:
        for row in csv.DictReader(fh):
            fn = (row.get("File Name") or "").strip()
            if fn:
                out[fn] = {"context": int(row["Context"]), "emitter": row["Emitter"].strip()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bouts", type=int, default=2000)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n-perm", type=int, default=150)
    args = ap.parse_args()

    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: PLC0415
    from bioacoustic_pipeline.diagnostics.context_binding import association_test  # noqa: PLC0415
    from bioacoustic_pipeline.diagnostics.probes import linear_probe  # noqa: PLC0415
    from bioacoustic_pipeline.embeddings.firewall import Use, default_firewall  # noqa: PLC0415
    from bioacoustic_pipeline.ingestion.manifest import load_species_profiles  # noqa: PLC0415
    from bioacoustic_pipeline.orchestration import segment_audio  # noqa: PLC0415
    from bioacoustic_pipeline.representation import SpectrogramBackbone  # noqa: PLC0415
    from bioacoustic_pipeline.utils.audio import load_audio  # noqa: PLC0415

    fw = default_firewall()
    fw.assert_allowed("acoustic_fingerprint", Use.TOKENIZE)
    fw.assert_allowed("spectrogram", Use.EMBEDDING)
    fw.assert_allowed("aves", Use.EMBEDDING)
    for m in ("acoustic_fingerprint", "spectrogram", "aves"):
        fw.assert_cross_species(m, "bat")
    print("firewall: all 3 representations legal for bat")

    annot = _load_annotations()
    profile = {**load_species_profiles()["bat"], **CALL_GRANULAR}

    # gather bat segments
    print(f"gathering bat segments ({args.n_bouts} bouts, call-granular)...")
    segs_data = []  # list of (audio, sr, context, emitter)
    for f in [x for x in sorted(BAT_AUDIO.rglob("*.wav")) if x.name in annot][: args.n_bouts]:
        try:
            audio, sr = load_audio(str(f), target_sr=profile["sample_rate"])
        except Exception:  # noqa: BLE001
            continue
        for seg in segment_audio(audio, sr, profile):
            segs_data.append((seg.audio, seg.sr, annot[f.name]["context"], annot[f.name]["emitter"]))
    print(f"  {len(segs_data)} segments across {args.n_bouts} bouts")

    contexts = np.array([d[2] for d in segs_data])
    emitters = [d[3] for d in segs_data]
    n_ctx = len(set(contexts.tolist()))

    # AVES load (GPU)
    print("loading AVES on GPU...")
    import avex  # noqa: PLC0415
    import torch  # noqa: PLC0415
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    aves = avex.load_model("esp_aves2_eat_bio", device=dev)
    aves.eval()

    spec_backbone = SpectrogramBackbone("bat")
    from scipy.signal import resample_poly  # noqa: PLC0415

    def _resample(audio, sr_from, sr_to=16000):
        g = np.gcd(int(sr_from), sr_to)
        return resample_poly(audio, sr_to // g, int(sr_from) // g).astype(np.float32)

    def _aves_embed(audio, sr):
        a = _resample(np.asarray(audio, dtype=np.float32), sr)
        if len(a) < 800:
            a = np.pad(a, (0, 800 - len(a)))
        with torch.no_grad():
            h = aves.forward(torch.from_numpy(a).unsqueeze(0).to(dev))
        return h.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)

    representations = {}
    # 1. fingerprint
    print("computing fingerprint embeddings...")
    representations["fingerprint"] = np.array(
        [afp.extract(d[0], d[1], species="bat").combined for d in segs_data], dtype=np.float32)
    # 2. spectrogram
    print("computing spectrogram embeddings...")
    representations["spectrogram"] = np.array(
        [spec_backbone.embed(d[0], d[1]) for d in segs_data], dtype=np.float32)
    # 3. AVES
    print("computing AVES embeddings (GPU)...")
    representations["aves"] = np.array(
        [_aves_embed(d[0], d[1]) for d in segs_data], dtype=np.float32)

    from sklearn.cluster import KMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    results = {}
    print(f"\n{'representation':15s} {'dim':>4s} {'nmi':>7s} {'V':>7s} {'p':>8s} "
          f"{'probe_acc':>9s} {'probe_lift':>10s}")
    print("-" * 65)
    for name, vecs in representations.items():
        scaler = StandardScaler().fit(vecs)
        km = KMeans(n_clusters=args.k, random_state=0, n_init=10).fit(scaler.transform(vecs))
        cents = km.cluster_centers_
        toks = np.array([int(np.argmin(((scaler.transform(v[None, :])[0] - cents) ** 2).sum(axis=1)))
                         for v in vecs])
        # context_binding: token<->Context, within-Emitter null
        binding = association_test(
            [[int(t)] for t in toks], contexts.tolist(),
            statistic="nmi", n_permutations=args.n_perm, seed=0,
            group=emitters, null_type="within_group").as_dict()
        # supervised probe: representation->Context
        probe = linear_probe(vecs, contexts, min_per_class=5).as_dict()
        results[name] = {
            "dim": int(vecs.shape[1]),
            "nmi": round(binding["observed_nmi"], 4),
            "cramers_v": round(binding["cramers_v"], 4),
            "p_value": binding["p_value"],
            "probe_acc": probe.get("accuracy"),
            "probe_lift": probe.get("lift_over_chance"),
            "codes_used": int(len(set(toks.tolist()))),
        }
        print(f"{name:15s} {vecs.shape[1]:>4d} {binding['observed_nmi']:>7.4f} "
              f"{binding['cramers_v']:>7.4f} {binding['p_value']:>8} "
              f"{probe.get('accuracy','?'):>9} {probe.get('lift_over_chance','?'):>10}")

    sig_count = sum(1 for r in results.values() if r["p_value"] < 0.05)
    print(f"\ncontext_binding significant under {sig_count}/{len(results)} representations")
    if sig_count == len(results):
        print("=> REPRESENTATION-ROBUST: bat context_binding survives all representations")
    elif sig_count > 0:
        print("=> PARTIAL: representation-dependent signal")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n_segments": len(segs_data), "n_contexts": n_ctx, "k": args.k,
        "results": results,
        "interpretation": (
            "representation-robust" if sig_count == len(results)
            else f"partial ({sig_count}/{len(results)} significant)"
        ),
    }, indent=2))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
