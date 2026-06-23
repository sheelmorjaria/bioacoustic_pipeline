#!/usr/bin/env python3
"""Extract AVES frame trajectories for the §21 unified LibriSpeech manifest.

AVES (Animal Vocalization Encoder; `esp_aves2_eat_bio`) is a multi-taxa SSL model —
firewall-legal for embedding (all-species) and 16 kHz (matches LibriSpeech, no resample).
The §27 DBA consensus-template sweep needs per-word AVES frame trajectories aligned with
`outputs/ls_unified/recs.pkl`. This re-reads the manifest's audio, AVES-encodes each file
once (GPU), and slices per-word by the manifest's spans (frame-rate derived from the output,
not assumed). Output: `outputs/ls_unified/aves_recs.pkl` — list of (T, 768) arrays in
manifest/word_spans order (= recs.pkl order), so they can be injected and used with
`ls_consensus_template.py --rep aves`.

Run:  python scripts/ls_aves_extract.py

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

LS = Path("/home/sheel/birdsong_analysis/data/LibriSpeech/train-clean-360")
MANIFEST = ROOT / "outputs" / "ls_unified" / "manifest.json"
OUT = ROOT / "outputs" / "ls_unified" / "aves_recs.pkl"


def main():
    m = json.loads(MANIFEST.read_text())
    file_ids = m["file_ids"]  # basenames
    word_spans = m["word_spans"]  # [file_idx, ws_ms, we_ms, word, speaker, dur, nphon]
    n_files = m["n_files"]
    n_words = m["n_words"]
    print(f"manifest: {n_files} files, {n_words} words", flush=True)

    # basename → full path (one-time glob of the LS tree)
    print("resolving file paths...", flush=True)
    all_flacs = glob.glob(str(LS / "**" / "*.flac"), recursive=True)
    base2path = {Path(f).name: f for f in all_flacs}
    missing = [fid for fid in file_ids if fid not in base2path]
    print(f"  {len(base2path)} flacs indexed | {len(missing)} manifest files missing", flush=True)

    # group word spans by file_idx
    by_file = {}
    for wi, span in enumerate(word_spans):
        by_file.setdefault(span[0], []).append((wi, span[1], span[2]))  # word_idx, ws_ms, we_ms

    print("loading AVES (esp_aves2_eat_bio)...", flush=True)
    import torch  # noqa: PLC0415
    import avex  # noqa: PLC0415

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    aves = avex.load_model("esp_aves2_eat_bio", device=dev).eval()
    print(f"  AVES on {dev}", flush=True)

    aves_recs = [None] * n_words
    n_done_files = 0
    for file_idx_str in sorted(by_file, key=lambda x: int(x)):
        fid = file_ids[file_idx_str]
        path = base2path.get(fid)
        if path is None:
            continue
        audio, sr = load_audio(path, target_sr=16000)
        if audio is None or len(audio) < 800:
            continue
        with torch.no_grad():
            wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(dev)
            h = aves.forward(wav)  # (1, T, 768)
        h = h.squeeze(0).cpu().numpy()  # (T, 768)
        T = h.shape[0]
        fps = T / (len(audio) / sr)  # actual frames/sec
        for wi, ws_ms, we_ms in by_file[file_idx_str]:
            fs = max(0, int(ws_ms / 1000 * fps))
            fe = max(fs + 1, int(we_ms / 1000 * fps))
            aves_recs[wi] = h[fs:fe].astype(np.float32)
        n_done_files += 1
        if n_done_files % 25 == 0:
            n_have = sum(1 for x in aves_recs if x is not None)
            print(f"  {n_done_files}/{len(by_file)} files → {n_have}/{n_words} words", flush=True)

    n_have = sum(1 for x in aves_recs if x is not None)
    with OUT.open("wb") as f:
        pickle.dump(aves_recs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"\nsaved {OUT} | {n_have}/{n_words} words have AVES frames "
        f"({100 * n_have / n_words:.0f}%)",
        flush=True,
    )
    # shape sanity
    for i, x in enumerate(aves_recs):
        if x is not None and x.shape[0] >= 2:
            print(f"  example word[{i}] AVES shape: {x.shape}")
            break
    print("LS AVES EXTRACT DONE")


if __name__ == "__main__":
    main()
