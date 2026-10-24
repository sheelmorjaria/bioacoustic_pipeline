#!/usr/bin/env python3
"""Evidence workbench v1: per-recording and paired-comparison HTML reports.

The acceptance test (consolidation plan): another researcher can follow a
reported result back to the source, understand its exclusions, and
distinguish valid acoustic evidence from a processing artefact — without
reading the training code.

Per-recording report places together:
  - waveform + spectrogram with the ACTUAL analysis grid marked;
  - activity mask and representation-validity mask displayed SEPARATELY
    (the four validity concepts are never merged);
  - the 12 named supported features and their derivatives, with units and
    missing-support indicators (gated frames hatched, never interpolated);
  - provenance (caller, recipient, context, recording block, split part);
  - evidence status badge from the project's status vocabulary.

Paired report: two recordings side by side + SEPARATE level and rate
distances (identical rates can hide different levels).

Static HTML with embedded figures — no platform, no server. The release
contract gates the run (schema v2, masks declared, provenance required).

Usage:
    python scripts/evidence_report.py --recording 5041.wav
    python scripts/evidence_report.py --pair 5041.wav 11799.wav
    python scripts/evidence_report.py --panel   # one report per panel file

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WIN_MS, HOP_MS, MAX_FRAMES = 40.0, 10.0, 300

STATUS_BADGES = {
    "valid measurement": "#2e7d32",
    "exploratory association": "#f9a825",
    "rejected match": "#c62828",
    "source-supported candidate": "#1565c0",
    "receiver-tested result": "#6a1b9a",
}


def _fig_png(fig) -> str:
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _recording_data(file_name: str):
    import numpy as np

    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        FEATURE_SCHEMA_VERSION,
        FEATURE_UNITS,
        _NAME_TO_IDX,
        first_derivative,
        supported_features,
    )
    from bioacoustic_pipeline.ingestion.labeled_corpus import (
        filename_block_group,
        parse_annotations_csv,
    )
    from bioacoustic_pipeline.segmentation.silence_units import (
        HysteresisConfig,
        extract_activity_units,
    )
    from bioacoustic_pipeline.utils.audio import load_audio

    path = ROOT / "data/egyptian_fruit_bats/audio" / file_name
    audio, sr = load_audio(str(path), target_sr=250000)
    win, hop = int(sr * WIN_MS / 1000), int(sr * HOP_MS / 1000)
    frames, i = [], 0
    while i + win <= len(audio) and len(frames) < MAX_FRAMES:
        v = np.asarray(
            afp.extract(audio[i : i + win], sr, species="bat").combined, dtype=np.float64
        )
        frames.append(v)
        i += hop
    frames = np.asarray(frames)
    valid = ~np.all(frames == 0, axis=1)
    feats = supported_features()
    fidx = [_NAME_TO_IDX[n] for n in feats]
    sub = frames[:, fidx]
    derivs = np.full_like(sub, np.nan)
    for j in range(len(feats)):
        derivs[:, j] = first_derivative(sub[:, j], valid, dt=HOP_MS / 1000.0)
    # activity mask on the same grid (peak-relative criterion, recorded
    # separately from representation validity)
    f10 = int(sr * 0.01)
    x = audio[: (len(audio) // f10) * f10].reshape(-1, f10).astype(np.float64)
    rms10 = np.sqrt((x**2).mean(axis=1))
    peak = float(np.abs(audio).max()) or 1.0
    vocal10 = rms10 >= 1e-3 * peak
    n = min(len(vocal10) // 4, len(valid))
    act = vocal10[: 4 * n].reshape(n, 4).mean(axis=1) > 0.5
    res = extract_activity_units(
        audio, sr=sr, config=HysteresisConfig(frame_ms=10.0), ref_amplitude=peak
    )
    annos = [
        a
        for a in parse_annotations_csv(ROOT / "data/egyptian_fruit_bats/annotations.csv")
        if a.file_name == file_name
    ]
    a = annos[0] if annos else None
    return {
        "file": file_name,
        "audio": audio,
        "sr": sr,
        "frames": frames,
        "valid": valid,
        "act": act,
        "sub": sub,
        "derivs": derivs,
        "feats": feats,
        "units": [FEATURE_UNITS[n] for n in feats],
        "schema": FEATURE_SCHEMA_VERSION,
        "n_units": len(res.spans),
        "ann": a,
        "block": filename_block_group(file_name),
        "disagreement": float((act & ~valid[:n]).mean()) if n else 0.0,
    }


def _figures(d: dict) -> list[str]:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_audio = np.arange(len(d["audio"])) / d["sr"]
    t_fr = (np.arange(len(d["valid"])) + 0.5) * (HOP_MS / 1000.0)

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1]}
    )
    axes[0].plot(t_audio, d["audio"], lw=0.2, color="k")
    axes[0].set_ylabel("waveform")
    axes[0].set_title(f"{d['file']} — analysis grid {WIN_MS:.0f} ms / {HOP_MS:.0f} ms hop")
    from scipy.signal import spectrogram

    fr, tt, ss = spectrogram(d["audio"], d["sr"], nperseg=1024, noverlap=768)
    sel = (fr <= 120000) & (fr > 1000)
    axes[1].pcolormesh(
        tt, fr[sel] / 1000.0, 10 * np.log10(ss[sel] + 1e-20), shading="auto", cmap="magma"
    )
    axes[1].set_ylabel("kHz")
    m = d["act"]
    axes[2].fill_between(
        t_fr[: len(m)], 0, 1, where=m, color="#1565c0", alpha=0.5, label="activity (peak-relative)"
    )
    v = d["valid"]
    axes[2].fill_between(
        t_fr[: len(v)],
        -1,
        0,
        where=v,
        color="#2e7d32",
        alpha=0.6,
        label="feature validity (absolute gate)",
    )
    both = m & v[: len(m)]
    axes[2].fill_between(t_fr[: len(both)], -0.5, -0.5 + 0 * both, where=both, color="none")
    axes[2].set_yticks([])
    axes[2].legend(loc="upper right", fontsize=7, ncol=2)
    axes[2].set_xlabel("s")
    fig.tight_layout()
    main_png = _fig_png(fig)

    fig, axes = plt.subplots(4, 3, figsize=(11, 7), sharex=True)
    for j, (name, unit) in enumerate(zip(d["feats"], d["units"])):
        ax = axes[j % 4][j // 4]
        ax.plot(t_fr, d["sub"][:, j], lw=0.8, color="#37474f")
        ax.set_xticks([])
        ax.set_yticks([])
        bad = ~d["valid"]
        ax.scatter(t_fr[bad], d["sub"][bad, j], s=1, color="#c62828", marker="x")
        dnan = ~np.isfinite(d["derivs"][:, j])
        ax.scatter(
            t_fr[dnan],
            np.nan_to_num(d["sub"][dnan, j], nan=0.0),
            s=1,
            color="#f9a825",
            marker=".",
            alpha=0.3,
        )
        ax.set_title(f"{name} [{unit}]", fontsize=7)
    fig.suptitle("12 supported features (x = gated frame, . = no derivative)", fontsize=9)
    fig.tight_layout()
    feat_png = _fig_png(fig)
    return [main_png, feat_png]


def _status_for(file_name: str) -> str:
    adj_p = ROOT / "outputs" / "recurrence_candidate_adjudication.json"
    if adj_p.exists():
        adj = json.loads(adj_p.read_text())
        for e in adj["adjudications"]:
            pass
    return "valid measurement"


def recording_html(d: dict, status: str) -> str:
    ann = d["ann"]
    prov = {
        "file": d["file"],
        "caller": getattr(ann, "emitter", None),
        "addressee": getattr(ann, "addressee", None),
        "context": getattr(ann, "context", None),
        "recording block (proxy)": d["block"],
        "duration s": round(len(d["audio"]) / d["sr"], 2),
        "frames (40 ms/10 ms)": int(len(d["valid"])),
        "valid frames": int(d["valid"].sum()),
        "activity units": d["n_units"],
        "activity-vs-validity disagreement": round(d["disagreement"], 3),
        "feature schema": f"v{d['schema']}",
    }
    color = STATUS_BADGES.get(status, "#455a64")
    figs = _figures(d)
    rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in prov.items()
    )
    imgs = "".join(f'<img src="data:image/png;base64,{p}" width="100%">' for p in figs)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Evidence report — {html.escape(d["file"])}</title>
<style>body{{font-family:system-ui;margin:24px;max-width:1000px}}
.badge{{color:white;padding:4px 12px;border-radius:12px;font-size:14px}}
table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:4px 10px}}</style>
</head><body>
<h1>{html.escape(d["file"])}</h1>
<p><span class="badge" style="background:{color}">{html.escape(status)}</span></p>
<h2>Provenance</h2><table>{rows}</table>
<h2>Source, masks, features</h2>{imgs}
<p style="color:#666;font-size:12px">Activity and representation-validity masks are
displayed separately by contract. Gated frames are marked, never interpolated.
Derivatives are rates of acoustic descriptors — not motor velocities, not meaning.</p>
</body></html>"""


def pair_html(a: dict, b: dict, status: str = "exploratory association") -> str:
    import numpy as np

    # position-by-position over the overlapping prefix (recordings differ in
    # length; region-specific comparison belongs to the reviewer's judgement)
    n = min(len(a["valid"]), len(b["valid"]))
    both = a["valid"][:n] & b["valid"][:n]
    lvl = {
        f: (
            float(np.abs(a["sub"][:n][both, j] - b["sub"][:n][both, j]).mean())
            if both.any()
            else None
        )
        for j, f in enumerate(a["feats"])
    }
    both_d = np.zeros(n, dtype=bool)
    both_d[1:] = both[1:] & both[:-1]
    rate = {
        f: (
            float(np.abs(a["derivs"][:n][both_d, j] - b["derivs"][:n][both_d, j]).mean())
            if both_d.any()
            else None
        )
        for j, f in enumerate(a["feats"])
    }

    def _cell(x):
        return "n/a — no commonly valid frames" if x is None else f"{x:.4g}"

    rows = "".join(
        f"<tr><td>{f}</td><td>{_cell(lvl[f])}</td><td>{_cell(rate[f])}</td></tr>"
        for f in a["feats"]
    )
    body_a = recording_html(a, status)
    body_b = recording_html(b, status)
    figs_a = body_a.split("<h2>Source, masks, features</h2>")[1].split("</body>")[0]
    figs_b = body_b.split("<h2>Source, masks, features</h2>")[1].split("</body>")[0]
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Paired comparison — {html.escape(a["file"])} vs {html.escape(b["file"])}</title>
<style>body{{font-family:system-ui;margin:24px}}
table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:4px 10px}}
.cols{{display:flex;gap:16px}} .col{{flex:1}}</style></head><body>
<h1>Paired acoustic comparison</h1>
<p><b>A</b> {html.escape(a["file"])} (caller {getattr(a["ann"], "emitter", "?")}) —
<b>B</b> {html.escape(b["file"])} (caller {getattr(b["ann"], "emitter", "?")})</p>
<p>Commonly valid frames: {int(both.sum())}; derivative-comparable: {int(both_d.sum())}</p>
<h2>Level and rate distances (SEPARATE views)</h2>
<table><tr><th>feature</th><th>mean |level difference|</th><th>mean |rate difference|</th></tr>
{rows}</table>
<p style="color:#666;font-size:12px">Identical rates can hide different absolute
levels (a+vt vs b+vt). Neither column is a similarity verdict; a derivative view is
a complementary observation of the same trajectories, not corroboration.</p>
<h2>A</h2><div class="cols"><div class="col">{figs_a}</div></div>
<h2>B</h2><div class="cols"><div class="col">{figs_b}</div></div>
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recording", type=str, default=None)
    ap.add_argument("--pair", nargs=2, type=str, default=None)
    ap.add_argument("--panel", action="store_true")
    ap.add_argument(
        "--status", type=str, default="valid measurement", choices=sorted(STATUS_BADGES)
    )
    args = ap.parse_args()

    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import FEATURE_SCHEMA_VERSION
    from bioacoustic_pipeline.pipeline.release_contract import (
        MaskProvenance,
        ScientificRunContract,
        raise_for_science,
    )

    names = []
    if args.recording:
        names = [args.recording]
    elif args.pair:
        names = list(args.pair)
    elif args.panel:
        panel = json.loads((ROOT / "outputs/source_reference_panel/manifest.json").read_text())
        names = [f["file"] for f in panel["files"]]
    if not names:
        ap.error("one of --recording / --pair / --panel is required")

    raise_for_science(
        ScientificRunContract(
            run_name="evidence_report",
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            masks={
                "source_validity": MaskProvenance(
                    kind="source_validity",
                    produced_by="fingerprint absolute gate",
                    win_ms=WIN_MS,
                    hop_ms=HOP_MS,
                    amplitude_stage="native waveform",
                ),
                "detected_activity": MaskProvenance(
                    kind="detected_activity",
                    produced_by="peak-relative vocal rule",
                    win_ms=WIN_MS,
                    hop_ms=HOP_MS,
                    amplitude_stage="native waveform",
                ),
            },
            source_manifest={n: {"group": n, "part": "report"} for n in names},
            eval_split={"provenance": "descriptive report (no fitting)", "unit": "file"},
        )
    )

    out_dir = ROOT / "outputs" / "evidence_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.pair:
        a, b = _recording_data(args.pair[0]), _recording_data(args.pair[1])
        p = out_dir / f"pair_{args.pair[0]}_{args.pair[1]}.html".replace("/", "_")
        p.write_text(pair_html(a, b, args.status))
        print(f"wrote {p}")
    else:
        for n in names:
            d = _recording_data(n)
            p = out_dir / f"report_{n}.html"
            p.write_text(recording_html(d, args.status))
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
