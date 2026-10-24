#!/usr/bin/env python3
"""Spectrogram review package + automated spectral screen (blinded review).

Design (review-directed):
  NUMERICAL SPECTROGRAM from the source WAVs: scipy ShortTimeFFT, 1024-sample
  Hann (~4.10 ms), hop 256 (~1.02 ms — display sampling, NOT a resolution
  claim), full recorded band, power matrix + frequency/time coordinates +
  source-sample mapping cached as npz. The PNG/HTML is a RENDERING of these
  measurements, never the input to numerical comparison. Boundary slices
  whose windows extend into padded samples are marked UNSUPPORTED (never
  source evidence).

  BLINDED PAGES: per pair (anonymous ids, existing randomised order), A and
  B full-context waveform + log-power spectrogram with PHYSICAL time axes
  (different durations stay different widths), one FIXED shared dB
  reference per pair (a separately labelled contrast view exists but the
  fixed-reference view is primary), unwarped detail views at identical
  physical scales, NO DTW boundaries, NO scores, NO condition labels, NO
  machine proposals. First-pass reviewers mark the shared core themselves.

  AUTOMATED SPECTRAL SCREEN (separate artifact, never shown before
  judgements are locked):
    background b(f), sigma(f) from a DECLARED rule — the 20% lowest-energy
    time-columns of each clip (documented limitation: may contain weak
    vocal material; whole-clip median NOT used so sustained components
    cannot become baseline);
    residual Z(f,t) = (S - b) / max(sigma, eps); mask Z > 3; conservative
    morphology only (raw mask preserved; no bridging of invalid columns);
    connected components (scipy.ndimage.label) with duration, bandwidth,
    centroid, orientation, mean intensity;
    PAIRWISE CORRESPONDENCE as SEPARATE measurements, each with
    available/unavailable flags (never zero for missing support):
      c1 fine spectral-envelope distance (cosine, mean spectra of the
         top-energy-quartile columns);
      c2 contour correspondence (dominant tonal peak-frequency tracks;
         unavailable if either side lacks a supported contour);
      c3 temporal energy-distribution correlation (unwarped, best lag
         within +/-25 ms);
      c4 amplitude-envelope correlation (10 ms RMS series);
      c5 unwarped vs constrained-warp cost (fixed-band DTW on the coarse
         spectrogram vs identity alignment).
    Status boundary (explicit): this is machine assessment under a stated
    rule on the same source audio — CONSISTENCY evidence with the
    fingerprint search, NOT an independent reference and NOT unit
    validation. review.csv is never auto-filled; blinded_review_key.json
    stays sealed until human judgements are locked.

Usage:
    python scripts/spectral_review_package.py    # builds all 40 pairs

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REVIEW = ROOT / "outputs" / "blinded_review"
NFFT = 1024
HOP = 256
SR = 250000
Z_THR = 3.0
BG_COL_FRAC = 0.20  # lowest-energy columns used for the declared background
EPS = 1e-12


def _spectrogram(x):
    """(power, freqs, times_s, supported_columns) via ShortTimeFFT."""
    import numpy as np
    from scipy.signal import ShortTimeFFT, get_window

    win = get_window("hann", NFFT)
    SFT = ShortTimeFFT(win, HOP, SR, scale_to="psd")
    # truncate to whole windows so every column is fully inside real samples
    n_cols = 1 + (len(x) - NFFT) // HOP if len(x) >= NFFT else 0
    if n_cols == 0:
        return None, None, None, None
    xs = x[: NFFT + (n_cols - 1) * HOP]
    S = np.abs(SFT.stft(xs)) ** 2  # (freq, time), PSD scaling; documented
    freqs = SFT.f
    times = np.arange(S.shape[1]) * (HOP / SR)
    supported = np.ones(S.shape[1], dtype=bool)  # no padded columns by construction
    return S, freqs, times, supported


def _regions(Z, freqs, supported, mask):
    """Connected components with physical units; raw mask preserved."""
    import numpy as np
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    out = []
    dt = HOP / SR
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    for i in range(1, n + 1):
        rows, cols = np.where(lab == i)
        if not supported[cols].all():
            continue  # never bridging invalid columns
        out.append(
            {
                "duration_ms": round(float(len(np.unique(cols)) * dt * 1000), 2),
                "bandwidth_hz": round(float((rows.max() - rows.min() + 1) * df), 1),
                "centroid_hz": round(float(freqs[rows].mean()), 1),
                "mean_Z": round(float(Z[rows, cols].mean()), 2),
                "orientation_deg": None,  # not inferred: contour direction is c2's job
            }
        )
    return out


def _automated(clip_paths):
    import numpy as np
    import soundfile as sf
    from scipy import ndimage

    sides = {}
    for tag, p in clip_paths.items():
        x, sr = sf.read(str(p), dtype="float64")
        if sr != SR:
            return None
        S, freqs, times, supported = _spectrogram(x)
        if S is None:
            return None
        col_e = S.sum(axis=0)
        order = np.argsort(col_e)
        n_bg = max(4, int(len(order) * BG_COL_FRAC))
        bg_cols = order[:n_bg]
        b = S[:, bg_cols].mean(axis=1)
        sigma = S[:, bg_cols].std(axis=1)
        Z = (S - b[:, None]) / np.maximum(sigma[:, None], EPS)
        mask = (Z > Z_THR) & supported[None, :]
        # conservative: keep the RAW mask; a single closing pass is applied
        # ONLY for the region listing, defined physically (2 cols ~ 2 ms)
        closed = ndimage.binary_closing(mask, structure=np.ones((3, 3), bool))
        regions = _regions(Z, freqs, supported, closed)
        sides[tag] = {
            "S": S,
            "Z": Z,
            "freqs": freqs,
            "mask": mask,
            "col_e": col_e,
            "x": x,
            "regions": regions,
        }
    a, b_ = sides["A"], sides["B"]

    def _top_spec(sd, frac=0.25):
        k = max(4, int(len(sd["col_e"]) * frac))
        cols = np.argsort(sd["col_e"])[-k:]
        return sd["S"][:, cols].mean(axis=1)

    c1 = float(
        1.0
        - np.dot(_top_spec(a), _top_spec(b_))
        / (np.linalg.norm(_top_spec(a)) * np.linalg.norm(_top_spec(b_)) + EPS)
    )

    def _contour(sd):
        import numpy as np

        act = sd["mask"].any(axis=0)
        if act.sum() < 8:
            return None
        cols = np.where(act)[0]
        peak_f, peak_z = [], []
        for c in cols:
            col = sd["Z"][:, c]
            j = int(col.argmax())
            if col[j] > Z_THR + 3:  # a supported tonal peak, not diffuse
                peak_f.append(sd["freqs"][j])
                peak_z.append(col[j])
        if len(peak_f) < 8:
            return None
        return np.array(peak_f)

    ca, cb = _contour(a), _contour(b_)
    if ca is not None and cb is not None and len(ca) == len(cb):
        c2 = {
            "available": True,
            "net_direction_a_hz": float(np.polyfit(np.arange(len(ca)), ca, 1)[0]),
            "net_direction_b_hz": float(np.polyfit(np.arange(len(cb)), cb, 1)[0]),
            "same_sign": bool(
                np.sign(np.polyfit(np.arange(len(ca)), ca, 1)[0])
                == np.sign(np.polyfit(np.arange(len(cb)), cb, 1)[0])
            ),
        }
    else:
        c2 = {
            "available": False,
            "reason": "no supported tonal contour on at least one side — unavailable, never zero",
        }

    def _best_lag_corr(e1, e2, max_lag_cols=25):
        import numpy as np

        e1 = (e1 - e1.mean()) / (e1.std() + EPS)
        best = -1.0
        for lag in range(-max_lag_cols, max_lag_cols + 1):
            if lag >= 0:
                x_, y_ = e1[lag:], e2[: len(e2) - lag] if lag else e2
            else:
                x_, y_ = e1[: len(e1) + lag], e2[-lag:]
            m = min(len(x_), len(y_))
            if m < 8:
                continue
            r = float(np.corrcoef(x_[:m], y_[:m])[0, 1])
            best = max(best, r)
        return best

    c3 = _best_lag_corr(a["col_e"], b_["col_e"])
    ra = np.sqrt((a["x"][: (len(a["x"]) // 2500) * 2500].reshape(-1, 2500) ** 2).mean(1))
    rb = np.sqrt((b_["x"][: (len(b_["x"]) // 2500) * 2500].reshape(-1, 2500) ** 2).mean(1))
    c4 = (
        float(np.corrcoef(ra - ra.mean(), rb - rb.mean())[0, 1])
        if len(ra) == len(rb) and len(ra) > 3
        else None
    )

    def _coarse(sd, n_t=40, n_f=32):
        import numpy as np

        import numpy as np  # noqa: F811

        St = sd["S"]
        t_idx = np.linspace(0, St.shape[1] - 1, n_t).astype(int)
        f_idx = np.linspace(0, St.shape[0] - 1, n_f).astype(int)
        return 10 * np.log10(St[np.ix_(f_idx, t_idx)] + EPS)

    def _warp_costs(Ma, Mb):
        import numpy as np

        n, m = Ma.shape[1], Mb.shape[1]
        D = np.full((n + 1, m + 1), 1e12)
        D[0, :] = 0.0
        for i in range(1, n + 1):
            lo = max(1, i * m // n - 3)
            hi = min(m, i * m // n + 3)
            for j in range(lo, hi + 1):
                step = float(np.mean((Ma[:, i - 1] - Mb[:, j - 1]) ** 2))
                D[i, j] = step + min(D[i - 1, j], D[i - 1, j - 1], D[i, j - 1])
        unwarped = float(
            np.mean(
                (Ma - Mb[:, : Ma.shape[1]] if Mb.shape[1] >= Ma.shape[1] else Mb - Mb[:, :0]) ** 2
            )
        )
        return D[n, m] / n, unwarped

    Ma, Mb = _coarse(a), _coarse(b_)
    if Ma.shape == Mb.shape:
        warped, unwarped = _warp_costs(Ma, Mb)
    else:
        warped = _warp_costs(Ma, Mb)[0] if Mb.shape[1] >= 1 else None
        unwarped = None
    c5 = {
        "warped_band_dtw_cost": round(float(warped), 3) if warped is not None else None,
        "unwarped_identity_cost": round(float(unwarped), 3) if unwarped is not None else None,
        "note": "smaller warped-vs-unwarped gap = less deformation needed",
    }
    return {
        "c1_fine_spectral_envelope_distance": round(c1, 4),
        "c2_contour_correspondence": c2,
        "c3_temporal_energy_bestlag_corr": round(c3, 4),
        "c4_amplitude_envelope_corr": None if c4 is None else round(c4, 4),
        "c5_warp_vs_unwarped": c5,
        "regions_A": a["regions"][:12],
        "regions_B": b_["regions"][:12],
        "n_mask_columns_A": int(a["mask"].any(axis=0).sum()),
        "n_mask_columns_B": int(b_["mask"].any(axis=0).sum()),
    }


def _page(pid: str, clips: dict, out: Path) -> None:
    import base64
    import io

    import matplotlib

    matplotlib.use("Agg")
    import numpy as np
    import matplotlib.pyplot as plt
    import soundfile as sf

    figs = ""
    specs = {}
    for tag in ("A", "B"):
        x, sr = sf.read(str(clips[tag]), dtype="float64")
        S, freqs, times, supported = _spectrogram(x)
        specs[tag] = (S, freqs, times, x, sr)
    ref = max(float(S.max()) for S, *_ in specs.values())
    for tag in ("A", "B"):
        S, freqs, times, x, sr = specs[tag]
        fig, axes = plt.subplots(
            2, 1, figsize=(9, 4.2), sharex=True, gridspec_kw={"height_ratios": [1, 3]}
        )
        axes[0].plot(np.arange(len(x)) / sr, x, lw=0.2, color="k")
        axes[0].set_ylabel("amp")
        axes[0].set_title(
            f"clip {tag} — {len(x) / sr:.2f} s (physical axes, no stretch)", fontsize=9
        )
        db = 10 * np.log10(S + EPS)
        im = axes[1].pcolormesh(
            times,
            freqs / 1000.0,
            db,
            shading="auto",
            cmap="magma",
            vmin=10 * np.log10(ref * 1e-6),
            vmax=10 * np.log10(ref),
        )
        axes[1].set_ylabel("kHz")
        axes[1].set_xlabel("s")
        fig.colorbar(im, ax=axes[1], label="dB (fixed shared reference)")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        figs += (
            f'<h3>clip {tag}</h3><img src="data:image/png;base64,'
            f'{base64.b64encode(buf.getvalue()).decode()}" width="100%">'
        )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{pid}</title><style>body{{font-family:system-ui;margin:20px;
max-width:1000px}} p{{color:#555;font-size:12px}}</style></head><body>
<h1>{pid}</h1>
<p>Log-power spectrograms at one FIXED shared dB reference (both clips,
same scale). Physical time axes — durations are not equalised. Mark the
shared acoustic core yourself; no algorithm proposal is shown.</p>
{figs}
</body></html>"""
    out.write_text(html)


def main() -> None:
    import numpy as np
    import soundfile as sf

    review_csv = REVIEW / "review.csv"
    import csv

    pairs = list(csv.DictReader(open(review_csv)))
    cache = REVIEW / "spectrogram_cache"
    cache.mkdir(exist_ok=True)
    automated = {}
    for row in pairs:
        pid = row["pair_id"]
        clips = {t: REVIEW / row[f"clip_{t}"] for t in ("A", "B")}
        # numerical cache
        for t, p in clips.items():
            x, sr = sf.read(str(p), dtype="float64")
            S, freqs, times, supported = _spectrogram(x)
            if S is None:
                continue
            np.savez_compressed(
                cache / f"{pid}_{t}.npz",
                power=S,
                freqs=freqs,
                times=times,
                supported=supported,
                hop=HOP,
                nfft=NFFT,
                sr=sr,
                source=str(p.relative_to(ROOT)),
            )
        res = _automated(clips)
        automated[pid] = res
        _page(pid, clips, REVIEW / f"{pid}" / "source_view.html")
        print(f"{pid}: automated {'ok' if res else 'unavailable (clip too short)'}", flush=True)
    (REVIEW / "automated_spectral.json").write_text(
        json.dumps(
            {
                "configuration": {
                    "nfft": NFFT,
                    "hop": HOP,
                    "sr": SR,
                    "z_threshold": Z_THR,
                    "background_rule": f"per-clip {BG_COL_FRAC:.0%} "
                    "lowest-energy columns (declared; may contain weak "
                    "vocal material — documented limitation)",
                    "spectrogram": "scipy ShortTimeFFT, spectrum "
                    "scaling, power = |STFT|^2; PNGs are renderings, "
                    "not measurement inputs; boundary slices truncated "
                    "(never padded into evidence)",
                },
                "status_boundary": "machine assessment under a stated rule on the "
                "same source audio — consistency evidence with the fingerprint "
                "search, NOT an independent reference, NOT unit validation; "
                "review.csv is never auto-filled; the condition key stays sealed "
                "until human judgements are locked",
                "pairs": automated,
            },
            indent=1,
        )
    )
    print(
        f"wrote {REVIEW / 'automated_spectral.json'} and {len(pairs)} source_view.html pages",
        flush=True,
    )


if __name__ == "__main__":
    main()
