"""Time–frequency scaling and speech-like acoustic organisation (bounded study).

Hypothesis tier implemented here (explicitly NOT a decoder claim): a FIXED,
invertible playback-rate expansion g_alpha with y(t) = x(t/alpha), alpha > 1,
changes how recorded information is PRESENTED — duration x alpha, frequency
/ alpha, genuine gaps x alpha — and adds none. The study asks whether such a
constrained rescaling moves well-audited animal vocalisations closer to a
human-speech reference under a declared acoustic metric, MORE than it moves
matched controls (spectrum-preserving surrogates, unit-reordered surrogates,
spectrum-matched synthetic noise, speech-rate AM noise).

Endpoint (prespecified):
    d_H(x)   : standardized distance from descriptor Phi(x) to a speech
               reference distribution
    Delta(x) = d_H(x) - d_H(g_a*(x))            (a* frozen on development data)
    Q_class  = E[Delta(animal) - Delta(control_class)]
A positive held-out Q means the transformation increases animal-speech
similarity more than for that control. Speech-likeness under a metric is an
acoustic-organisation statement — never evidence of linguistic content,
meaning, or a "compressed message" (that tier requires context/receiver
prediction on unseen examples; Track C biology).

Firewall: Phi is a deterministic measurement computed identically on human
and animal material at a common 16 kHz analysis rate. No human-trained model
weights enter the animal analysis; no semantic verdict is produced. Human
speech audio is a REFERENCE DISTRIBUTION (data), not a transferred model.

Measurement-contract notes:
- The analysis rate (16 kHz) means ultrasonic source content is invisible to
  Phi until expansion divides it into the analysis band. That is the point of
  the manipulation, and the reason controls receive the identical transform.
- Modulation statistics are computed from multi-second envelopes (whole
  recordings), never from the 40 ms short-window modulation coordinate whose
  support limitation is documented in acoustic_derivatives.
- Global time scaling preserves event order and expands gaps proportionally;
  it cannot change token-transition structure. Any apparent organisational
  gain that comes only from denser analysis framing is a resolution effect,
  and per-source recording counts (not frame counts) ground the statistics.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np

ANALYSIS_SR = 16000
ALPHA_GRID = (1, 2, 4, 8, 16, 32)
MIN_ANALYSIS_SECONDS = 1.0

# canonical amplitude-modulation bands (Hz) — syllable-rate and neighbours
AM_BANDS_HZ = ((0.5, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, 32.0))

DESCRIPTOR_NAMES = (
    # log-band spectral envelope shape (12 log-spaced bands, 100 Hz - 7.5 kHz)
    *(f"band_{i:02d}" for i in range(12)),
    "spectral_slope",
    # amplitude-envelope modulation energy, normalised per recording
    *(f"am_{lo:g}_{hi:g}hz" for lo, hi in AM_BANDS_HZ),
    # |d(centroid)/dt| quantiles (gap-safe backward differences)
    "fm_q25",
    "fm_q50",
    "fm_q75",
    # harmonic proxies (deterministic)
    "spectral_flatness",
    "peak_prominence",
    # spectro-temporal organisation
    "temporal_continuity",
    "band_variance",
)


def scaling_study_spec() -> dict:
    """The prespecified study contract (immutable inputs for the driver)."""
    return {
        "study": "time-frequency scaling and speech-like acoustic organisation",
        "transformation": "playback-rate expansion y(t)=x(t/alpha), alpha>1 "
        "(fixed, invertible, gap-preserving; duration x alpha, freq / alpha)",
        "alpha_grid": list(ALPHA_GRID),
        "alpha_selection": "argmin mean d_H over DEVELOPMENT animal "
        "recordings; identical grid and budget applied to every control "
        "class; frozen before any EVAL row is scored",
        "descriptor": "25-D deterministic Phi at 16 kHz analysis rate: "
        "12 log-band envelope + slope, 5 AM bands from multi-second "
        "envelopes, 3 gap-safe FM quantiles, 2 harmonic proxies, "
        "temporal continuity + band variance",
        "distance": "mean squared standardized deviation from the speech "
        "reference (diagonal Mahalanobis, variance floored at 1e-3 x mean var)",
        "endpoint": "Delta(x) = d_H(x) - d_H(g_a*(x)); Q_class = mean Delta"
        "(animal) - mean Delta(control); file-clustered bootstrap + "
        "label-permutation p on EVAL",
        "controls": {
            "phase_surrogate": "within-recording, preserves exact spectral "
            "magnitudes, disrupts temporal structure and envelope phase",
            "unit_reorder": "shuffles decoded activity units (with their "
            "following gaps, 5 ms crossfades), preserves local acoustic "
            "material + unit multiset, disrupts order",
            "spec_noise": "white noise shaped to the recording's smoothed "
            "long-term spectrum, preserves coarse spectrum only",
            "am_noise": "spec_noise x 4 Hz sinusoidal AM, adds speech-rate "
            "modulation to a structureless carrier",
        },
        "positive_control": "human speech compressed by a* then expanded by "
        "a* must recover d_H (>=80% of eval clips with Delta >= 0)",
        "partitions": "alpha chosen on DEV animal recordings only; EVAL "
        "callers/files disjoint; speech reference speakers disjoint from "
        "speech-eval speakers; all transforms of one source stay in the "
        "source's partition",
        "decision_rule": "study-positive iff Q > 0 with bootstrap CI "
        "excluding 0 for EVERY control class on EVAL, positive control "
        "passes, AND speech-distance reduction exceeds noise-reference "
        "reduction (specificity)",
        "explicitly_excluded": [
            "ASR transcripts (hallucination-prone on nonspeech; not primary)",
            "neural speech synthesis / vocoders / unconstrained per-clip warp",
            "any semantic or 'compressed message' claim",
            "pitch-preserving stretch (separate later operation, own artefact controls)",
        ],
        "firewall": "deterministic measurements only; human speech is a "
        "reference distribution, not transferred weights; no semantic verdict",
        "gap_note": "human song/humming comparison class requires a source "
        "not present locally; deferred and must be documented in any report",
    }


# ------------------------------------------------------------ transformations


def expand_playback(audio, sr: int, alpha: float):
    """Playback-rate expansion y(t) = x(t/alpha): duration x alpha, freq/alpha.

    Polyphase resampling — deterministic and invertible in the information
    sense (no structure is synthesised). Genuine gaps expand proportionally
    because the whole observed timeline is scaled.
    """
    from fractions import Fraction  # noqa: PLC0415

    from scipy.signal import resample_poly  # noqa: PLC0415

    if alpha <= 0:
        raise ValueError("alpha must be positive")
    x = np.asarray(audio, dtype=np.float64)
    frac = Fraction(alpha).limit_denominator(1000)
    return resample_poly(x, frac.numerator, frac.denominator).astype(np.float32)


def compress_playback(audio, sr: int, alpha: float):
    """Playback-rate compression y(t) = x(alpha*t) (positive control only)."""
    from fractions import Fraction  # noqa: PLC0415

    from scipy.signal import resample_poly  # noqa: PLC0415

    if alpha <= 0:
        raise ValueError("alpha must be positive")
    x = np.asarray(audio, dtype=np.float64)
    frac = Fraction(alpha).limit_denominator(1000)
    return resample_poly(x, frac.denominator, frac.numerator).astype(np.float32)


def to_analysis_rate(audio, sr: int, target: int = ANALYSIS_SR):
    """Anti-aliased resample to the common analysis rate."""
    from math import gcd  # noqa: PLC0415

    from scipy.signal import resample_poly  # noqa: PLC0415

    x = np.asarray(audio, dtype=np.float64)
    g = gcd(int(sr), int(target))
    return resample_poly(x, target // g, sr // g).astype(np.float32)


# -------------------------------------------------------------- descriptor Phi


def _log_bands(sr: int, n_bands: int = 12, f_lo: float = 100.0, f_hi: float = 7500.0):
    """Log-spaced band edges mapped to spectrum bin index ranges."""
    edges = np.geomspace(f_lo, min(f_hi, sr / 2 - 1), n_bands + 1)
    return edges


def long_window_descriptor(audio, sr: int = ANALYSIS_SR) -> dict:
    """25-D deterministic descriptor Phi on (multi-second) audio.

    Returns {"vector": (25,), "names": tuple, "n_valid_frames": int,
    "duration_s": float}. Frames: 25 ms windows, 10 ms hop.
    """
    x = np.asarray(audio, dtype=np.float64)
    if sr != ANALYSIS_SR:
        raise ValueError(f"descriptor requires the {ANALYSIS_SR} Hz analysis rate")
    win, hop = int(sr * 0.025), int(sr * 0.010)
    if len(x) < win:
        return {
            "vector": np.full(len(DESCRIPTOR_NAMES), np.nan),
            "names": DESCRIPTOR_NAMES,
            "n_valid_frames": 0,
            "duration_s": len(x) / sr,
        }

    n_frames = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(win)[None, :]

    # --- spectrum per frame; 12 log bands
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    edges = _log_bands(sr)
    band_pow = np.stack(
        [
            spec[:, (freqs >= edges[i]) & (freqs < edges[i + 1])].sum(axis=1)
            for i in range(len(edges) - 1)
        ],
        axis=1,
    )
    total = band_pow.sum(axis=1, keepdims=True) + 1e-30
    band_db = 10.0 * np.log10(band_pow / total + 1e-30)
    band_shape = band_db - band_db.mean(axis=1, keepdims=True)  # level-invariant

    # --- frame RMS (validity + amplitude envelope)
    rms = np.sqrt((x[idx] ** 2).mean(axis=1))
    rmax = float(rms.max()) + 1e-30
    valid = rms > 1e-4 * rmax

    # --- FM: gap-safe |d centroid/dt| quantiles on valid frames
    bf = (edges[:-1] + edges[1:]) / 2.0
    centroid = (band_pow * bf[None, :]).sum(axis=1) / (band_pow.sum(axis=1) + 1e-30)
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (  # noqa: PLC0415
        first_derivative,
    )

    dc = first_derivative(centroid, valid, dt=hop / sr)
    rate = np.abs(dc[np.isfinite(dc)])
    if rate.size >= 5:
        fm = np.percentile(rate, [25, 50, 75])
    else:
        fm = np.full(3, np.nan)

    # --- AM spectrum of the envelope (multi-second support)
    env = rms - rms.mean()
    n_fft = 1 << max(9, int(np.ceil(np.log2(max(len(env), 16)))))
    amp = np.abs(np.fft.rfft(env * np.hanning(len(env)), n=n_fft))
    mfreqs = np.fft.rfftfreq(n_fft, hop / sr)
    am_bands = []
    for lo, hi in AM_BANDS_HZ:
        m = (mfreqs >= lo) & (mfreqs < hi)
        am_bands.append(float(amp[m].sum()) if m.any() else 0.0)
    am_total = sum(am_bands) + 1e-30
    am = np.array(am_bands) / am_total

    # --- harmonic proxies
    spec_flat = np.exp(np.mean(np.log(band_pow + 1e-30), axis=1)) / (band_pow.mean(axis=1) + 1e-30)
    flatness = float(spec_flat[valid].mean()) if valid.any() else np.nan
    prominence = (
        float((band_shape[valid].max(axis=1) - np.median(band_shape[valid], axis=1)).mean())
        if valid.any()
        else np.nan
    )

    # --- spectro-temporal organisation
    if valid.sum() >= 3:
        b = band_shape[valid]
        norms = np.linalg.norm(b, axis=1) + 1e-30
        corr = (b[:-1] * b[1:]).sum(axis=1) / (norms[:-1] * norms[1:])
        continuity = float(np.clip(corr, -1, 1).mean())
        band_var = float(b.std(axis=0).mean())
    else:
        continuity = band_var = np.nan

    slope = (
        float(np.polyfit(np.arange(12), band_shape[valid].mean(axis=0), 1)[0])
        if valid.any()
        else np.nan
    )

    vec = np.concatenate(
        [
            band_shape[valid].mean(axis=0) if valid.any() else np.full(12, np.nan),
            [slope],
            am,
            fm,
            [flatness, prominence],
            [continuity, band_var],
        ]
    )
    return {
        "vector": vec,
        "names": DESCRIPTOR_NAMES,
        "n_valid_frames": int(valid.sum()),
        "duration_s": len(x) / sr,
    }


def build_reference(vectors) -> dict:
    """Reference distribution: per-feature mean/var (var floored)."""
    v = np.asarray(vectors, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] < 5:
        raise ValueError("reference needs >= 5 descriptor rows")
    keep = np.isfinite(v).all(axis=1)
    v = v[keep]
    mean = v.mean(axis=0)
    var = v.var(axis=0)
    var = np.maximum(var, 1e-3 * max(var.mean(), 1e-12))
    return {"names": DESCRIPTOR_NAMES, "mean": mean, "var": var, "n": int(len(v))}


def standardized_distance(vector, reference: dict) -> float:
    """Mean squared standardized deviation from the reference distribution."""
    v = np.asarray(vector, dtype=np.float64)
    if not np.isfinite(v).all():
        return float("nan")
    z = (v - reference["mean"]) / np.sqrt(reference["var"])
    return float((z**2).mean())


# ---------------------------------------------------------------- surrogates


def phase_randomized_surrogate(audio, seed: int):
    """Within-recording surrogate: exact spectral magnitudes, random phase.

    Preserves the power spectrum; disrupts temporal structure, envelope
    organisation and phase-sensitive harmonic relations.
    """
    x = np.asarray(audio, dtype=np.float64)
    rng = np.random.RandomState(seed)
    spec = np.fft.rfft(x)
    ph = rng.uniform(0, 2 * np.pi, size=spec.shape)
    # magnitudes are preserved EXACTLY; only the phase is randomised (no
    # rescale — any rescale would break the preserved-spectrum contract)
    out = np.fft.irfft(np.abs(spec) * np.exp(1j * ph), n=len(x))
    return out


def spec_matched_noise(audio, seed: int, sr: int, smoothing_octaves: float = 1.0 / 3):
    """White noise shaped to the recording's SMOOTHED long-term spectrum.

    Distinct from the phase surrogate: fine spectral structure is smoothed
    away, so only the coarse spectral envelope is preserved.
    """
    x = np.asarray(audio, dtype=np.float64)
    n_fft = 1 << max(10, int(np.ceil(np.log2(len(x)))))
    mag = np.abs(np.fft.rfft(x, n=n_fft))
    # 1/N-octave smoothing of the magnitude envelope — cumsum moving average
    # (the kernel can be tens of thousands of taps; direct convolution is
    # quadratic in practice)
    n_pts = min(len(mag), max(8, int(len(mag) * np.log(2) / max(smoothing_octaves, 1e-3) / 12)))
    cs = np.cumsum(np.concatenate([[0.0], mag]))
    lo = np.maximum(np.arange(len(mag)) - n_pts // 2, 0)
    hi = np.minimum(np.arange(len(mag)) + (n_pts + 1) // 2, len(mag))
    sm = (cs[hi] - cs[lo]) / (hi - lo) + 1e-12
    rng = np.random.RandomState(seed)
    ph = rng.uniform(0, 2 * np.pi, size=sm.shape)
    out = np.fft.irfft(sm * np.exp(1j * ph), n=n_fft)[: len(x)]
    scale = np.sqrt((x**2).mean() + 1e-30) / np.sqrt((out**2).mean() + 1e-30)
    return (out * scale).astype(np.float32)


def am_modulated(audio_like_noise, sr: int, rate_hz: float = 4.0, depth: float = 0.5):
    """Apply sinusoidal AM (speech-rate) to a carrier (the spec-noise control)."""
    x = np.asarray(audio_like_noise, dtype=np.float64)
    t = np.arange(len(x)) / sr
    return (x * (1.0 + depth * np.sin(2 * np.pi * rate_hz * t))).astype(np.float32)


def unit_reorder_surrogate(audio, sr: int, seed: int = 0):
    """Shuffle decoded activity units (each glued to its following gap).

    Preserves: unit waveforms, unit multiset, total gap budget, duration.
    Disrupts: order. Joins use 5 ms raised-cosine crossfades so shuffling
    clicks cannot identify the control.
    """
    from bioacoustic_pipeline.segmentation.silence_units import (  # noqa: PLC0415
        HysteresisConfig,
        extract_activity_units,
    )

    x = np.asarray(audio, dtype=np.float64)
    cfg = HysteresisConfig(frame_ms=10.0)
    res = extract_activity_units(x, sr=sr, config=cfg, ref_amplitude=float(np.abs(x).max()))
    spans = res.spans
    if len(spans) < 3:
        return None  # too few units to reorder: control unavailable, never faked
    # chunk = [unit_start, next_unit_start) so gaps travel with their unit;
    # the leading pre-first-unit segment is its own chunk so the total
    # duration is preserved EXACTLY (leading silence is observed timeline)
    bounds = [0] + [s.start_sample for s in spans] + [len(x)]
    chunks = [x[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(chunks))
    xf = int(sr * 0.005)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, xf)) if xf > 1 else None

    def _edge_fades(c):
        # 5 ms cosine edge-fades: length-preserving click prevention (a true
        # crossfade would shorten the signal by one fade per join)
        if ramp is None or len(c) < 2 * xf:
            return c
        e = c.copy()
        e[:xf] *= ramp
        e[-xf:] *= ramp[::-1]
        return e

    y = np.concatenate([_edge_fades(chunks[i]) for i in order])
    assert len(y) == len(x), "unit reorder must preserve total duration exactly"
    return y


MAX_ANALYSIS_SECONDS = 20.0


def transform_and_describe(audio, sr: int, alpha: float, *, pre_render=None):
    """g_alpha -> analysis rate -> Phi. ``pre_render`` applies an identical
    rendering hook (surrogate construction) BEFORE the shared transform.

    The descriptor uses a deterministic 20 s PREFIX of the transformed signal
    (every class and control treated identically), so deeply expanded
    recordings stay bounded; statistics remain grounded per source recording.
    """
    x = np.asarray(audio, dtype=np.float64)
    if pre_render is not None:
        x = pre_render(x, sr)
        if x is None:
            return None
    y = expand_playback(x, sr, alpha) if alpha != 1 else x.astype(np.float32)
    z = to_analysis_rate(y, sr) if sr != ANALYSIS_SR else y
    if len(z) < int(MIN_ANALYSIS_SECONDS * ANALYSIS_SR):
        return None  # below the modulation-support minimum: unavailable
    z = z[: int(MAX_ANALYSIS_SECONDS * ANALYSIS_SR)]
    return long_window_descriptor(z)
