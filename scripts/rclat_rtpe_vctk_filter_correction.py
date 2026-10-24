"""Single frozen FIR correction audit using synthetic data only."""

import json
from pathlib import Path
import numpy as np
from scipy.signal import freqz
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha, write
from bioacoustic_pipeline.diagnostics.rclat_filter_response import response_metrics
from bioacoustic_pipeline.representation.rclat_causal_resample_v2 import (
    causal_decimate_v2,
    mel_features_v2,
    target_separation_v2,
)
from bioacoustic_pipeline.representation.rclat_ctc import acoustic_features

ROOT = Path("outputs/rclat_rtpe_vctk_filter_correction_v2")


def main():
    plan = json.loads((ROOT / "plan.json").read_text())
    assert all(sha(p) == h for p, h in plan["dependencies"].items())
    taps = causal_decimate_v2(np.zeros(600), 48000)["taps"]
    response = response_metrics(taps)
    f = np.array(plan["independent_frequencies_hz"])
    reference = np.exp(-2j * np.pi * f[:, None] * np.arange(193) / 48000) @ taps
    _, actual = freqz(taps, worN=f, fs=48000)
    dft_error = float(np.max(np.abs(actual - reference)))
    assert dft_error < 1e-12
    tones = []
    for frequency in plan["tone_frequencies_hz"]:
        source = np.cos(2 * np.pi * frequency * np.arange(48000) / 48000)
        values = causal_decimate_v2(source, 48000)["values"][200:]
        k = np.arange(16000)[200:]
        cosine = np.cos(2 * np.pi * frequency * k / 16000)
        if frequency == 8000:
            amplitude = abs(float(np.mean(values * cosine)))
        else:
            sine = np.sin(2 * np.pi * frequency * k / 16000)
            amplitude = float(
                np.linalg.norm(
                    np.linalg.lstsq(np.column_stack([cosine, sine]), values, rcond=None)[0]
                )
            )
        expected = float(
            abs(np.sum(taps * np.exp(-2j * np.pi * frequency * np.arange(193) / 48000)))
        )
        assert abs(amplitude - expected) < 1e-10
        tones.append(
            dict(
                frequency_hz=frequency,
                amplitude=amplitude,
                expected_amplitude=expected,
                error=abs(amplitude - expected),
            )
        )
    x = np.random.default_rng(19).normal(size=48000)
    changed = x.copy()
    changed[30001:] += 10
    a = causal_decimate_v2(x, 48000)
    b = causal_decimate_v2(changed, 48000)
    assert np.array_equal(a["values"][:10001], b["values"][:10001])
    features = mel_features_v2(x, 48000)
    later = mel_features_v2(changed, 48000)
    direct = acoustic_features(a["values"], 16000)
    assert np.array_equal(features["mel"], direct["mel"])
    prefix = features["source_last_samples"] <= 30000
    assert np.array_equal(features["mel"][prefix], later["mel"][prefix])
    assert not a["valid"][:64].any() and a["valid"][64:].all()
    assert not features["mel_valid"][0].any() and features["mel_valid"][1:].all()
    gaps = [target_separation_v2(32, h) for h in [4, 8, 16]]
    assert not gaps[0]["disjoint"] and all(g["disjoint"] for g in gaps[1:])
    checkpoint_policy = json.loads(
        Path("outputs/rclat_rtpe_vctk_horizon_policy_v1/policy.json").read_text()
    )
    checked = 0
    for model in checkpoint_policy["models"]:
        for record in model["files"].values():
            assert sha(record["path"]) == record["sha256"]
            checked += 1
    metrics = {k: v for k, v in response.items() if k not in ["frequency_hz", "complex_response"]}
    passed = metrics["minimum_engineering_requirements_passed"]
    audit = dict(
        status="CORRECTIVE_FILTER_BOUNDED_ENGINEERING_PASSED"
        if passed
        else "CORRECTIVE_FILTER_ENGINEERING_FAILED",
        **metrics,
        direct_DFT_max_error=dft_error,
        synthetic_tones=tones,
        source_support=gaps,
        source_history_span_samples=192,
        source_history_span_ms=4.0,
        group_delay_source_samples=96,
        group_delay_ms=2.0,
        unsupported_initial_decimated_samples=64,
        earliest_eligible_origin=32,
        prefix_invariance=True,
        synthetic_mel_prefix_invariance=True,
        original_mel_values_bit_identical=True,
        synthetic_feature_shape=list(features["mel"].shape),
        checkpoint_files_verified=checked,
        real_source_support_certified=False,
        real_feature_distribution_compatibility_certified=False,
        microphone_policy="Previously frozen mic2 only,no fallback",
        audio_acquired=False,
        model_inference=False,
        scientific_scoring=False,
    )
    write(ROOT / "audit.json", audit)
    np.savez_compressed(
        ROOT / "response_and_support.npz",
        taps=taps,
        frequency_hz=response["frequency_hz"],
        complex_response=response["complex_response"],
        independent_frequency_hz=f,
        independent_complex_response=reference,
        synthetic_mel=features["mel"],
        synthetic_valid=features["mel_valid"],
        source_first=features["source_first_samples"],
        source_last=features["source_last_samples"],
    )
    report = f"""# VCTK corrective FIR v2 — {"PASSED bounded engineering checks" if passed else "FAILED engineering requirement"}

A single separately frozen candidate used 193 taps, cutoff 7 kHz, Kaiser beta 8.6 and causal phase-zero decimation from 48 to 16 kHz. No delay compensation, filter search or post-result retuning occurred. The failed 97-tap candidate is preserved.

| Frequency-response metric | Result | Frozen requirement |
|---|---:|---:|
| Maximum absolute deviation, 0–6 kHz | {metrics["passband_max_abs_dB"]:.6f} dB | <=0.1 dB |
| Worst grid suppression, 8–24 kHz | {metrics["stopband_attenuation_dB"]:.6f} dB | >=60 dB |

The 131,073-point response grid and direct complex DFT at nine fixed frequencies agreed within {dft_error:.3g}. Five synthetic steady-state tone amplitudes agreed within 1e-10. These are sampled-grid engineering findings, not rigorous continuous-frequency bounds or a universal audio quality certificate.

## Renewed source support and frontend contract

Each output k conservatively consumes source samples [3k-192,3k]. Group delay is 96 source samples (2 ms); full historical support spans 192 samples (4 ms). The first 64 decimated outputs are unsupported. The first mel frame is explicitly masked unsupported; complete 32-frame histories still require origin >=32. Target bounds remain relative to the distributed 48 kHz waveform, not original live acquisition.

| Model horizon | Source gap | Disjoint | Scoring policy |
|---|---:|---|---|
"""
    for g in gaps:
        report += f"| {g['horizon_frames'] * 10} ms | {g['gap_source_samples']} samples / {g['gap_ms']:.6f} ms | {'Yes' if g['disjoint'] else 'No'} | {'Unscored' if g['horizon_frames'] == 4 else 'Primary' if g['horizon_frames'] == 8 else 'Secondary'} |\n"
    report += """
Synthetic future perturbations preserved the resampled and mel prefixes. The new wrapper calls the unchanged original acoustic_features routine and returns bit-identical 64D mel values for that resampled waveform, while adding explicit source support and startup masks. This is format/arithmetic compatibility, **not** equality to native LibriSpeech features or proof of real VCTK distribution compatibility.

All 15 bound GRU/Ridge/scaler files were hash-verified unchanged. No checkpoint parameters were loaded for inference. Mic2-only/no-fallback policy remains unchanged, with actual archive mapping still unverified. The original 97-tap policy is not overwritten: a future scientific protocol must explicitly bind this v2 module, coefficients and updated timing instead of silently substituting them.

This bounded engineering audit does not certify real VCTK waveforms, complete corpus eligibility, global speaker identity or forecasting transfer. No VCTK audio was acquired, decoded or processed; no model inference, adaptation or scientific scoring occurred. RTPE-1/R1 positive results are preserved. P100/P960 and MACDE–CAVEM Stage 2 remain blocked.
"""
    with (ROOT / "RESULTS.md").open("x") as out:
        out.write(report)
    write(ROOT / "manifest.json", {str(p): sha(p) for p in ROOT.iterdir() if p.is_file()})
    write(
        ROOT / "report_manifest.json",
        {
            str(Path(__file__)): sha(__file__),
            str(ROOT / "manifest.json"): sha(ROOT / "manifest.json"),
        },
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
