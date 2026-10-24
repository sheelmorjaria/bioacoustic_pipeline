"""Frozen FIR response and documentation audit; no real waveform access."""

import json
from pathlib import Path
import numpy as np
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha, write
from bioacoustic_pipeline.diagnostics.rclat_filter_response import response_metrics
from bioacoustic_pipeline.representation.rclat_causal_resample import causal_decimate

ROOT = Path("outputs/rclat_rtpe_vctk_filter_mic_audit_v1")


def main():
    plan = json.loads((ROOT / "plan.json").read_text())
    assert all(sha(p) == h for p, h in plan["dependencies"].items())
    taps = causal_decimate(np.zeros(600), 48000)["taps"]
    r = response_metrics(taps)
    frequencies = np.array([0, 1000, 6000, 7200, 8000, 12000, 24000])
    direct = np.exp(-2j * np.pi * frequencies[:, None] * np.arange(97) / 48000) @ taps
    from scipy.signal import freqz

    _, reference = freqz(taps, worN=frequencies, fs=48000)
    error = float(np.max(np.abs(direct - reference)))
    assert error < 1e-12
    tone_records = []
    for f in [1000, 9000, 12000, 18000]:
        t = np.arange(48000)
        x = np.cos(2 * np.pi * f * t / 48000)
        out = causal_decimate(x, 48000)["values"]
        k = np.arange(len(out))[100:]
        values = out[100:]
        design = np.column_stack(
            [np.cos(2 * np.pi * f * k / 16000), np.sin(2 * np.pi * f * k / 16000)]
        )
        amplitude = float(np.linalg.norm(np.linalg.lstsq(design, values, rcond=None)[0]))
        expected = float(abs(np.sum(taps * np.exp(-2j * np.pi * f * np.arange(97) / 48000))))
        assert abs(amplitude - expected) < 1e-10
        tone_records.append(
            dict(
                input_frequency_hz=f,
                output_amplitude=amplitude,
                attenuation_dB=float(-20 * np.log10(max(amplitude, 1e-300))),
                reference_absolute_error=abs(amplitude - expected),
            )
        )
    metrics = {k: v for k, v in r.items() if k not in ["frequency_hz", "complex_response"]}
    audit = dict(
        status="FILTER_MINIMUM_REQUIREMENTS_FAILED_MIC2_MAPPING_DOCUMENTED",
        **metrics,
        independent_DFT_max_complex_error=error,
        exact_8000Hz_attenuation_dB=float(-20 * np.log10(abs(direct[4]))),
        synthetic_tones=tone_records,
        microphone=dict(
            suffix="mic2",
            physical_microphone="Sennheiser MKH800",
            mapping_source="https://www.tensorflow.org/datasets/catalog/vctk",
            source_type="Official TensorFlow Datasets implementation documentation;not a new publisher archive inspection",
            publisher_cross_check="Official VCTK README names same hardware and p280/p315 issues but does not itself map suffixes",
            fallback=False,
            actual_archive_file_mapping_verified=False,
        ),
        audio_acquired=False,
        real_audio_processed=False,
        model_inference=False,
        scientific_evaluation=False,
        filter_retuned=False,
    )
    np.savez_compressed(
        ROOT / "synthetic_frequency_response.npz",
        frequency_hz=r["frequency_hz"],
        complex_response=r["complex_response"],
        taps=taps,
        independent_frequencies=frequencies,
        independent_complex_response=direct,
    )
    write(ROOT / "audit.json", audit)
    report = f"""# VCTK filter/microphone audit — filter requirement FAILED

The existing 97-tap candidate was tested unchanged against a locally frozen engineering requirement: maximum absolute passband deviation <=0.1 dB over 0–6 kHz, and stopband attenuation >=60 dB over 8–24 kHz. These are audit-specific requirements, not a universal audio standard or an RTPE predictive gate. No thresholds were changed after measuring the candidate.

| Quantity | Observed | Requirement | Outcome |
|---|---:|---:|---|
| Maximum absolute passband deviation | {metrics["passband_max_abs_dB"]:.6f} dB | <=0.1 dB | PASS |
| Worst grid stopband suppression | {metrics["stopband_attenuation_dB"]:.6f} dB | >=60 dB | FAIL |
| Suppression at exactly 8 kHz | {audit["exact_8000Hz_attenuation_dB"]:.6f} dB | >=60 dB | FAIL |

The response grid contains 131,073 equally spaced frequencies including DC and 24 kHz. Direct complex DFT checks at seven fixed frequencies agreed with SciPy within {error:.3g}. Four synthetic steady-state tones agreed with independently computed FIR amplitudes. Frequency response arrays and taps are saved for recomputation. This is a sampled-grid numerical assessment, not a rigorous continuous-frequency bound or a claim about actual VCTK signal energy. The transition from 6–8 kHz is reported outside the passband; poor stopband suppression above the new Nyquist remains a preprocessing limitation. **No alias-adequacy certificate is issued.** The filter was not retuned or replaced.

## Microphone provenance and prospective policy

[Official TensorFlow Datasets documentation](https://www.tensorflow.org/datasets/catalog/vctk) explicitly maps mic1 to DPA 4035 and mic2 to Sennheiser MKH 800. It notes very-low-frequency noise for mic1 and recording issues for mic2 speakers p280/p315. This mapping is direct documentation from the dataset-tool maintainer, not inferred solely from missing speakers. The downloaded publisher README independently confirms hardware and the latter issues, but does not itself name the suffix mapping.

The locally frozen audit prospectively selects **mic2 only**, with **no mic1 fallback**. Actual channel/file mapping must still be cross-checked against archive metadata. Missing selected-channel files are documented source exclusions; no actual exclusion counts are claimed before acquisition. Missing transcripts alone do not exclude recordings. Both microphone versions of a spoken utterance remain one event. No quality or forecasting results informed microphone selection.

## Interpretation and next step

The filter result is an engineering requirement failure, not a negative transfer result. The original filter and horizon policies remain preserved. A separately specified corrective filter design is required if these anti-aliasing requirements are retained; any change must recheck startup support, retained-horizon disjointness and feature compatibility. Do not silently replace the frozen candidate or use real evaluation outcomes to select a filter. The 40 ms output remains unscored; changing filter support could also alter the retained 80/160 ms source gaps.

No VCTK audio was acquired, decoded or processed. No model was fitted or scored. Independent-source transfer remains unevaluated; RTPE-1/R1 positive results and all downstream blocks remain unchanged.
"""
    with (ROOT / "RESULTS.md").open("x") as f:
        f.write(report)
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
