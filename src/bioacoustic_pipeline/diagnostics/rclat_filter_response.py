"""Fixed-grid engineering response diagnostic; not a universal audio standard."""

import numpy as np
from scipy.signal import freqz


def response_metrics(taps, *, points=131073):
    taps = np.asarray(taps, dtype=float)
    if taps.ndim != 1 or not len(taps) or not np.isfinite(taps).all() or points < 3:
        raise ValueError("Finite FIR and supported response grid required")
    frequencies = np.linspace(0, 24000, points)
    _, response = freqz(taps, worN=frequencies, fs=48000)
    db = 20 * np.log10(np.maximum(np.abs(response), 1e-300))
    p = frequencies <= 6000
    s = frequencies >= 8000
    maximum = float(np.max(np.abs(db[p])))
    attenuation = float(-np.max(db[s]))
    return dict(
        frequency_hz=frequencies,
        complex_response=response,
        passband_max_abs_dB=maximum,
        passband_peak_to_peak_dB=float(np.ptp(db[p])),
        stopband_attenuation_dB=attenuation,
        worst_stopband_frequency_hz=float(frequencies[s][np.argmax(db[s])]),
        minimum_engineering_requirements_passed=bool(maximum <= 0.1 and attenuation >= 60),
    )
