"""Human-perceptual layer of the fingerprint (Mel-MFCC) — opt-in, NOT cross-species.

The Mel scale is a *human-auditory* frequency warp. It is appropriate for human speech
and roughly for audio within the human hearing band, but it is **anthropocentric**:
on ultrasonic taxa (e.g. bats at 250 kHz) it compresses exactly the high frequencies
that matter most. For that reason Mel-MFCC is kept OUT of the default cross-species
``combined`` view and is computed only on request. The species-neutral core
(``physics`` + ``macrotexture`` + ``microtexture``, all *linear*) is the fingerprint's
default and the one registered in the embeddings bank.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np

from . import _dsp


def perceptual(audio, sr, *, n_mfcc: int = 13) -> np.ndarray:
    """Mel-frequency cepstral coefficients (mean over frames). Human-perceptual."""
    return _dsp.mfcc(audio, sr, n_mfcc=n_mfcc)
