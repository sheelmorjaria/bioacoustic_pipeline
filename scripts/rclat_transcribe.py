#!/usr/bin/env python3
"""Transcribe native mono 16 kHz audio with a trained Study B checkpoint."""

import argparse

import numpy as np
import soundfile as sf
import torch

from bioacoustic_pipeline.diagnostics.frozen_acoustic_transfer import frozen_scale
from bioacoustic_pipeline.representation.rclat_ctc import (
    CausalCTC,
    LABELS,
    acoustic_features,
    decode_logits,
)
from bioacoustic_pipeline.utils.devices import resolve_device


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("audio")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved["labels"] != list(LABELS):
        raise ValueError("Checkpoint alphabet mismatch")
    audio, sr = sf.read(args.audio)
    features = acoustic_features(audio, sr)
    arm = saved["arm"]
    if arm == "B3":
        raise ValueError("B3 requires its separate frozen offline pretrained feature extractor")
    if arm == "B2":
        x = np.concatenate([features["spectral"], features["physical"]], axis=1)
        mask = np.concatenate([features["spectral_valid"], features["physical_valid"]], axis=1)
    else:
        key = {"B0": "mel", "B1": "physical"}[arm]
        x, mask = features[key], features[key + "_valid"]
    x, mask = frozen_scale(x, mask, saved["scaler"])
    device = resolve_device(args.device)
    model = CausalCTC(saved["model_arm"], saved["input_dim"], saved["width"])
    model.load_state_dict(saved["state_dict"])
    model.eval().to(device)
    with torch.inference_mode():
        logits, _ = model(
            torch.from_numpy(x)[None].to(device), torch.from_numpy(mask)[None].to(device)
        )
    print(decode_logits(logits[0], len(x)))


if __name__ == "__main__":
    main()
