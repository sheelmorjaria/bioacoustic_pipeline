"""Protected source reconstruction and frozen-state restoration;never fits."""

import hashlib
import io
import json
import zipfile
import numpy as np
import soundfile as sf
import torch
from ..representation.rclat_causal_resample_v2 import mel_features_v2
from ..representation.rclat_rtpe_guarded import GuardedPoolTemporalPredictionEngine
from ..representation.rclat_rtpe_weighted import WeightedHistoryRidge, WeightedTrainingScaler
from .rclat_rtpe_execution import sha


def build_source_arrays(archive, inventory, *, check=lambda: None, progress=lambda i, n: None):
    n = sum(len(row["selected_origins"]) for row in inventory)
    x = np.empty((n, 32, 64), np.float32)
    v = np.empty_like(x, dtype=bool)
    y = np.zeros((n, 3, 64), np.float32)
    m = np.zeros_like(y, dtype=bool)
    ids = []
    index = 0
    with zipfile.ZipFile(archive) as z:
        for number, row in enumerate(inventory, 1):
            check()
            payload = z.read(row["member"])
            if hashlib.sha256(payload).hexdigest() != row["source_bytes_sha256"]:
                raise ValueError("Selected source byte hash changed")
            info = sf.info(io.BytesIO(payload))
            audio, rate = sf.read(io.BytesIO(payload), dtype="float64")
            if (
                info.channels != 1
                or rate != 48000
                or audio.ndim != 1
                or len(audio) != row["frames"]
                or not np.isfinite(audio).all()
            ):
                raise ValueError("Selected source header/PCM changed")
            pcm = hashlib.sha256(
                json.dumps([rate, info.channels], separators=(",", ":")).encode()
                + np.asarray(audio, dtype="<f8").tobytes()
            ).hexdigest()
            if pcm != row["decoded_pcm_sha256"]:
                raise ValueError("Selected source PCM hash changed")
            f = mel_features_v2(audio, rate)
            if (
                hashlib.sha256(f["mel"].tobytes()).hexdigest() != row["mel_sha256"]
                or hashlib.sha256(f["mel_valid"].tobytes()).hexdigest() != row["valid_sha256"]
            ):
                raise ValueError("Selected source feature/mask hash changed")
            valid = f["mel_valid"].all(-1)
            supported = np.array(
                [
                    o
                    for o in range(32, max(32, len(valid) - 16))
                    if valid[o - 31 : o + 1].all() and valid[o + 8] and valid[o + 16]
                ],
                dtype="<i8",
            )
            if hashlib.sha256(supported.tobytes()).hexdigest() != row["eligible_origin_sha256"]:
                raise ValueError("Selected source origin support hash changed")
            allowed = set(supported.tolist())
            for origin in row["selected_origins"]:
                if (
                    origin not in allowed
                    or f["source_first_samples"][origin + 8] <= f["source_last_samples"][origin]
                    or f["source_first_samples"][origin + 16] <= f["source_last_samples"][origin]
                ):
                    raise ValueError("Selected origin has unsupported history/target support")
                x[index] = f["mel"][origin - 31 : origin + 1]
                v[index] = f["mel_valid"][origin - 31 : origin + 1]
                for column, h in [(1, 8), (2, 16)]:
                    y[index, column] = f["mel"][origin + h]
                    m[index, column] = f["mel_valid"][origin + h]
                ids.append(dict(speaker=row["speaker"], event=row["event"], origin=origin))
                index += 1
            progress(number, len(inventory))
    return dict(x=x, v=v, y=y, m=m), ids


def load_frozen(info, device="cpu"):
    for record in info["files"].values():
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("Frozen model artifact changed")
    scaler_state = torch.load(
        info["files"]["scaler"]["path"], map_location="cpu", weights_only=True
    )
    if (
        not all(t.shape == (64,) and torch.isfinite(t).all() for t in scaler_state.values())
        or not (scaler_state["scale"] > 0).all()
    ):
        raise ValueError("Frozen scaler invalid")
    scaler = WeightedTrainingScaler()
    scaler.mean = scaler_state["mean"]
    scaler.scale = scaler_state["scale"]
    ridge_state = torch.load(
        info["files"]["ridge_checkpoint"]["path"], map_location="cpu", weights_only=True
    )
    if tuple(ridge_state["history_shape"]) != (32, 64) or tuple(ridge_state["output_shape"]) != (
        3,
        64,
    ):
        raise ValueError("Frozen Ridge shape differs")
    ridge = WeightedHistoryRidge(ridge_state["alpha"])
    for key, value in ridge_state.items():
        setattr(ridge, key, value)
    checkpoint = torch.load(
        info["files"]["gru_checkpoint"]["path"], map_location="cpu", weights_only=True
    )
    if checkpoint["step"] != info["selected_update"]:
        raise ValueError("Selected checkpoint step changed")
    model = GuardedPoolTemporalPredictionEngine(
        64, hidden=64, scales=(1, 4, 16), horizons=(4, 8, 16)
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    for parameter in model.order.parameters():
        parameter.requires_grad_(False)
    return model, ridge, scaler, dict(scaler=scaler_state, ridge=ridge_state)


def equal_states(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(equal_states(a[k], b[k]) for k in a)
        )
    if isinstance(a, (tuple, list)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(equal_states(x, y) for x, y in zip(a, b))
        )
    return type(a) is type(b) and a == b
