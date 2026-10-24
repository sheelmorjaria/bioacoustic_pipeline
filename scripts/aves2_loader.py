#!/usr/bin/env python3
"""AVES v2 (esp-aves2-eat-bio) loader — HF checkpoint integration layer.

INTEGRATION NOTES (2026-10-20, environment-specific):
  - avex 1.4.0 has no `build_model_from_config`; the esp-aves2-eat-bio repo
    ships raw safetensors (ImageBind-style key layout) with no config.json.
  - The architecture is built from the worstchan/EAT-base HF remote code
    (Data2Vec-multi style, trust_remote_code); the checkpoint is then
    remapped onto it:
        blocks.*                                    -> model.blocks.*
        modality_encoders.IMAGE.local_encoder.*     -> model.local_encoder.*
        modality_encoders.IMAGE.extra_tokens        -> model.extra_tokens
        modality_encoders.IMAGE.fixed_positional_encoder.*
                                                    -> model.fixed_positional_encoder.*
        modality_encoders.IMAGE.context_encoder.norm.*
                                                    -> model.pre_norm.*
    (decoder.* tensors are unused — encoder-only use).
  - torchvision 0.29.1's compiled ops do not register against the Intel
    xpu torch, so remote code that registers fake torchvision ops fails;
    the missing op schemas are shimmed (never called for audio).
  - Extraction policy (advisor's padding caveat): waveform -> EAT mel
    fbank (16 kHz, 128 mels, target_length frames, checkpoint's own
    norm -7.343/4.41212) -> patch embeddings (each ~20 ms of audio).

VALIDATION GATE (--validate): encodes real annotated Bengalese syllables
and requires mean within-class cosine > between-class cosine for the
top classes — the loader must demonstrate non-trivial acoustic structure
before any downstream probe uses it.

Usage:
    python scripts/aves2_loader.py --validate [--device xpu:1]

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CKPT = (
    "/home/sheel/.cache/huggingface/hub/models--EarthSpeciesProject--esp-aves2-eat-bio/"
    "snapshots/2eb5ad51de455449800e2486fa91f8dbdf5bc499/esp-aves2-eat-bino.safetensors"
)
CKPT = CKPT.replace("eat-bino", "eat-bio")
ARCH = "worstchan/EAT-base_epoch30_pretrain"
NORM_MEAN, NORM_STD = -7.343, 4.41212  # from the checkpoint's train_config.yaml

_SCHEMAS = {
    "torchvision::nms": "(Tensor dets, Tensor scores, float iou_threshold) -> Tensor",
    "torchvision::qnms": "(Tensor dets, Tensor scores, float iou_threshold) -> Tensor",
    "torchvision::roi_align": "(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio, bool aligned) -> Tensor",
    "torchvision::roi_pool": "(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width) -> Tensor",
    "torchvision::_new_empty_tensor_op": "(Tensor self, int[] size) -> Tensor",
}

_MODEL = None


def _shim_imports():
    """Import avex/torchvision, defining missing op schemas as needed."""
    import torch

    for _ in range(12):
        try:
            import avex.models.eat_hf  # noqa: F401
            import torchvision  # noqa: F401

            return
        except RuntimeError as e:
            m = re.search(r"operator ([\w:]+) does not exist", str(e))
            if not m:
                raise
            torch.library.define(
                m.group(1), _SCHEMAS.get(m.group(1), "(Tensor[] tensors) -> Tensor[]")
            )


def load_aves2(device="xpu:1"):
    """Build the architecture, remap and load the esp-aves2 weights, eval mode."""
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    global _MODEL
    if _MODEL is not None:
        return _MODEL
    _shim_imports()
    cfg = AutoConfig.from_pretrained(ARCH, trust_remote_code=True)
    cls = get_class_from_dynamic_module(cfg.auto_map["AutoModel"], ARCH)
    net = cls(cfg)
    sd = load_file(CKPT)
    remapped = {f"model.{k}": v for k, v in sd.items() if k.startswith("blocks.")}
    for k, v in sd.items():
        if k.startswith("modality_encoders.IMAGE."):
            nk = k.replace("modality_encoders.IMAGE.", "")
            if nk.startswith("decoder."):
                continue  # encoder-only use
            if nk.startswith("context_encoder.norm."):
                nk = nk.replace("context_encoder.norm.", "pre_norm.")
            remapped[f"model.{nk}"] = v
    missing, unexpected = net.load_state_dict(remapped, strict=False)
    real_missing = [k for k in missing if "decoder" not in k]
    if real_missing:
        raise RuntimeError(f"AVES2 load incomplete; missing {real_missing[:5]}")
    net = net.to(device).eval()
    _MODEL = (net, device)
    return _MODEL


def embed_audio(audio_16k, target_frames=1024):
    """Waveform (float, 16 kHz, seconds) -> patch embeddings (n_patches, 768).

    Padding policy DECLARED: the fbank is zero-padded/truncated to
    target_frames — embeddings of padded regions are returned as-is and
    it is the caller's job to mask by real duration (extraction-stage
    validity contract per the advisor's AVES caveat).
    """
    import numpy as np
    import torch

    from avex.models.eat_hf import EATAudioProcessor

    net, device = load_aves2()
    proc = EATAudioProcessor(
        sample_rate=16000,
        target_length=target_frames,
        n_mels=128,
        norm_mean=NORM_MEAN,
        norm_std=NORM_STD,
    )
    a = np.asarray(audio_16k, dtype=np.float32)
    feats = np.asarray(proc(a))
    f = torch.as_tensor(feats, dtype=torch.float32)
    if f.dim() == 2:  # (T, mels) -> (1, 1, T, mels)
        f = f.unsqueeze(0).unsqueeze(0)
    elif f.dim() == 3:  # (B, T, mels) -> (B, 1, T, mels)
        f = f.unsqueeze(1)
    f = f.to(device)
    with torch.no_grad():
        out = net(f)
    o = out[0] if isinstance(out, (tuple, list)) else out
    return o.squeeze(0).cpu().numpy()


def _validate():
    """Gate: real Bengalese syllables must cluster better than chance."""
    import numpy as np

    from bengalese_natural_benchmark import DATA, parse_annotation, _load_wav

    np.random.RandomState(0)
    results = {}
    for bird in (0,):
        bdir = DATA / f"Bird{bird}"
        bouts = parse_annotation(bdir / "Annotation.xml")[:20]
        # gather syllable audio by class from the annotation (samples)
        by_class: dict = {}
        for wav_name, notes in bouts:
            audio = _load_wav(bdir / "Wave" / wav_name)
            if audio is None:
                continue
            for abs_start, length, label in notes:
                by_class.setdefault(label, []).append(audio[abs_start : abs_start + length])
        top = sorted(by_class, key=lambda k: -len(by_class[k]))[:4]

        # resample 32k -> 16k (AVES input rate): linear interpolation
        def _to16k(a):
            if len(a) < 64:
                return None
            x = np.linspace(0, 1, len(a))
            xi = np.linspace(0, 1, max(len(a) // 2, 1))
            return np.interp(xi, x, a).astype(np.float32)

        embs, labels = [], []
        for lab in top:
            for a in by_class[lab][:6]:
                a16 = _to16k(a)
                if a16 is None or len(a16) < 160:
                    continue
                e = embed_audio(a16, target_frames=256)
                embs.append(e.mean(axis=0))
                labels.append(lab)
        E = np.stack(embs)
        E = E - E.mean(axis=0, keepdims=True)  # CENTERED: remove the
        # dominant mean direction — raw cosines are ~0.998 for everything
        # and a pass on them is not evidence of class structure
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        labels = np.array(labels)
        cos = E @ E.T
        same, diff = [], []
        for i in range(len(E)):
            for j in range(i + 1, len(E)):
                (same if labels[i] == labels[j] else diff).append(cos[i, j])
        w, b = float(np.mean(same)), float(np.mean(diff))
        results[f"Bird{bird}"] = {
            "n_segments": len(E),
            "classes": top,
            "within_class_cosine": round(w, 4),
            "between_class_cosine": round(b, 4),
            "gate_passed": w > b,
        }
    print(results, flush=True)
    ok = all(v["gate_passed"] for v in results.values())
    print("VALIDATION GATE:", "PASSED" if ok else "FAILED", flush=True)
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    if args.validate:
        raise SystemExit(_validate())
    print(load_aves2()[0].__class__.__name__, "loaded", flush=True)


if __name__ == "__main__":
    main()
