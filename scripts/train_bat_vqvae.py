#!/usr/bin/env python3
"""Train a bat VQ-VAE tokenizer on bat CPC latents.

Unblocks Track B: once trained, the pipeline's discrete-token diagnostics
(LZ complexity, functional aliasing, call unitization, motif mining) work
end-to-end on bat vocalizations.

Extracts CPC latents from bat audio using the bat_cpc checkpoint, then trains
a VQ-VAE (same architecture as the human one: EMA codebook, 256 tokens).

Usage:
    python scripts/train_bat_vqvae.py --audio-dir /path/to/bat/audio --n-files 500

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CKPT_OUT = ROOT / "checkpoints" / "bat_vqvae" / "best_model.pt"


def main():
    ap = argparse.ArgumentParser(description="Train bat VQ-VAE on CPC latents")
    ap.add_argument("--audio-dir", type=Path, required=True, help="directory of bat audio")
    ap.add_argument("--n-files", type=int, default=500)
    ap.add_argument("--codebook-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()

    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader, TensorDataset

    from bioacoustic_pipeline.embeddings.species_cpc_backbone import SpeciesCPCBackbone
    from bioacoustic_pipeline.tokenization.syntactic_vqvae import SyntacticVQVAE
    from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals

    # 1. Load bat CPC + extract latents
    print("Loading bat CPC...")
    cpc = SpeciesCPCBackbone(name="bat_cpc", provenance_name="bat_cpc")
    if not cpc.available:
        raise RuntimeError(f"bat CPC unavailable: {cpc.load_error}")

    # Collect audio files
    audio_files = []
    for ext in ("*.wav", "*.flac", "*.mp3"):
        audio_files.extend(sorted(args.audio_dir.rglob(ext)))
    audio_files = audio_files[: args.n_files]
    print(f"Processing {len(audio_files)} bat audio files...")

    from bioacoustic_pipeline.utils.audio import load_audio
    from bioacoustic_pipeline.ingestion.manifest import load_species_profiles

    profile = load_species_profiles().get("bat", {})
    latents = []
    for i, f in enumerate(audio_files):
        try:
            audio, sr = load_audio(str(f), target_sr=profile.get("sample_rate", 250000))
            if len(audio) < sr * 0.02:
                continue
            sig = compute_signals(audio, sr, hop_ms=profile.get("hop_ms", 10.0))
            proposals = generate_proposals(sig, floor_energy=profile.get("energy_threshold", 0.20),
                                           floor_flux=profile.get("spectral_threshold", 0.50),
                                           debounce_ms=profile.get("debounce_ms", 200.0))
            for prop in proposals:
                s = int(prop.time_ms / 1000 * sr)
                e = min(s + int(sr * 0.1), len(audio))
                if e - s > sr * 0.02:
                    latent = cpc.embed(audio[s:e], sr)
                    latents.append(latent)
        except Exception:  # noqa: BLE001
            continue
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(audio_files)} files, {len(latents)} segments")

    if len(latents) < 100:
        print(f"Only {len(latents)} segments — need more audio for VQ-VAE training")
        return

    latents = torch.tensor(np.array(latents), dtype=torch.float32)
    mean = latents.mean(0, keepdim=True)
    std = latents.std(0, keepdim=True) + 1e-6
    latents = (latents - mean) / std
    print(f"Training VQ-VAE on {len(latents)} bat CPC latents ({latents.shape[1]}D)")

    # 2. Train VQ-VAE
    vqvae = SyntacticVQVAE(
        input_dim=latents.shape[1], codebook_size=args.codebook_size,
        codebook_dim=64, hidden_dim=128,
    )
    opt = torch.optim.Adam(vqvae.parameters(), lr=1e-3)
    ds = TensorDataset(latents)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    best_loss = float("inf")
    for epoch in range(args.epochs):
        vqvae.train()
        total = 0
        for (batch,) in loader:
            x_recon, z_q, token_ids, _ = vqvae(batch)
            z = vqvae.encode(batch)
            loss, recon_l, commit_l = vqvae.loss_function(batch, x_recon, z, z_q)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(ds)
        if avg < best_loss:
            best_loss = avg
            CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": vqvae.state_dict(),
                "config": {"input_dim": latents.shape[1], "codebook_size": args.codebook_size,
                           "codebook_dim": 64, "hidden_dim": 128},
                "norm_mean": mean.numpy(), "norm_std": std.numpy(),
                "best_loss": best_loss,
            }, CKPT_OUT)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}: loss={avg:.4f} best={best_loss:.4f}")
    print(f"Done. Best loss={best_loss:.4f} -> {CKPT_OUT}")


if __name__ == "__main__":
    main()
