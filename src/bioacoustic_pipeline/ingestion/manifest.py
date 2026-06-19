"""Corpus ingestion: species profiles, domain manifest, corpus QA.

Loads configs/species_profiles.yaml (NBD priors per species), builds a manifest
of audio files for a target species, runs QA (loadable, duration, sample rate),
and reports which embedding backbones the firewall permits for that species.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILES = REPO_ROOT / "configs" / "species_profiles.yaml"


@dataclass
class DomainEntry:
    path: str
    species: str
    duration_s: float
    sample_rate: int


def load_species_profiles(path: str | Path | None = None) -> dict:
    """Parse configs/species_profiles.yaml -> {species: {thresholds...}}."""
    import yaml  # noqa: PLC0415

    p = Path(path) if path else DEFAULT_PROFILES
    with open(p) as f:
        return yaml.safe_load(f) or {}


def build_manifest(audio_dir: str | Path, species: str,
                   profiles: dict | None = None) -> list[DomainEntry]:
    """Enumerate audio files under audio_dir, QA each, return valid entries."""
    from bioacoustic_pipeline.utils.audio import load_audio  # noqa: PLC0415

    profiles = profiles or load_species_profiles()
    target_sr = profiles.get(species, {}).get("sample_rate")
    audio_dir = Path(audio_dir)
    entries: list[DomainEntry] = []
    for ext in ("*.wav", "*.flac", "*.mp3"):
        for p in sorted(audio_dir.rglob(ext)):
            try:
                audio, sr = load_audio(str(p), target_sr=target_sr)
            except Exception:  # noqa: BLE001 — QA skips unreadable files
                continue
            if len(audio) < int((target_sr or sr) * 0.05):
                continue  # too short
            entries.append(DomainEntry(
                path=str(p), species=species,
                duration_s=len(audio) / (target_sr or sr), sample_rate=target_sr or sr,
            ))
    return entries


def allowed_backbones(species: str, firewall=None) -> list[str]:
    """Which registered models the firewall permits to EMBED for this species."""
    from bioacoustic_pipeline.embeddings.firewall import Use, default_firewall  # noqa: PLC0415

    fw = firewall or default_firewall()
    out = []
    for prov in fw.audit():
        name = prov["model_name"]
        if (
            fw.is_allowed(name, Use.EMBEDDING)
            and (not prov["applies_to_species"] or species in prov["applies_to_species"])
        ):
            out.append(name)
    return out
