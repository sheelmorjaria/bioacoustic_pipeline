"""Canonical serialization + SHA-256 hashing for reproducibility.

The tri-run reproducibility claim rests on *byte-identical* artifacts across runs.
Two pieces make that robust:

* ``canonical_json`` — key-order-independent JSON (``sort_keys=True`` + compact
  separators). A dict assembled in a different insertion order still serialises to
  the same bytes, so hashing is resilient to nondeterministic key order.
* ``sha256_file`` / ``hash_artifacts`` — content-address each output artifact so the
  tri-run table compares fingerprints, not "the numbers look the same".

NumPy scalars/arrays that sneak into a report or trace are coerced to plain Python
in the JSON default handler (the heavy ``.npy`` arrays are written separately via
``np.save`` and hashed as files).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

_CHUNK = 1024 * 1024  # 1 MiB read chunks


def _json_default(obj):
    """Coerce numpy types/arrays to JSON-serialisable Python (for canonical_json)."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def canonical_json(obj) -> str:
    """Deterministic, key-order-independent JSON string.

    ``sort_keys=True`` + compact separators + numpy coercion => the same logical
    object always serialises to identical bytes regardless of dict insertion order.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def write_canonical_json(path: str | Path, obj) -> str:
    """Write ``obj`` as canonical JSON (compact, sorted keys). Returns the string written."""
    s = canonical_json(obj)
    Path(path).write_text(s, encoding="utf-8")
    return s


def write_canonical_jsonl(path: str | Path, records) -> None:
    """Write a list of records as canonical JSON-lines (one canonical object per line)."""
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(canonical_json(rec))
            fh.write("\n")


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of a byte string."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """SHA-256 hex digest of a file's bytes (streamed in 1 MiB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_string(s: str) -> str:
    """SHA-256 hex digest of a UTF-8 string."""
    return sha256_bytes(s.encode("utf-8"))


def hash_artifacts(out_dir: str | Path, names) -> dict[str, str]:
    """Map each existing artifact name -> SHA-256 of ``out_dir / name``.

    Missing files are skipped (the DSP-core path emits no ``affect.npy``; the table
    only lists artifacts that path actually produced).
    """
    base = Path(out_dir)
    return {name: sha256_file(base / name) for name in names if (base / name).exists()}
