"""Complete a timed-out official transfer using explicitly bounded official ranges.

Preserves the incomplete original and every range; accepts only published full MD5.
"""

import hashlib
from pathlib import Path
import shutil

from rclat_p10_protocol import ARCHIVES, ROOT, freeze, sha

RANGES = [
    (2000000000, 3096827374),
    (3096827375, 4193654749),
    (4193654750, 5290482124),
    (5290482125, 6387309498),
]


def assemble():
    receipt = ROOT / "archive_integrity.json"
    if receipt.exists():
        raise RuntimeError("Archive receipt already exists")
    archive = ROOT / "train-clean-100.tar.gz"
    original = ROOT / "train-clean-100.partial-original.tar.gz"
    assembled = ROOT / "train-clean-100.assembled.tar.gz"
    if original.exists() or assembled.exists():
        raise RuntimeError("Previous assembly exists; inspect without overwriting")
    if archive.stat().st_size < RANGES[0][0]:
        raise ValueError("Insufficient original prefix")
    parts = []
    for i, (start, end) in enumerate(RANGES):
        path = ROOT / f"train100_tail_{i}.part"
        if path.stat().st_size != end - start + 1:
            raise ValueError(f"Incomplete range {i}")
        parts.append(
            {
                "path": str(path),
                "start": start,
                "end_inclusive": end,
                "sha256": sha(path),
                "url": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
            }
        )
    prefix_stat = archive.stat()
    prefix_digest = hashlib.sha256()
    with assembled.open("xb") as dst, archive.open("rb") as src:
        remaining = RANGES[0][0]
        while remaining:
            block = src.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError("Prefix truncated")
            dst.write(block)
            prefix_digest.update(block)
            remaining -= len(block)
        for part in parts:
            with Path(part["path"]).open("rb") as source:
                shutil.copyfileobj(source, dst, 8 * 1024 * 1024)
    after = archive.stat()
    if (after.st_size, after.st_mtime_ns, after.st_ino) != (
        prefix_stat.st_size,
        prefix_stat.st_mtime_ns,
        prefix_stat.st_ino,
    ):
        raise RuntimeError("Original transfer is still changing")
    with assembled.open("rb") as source:
        md5 = hashlib.file_digest(source, "md5").hexdigest()
    if md5 != ARCHIVES["train-clean-100"]:
        raise ValueError("Published full-archive MD5 mismatch; no extraction allowed")
    digest = sha(assembled)
    archive.rename(original)
    assembled.rename(archive)
    freeze(
        receipt,
        {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "md5": md5,
            "sha256": digest,
            "source": "https://openslr.trmal.net/resources/12/train-clean-100.tar.gz",
            "attempt": "Original curl reached600s timeout (exit28); partial retained. Parallel ranges from official www.openslr.org mirror reassembled; full published checksum matched.",
            "retained_original_partial": str(original),
            "prefix_bytes_used": RANGES[0][0],
            "prefix_sha256": prefix_digest.hexdigest(),
            "ranges": parts,
        },
    )
    print(
        "train-clean-100 complete; official full-file checksum verified; all transfer parts retained."
    )


if __name__ == "__main__":
    assemble()
