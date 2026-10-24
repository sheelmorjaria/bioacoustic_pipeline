#!/usr/bin/env python3
"""Segmented parallel downloader for large archive.org files.

Per-connection throttling at archive.org makes single-stream downloads crawl
(~0.4 MB/s). This splits the file into segments fetched by N parallel range
connections into per-segment part files, then concatenates. Resumable: a
segment whose part file is already complete is skipped.

Usage: python3 segmented_download.py <url> <out_path> [n_workers] [seg_mb]
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CHUNK = 1 << 20  # 1 MiB read chunks


def total_size(url: str) -> int:
    for _ in range(5):
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=60) as r:
                return int(r.headers["Content-Length"])
        except Exception:
            time.sleep(5)
    raise RuntimeError("HEAD failed")


def fetch_segment(url: str, start: int, end: int, part: Path) -> tuple[int, bool]:
    """Fetch byte range [start, end] into part, resuming a partial part."""
    have = part.stat().st_size if part.exists() else 0
    want = end - start + 1
    if have >= want:
        return len(str(part)), True
    for attempt in range(12):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})
            with urllib.request.urlopen(req, timeout=90) as r, open(part, "ab") as fh:
                while True:
                    block = r.read(CHUNK)
                    if not block:
                        break
                    fh.write(block)
                    have += len(block)
            if have >= want:
                return len(str(part)), True
        except Exception:
            time.sleep(min(2**attempt, 60))
    return len(str(part)), False


def main() -> None:
    url, out = sys.argv[1], Path(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    seg_mb = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    total = total_size(url)
    seg = seg_mb * (1 << 20)
    ranges = [(i, min(i + seg - 1, total - 1)) for i in range(0, total, seg)]
    parts_dir = out.parent / f".{out.name}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    print(f"{total / 1e9:.1f} GB in {len(ranges)} segments, {workers} workers", flush=True)

    def job(idx: int) -> tuple[int, bool]:
        s, e = ranges[idx]
        return fetch_segment(url, s, e, parts_dir / f"{idx:05d}.part")

    t0, done_bytes = time.time(), 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(job, i): i for i in range(len(ranges))}
        ok = True
        for fut in as_completed(futs):
            _, good = fut.result()
            i = futs[fut]
            done_bytes += ranges[i][1] - ranges[i][0] + 1
            if not good:
                ok = False
                print(f"SEGMENT {i} FAILED", flush=True)
            if done_bytes % (seg * 4) < seg:
                rate = done_bytes / max(time.time() - t0, 1) / 1e6
                print(f"{done_bytes / 1e9:.1f}/{total / 1e9:.1f} GB  {rate:.1f} MB/s", flush=True)
    if not ok:
        sys.exit("some segments failed — rerun to resume")

    with open(out, "wb") as out_fh:
        for i in range(len(ranges)):
            with open(parts_dir / f"{i:05d}.part", "rb") as pf:
                while block := pf.read(1 << 22):
                    out_fh.write(block)
    got = out.stat().st_size
    if got != total:
        sys.exit(f"size mismatch after concat: {got} != {total}")
    for pf in parts_dir.glob("*.part"):
        pf.unlink()
    parts_dir.rmdir()
    print(f"DONE {out} {got / 1e9:.1f} GB in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
