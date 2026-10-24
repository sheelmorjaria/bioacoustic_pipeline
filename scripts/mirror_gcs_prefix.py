#!/usr/bin/env python3
"""Mirror public GCS prefixes (alp-data corpus trees) into data/alp.

Lists objects under each prefix via the public JSON API (paginated) and
downloads them concurrently; existing same-size files are skipped, so the
script is resumable. Mirrors the exact object paths so alp-data's Dataset
classes read straight off ALP_DATA_HOME=data/alp.

Usage: python mirror_gcs_prefix.py <prefix> [<prefix> ...]
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET = "esp-data-274503"
BASE = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
DL = f"https://storage.googleapis.com/{BUCKET}/"
DEST = Path(os.environ.get("ALP_DATA_HOME", "data/alp"))


def list_objects(prefix: str) -> list[dict]:
    out, token = [], None
    while True:
        url = f"{BASE}?prefix={urllib.parse.quote(prefix)}&maxResults=1000"
        if token:
            url += f"&pageToken={token}"
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
        out.extend(d.get("items", []))
        token = d.get("nextPageToken")
        if not token:
            return out


def fetch(obj: dict) -> tuple[str, bool]:
    name, size = obj["name"], int(obj["size"])
    target = DEST / name
    if target.exists() and target.stat().st_size == size:
        return name, True
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(8):
        try:
            req = urllib.request.Request(DL + urllib.parse.quote(name))
            with urllib.request.urlopen(req, timeout=120) as r, open(target, "wb") as fh:
                while block := r.read(1 << 20):
                    fh.write(block)
            if target.stat().st_size == size:
                return name, True
        except Exception:
            time.sleep(min(2**attempt, 30))
    return name, False


def main() -> None:
    import urllib.parse

    for prefix in sys.argv[1:]:
        objs = list_objects(prefix)
        total = sum(int(o["size"]) for o in objs)
        print(f"{prefix}: {len(objs)} objects, {total / 1e9:.2f} GB", flush=True)
        done = failed = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=40) as pool:
            futs = {pool.submit(fetch, o): o for o in objs}
            for fut in as_completed(futs):
                _, ok = fut.result()
                done += 1
                failed += 0 if ok else 1
                if done % 250 == 0:
                    gb = sum(int(o['size']) for o in objs[:done]) / 1e9
                    print(f"  {done}/{len(objs)} ({gb:.2f} GB) {failed} failed", flush=True)
        print(f"{prefix} DONE: {done} objects, {failed} failed, {(time.time()-t0)/60:.1f} min", flush=True)
    print("MIRROR-DONE", flush=True)


if __name__ == "__main__":
    main()
