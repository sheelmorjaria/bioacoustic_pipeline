#!/usr/bin/env python3
"""Download selected alp-data corpora for caller-ID cross-corpus work.

Run inside the .venv-alp environment with ALP_DATA_HOME=data/alp. Downloads
(by iterating fully, which forces the fetch) and reports per-dataset item
counts. Chosen corpora (licenses recorded in data/alp/README.md):
chiffchaff_id / pipit_id / littleowl_id (Stowell 2018, CC-BY-4.0, canonical
within/across-year individual-ID splits), infant_marmosets_vox (caller +
call type), dinardo_dolphin_whistles. corvid_wascher EXCLUDED (private/MOU).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import os
import sys
import time

DATASETS = [
    ("chiffchaff_id", "train_within_year"),
    ("chiffchaff_id", "test_within_year"),
    ("chiffchaff_id", "train_across_year"),
    ("chiffchaff_id", "test_across_year"),
    ("pipit_id", "train_within_year"),
    ("pipit_id", "test_within_year"),
    ("pipit_id", "train_across_year"),
    ("pipit_id", "test_across_year"),
    ("littleowl_id", "train_across_year"),
    ("littleowl_id", "test_across_year"),
    ("infant_marmosets_vox", "all"),
    ("dinardo_dolphin_whistles", "all"),
]


def main() -> None:
    from alp_data.datasets import (
        ChiffchaffId,
        DinardoDolphinWhistles,
        InfantMarmosetsVox,
        LittleOwlId,
        PipitId,
    )

    classes = {
        "chiffchaff_id": ChiffchaffId,
        "pipit_id": PipitId,
        "littleowl_id": LittleOwlId,
        "infant_marmosets_vox": InfantMarmosetsVox,
        "dinardo_dolphin_whistles": DinardoDolphinWhistles,
    }
    for name, split in DATASETS:
        cls = classes[name]
        t0 = time.time()
        try:
            ds = cls(split=split, sample_rate=None)  # native sr, no decode cost
            n = 0
            fields = None
            for item in ds:
                n += 1
                if fields is None and hasattr(item, "keys"):
                    fields = list(item.keys())
            print(f"OK {name}[{split}]: {n} items in {time.time()-t0:.0f}s; fields={fields}", flush=True)
        except Exception as exc:  # noqa: BLE001 — report and continue with others
            print(f"FAIL {name}[{split}]: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    print("ALL-DONE", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("ALP_DATA_HOME", os.path.join(os.path.dirname(__file__), "..", "data", "alp"))
    sys.exit(main())
