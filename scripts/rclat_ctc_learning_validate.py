"""Verify completed diagnostic artifacts without inference or checkpoint loading."""

import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np

OUT = Path("outputs/rclat_ctc_learning_diagnostic_v1")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for x in iter(lambda: f.read(2**20), b""):
            h.update(x)
    return h.hexdigest()


def main():
    cohort = json.loads((OUT / "cohort.json").read_text())
    verified = 0
    templates = {}
    for stage, steps in [("D0", [1, 16, 30]), ("D1", [0, 100, 250, 500, 1000, 1500, 2000])]:
        for step in steps:
            e = json.loads((OUT / stage / f"step_{step:04}" / "evaluation.json").read_text())
            for split in ["training", "validation"]:
                rows = e[split]["predictions"]
                expected = {r["uid"] for r in cohort[split]}
                assert len(rows) == len(expected) and {r["uid"] for r in rows} == expected
                for row in rows:
                    path = Path(row["raw_path"])
                    assert sha(path) == row["raw_sha256"]
                    with np.load(path, allow_pickle=False) as data:
                        logits = data["logits"]
                        gamma = data["conditioned_occupancy"]
                        assert logits.shape == gamma.shape == (row["output_length"], 29)
                        assert logits.dtype == np.float32 and gamma.dtype == np.float64
                        assert len(data["target_ids"]) == row["target_length"]
                        assert int(data["native_length"]) == row["native_length"]
                        assert np.isfinite(logits).all() and np.isfinite(gamma).all()
                        assert gamma.min() >= -1e-7 and gamma.max() <= 1 + 1e-7
                        assert np.max(np.abs(gamma.sum(-1) - 1)) <= 1e-7
                    verified += 1
                if stage == "D1" and step == 2000:
                    c = Counter(r["hypothesis"] for r in rows)
                    templates[split] = dict(
                        distinct_hypotheses=len(c), top_hypotheses=c.most_common()
                    )
    for step in [0, 100, 250, 500, 1000, 1500, 2000]:
        r = json.loads((OUT / f"D1_step_{step:04}.json").read_text())
        assert sha(r["checkpoint"]) == r["checkpoint_sha256"]
    events = []
    for line in (OUT / "D1_journal.jsonl").open():
        r = json.loads(line)
        if r.get("event") == "step_complete":
            events.append(r["step"])
    assert events == list(range(1, 2001)) and verified == 960
    assert not list(OUT.glob("*FAILURE.json"))
    result = dict(
        raw_records_verified=verified,
        D1_checkpoint_hashes_verified=7,
        D1_updates_verified=2000,
        posterior_arrays_checked=True,
        final_hypothesis_templates=templates,
        new_inference=False,
    )
    (OUT / "artifact_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Verified960rawrecords,7checkpoint hashes,2000sequential updates.")


if __name__ == "__main__":
    main()
