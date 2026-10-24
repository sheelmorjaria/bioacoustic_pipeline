"""Integrity checks for completed synthetic artifacts; no model execution."""

from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import numpy as np

OUT = Path("outputs/rclat_ctc_synthetic_diagnostic_v1")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    cohort = json.loads((OUT / "cohort.json").read_text())
    expected = {
        s: {r["uid"] for r in cohort["records"] if r["split"] == s}
        for s in ["training", "validation"]
    }
    count = 0
    template = {}
    for step in [0, 100, 250, 500, 1000, 1500, 2000]:
        result = json.loads((OUT / "SYNTHETIC" / f"step_{step:04}" / "evaluation.json").read_text())
        for split in ["training", "validation"]:
            rows = result[split]["predictions"]
            assert len(rows) == len(expected[split]) and {r["uid"] for r in rows} == expected[split]
            for row in rows:
                path = Path(row["raw_path"])
                assert sha(path) == row["raw_sha256"]
                with np.load(path, allow_pickle=False) as f:
                    assert (
                        f["logits"].shape
                        == f["conditioned_occupancy"].shape
                        == (row["output_length"], 29)
                    )
                    assert (
                        f["logits"].dtype == np.float32
                        and f["conditioned_occupancy"].dtype == np.float64
                    )
                    assert len(f["target_ids"]) == row["target_length"]
                    gamma = f["conditioned_occupancy"]
                    assert np.isfinite(f["logits"]).all() and np.isfinite(gamma).all()
                    assert (
                        gamma.min() >= -1e-7
                        and gamma.max() <= 1 + 1e-7
                        and np.max(np.abs(gamma.sum(-1) - 1)) <= 1e-7
                    )
                count += 1
            if step == 2000:
                by = defaultdict(Counter)
                for row in rows:
                    by[row["native_length"]][row["hypothesis"]] += 1
                template[split] = {str(k): dict(v) for k, v in by.items()}
        receipt = json.loads((OUT / f"step_{step:04}.json").read_text())
        assert sha(receipt["checkpoint"]) == receipt["checkpoint_sha256"]
    steps = []
    for line in (OUT / "SYNTHETIC_journal.jsonl").open():
        r = json.loads(line)
        if r.get("event") == "step_complete":
            steps.append(r["step"])
    assert count == 672 and steps == list(range(1, 2001)) and not list(OUT.glob("*FAILURE.json"))
    output = dict(
        raw_records_verified=count,
        checkpoint_hashes_verified=7,
        completed_updates_verified=2000,
        final_hypotheses_by_native_length=template,
        all_same_length_utterances_have_identical_final_hypotheses=all(
            len(v) == 1 for group in template.values() for v in group.values()
        ),
        new_inference=False,
    )
    (OUT / "artifact_validation.json").write_text(json.dumps(output, indent=2) + "\n")
    print("Verified672rawrecords,7checkpoints,2000sequentialupdates.")


if __name__ == "__main__":
    main()
