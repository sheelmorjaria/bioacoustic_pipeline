"""Read-only full-training-population inference at fixed closed checkpoints."""

import argparse
import json
from pathlib import Path
import time
import traceback

import torch
import rclat_bilstm_p10_train as training

core = training.core
ROOT = Path("outputs/rclat_bilstm_p10_error_diagnostic_v1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, choices=["xpu:0", "xpu:1"])
    a = parser.parse_args()
    plan = json.loads((ROOT / "protocol.json").read_text())
    for p, digest in plan["dependencies"].items():
        if core.sha(p) != digest:
            raise ValueError("Changed dependency: " + p)
    entry = plan["cases"][a.device]
    epoch = entry["epoch"]
    training.setup(a.device)
    _, population = training.check()
    receipt = json.loads((training.ROOT / "input_verification.json").read_text())
    inventory = json.loads(Path("outputs/rclat_p10/cache_integrity.json").read_text())["files"]
    cohort = {}
    for split, key in [("training", "selected_training"), ("validation", "selected_validation")]:
        rows = sorted(population[key], key=lambda r: r["uid"])
        if split == "validation":
            rows = rows[:4]
        cohort[split] = []
        for row in rows:
            cache = Path("cache/rclat_p10") / (row["uid"] + ".npz")
            if (
                core.sha(cache) != inventory[str(cache)]
                or core.sha(row["path"]) != row["source_sha256"]
            ):
                raise ValueError("Source/cache changed")
            cohort[split].append(row)
    assert len(cohort["training"]) == 2814
    assert all(r["uid"] in receipt["records"] for rows in cohort.values() for r in rows)
    out = ROOT / a.device.replace(":", "_")
    core.OUT = out
    core.write(
        out / "STARTED.json",
        {
            "epoch": epoch,
            "device": a.device,
            "protocol_sha256": core.sha(ROOT / "protocol.json"),
            "optimizer_updates": 0,
        },
    )
    checkpoint = torch.load(entry["checkpoint"], weights_only=True, map_location="cpu")
    model = core.build()
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model = model.to(a.device).eval()
    digest = core.predecessor.initial_digest(model)
    start = time.monotonic()
    with (out / "journal.jsonl").open("x", buffering=1) as journal:
        evaluation = core.evaluate(
            model, training.batcher(a.device), cohort, "epoch_probe", epoch, journal
        )
    assert core.predecessor.initial_digest(model) == digest
    assert core.sha(entry["checkpoint"]) == entry["checkpoint_sha256"]
    saved = json.loads(Path(entry["saved_validation"]).read_text())["predictions"]
    lookup = {r["uid"]: r for r in saved}
    reproduced = all(
        r["hypothesis"] == lookup[r["uid"]]["hypothesis"]
        for r in evaluation["validation"]["predictions"]
    )
    core.write(
        out / "result.json",
        {
            "completed": True,
            "epoch": epoch,
            "training_utterances": 2814,
            "validation_anchor_hypotheses_reproduced": reproduced,
            "optimizer_updates": 0,
            "model_and_checkpoint_unchanged": True,
            "elapsed_seconds": time.monotonic() - start,
        },
    )
    if not reproduced:
        raise ValueError("Validation anchor reproduction failed; do not interpret")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import sys

        device = sys.argv[sys.argv.index("--device") + 1]
        path = ROOT / device.replace(":", "_") / "FAILURE.json"
        if not path.exists():
            core.write(path, {"traceback": traceback.format_exc(), "automatic_retry": False})
        raise
