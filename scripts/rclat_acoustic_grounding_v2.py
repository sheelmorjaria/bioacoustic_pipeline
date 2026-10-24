"""Inference-only, fixed-checkpoint acoustic grounding controls."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import rclat_ctc_learning_diagnostic as core

ROOT = Path("outputs/rclat_acoustic_grounding_v2")


def transform(values, masks, native, condition, donor_inputs=None):
    values, masks = values.clone(), masks.clone()
    if condition == "swapped":
        dv, dm, dn = donor_inputs
        for i, n in enumerate(native.tolist()):
            idx = torch.linspace(0, int(dn[i]) - 1, n, device=values.device).round().long()
            values[i, :n] = dv[i, idx]
            masks[i, :n] = dm[i, idx]
    elif condition == "reversed":
        for i, n in enumerate(native.tolist()):
            values[i, :n] = values[i, :n].flip(0)
            masks[i, :n] = masks[i, :n].flip(0)
    elif condition == "zeroed":
        values.zero_()
    elif condition != "original":
        raise ValueError(condition)
    return values, masks, native


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, choices=["xpu:0", "xpu:1"])
    args = parser.parse_args()
    plan = json.loads((ROOT / "execution_plan.json").read_text())
    if os.environ.get("ONEDNN_PRIMITIVE_CACHE_CAPACITY") != "16384":
        raise ValueError("Missing frozen cache policy")
    for path, digest in plan["dependencies"].items():
        if core.sha(path) != digest:
            raise ValueError("Changed dependency: " + path)
    core.OUT = ROOT / args.device.replace(":", "_")
    _, _, cohort, batch, _ = core.prepare("grounding", args.device)
    model = core.build()
    checkpoint = torch.load(plan["checkpoint"], weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model = model.to(args.device).eval()
    initial = core.predecessor.initial_digest(model)
    donors = plan["donors"]
    rows_by_uid = {r["uid"]: r for name in ["training", "validation"] for r in cohort[name]}
    start = time.monotonic()
    results = {}
    with (core.OUT / "journal.jsonl").open("x", buffering=1) as journal:
        for condition in plan["conditions"]:
            if time.monotonic() - start > plan["max_seconds"]:
                raise TimeoutError("Diagnostic bound")

            def controlled_batch(rows, where=args.device):
                inputs, targets = batch(rows, where)
                donor_inputs = None
                if condition == "swapped":
                    donor_inputs, _ = batch([rows_by_uid[donors[r["uid"]]] for r in rows], where)
                return transform(*inputs, condition, donor_inputs), targets

            results[condition] = core.evaluate(model, controlled_batch, cohort, condition, 2000, journal)
    if core.predecessor.initial_digest(model) != initial:
        raise ValueError("Inference mutated model state")
    reproduction = {}
    for split in ["training", "validation"]:
        old = json.loads(Path(plan["saved_evaluation"]).read_text())[split]["predictions"]
        new = results["original"][split]["predictions"]
        checks = []
        for a, b in zip(old, new):
            assert a["uid"] == b["uid"]
            with np.load(a["raw_path"], allow_pickle=False) as f, np.load(b["raw_path"], allow_pickle=False) as g:
                x, y = f["logits"].astype(float), g["logits"].astype(float)
                rel = float(np.linalg.norm(x-y) / max(np.linalg.norm(x), 1e-30))
            checks.append({"uid":a["uid"],"logit_relative_l2":rel,"hypothesis_equal":a["hypothesis"]==b["hypothesis"]})
        reproduction[split] = checks
    passed = all(c["hypothesis_equal"] and c["logit_relative_l2"] <= 1e-4 for checks in reproduction.values() for c in checks)
    if core.sha(plan["checkpoint"]) != plan["dependencies"][plan["checkpoint"]]:
        raise ValueError("Checkpoint bytes changed")
    core.write(core.OUT / "result.json", {"completed":True,"original_reproduction_passed":passed,"reproduction":reproduction,"elapsed_seconds":time.monotonic()-start,"optimizer_updates":0,"checkpoint_unchanged":True,"P100_authorized":False,"P960_authorized":False,"stage2_ready":False})
    if not passed:
        raise ValueError("Original prediction/logit reproduction prerequisite failed")


if __name__ == "__main__":
    main()
