"""Native-length inference and independently scored transcript-pairing control."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import rclat_simple_ctc_diagnostic_v2 as core

ROOT = Path("outputs/rclat_bilstm_matched_grounding_v1")


def pairing_scores(predictions, donors):
    by_uid = {r["uid"]: r for r in predictions}
    if set(donors.values()) != set(donors) or not set(by_uid).issubset(donors):
        raise ValueError("Donor map must be a bijection covering predictions")
    scores = {"matched": [], "native_swapped": []}
    for row in predictions:
        for condition, source in [("matched", row), ("native_swapped", by_uid[donors[row["uid"]]])]:
            reference = row["reference"]
            hypothesis = source["hypothesis"]
            char = core.edit_counts(reference.replace(" ", ""), hypothesis.replace(" ", ""))
            word = core.edit_counts(reference.split(), hypothesis.split())
            scores[condition].append(
                {
                    "uid": row["uid"],
                    "speaker": row["speaker"],
                    "input_uid": source["uid"],
                    "input_native_length": source["native_length"],
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "reference_characters": len(reference.replace(" ", "")),
                    "reference_words": len(reference.split()),
                    "character_errors": char["errors"],
                    "word_errors": word["errors"],
                    "character_decomposition": char,
                    "word_decomposition": word,
                    "cer": char["errors"] / len(reference.replace(" ", "")),
                    "wer": word["errors"] / len(reference.split()),
                }
            )
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, choices=["xpu:0", "xpu:1"])
    args = parser.parse_args()
    plan = json.loads((ROOT / "execution_plan.json").read_text())
    for p, digest in plan["dependencies"].items():
        if core.sha(p) != digest:
            raise ValueError("Changed dependency: " + p)
    if os.environ.get("ONEDNN_VERBOSE") != "profile_create,filter=conv":
        raise ValueError("Required oneDNN logging absent")
    core.OUT = ROOT / args.device.replace(":", "_")
    _, _, cohort, batch, _ = core.prepare("native_grounding", args.device)
    checkpoint = torch.load(plan["checkpoint"], weights_only=True, map_location="cpu")
    model = core.build()
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model = model.to(args.device).eval()
    before = core.predecessor.initial_digest(model)
    start = time.monotonic()
    with (core.OUT / "journal.jsonl").open("x", buffering=1) as journal:
        results = core.evaluate(model, batch, cohort, "native", 2000, journal)
    if core.predecessor.initial_digest(model) != before:
        raise ValueError("Model state mutated")
    saved = json.loads(Path(plan["saved_evaluation"]).read_text())
    reproduction = []
    for split in ["training"]:
        for a, b in zip(saved[split]["predictions"], results[split]["predictions"]):
            assert a["uid"] == b["uid"] and a["native_length"] == b["native_length"]
            with (
                np.load(a["raw_path"], allow_pickle=False) as f,
                np.load(b["raw_path"], allow_pickle=False) as g,
            ):
                x, y = f["logits"].astype(float), g["logits"].astype(float)
                rel = float(np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-30))
            reproduction.append(
                {
                    "uid": a["uid"],
                    "hypothesis_equal": a["hypothesis"] == b["hypothesis"],
                    "logit_relative_L2": rel,
                }
            )
    passed = all(r["hypothesis_equal"] and r["logit_relative_L2"] <= 1e-4 for r in reproduction)
    assert core.sha(plan["checkpoint"]) == plan["dependencies"][plan["checkpoint"]]
    core.write(
        core.OUT / "result.json",
        {
            "completed": True,
            "original_reproduction_passed": passed,
            "reproduction": reproduction,
            "checkpoint_unchanged": True,
            "optimizer_updates": 0,
            "elapsed_seconds": time.monotonic() - start,
        },
    )
    if not passed:
        raise ValueError("Reproduction prerequisite failed")


if __name__ == "__main__":
    main()
