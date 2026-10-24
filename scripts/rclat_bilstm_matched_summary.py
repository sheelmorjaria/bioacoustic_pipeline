"""Frozen speaker-cluster paired analysis; no tuning or progression decision."""

import json

import numpy as np
import rclat_bilstm_matched_grounding as runner

core = runner.core
ROOT = runner.ROOT


def paired_interval(values, seed, replicates):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid speaker contrasts")
    indices = np.random.default_rng(seed).integers(0, len(values), size=(replicates, len(values)))
    boot = values[indices].mean(axis=1)
    return {
        "point_estimate": float(values.mean()),
        "percentile95": np.quantile(boot, [0.025, 0.975], method="linear").tolist(),
        "replicates": replicates,
        "seed": seed,
        "speaker_clusters": len(values),
    }


def main():
    plan = json.loads((ROOT / "execution_plan.json").read_text())
    for p, digest in plan["dependencies"].items():
        assert core.sha(p) == digest, p
    results = {}
    raw = 0
    for device in ["xpu_0", "xpu_1"]:
        out = ROOT / device
        receipt = json.loads((out / "result.json").read_text())
        assert receipt["completed"] and receipt["original_reproduction_passed"]
        evaluation = json.loads((out / "native/step_2000/evaluation.json").read_text())
        for split in ["training", "validation"]:
            for r in evaluation[split]["predictions"]:
                assert core.sha(r["raw_path"]) == r["raw_sha256"]
                raw += 1
        scores = runner.pairing_scores(evaluation["validation"]["predictions"], plan["donors"])
        metrics = {k: core.predecessor.original.aggregate(v) for k, v in scores.items()}
        speakers = sorted({r["speaker"] for r in scores["matched"]})
        effects = []
        for speaker in speakers:
            units = {}
            for condition, rows in scores.items():
                group = [r for r in rows if r["speaker"] == speaker]
                assert len(group) == 16
                units[condition] = {}
                for unit, den in [
                    ("character", "reference_characters"),
                    ("word", "reference_words"),
                ]:
                    total = sum(r[den] for r in group)
                    units[condition][unit] = {
                        component: sum(r[unit + "_decomposition"][component] for r in group) / total
                        for component in ["errors", "substitutions", "deletions", "insertions"]
                    }
            effects.append(
                {
                    "speaker": speaker,
                    "matched": units["matched"],
                    "native_swapped": units["native_swapped"],
                    "CER_increase": units["native_swapped"]["character"]["errors"]
                    - units["matched"]["character"]["errors"],
                    "WER_increase": units["native_swapped"]["word"]["errors"]
                    - units["matched"]["word"]["errors"],
                }
            )
        assert len(effects) == 24
        interval = paired_interval(
            [r["CER_increase"] for r in effects],
            plan["bootstrap_seed"],
            plan["bootstrap_replicates"],
        )
        results[device] = {
            "metrics": metrics,
            "primary_speaker_macro_CER_difference": interval,
            "positive_speakers": sum(r["CER_increase"] > 0 for r in effects),
            "speaker_effects": effects,
            "per_utterance_scores": scores,
            "training_anchor_metrics": evaluation["training"]["metrics"],
        }
    summary = {
        "completed": True,
        "raw_records_verified": raw,
        "results": results,
        "devices_metrics_identical": results["xpu_0"] == results["xpu_1"],
        "optimizer_updates": 0,
        "descriptive_not_confirmatory": True,
        "P100_authorized": False,
        "P960_authorized": False,
        "stage2_ready": False,
    }
    core.write(ROOT / "summary.json", summary)
    r = results["xpu_0"]
    ci = r["primary_speaker_macro_CER_difference"]
    lines = [
        "# RCLAT duration/character-matched grounding evaluation",
        "",
        "Frozen checkpoint; 384 native-length utterances, 192 disjoint reciprocal pairs, 24 unseen training speakers in exposed dev-other. Each pair meets both prospective duration and character-count max/min ratios <=1.10.",
        "",
        "| Pairing | Speaker-macro CER | Speaker-macro WER |",
        "|---|---:|---:|",
    ]
    for condition, m in r["metrics"].items():
        lines.append(
            f"| {condition} | {m['speaker_macro_cer']:.2%} | {m['speaker_macro_wer']:.2%} |"
        )
    lines += [
        "",
        f"Primary descriptive mismatched-minus-matched CER: **{ci['point_estimate'] * 100:.2f} percentage points**. Paired speaker-cluster percentile 95% bootstrap interval: **[{ci['percentile95'][0] * 100:.2f}, {ci['percentile95'][1] * 100:.2f}] points**; {r['positive_speakers']}/24 positive speaker contrasts.",
        "",
        f"Bootstrap: {ci['replicates']} replicates; NumPy default_rng seed {ci['seed']}; sample 24 speaker clusters with replacement, preserving all eight pairs per selected speaker; linear percentile quantiles. Both conditions share the same speaker draws. No utterance-independence assumption or significance-based progression rule.",
        "",
        "Original 64-utterance training anchors reproduced saved hypotheses exactly and logits within 1e-4 relative L2 on both devices. All 896 raw records, model-state preservation and frozen dependency/checkpoint hashes verified. No training or model selection occurred.",
        "",
        "Matching reduces two measurable confounds; it does not equate phonetic difficulty, language distributions, recording properties or all length effects. The interval quantifies clustered sampling variation for the selected development population and fixed seed/model, not independent confirmatory evidence, model-seed uncertainty or a causal phonetic mechanism. No final-test data accessed. Prior results remain closed; P100/P960 and MACDE–CAVEM Stage 2 remain blocked.",
        "",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    core.write(
        ROOT / "closeout.json",
        {
            "completed": True,
            "raw_records_verified": raw,
            "anchor_reproduction_passed_both": True,
            "optimizer_updates": 0,
            "devices_metrics_identical": summary["devices_metrics_identical"],
            "primary": ci,
            "P100_authorized": False,
            "P960_authorized": False,
            "stage2_ready": False,
        },
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
