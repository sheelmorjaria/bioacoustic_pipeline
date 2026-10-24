"""Validate completed Study B provenance, common population, and saved predictions."""

import json
from pathlib import Path

import rclat_study_b as study


def main():
    spec = study.check()
    path = Path("outputs/rclat_study_b_results.json")
    results = json.loads(path.read_text())
    sources = json.loads(study.AUDIT.read_text())["records"]
    verified = []
    for arm, result in results["arms"].items():
        if len(result["training_loss_trace"]) != spec["training"]["epochs"]:
            raise ValueError("Incomplete training budget")
        if result["device"] != spec["device_assignment"][arm]:
            raise ValueError("Device assignment changed")
        if result["spec_sha256"] != study.prior.sha(study.SPEC):
            raise ValueError("Protocol mismatch")
        if (
            study.prior.sha(study.ROOT / result["checkpoint"]["path"])
            != result["checkpoint"]["sha256"]
        ):
            raise ValueError("Checkpoint changed")
        if not result["checkpoint_prediction_reload_passed"]:
            raise ValueError("Checkpoint reload did not pass")
        for split in ["validation", "evaluation"]:
            expected = {r["uid"]: r for r in sources if r["split"] == split}
            observed = result["evaluations"][split]["rows"]
            if len(observed) != len(expected) or {r["uid"] for r in observed} != set(expected):
                raise ValueError("Arm evaluation population differs")
            for row in observed:
                source = expected[row["uid"]]
                if source["transcript"] != row["reference"] or source["speaker"] != row["speaker"]:
                    raise ValueError("Reference or identity changed")
                if study.text_errors(row["reference"], row["prediction"]) != row["metrics"]:
                    raise ValueError("Saved transcript metrics mismatch")
        verified.append(arm)
    primary = results["primary_B0_minus_B2_macro_wer"]
    output = Path("outputs/rclat_study_b_closeout.json")
    study.prior.freeze_json(
        output,
        {
            "study": spec["study"],
            "protocol_sha256": study.prior.sha(study.SPEC),
            "results_sha256": study.prior.sha(path),
            "all_four_arms_completed_24_epochs": len(verified) == 4,
            "checkpoint_checksums_and_prediction_reloads_verified": verified,
            "common_source_population_and_transcript_metrics_verified": True,
            "primary_comparison": primary,
            "registered_primary_superiority_criterion_met": primary["ci95"][0] > 0,
            "previous_research_dependencies_unchanged": True,
            "post_evaluation_training_or_tuning_performed": False,
            "competitive_ASR_demonstrated": False,
            "descriptive_decoding_counts": {
                arm: {
                    "empty_hypotheses": sum(
                        not row["prediction"] for row in result["evaluations"]["evaluation"]["rows"]
                    ),
                    "evaluation_utterances": len(result["evaluations"]["evaluation"]["rows"]),
                }
                for arm, result in results["arms"].items()
            },
            "human_word_availability_validated": False,
            "semantic_reversibility_demonstrated": False,
            "stage2_ready": False,
            "scope": "Bounded English pilot: 136 training utterances, 8 evaluation speakers and 32 evaluation utterances. B3 is a supervised-pretrained offline reference; no population-wide language inference.",
        },
    )
    print("Completed study verified; primary criterion met:", primary["ci95"][0] > 0)


if __name__ == "__main__":
    main()
