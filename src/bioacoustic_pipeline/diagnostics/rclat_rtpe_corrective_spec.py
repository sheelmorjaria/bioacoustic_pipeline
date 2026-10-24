"""Scientific-successor specification only; provides no execution capability."""

import copy


def corrective_protocol(original, frozen_utc, predecessor_sha):
    required = {
        "seed_pairs",
        "model_budget_per_seed",
        "replication_criterion",
        "study",
        "status",
        "frozen_utc",
    }
    if not required <= original.keys():
        raise ValueError("Complete predecessor scientific specification required")
    result = copy.deepcopy(original)
    result.update(
        study="RTPE-R1 corrective WHERE-guard successor specification v2",
        status="LOCALLY_FROZEN_SPECIFICATION_ONLY_EXECUTION_NOT_AUTHORIZED",
        frozen_utc=frozen_utc,
    )
    result["correction"] = dict(
        predecessor_protocol="outputs/rclat_rtpe_multiseed_v1/protocol.json",
        predecessor_protocol_sha256=predecessor_sha,
        predecessor_outcome="PRE_FLIGHT_FAILED_SCIENTIFIC_CRITERION_UNEVALUATED",
        production_guard="WHERE",
        candidate_module="bioacoustic_pipeline.representation.rclat_rtpe_guarded",
        model_class="GuardedPoolTemporalPredictionEngine",
        loss_function="guarded_minibatch_mse",
        scope="Replace only observed-cell finite guards in input validation and weighted-loss validation;forecast and weighted-residual mathematics unchanged",
        inherited_changes_field="The inherited changes field describes the original seed-replication design;this correction is an additional execution-only guard substitution",
        scientific_execution_authorized=False,
        runner_status="PENDING_SEPARATE_BINDING_AND_VERIFICATION",
        required_before_execution=[
            "Dedicated successor runner bound to guarded model and guarded loss in numerical preflight,training and validation paths",
            "Independent scientific-math equivalence and evaluation-lock tests",
            "Pinned execution manifest and current source/environment verification",
            "Explicit scientific execution authorization",
        ],
        engineering_certificate="outputs/rclat_rtpe_guard_execution_v1/certificate.json",
        no_engineering_checkpoint_reuse=True,
        all_original_scientific_fields_preserved=True,
    )
    return result
