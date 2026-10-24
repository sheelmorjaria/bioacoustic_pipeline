"""Locally freeze P10-C3; never starts training or scores validation."""

import copy
import datetime
import json
from pathlib import Path
import rclat_p10_c2_train as prior
import rclat_p10_c3_train as current


def main():
    spec, _ = prior.check()
    destination = current.OUT / "preregistration.json"
    if destination.exists():
        raise RuntimeError("C3 already frozen; no replacement")
    new = copy.deepcopy(spec)
    engineering = Path("outputs/rclat_p10_cache16384_fullworkload_engineering_v1")
    new.update(
        study="RCLAT P10-C3: fresh scientific development execution with requested cache capacity16384",
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose="Separate30epoch scientific learning-validation execution following passed12epoch engineering prerequisite;fresh start,no engineering-state reuse.",
        predecessor=dict(
            protocol=str(prior.OUT / "preregistration.json"),
            protocol_sha256=prior.original.protocol.sha(prior.OUT / "preregistration.json"),
            status="P10-C2 interrupted at step7751;closed originalP10 and all prior outcomes unchanged.",
            continuation_or_resume=False,
        ),
        engineering_results={
            arm: str(engineering / f"{arm}_result.json") for arm in ["mel", "combined"]
        },
        engineering_parity={
            arm: str(engineering / f"{arm}_parity.json") for arm in ["mel", "combined"]
        },
        engineering_interpretation="Both16384arms passed frozen12epoch/8448update gate with native<64 and RSS<128MiB/epoch slopes;historicalfailure point crossed;six real-data parity cases passed. Not30epoch stability or transcription-learning evidence. Internalcachemechanism/rootcause unresolved.",
        runtime_policy=dict(
            requested_ONEDNN_PRIMITIVE_CACHE_CAPACITY=16384,
            ONEDNN_VERBOSE="profile_create,filter=conv",
            legacy_DNNL_override="prohibited",
            numeric_capacity_readback=None,
            change="Single principal runtime intervention relative toC2;native telemetry retained;no model mathematics change",
        ),
        scientific_policy_changes=dict(
            relative_to="P10-C2",
            model_mathematics_changed=False,
            requested_primitive_cache_capacity=16384,
            exact_historical_fault_cause_established=False,
            data_targets_order_budget_validation_and_learning_gates_changed=False,
            added_native_allocation_telemetry=True,
        ),
        additional_development_exposure="OriginalP10/C1/C2 and engineering diagnostics used the unchanged source population. C1/C2 evaluated validation epochs1..11;engineering16384 validationforwarded all12epochs without decoding/scoring. Dev-other is exposed development validation,not untouched finalconfirmation. Original scientific settings and learning gates unchanged.",
        checkpoint_policy=dict(
            save="All30epoch model/optimizer/step/RNG checkpoints with protocol/source provenance;per-epoch predictions;select minimum speaker-macroCER,ties earliest,afterall30epochs;reportfinalepoch too",
            reuse="Fresh epoch0 only;no study/engineering checkpoint loaded;no automatic retry/resume;interruption leaves pairedgate unpassed;any continuation requires another prospective specification",
        ),
        resource_interpretation="Monitor native/RSS/device/runtime counters;engineeringbounds are prior prerequisites,not new outcome-driven scientific epoch selection or early stopping. Nonfinite/OOM/runtime failures halt unchanged.30epoch stability unestablished until completion.",
        scientific_execution_authorized_by_freeze=False,
        automatic_scientific_start=False,
        scientific_training_started=False,
        training_started=False,
        stage2_ready=False,
    )
    for key in [
        "selection",
        "frontends",
        "features",
        "architecture",
        "initialization",
        "training",
        "validation",
        "learning_gate",
        "source_population",
        "final_test",
        "restart_policy",
    ]:
        assert new[key] == spec[key], key
    dependencies = [
        Path(__file__),
        Path("scripts/rclat_p10_c3_train.py"),
        prior.OUT / "preregistration.json",
        engineering / "plan.json",
        engineering / "status_closeout.json",
        Path("outputs/rclat_p10_reduced_cache_intervention_v1/closeout.json"),
    ]
    for arm in ["mel", "combined"]:
        dependencies.extend(
            [
                engineering / f"{arm}_result.json",
                engineering / f"{arm}_parity.json",
                engineering / f"{arm}_initial_state.json",
            ]
        )
    new["dependencies"].update({str(p): prior.original.protocol.sha(p) for p in dependencies})
    prior.original.protocol.freeze(destination, new)
    current.check()
    print("P10-C3 locally frozen; training not started:", prior.original.protocol.sha(destination))


if __name__ == "__main__":
    main()
