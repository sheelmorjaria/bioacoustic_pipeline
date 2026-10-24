"""Freeze a separate fresh-start P10-C2 development execution; no fault-fix claim."""

import copy
import json
from pathlib import Path
import time

import rclat_p10_corrected_train as prior
import rclat_p10_c2_train as current


def main():
    spec, _ = prior.check()
    if (current.OUT / "preregistration.json").exists():
        raise RuntimeError("C2 already frozen")
    new = copy.deepcopy(spec)
    new.update(
        study="RCLAT P10-C2: fresh epoch-0 execution with resource telemetry",
        frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        purpose="New developmental execution after interrupted C1; fresh seed-0 initialization, not continuation or evidence that the backend failure is repaired.",
        predecessor=dict(
            protocol=str(prior.OUT / "preregistration.json"),
            protocol_sha256=prior.original.protocol.sha(prior.OUT / "preregistration.json"),
            status="Interrupted after 11 completed epochs; step 7751 forward primitive creation failed in both arms; learning gate not passed.",
            continuation_or_resume=False,
        ),
        scientific_policy_changes=dict(
            relative_to="P10-C1",
            model_mathematics_changed=False,
            backend_convolution_policy_changed=False,
            runtime_fault_fixed=False,
            added_resource_and_shape_telemetry=True,
            data_targets_order_budget_validation_and_learning_gates_changed=False,
        ),
        resource_telemetry=dict(
            training="Before forward at step1, every100steps, and every step7700..7800; per-batch shape/length/UID journal throughout.",
            validation="Per-batch shape/length/UID journal; resource snapshots at validation begin/end and after checkpoint save.",
            metrics="Host RSS/virtual/swap/threads,FD/map counts; XPU allocator current/peak counters; raw runtime free-memory reading (device1 previously unreliable); no cache occupancy claim.",
        ),
        additional_development_exposure="C1 validation evaluated after epochs1..11; this is a development re-execution on the same exposed validation population, not fresh confirmation. No architecture, threshold, learning-rate or epoch changes are made from those outcomes.",
        restart_policy="Fresh construction and optimizer at epoch0,step0. Initial-state hash must match the predecessor fresh seed0 state. No checkpoint loading, retry/resume, batch dropping or automatic replacement run.",
        engineering_interpretation="Prior bounded numerical checks and 7751-step reduced convolution-shape replay passed; historical primitive failure not reproduced and root cause unresolved. Full-run stability is unestablished.",
    )
    for key in (
        "selection",
        "frontends",
        "features",
        "architecture",
        "initialization",
        "training",
        "validation",
        "learning_gate",
        "source_population",
    ):
        assert new[key] == spec[key], key
    dependencies = [
        Path(__file__),
        Path("scripts/rclat_p10_c2_train.py"),
        Path("scripts/rclat_p10_c1_shape_replay.py"),
        prior.OUT / "preregistration.json",
        Path("outputs/rclat_p10_c1_shape_replay/closeout.json"),
        Path("outputs/rclat_p10_c1_convolution_diagnostic/closeout.json"),
    ]
    for arm in ("mel", "combined"):
        dependencies.extend(
            [prior.OUT / f"{arm}_FAILURE.json", prior.OUT / f"{arm}_initial_state.json"]
        )
    new["dependencies"].update({str(p): prior.original.protocol.sha(p) for p in dependencies})
    prior.original.protocol.freeze(current.OUT / "preregistration.json", new)
    current.check()
    print(
        "P10-C2 locally frozen:", prior.original.protocol.sha(current.OUT / "preregistration.json")
    )


if __name__ == "__main__":
    main()
