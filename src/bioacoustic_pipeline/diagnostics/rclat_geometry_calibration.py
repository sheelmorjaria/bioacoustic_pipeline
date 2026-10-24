"""Synthetic responses on a fixed, identity-only listener–stimulus graph."""

import numpy as np
from .rclat_bootstrap_calibration import GRID, _predictions, _sigmoid


def extract_geometry(stimuli, responses):
    keys = ("stimulus_id", "source_filename", "speaker_id", "text_id")
    inventory = sorted([{k: s[k] for k in keys} for s in stimuli], key=lambda s: s["stimulus_id"])
    names = {s["source_filename"] for s in inventory}
    if len(names) != len(inventory) or len({s["stimulus_id"] for s in inventory}) != len(inventory):
        raise ValueError("Duplicate stimulus")
    edges = sorted(
        [
            {"source_filename": r["source_filename"], "listener_id": r["listener_id"]}
            for r in responses
        ],
        key=lambda r: (r["source_filename"], r["listener_id"]),
    )
    if len({(e["source_filename"], e["listener_id"]) for e in edges}) != len(edges):
        raise ValueError("Duplicate edge")
    if {e["source_filename"] for e in edges} != names:
        raise ValueError("Unknown or unsupported stimulus")
    for speaker in {s["speaker_id"] for s in inventory}:
        n = sum(s["speaker_id"] == speaker for s in inventory)
        if n % 4:
            raise ValueError("Balanced four-point grid requires multiples of four")
    return {"stimuli": inventory, "edges": edges}


def simulate_geometry(geometry, scenario, *, seed):
    if min(scenario["speaker_sd"], scenario["listener_sd"]) < 0:
        raise ValueError("Negative scale")
    rng = np.random.default_rng(seed)
    stimuli = geometry["stimuli"]
    speakers = sorted({s["speaker_id"] for s in stimuli})
    listeners = sorted({e["listener_id"] for e in geometry["edges"]})
    se = dict(zip(speakers, rng.normal(0, scenario["speaker_sd"], len(speakers))))
    le = dict(zip(listeners, rng.normal(0, scenario["listener_sd"], len(listeners))))
    eta = {}
    for speaker in speakers:
        subset = [s for s in stimuli if s["speaker_id"] == speaker]
        eta.update({s["source_filename"]: GRID[i % 4] for i, s in enumerate(subset)})
    by_name = {s["source_filename"]: s for s in stimuli}
    c1, c2 = {}, {}
    for s in stimuli:
        a, b = _predictions(np.array([eta[s["source_filename"]]]), scenario["alternative"])
        c1[s["stimulus_id"]], c2[s["stimulus_id"]] = float(a[0]), float(b[0])
    responses = []
    for edge in geometry["edges"]:
        name, listener = edge["source_filename"], edge["listener_id"]
        p = _sigmoid(eta[name] + se[by_name[name]["speaker_id"]] + le[listener])
        responses.append({**edge, "perceived_request": int(rng.binomial(1, p))})
    return dict(stimuli=stimuli, responses=responses, c1=c1, c2=c2)
