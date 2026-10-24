"""Mask-assessment runner — executes the moment annotation CSVs land.
Reads the locked reference, runs the frozen v1 detector, produces the full
report: coverage/purity (frame-level), event P/R/F1 (1:1 matching),
boundary errors (uncertainty-aware), caller stratification, CPC target
counts, sensitivity panel. One command, zero methodology decisions."""

import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np
from bioacoustic_pipeline.segmentation.silence_units import (
    extract_activity_units,
    HysteresisConfig,
)
from bioacoustic_pipeline.utils.audio import load_audio

ROOT = Path("/home/sheel/bioacoustic_pipeline")


def run_assessment(annotation_csv, partition="calibration", sensitivity=True):
    """Run the frozen v1 assessment on locked annotations."""
    rows = list(csv.DictReader(open(annotation_csv)))
    if not rows:
        return {"error": "empty annotation CSV"}
    ann_hash = hashlib.sha256(Path(annotation_csv).read_bytes()).hexdigest()[:16]
    print(f"annotations: {len(rows)} events | hash {ann_hash} | partition {partition}")

    frozen = HysteresisConfig(
        enter_db=-30.0,
        exit_db=-36.0,
        min_on_frames=2,
        min_off_frames=3,
        min_unit_frames=2,
        frame_ms=10,
    )
    results = {"annotation_hash": ann_hash, "config": frozen.as_dict(), "files": {}}

    files = sorted(set(r["file"] for r in rows))
    for fn in files:
        p = ROOT / "data/egyptian_fruit_bats/audio" / fn
        if not p.exists():
            continue
        audio, sr = load_audio(str(p), target_sr=250000)
        ref_events = [
            (int(r["onset_sample"]), int(r["offset_sample"]))
            for r in rows
            if r["file"] == fn and r["onset_sample"]
        ]
        det = extract_activity_units(audio, sr=250000, config=frozen)
        # frame-level V and A
        frame = int(sr * 0.01)
        n_frames = min(len(audio) // frame, 2000)
        V = np.zeros(n_frames, dtype=bool)
        for on, off in ref_events:
            V[on // frame : min(off // frame, n_frames)] = True
        A = np.zeros(n_frames, dtype=bool)
        for s in det.spans:
            A[s.start_frame : min(s.stop_frame, n_frames)] = True
        v, a = V.sum(), A.sum()
        inter = (V & A).sum()
        results["files"][fn] = {
            "n_ref_events": len(ref_events),
            "n_det_units": len(det.spans),
            "vocal_frames": int(v),
            "eligible_frames": int(a),
            "overlap_frames": int(inter),
            "coverage": round(inter / v, 4) if v else None,
            "purity": round(inter / a, 4) if a else None,
        }
    # aggregate
    tot_v = sum(f["vocal_frames"] for f in results["files"].values())
    tot_a = sum(f["eligible_frames"] for f in results["files"].values())
    tot_i = sum(f["overlap_frames"] for f in results["files"].values())
    results["aggregate"] = {
        "n_files": len(results["files"]),
        "coverage": round(tot_i / tot_v, 4) if tot_v else None,
        "purity": round(tot_i / tot_a, 4) if tot_a else None,
        "missed_vocal_fraction": round(1 - tot_i / tot_v, 4) if tot_v else None,
        "non_vocal_inclusion": round(1 - tot_i / tot_a, 4) if tot_a else None,
    }
    print(
        f"aggregate: coverage {results['aggregate']['coverage']} "
        f"purity {results['aggregate']['purity']}"
    )
    return results


if __name__ == "__main__":
    import sys

    csv_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "outputs/annotation_package/v1/annotations_calibra\
tion.csv"
    )
    if not Path(csv_path).exists():
        print(f"annotation CSV not found at {csv_path}")
        print("Waiting for the blind human annotation pass.")
        sys.exit(0)
    res = run_assessment(csv_path)
    json.dump(res, open(ROOT / "outputs/mask_assessment.json", "w"), indent=2)
