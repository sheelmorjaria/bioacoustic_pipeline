"""Prepare independently defined synthetic symbol trajectories; no model execution."""

import hashlib
import json
from pathlib import Path
import numpy as np

OUT = Path("outputs/rclat_ctc_synthetic_diagnostic_v1")


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / "cohort.json").exists():
        raise RuntimeError("Fixture already prepared")
    rng = np.random.default_rng(20261022)
    seen = set()
    records = []
    # Unique symbolic messages; waveform-like feature codes intentionally explicit.
    for split, count in [("training", 64), ("validation", 32)]:
        for i in range(count):
            length = 4 + i % 9
            while True:
                symbols = tuple(map(int, rng.integers(0, 4, size=length)))
                if symbols not in seen:
                    break
            seen.add(symbols)
            frames = 24 * length
            values = rng.normal(0, 0.05, (frames, 64)).astype(np.float32)
            intervals = []
            for j, symbol in enumerate(symbols):
                start = 24 * j
                values[start : start + 16, symbol] += 3.0
                intervals.append(
                    dict(
                        symbol="ABCD"[symbol],
                        start_frame=start,
                        stop_frame_exclusive=start + 16,
                        blank_gap_start=start + 16,
                        blank_gap_stop_exclusive=start + 24,
                    )
                )
            target_ids = np.array([2 + s for s in symbols], dtype=np.int64)
            minimum = length + sum(a == b for a, b in zip(symbols, symbols[1:]))
            output_frames = (frames + 3) // 4
            assert output_frames >= minimum and np.isfinite(values).all()
            uid = f"{split}_{i:03}"
            path = OUT / "fixtures" / f"{uid}.npz"
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(
                path,
                features=values,
                valid=np.ones_like(values, dtype=bool),
                target_ids=target_ids,
                native_length=frames,
            )
            records.append(
                dict(
                    uid=uid,
                    split=split,
                    text="".join("ABCD"[s] for s in symbols),
                    target_length=length,
                    native_frames=frames,
                    ctc_frames=output_frames,
                    ctc_minimum_frames=minimum,
                    intervals=intervals,
                    fixture_path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
    write = dict(
        seed=20261022,
        training_count=64,
        validation_count=32,
        unique_messages=True,
        speakers="Not applicable;synthetic feature task,not speech",
        generation="64D FP32noiseNormal(0,.05);symbolA-D adds3toitsfirst4dimensions for16frames;8noise-onlygapframes;4..12symbols;allmasksvalid",
        records=records,
        model_execution=False,
    )
    with (OUT / "cohort.json").open("x") as f:
        json.dump(write, f, indent=2)
    print("Prepared96unique synthetic trajectories;no model or training executed.")


if __name__ == "__main__":
    main()
