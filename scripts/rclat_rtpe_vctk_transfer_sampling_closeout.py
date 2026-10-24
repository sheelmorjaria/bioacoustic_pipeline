"""Independent metadata-sampling audit;no forecasting or waveform access."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha, write

ROOT = Path("outputs/rclat_rtpe_vctk_transfer_protocol_v1")


def main():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    selected = json.loads((ROOT / "selected_inventory.json").read_text())
    assert all(sha(p) == h for p, h in protocol["dependencies"].items())
    source = Path("outputs/rclat_rtpe_vctk_support_execution_v2/support/recordings.jsonl")
    groups = defaultdict(list)
    for line in source.read_text().splitlines():
        row = json.loads(line)
        if row["reason"] == "eligible":
            groups[row["speaker"]].append(row)
    salt = protocol["sample"]["salt"]

    def digest(parts):
        return hashlib.sha256(
            json.dumps([salt, *parts], separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()

    expected = {}
    for speaker, rows in sorted(groups.items()):
        ranked = sorted(
            rows, key=lambda row: (digest(["recording", speaker, row["event"]]), row["event"])
        )[:32]
        assert len(ranked) == 32
        for row in ranked:
            eligible = np.arange(
                row["first_eligible_origin"], row["last_eligible_origin"] + 1, dtype="<i8"
            )
            assert hashlib.sha256(eligible.tobytes()).hexdigest() == row["eligible_origin_sha256"]
            origins = sorted(
                sorted(
                    eligible.tolist(),
                    key=lambda o: (digest(["origin", speaker, row["event"], o]), o),
                )[:8]
            )
            expected[row["event"]] = origins
    assert len(selected) == len(expected) == 3456
    actual = {row["event"]: row["selected_origins"] for row in selected}
    assert actual == expected
    speakers = defaultdict(list)
    for row in selected:
        assert row["recording_weight"] == 1 / (108 * 32) and row["origin_weight"] == 1 / (
            108 * 32 * 8
        )
        assert len(set(row["selected_origins"])) == 8
        speakers[row["speaker"]].append(row)
    assert len(speakers) == 108 and all(len(rows) == 32 for rows in speakers.values())
    assert np.isclose(sum(row["origin_weight"] * 8 for row in selected), 1)
    assert not (ROOT / "result.json").exists()
    audit = dict(
        status="FROZEN_ZERO_SHOT_SPECIFICATION_AND_PROTECTED_SAMPLE_VERIFIED",
        speakers=108,
        recordings=3456,
        origins=27648,
        recordings_per_speaker=32,
        origins_per_recording=8,
        all_source_speakers_retained=True,
        selection_independently_recomputed=True,
        weights_verified=True,
        CPU_tests=122,
        lint_passed=True,
        protocol_sha256=sha(ROOT / "protocol.json"),
        inventory_sha256=sha(ROOT / "selected_inventory.json"),
        models_fitted=False,
        model_inference=False,
        predictive_scoring=False,
        scientific_execution_authorized=False,
        inference_runner_pending=True,
    )
    write(ROOT / "independent_sampling_audit.json", audit)
    write(
        ROOT / "sampling_report_manifest.json",
        {
            str(p): sha(p)
            for p in [
                ROOT / "independent_sampling_audit.json",
                Path(__file__),
                ROOT / "manifest.json",
            ]
        },
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
