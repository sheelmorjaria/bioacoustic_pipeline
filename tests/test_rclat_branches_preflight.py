import importlib
import json
from pathlib import Path
import sys


def test_preflight_command_reports_blockers_without_fitting(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    command = importlib.import_module("rclat_branches_preflight")
    monkeypatch.chdir(tmp_path)

    def write(path, value):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))

    write(
        "outputs/rclat_study_c_source_audit_stimuli.json",
        {
            "stimuli": [
                {
                    "stimulus_id": "s",
                    "source_filename": "s.wav",
                    "speaker_id": "v",
                    "text_id": "temperature",
                }
            ]
        },
    )
    write(
        "outputs/rclat_study_c_source_audit_human_judgments.json",
        {
            "model_generated_labels": False,
            "responses": [
                {
                    "source_filename": "s.wav",
                    "speaker_id": "v",
                    "text_id": "temp",
                    "listener_id": "l",
                    "perceived_request": 1,
                }
            ],
        },
    )
    base = Path("outputs/rclat_request_protocol")
    write(base / "build_manifest.json", {"source_files": {}, "code": {}})
    write(
        base / "custodian_only/wording_crosswalk.json",
        {"stimuli": [{"review_id": "r", "script_text": "my car"}]},
    )
    (base / "custodian_only/adjudication.csv").write_text("review_id,status\n")
    for name in ["reviewer_A", "reviewer_B"]:
        write(base / name / "metadata.json", {})
        (base / name / "review.csv").write_text("review_id,status,heard_text\nr,pending,\n")
    output = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["preflight", "--output", str(output)])
    command.main()
    r = json.loads((output / "readiness.json").read_text())
    assert r["models_fitted"] == 0 and not r["communication"]["ready"]
    assert r["communication"]["alias_response_counts"]["temp"] == 1
    assert r["communication"]["wording_unresolved"] == 1
