"""Package immutable evidence for independent review;never performs inference."""

from pathlib import Path
import json
import zipfile
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha

REPO = Path.cwd().resolve()
SCI = Path("outputs/rclat_rtpe_vctk_zero_shot_execution_v1")
SPEC = Path("outputs/rclat_rtpe_vctk_transfer_protocol_v1")
OUT = Path("outputs/rclat_rtpe_vctk_review_bundle_v1")


def arcpath(path):
    resolved = Path(path).resolve()
    return "workspace/" + resolved.relative_to(REPO).as_posix()


def main():
    original = json.loads((SCI / "final_scientific_artifact_manifest.json").read_text())
    assert len(original) == 110 and all(sha(p) == h for p, h in original.items())
    protocol = json.loads((SPEC / "protocol.json").read_text())
    execution = json.loads((SCI / "execution_manifest.json").read_text())
    assert all(sha(p) == h for p, h in protocol["dependencies"].items()) and all(
        sha(p) == h for p, h in execution["dependencies"].items()
    )
    files = {arcpath(p): Path(p) for p in original}
    extra = [
        SCI / "final_scientific_artifact_manifest.json",
        SCI / "final_scientific_report_manifest.json",
        SPEC / "protocol.json",
        SPEC / "selected_inventory.json",
        SPEC / "sampling_summary.json",
        SPEC / "independent_sampling_audit.json",
        Path("scripts/rclat_rtpe_vctk_zero_shot_closeout.py"),
        Path("scripts/rclat_rtpe_vctk_zero_shot_closeout_v2.py"),
        Path("scripts/rclat_rtpe_vctk_zero_shot_readiness.py"),
    ]
    extra += [Path(p) for p in protocol["dependencies"]] + [
        Path(p) for p in execution["dependencies"]
    ]
    for model in protocol["models"]:
        for record in model["files"].values():
            assert sha(record["path"]) == record["sha256"]
            extra.append(Path(record["path"]))
    extra += [
        Path("src/bioacoustic_pipeline/__init__.py"),
        Path("src/bioacoustic_pipeline/diagnostics/__init__.py"),
        Path("src/bioacoustic_pipeline/representation/__init__.py"),
        Path("src/bioacoustic_pipeline/diagnostics/macde_data.py"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat.py"),
        Path("data/LibriSpeech/LICENSE.TXT"),
        Path("outputs/rclat_rtpe_vctk_preacquisition_v1/official_README.txt"),
        Path("outputs/rclat_rtpe_vctk_preacquisition_v1/official_license.txt"),
    ]
    for path in extra:
        if not path.is_file():
            raise FileNotFoundError(path)
        files[arcpath(path)] = path
    docs = {
        "SCIENTIFIC_RESULTS.md": SCI / "SCIENTIFIC_RESULTS.md",
        "independent_scientific_closeout.json": SCI / "independent_scientific_closeout.json",
        "scientific_result.json": SCI / "scientific_run/result.json",
        "SCORING_PRECISION_CORRECTION.md": SCI / "SCORING_PRECISION_CORRECTION.md",
        "final_scientific_artifact_manifest.json": SCI / "final_scientific_artifact_manifest.json",
        "final_sealing_record.json": SCI / "final_sealing_record.json",
        "protocol.json": SPEC / "protocol.json",
        "execution_manifest.json": SCI / "execution_manifest.json",
    }
    files.update(docs)
    files["verify_review_bundle.py"] = OUT / "verify_review_bundle.py"
    start = """# RTPE-ZS1 independent review package

Read SCIENTIFIC_RESULTS.md,independent_scientific_closeout.json andscientific_result.json first. The documents-only archive contains these records but cannot reproduce numerical scores without the prediction artifacts. The full archive includes all110files referenced by the original final scientific manifest,plus that manifest,the frozen protocol/sample,selected model/Ridge/scaler files andsupporting code/dependencies.

## Verify without extracting the full archive

Obtain verify_review_bundle.py from either archive (or the separately supplied file),then run:

    python verify_review_bundle.py --zip RTPE-ZS1_full_review.zip

Python's standard library handles hashes;NumPy>=1.22 handles scores (the pinned run used2.5.4). Use --hashes-only if NumPy is unavailable. Alternatively,extract the full archive and run python verify_review_bundle.py from its root. The verifier firstchecks bundle hashes and all110original scientific references,then recomputes per-origin/recording/speaker80msand160ms losses,10,000drawbootstrapintervals andthe all-five primary decision. It explicitly reproduces the frozen FP32subtraction/FP64reduction behaviour. It never unpickles Torch checkpoints,executes the pipeline,loads a model or performs inference.

## Evidence boundary

This verifies consistency of supplied bytes and numerical scores from saved predictions and targets. It does not independently prove who generated the predictions,authenticate an unsigned package,verify raw waveform provenance or replay model inference. Compare the original final-manifest SHA256 with your previously trusted copy. The archive's own SHA256 provides transfer integrity,not independent origin authentication.

The11.75GBraw VCTK ZIP is not included. Its official provenance and hashes are recorded in the source-support records;obtaining it requires its own licence/provenance handling. The full original repository and pinned runtime may still be needed for model-inference replay. This package does not perform or authorise that replay.

SCORING_PRECISION_CORRECTION.md documents the first strict host-audit mismatch and its resolution. final_sealing_record.json identifies the final manifest as authoritative;the retained provisional host manifest includes a console that finished after its snapshot. Those historical records are not silently rewritten.

Dataset attribution and licence files are retained. No scientific artifact was changed,new model inference run,or downstream progression authorised during packaging.
"""
    (OUT / "START_HERE.md").write_text(start)
    (OUT / "ATTRIBUTION.md").write_text("""# Dataset attribution

CSTR VCTK Corpus0.92:Yamagishi,Junichi;Veaux,Christophe;MacDonald,Kirsten(2019). University of Edinburgh,Centre for Speech Technology Research. DOI:https://doi.org/10.7488/ds/2645. Publisher licence:Creative Commons Attribution4.0;the original README andlicence are retained underworkspace/outputs/rclat_rtpe_vctk_preacquisition_v1/. Included derived material reflects193tapcausal48-to16kHzresampling,64Dlog-melfeatures,frozen training-scaler transformations andsaved forecasting outputs. No rawVCTKwaveform archive is included.

LibriSpeech copyright2014VassilPanayotov. The retained source licence is workspace/data/LibriSpeech/LICENSE.TXT (CCBY4.0). The included RTPE models,Ridge states,scalers andnumerical fixtures were developed using the documented LibriSpeech population.

These preserved dataset notices do not assign a new licence to repository code or third-party dependencies.
""")
    files["START_HERE.md"] = OUT / "START_HERE.md"
    files["ATTRIBUTION.md"] = OUT / "ATTRIBUTION.md"
    manifest = dict(
        format="RTPE-ZS1_PORTABLE_REVIEW_BUNDLE_V1",
        files={name: sha(p) for name, p in sorted(files.items())},
        original_final_manifest=arcpath(SCI / "final_scientific_artifact_manifest.json"),
        original_reference_map={p: arcpath(p) for p in original},
        scientific_protocol=arcpath(SPEC / "protocol.json"),
        scientific_run=arcpath(SCI / "scientific_run"),
        raw_waveform_archive_included=False,
        original_final_references_included=110,
    )
    (OUT / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    full = OUT / "RTPE-ZS1_full_review.zip"
    with zipfile.ZipFile(full, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for name, path in sorted(files.items()):
            z.write(path, name)
        z.write(OUT / "BUNDLE_MANIFEST.json", "BUNDLE_MANIFEST.json")
    small = OUT / "RTPE-ZS1_documents.zip"
    with zipfile.ZipFile(small, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, path in docs.items():
            z.write(path, name)
        for name in ["START_HERE.md", "ATTRIBUTION.md", "verify_review_bundle.py"]:
            z.write(OUT / name, name)
    checksums = {full.name: sha(full), small.name: sha(small)}
    (OUT / "SHA256SUMS.txt").write_text("".join(f"{h}  {name}\n" for name, h in checksums.items()))
    print(
        json.dumps(
            dict(
                original_manifest_entries_verified=110,
                bundle_files=len(files),
                full_bytes=full.stat().st_size,
                documents_bytes=small.stat().st_size,
                checksums=checksums,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
