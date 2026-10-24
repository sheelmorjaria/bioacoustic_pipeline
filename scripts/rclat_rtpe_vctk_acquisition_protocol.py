"""Pin acquisition/support policy only; no download, extraction or audio reads."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha, write

ROOT = Path("outputs/rclat_rtpe_vctk_acquisition_protocol_v1")


def main():
    corrective = Path("outputs/rclat_rtpe_vctk_filter_correction_v2")
    assert (
        json.loads((corrective / "audit.json").read_text())["status"]
        == "CORRECTIVE_FILTER_BOUNDED_ENGINEERING_PASSED"
    )
    for filename in ["manifest.json", "report_manifest.json"]:
        assert all(sha(p) == h for p, h in json.loads((corrective / filename).read_text()).items())
    paths = [
        Path(__file__),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_vctk_source_contract.py"),
        Path("tests/test_rclat_vctk_source_contract.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_causal_resample_v2.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_ctc.py"),
        corrective / "audit.json",
        corrective / "response_and_support.npz",
        Path("outputs/rclat_rtpe_vctk_horizon_policy_v1/policy.json"),
        Path("outputs/rclat_rtpe_vctk_filter_mic_audit_v1/plan.json"),
        Path("outputs/rclat_rtpe_vctk_preacquisition_v1/official_README.txt"),
        Path("outputs/rclat_rtpe_vctk_preacquisition_v1/official_license.txt"),
    ]
    ROOT.mkdir(exist_ok=False)
    policy = dict(
        status="LOCALLY_FROZEN_ACQUISITION_SUPPORT_POLICY_NOT_TRANSFER_EVALUATION_PROTOCOL",
        date=datetime.now(timezone.utc).isoformat(),
        source=dict(
            doi="10.7488/ds/2645",
            official_landing="https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928",
            archive_name="VCTK-Corpus-0.92.zip",
            published_md5="8a6ba2946b36fcbef0212cad601f4bfa",
            published_display_size="10.94GB",
            license="CC BY4.0;officialtextretained",
            attribution="Yamagishi,Veaux,MacDonald(2019),UniversityofEdinburghCSTR,DOI10.7488/ds/2645;retainnoticesanddocumentprocessing",
        ),
        acquisition=dict(
            destination="data/rclat_vctk/VCTK-Corpus-0.92.zip",
            official_url_resolution="Resolve archive link from pinned publisher landing;record finalURLandheaders;no mirror substitution",
            maximum_archive_bytes=20 * 2**30,
            minimum_free_bytes=80 * 2**30,
            download_policy="Oneattempt;preservepartialfailurewithoutautomatedretry/resume;streamMD5andSHA256;no extractionbeforeMD5match",
            overwrite=False,
        ),
        archive_support=dict(
            maximum_expanded_bytes=64 * 2**30,
            reject=[
                "absolute paths",
                "path traversal",
                "Windows separator/drive names",
                "symlinks",
                "duplicate normalized names",
                "expandedbudgetoverflow",
            ],
            microphone="mic2 only,no fallback",
            identity="speaker directory+filename must agree;event=speaker+utterance;mic1/mic2 grouped",
            selection="Metadata audit first;list actualsuffixmapping/missingchannels before decoding",
            transcripts="Optional;absence alone not exclusion",
            waveform="Selectedmic2 finite mono48000Hz;native framecount/sourcehash/PCMhash;no stereo downmix,ratefallback or repair",
            duplicates="Hash rawPCMwithrate/channelmetadata;within-speaker duplicates one event representative deterministically;cross-speaker duplicates flagged and withheld pending independent audit",
        ),
        preprocessing=dict(
            module="bioacoustic_pipeline.representation.rclat_causal_resample_v2",
            resample_function="causal_decimate_v2",
            feature_function="mel_features_v2",
            taps=193,
            cutoff_hz=7000,
            kaiser_beta=8.6,
            phase=0,
            delay_compensation=False,
            coefficient_artifact=str(corrective / "response_and_support.npz"),
            startup="first64outputssamplesunsupported;frame0masked;32framehistoryorigin>=32",
            primary_horizon=8,
            secondary_horizon=16,
            excluded_horizon=4,
            source_gaps_samples_48k={"8": 1730, "16": 5570},
        ),
        eligibility_audit=dict(
            scope="All mic2 metadata;decode actual selected files only after acquisition/integrity;no model invocation",
            supported_origins="Complete32framehistoryandobserved64Dmel support;both retained targets withinrecordingandobserved;finite observed values;source-supportdisjoint",
            report=[
                "archive/source/decodedhashes",
                "speaker/event/microphonecounts",
                "missingchannelcounts",
                "duplicategroups",
                "duration/rate/channeldistributions",
                "supportedorigin counts perrecording/speaker",
                "reasoncoded exclusions",
                "causaltiming and startup checks",
            ],
            scientific_sampling="Pendingseparatefreeze;do not select protectedoriginsyet",
            support_threshold="Report actual support;scientific minimumspeaker/recordingcounts tobe specified beforeevaluation,not assumedhere",
        ),
        prohibited=[
            "RTPEcheckpointinference",
            "Ridgefitting",
            "scaleradaptation",
            "checkpointselection",
            "forecastlosses",
            "scientificgateevaluation",
            "performance-drivenchannel/filterchoices",
        ],
        scientific_models="Allfive originalselected models remain frozen;noauditorcheckpointloadsneeded",
        no_download_performed=True,
        no_real_audio_processing=True,
        source_support_certified=False,
        scientific_inference_authorized=False,
        P100_authorized=False,
        P960_authorized=False,
        stage2_ready=False,
        dependencies={str(p): sha(p) for p in paths},
    )
    write(ROOT / "protocol.json", policy)
    free = shutil.disk_usage(".").free
    write(
        ROOT / "resource_preflight.json",
        dict(
            free_bytes=free,
            required_bytes=80 * 2**30,
            passed=free >= 80 * 2**30,
            scope="Point-in-time local capacity only;not promisedfutureavailability",
        ),
    )
    (ROOT / "README.md").write_text("""# VCTK acquisition/support protocol v1 — locally frozen

This version explicitly binds the successful 193-tap FIR v2, its saved coefficients, startup masks and renewed source-support calculations. It preserves mic2-only/no-fallback and 80 ms primary/160 ms secondary treatment;40 ms remains unscored. It does not rewrite the failed97-tap or previous horizon policies.

The protocol stages official archive acquisition/checksums, archive-member safety and microphone/event metadata, then finite mono48kHz waveform/duplicate/feature support auditing. These stages must not invoke forecasting models, fit scalers or compute predictive losses. Source-support reports will enumerate real counts and exclusion reasons rather than assuming publisher totals. No new protected origin inventory or scientific decision rule is certified here.

The point-in-time capacity preflight reserves80GiB headroom;actual future acquisition must recheck it. Archive/expanded budgets, paths, attribution and failure-preservation rules are pinned. Two TDD tests cover unsafe/duplicate/symlink entries and microphone event grouping without fallback. The preparation script provides no download or decoding command. No archive was acquired,extracted or decoded at this milestone.

Next implementation: a separately pinned acquisition/support runner, tested with synthetic archives/waveforms, that records receipts and enforces these boundaries. Real-data processing remains pending. After actual support is established,freeze origin sampling,speaker-level uncertainty and external-transfer decision criteria before any inference. Original RTPE-1/R1 results remain unchanged;P100/P960 and MACDE–CAVEM Stage2 remain blocked.
""")
    write(ROOT / "manifest.json", {str(p): sha(p) for p in ROOT.iterdir() if p.is_file()})
    print("Acquisition/support specification frozen;no download,decoding or inference.")


if __name__ == "__main__":
    main()
