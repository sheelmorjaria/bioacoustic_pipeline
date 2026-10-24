"""Freeze zero-shot specification and performance-blind metadata sample only."""

from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from bioacoustic_pipeline.diagnostics.rclat_vctk_transfer_sampling import select_inventory
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha, write

ROOT = Path("outputs/rclat_rtpe_vctk_transfer_protocol_v1")
SUPPORT = Path("outputs/rclat_rtpe_vctk_support_execution_v2")
SALT = "RCLAT-RTPE-VCTK-ZS1-20261024-record32-origin8-v1"


def main():
    for name in ["source_artifact_manifest.json", "support_closeout_manifest.json"]:
        assert all(sha(p) == h for p, h in json.loads((SUPPORT / name).read_text()).items())
    closeout = json.loads((SUPPORT / "support_closeout.json").read_text())
    assert closeout["status"] == "SOURCE_ONLY_WAVEFORM_AND_FEATURE_SUPPORT_AUDIT_COMPLETED_VERIFIED"
    rows = [
        json.loads(line) for line in (SUPPORT / "support/recordings.jsonl").read_text().splitlines()
    ]
    eligible = [row for row in rows if row["reason"] == "eligible"]
    speakers = sorted({row["speaker"] for row in eligible})
    assert len(speakers) == 108
    model_policy = Path("outputs/rclat_rtpe_vctk_horizon_policy_v1/policy.json")
    models = json.loads(model_policy.read_text())["models"]
    for model in models:
        for record in model["files"].values():
            assert sha(record["path"]) == record["sha256"]
    paths = [
        Path(__file__),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_vctk_transfer_sampling.py"),
        Path("tests/test_rclat_vctk_transfer_sampling.py"),
        SUPPORT / "support/recordings.jsonl",
        SUPPORT / "support/support_summary.json",
        SUPPORT / "support_closeout.json",
        SUPPORT / "source_runtime_context.json",
        model_policy,
        Path("outputs/rclat_rtpe_vctk_filter_correction_v2/audit.json"),
        Path("outputs/rclat_rtpe_vctk_filter_correction_v2/response_and_support.npz"),
        Path("src/bioacoustic_pipeline/representation/rclat_causal_resample_v2.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_ctc.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe_guarded.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe_pool.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe_weighted.py"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_vctk_horizons.py"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_rtpe_protocol.py"),
        Path("outputs/rclat_rtpe1_protocol_v2/training_numerical_fixtures.npz"),
    ]
    ROOT.mkdir(exist_ok=False)
    protocol = dict(
        study="RTPE-ZS1 VCTK finite-five-model zero-shot external-source forecasting",
        status="LOCALLY_FROZEN_NOT_EXTERNAL_REGISTRATION",
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        research_question="Does the previously replicated80ms causal-GRU advantage over frozenhistory-matchedRidge persist on independently collectedVCTK speech without adaptation?",
        population=dict(
            source="CSTR VCTK0.92 mic2 only",
            source_support_recordings=43873,
            source_support_speakers=108,
            speakers=speakers,
            global_cross_corpus_human_identity="Unresolved;corpus-local roster IDs only",
            exposure="Source support and preprocessing audited;no RTPE predictive outcomes on VCTK inspected",
        ),
        sample=dict(
            salt=SALT,
            recordings_per_speaker=32,
            origins_per_recording=8,
            speaker_requirement="All108verifiedsupported speakers must meet32recordingseachwith>=8verifiedorigins;otherwiseABSTAIN;no threshold relaxation",
            selection="SHA256 canonical JSON[salt,recording,speaker,event] ranking;origins use[salt,origin,speaker,event,index];ties by event/index;rankselect then sort IDs/origins",
            expected_recordings=3456,
            expected_origins=27648,
            justification="Retain allavailable speakerunits;32recordings per speaker bounds feature regeneration;8origins matches prior RTPE density. Not chosen using forecasting variance or power results",
            population_scope="Fixed selected recordings/origins;not whole12.5million-origin performance",
        ),
        models=models,
        models_policy="All five original LibriSpeech validation-selected checkpoints;seed17excluded;no VCTK model/seed/checkpointselection;no ensembling",
        ridge_policy="Each seed uses its frozen selectedRidge/scaler;verify identical scaler andRidge numerical states and bit-identical frozen comparator predictions;analyse as one shared deterministic baseline,notfiveindependentRidges",
        preprocessing=dict(
            module="bioacoustic_pipeline.representation.rclat_causal_resample_v2",
            function="mel_features_v2",
            input_hz=48000,
            output_hz=16000,
            FIR=dict(taps=193, cutoff_hz=7000, beta=8.6, causal=True, delay_compensated=False),
            coefficients="outputs/rclat_rtpe_vctk_filter_correction_v2/response_and_support.npz",
            features=64,
            history=32,
            scaling="Original per-seed LibriSpeech training scaler only;no VCTK fitting or per-recording amplitude/feature normalisation",
            startup="First64outputs unsupported;mel frame0masked;complete history origin>=32",
            sample_support="Reverify member/source/PCM/mel/mask/origin hashes before inference;invalid selected record/origin causes incomplete execution,not replacement",
            source_gap_ms={"80": 1730 / 48, "160": 5570 / 48},
            source_timing_scope="Distributed48kHzwaveform support only;not original live acquisition",
        ),
        endpoints=dict(
            primary_horizon_frames=8,
            secondary_horizon_frames=[16],
            unscored_horizon_frames=[4],
            metric="Per-origin mean squared error across observed standardised feature cells;equal origins withinrecording;equal recordings withinspeaker;equal speaker weights",
            mask_policy="Require nonempty finite observed cells for each retained horizon/origin;masked nonfinite ignored;40ms target/output never entersloss,uncertainty,modelselection or decision",
            contrast="Speaker-macro MSE_Ridge minus MSE_GRU;positivefavoursGRU",
        ),
        uncertainty=dict(
            unit="Speaker",
            replicates=10000,
            seed=2026102403,
            interval="Pairedspeakerpercentile95%,linearquantiles .025/.975;preserve eachspeaker32recordingsand8origins;shared drawindices across models",
            scope="Per-seed conditional fixed-model/fixed-recording/fixed-origin speaker intervals;no simultaneousfiveintervalcoverage or jointseed-speaker/trainingseedpopulationclaim",
        ),
        decision=dict(
            per_seed="PASS iff completed and primarypoint>0 and lower95bound>0;zero/equalitynotpass",
            all_five="PASS iff everyregisteredmodelcompletedandallfivepassed;allcompletewithanyseedfailure=>FAIL;incomplete=>UNEVALUATED",
            aggregate="Report equal-seedmeancontrasts,sampleSD/range and all per-speaker/modelcontrasts;descriptiveonly;notalternativeprimarygate",
            secondary="160ms scores/conditionalintervals descriptive;cannotrescue80ms gate",
            multiplicity="All-five rule;no selectedseed/horizon/ensemble resultsubstitution",
        ),
        execution=dict(
            training=False,
            optimizer_updates=0,
            adaptation=False,
            device="GRU XPU0 sequential;Ridge CPU float64;GRU FP32;no mixedprecision",
            GRU_batch_origins=128,
            Ridge_batch_origins=512,
            CPU_threads=2,
            OPENBLAS_NUM_THREADS=2,
            OMP_NUM_THREADS=2,
            ONEDNN_PRIMITIVE_CACHE_CAPACITY=16384,
            max_walltime_seconds=7200,
            peak_host_RSS_GiB=8,
            ordering="Freeze dataset/features and all models before prediction;save/hash all five prediction sets thenlock completion before primary orsecondaryscoring",
            numerical_preflight="Before VCTK predictions:all5models x3frozen originaltraining fixtures,CPU-XPUforecast/weightedloss/input-andparametergradient relative errors<1e-4;zero optimiserupdates;durable case records;no VCTK scores",
            preconditions="Dedicated tested inference runner,separatelypinnedexecutionmanifest,environment/source/weight-state/featureverification and explicit scientific execution authorisation",
        ),
        failure="No automaticretry,resume,budgetextension,recording/originreplacement,modeladaptation,seedselection,thresholdrelaxation,precisionchange or outcome-driven preprocessing;preserve everyfailure",
        interpretation="External-source zero-shot acousticforecasting on selected VCTK mic2population;not temporal-ordercausality,linguistic/semanticrecovery,animaltransfer or behaviouralutility",
        scientific_inference_started=False,
        scientific_execution_authorized=False,
        inference_runner_implemented=False,
        automatic_start=False,
        P100_authorized=False,
        P960_authorized=False,
        stage2_ready=False,
        dependencies={str(p): sha(p) for p in paths},
    )
    # Freeze rules before deriving the protected sample. No target/prediction arrays read.
    write(ROOT / "protocol.json", protocol)
    inventory = select_inventory(rows, salt=SALT, recordings=32, origins=8)
    assert len(inventory) == 3456 and sum(len(r["selected_origins"]) for r in inventory) == 27648
    assert sorted({r["speaker"] for r in inventory}) == speakers
    write(ROOT / "selected_inventory.json", inventory)
    per_speaker = {
        s: dict(
            recordings=sum(r["speaker"] == s for r in inventory),
            origins=sum(len(r["selected_origins"]) for r in inventory if r["speaker"] == s),
        )
        for s in speakers
    }
    assert all(v == dict(recordings=32, origins=256) for v in per_speaker.values())
    assert np.isclose(sum(r["recording_weight"] for r in inventory), 1)
    assert np.isclose(sum(r["origin_weight"] * len(r["selected_origins"]) for r in inventory), 1)
    summary = dict(
        completed=True,
        speakers=108,
        recordings=3456,
        origins=27648,
        recordings_per_speaker=32,
        origins_per_recording=8,
        origins_per_speaker=256,
        per_speaker=per_speaker,
        protocol_sha256=sha(ROOT / "protocol.json"),
        inventory_sha256=sha(ROOT / "selected_inventory.json"),
        source_fields="Metadata-only source identifiers/hashes and validity-derived origin ranges;no features/predictions read",
        model_inference=False,
        predictive_scores=False,
        scientific_execution_authorized=False,
    )
    write(ROOT / "sampling_summary.json", summary)
    write(
        ROOT / "preparation_manifest.json",
        dict(
            status="SPECIFICATION_AND_PROTECTED_SAMPLE_ONLY_NOT_EXECUTION_MANIFEST",
            protocol_sha256=sha(ROOT / "protocol.json"),
            inventory_sha256=sha(ROOT / "selected_inventory.json"),
            scientific_execution_authorized=False,
            inference_runner_pending=True,
        ),
    )
    (ROOT / "README.md").write_text("""# RTPE-ZS1 VCTK zero-shot protocol v1 — locally frozen

The specification and performance-blind protected sample are frozen before forecasting. All108supported VCTK speakers contribute32hash-selected recordings and8hash-selected origins each:**3,456recordings,27,648origins**. This preserves equal speaker/recording/origin weighting while bounding computation. No speaker was excluded by the32recording threshold (minimum available123). No outcome-based precision/power selection was performed. Sampling uses source metadata and verified origin hashes only;no waveform/feature or predictive array was accessed in preparation.

All five previously selected RTPE-R1 models,scalers and Ridge states are hash-bound. No training,adaptation,retuning,new checkpoint selection or ensembling is allowed. The existing193tapcausalFIR-v2 and mic2-only policies remain authoritative.80ms is primary;160ms is descriptive secondary;40ms is unscored and cannot influence the scientific decision. Nominal horizons and actual source-support gaps remain distinct.

Each model receives a paired10,000drawspeaker-bootstrap95%percentile interval,seed2026102403. Primary transfer criterion requires all five positive point contrasts and strictly positive lower bounds;allcompletewithanymiss isFAIL,incomplete isUNEVALUATED. Equal-seedmean/SD/range are descriptive,not alternative gates. Intervals condition on models,recordings andorigins;there is no jointseed-speaker or globalidentity claim.

The runtime is inference-only,GRU XPU0/FP32 and RidgeCPU/float64,with bounded batches,two CPUthreads,8GiBhostRSS and7,200seconds. Every selected source/feature/mask/model binding must be verified. All five prediction sets must be persisted and locked before scoring. The40ms output remains excluded even if its values differ or are nonfinite while retained horizons are valid. No unsupported selected record may be replaced.

**Pending:** dedicated inference runner and synthetic lock/score/weight tests,separate execution manifest,source/environment/state compatibility and numerical preflight,explicit scientific execution authorisation. This preparation manifest is not an execution manifest. No inference or predictive scoring occurred. Source support alone and this sample do not establish transfer. RTPE-1/R1 remain closed and positive;P100/P960 and MACDE–CAVEM Stage2 remain blocked.
""")
    write(ROOT / "manifest.json", {str(p): sha(p) for p in ROOT.iterdir() if p.is_file()})
    print("Zero-shot specification and protected metadata sample frozen;no inference or scores.")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_speaker"}, indent=2))


if __name__ == "__main__":
    main()
