"""Metadata-only localisation of incomplete VCTK support audit; no audio decode."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import zipfile
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha, write
from bioacoustic_pipeline.diagnostics.rclat_vctk_source_contract import recording_identity

ROOT = Path("outputs/rclat_rtpe_vctk_identifier_diagnostic_v1")
EXEC = Path("outputs/rclat_rtpe_vctk_support_execution_v1")


def main():
    ROOT.mkdir(exist_ok=False)
    failure = json.loads((EXEC / "support" / "FAILURE.json").read_text())
    assert "Unexpected audio identifier" in failure["traceback"]
    assert not (EXEC / "support" / "waveform_journal.jsonl").exists()
    assert not (EXEC / "source_support_complete.json").exists()
    with zipfile.ZipFile("data/rclat_vctk/VCTK-Corpus-0.92.zip") as z:
        infos = [i for i in z.infolist() if not i.is_dir() and i.filename.endswith(".flac")]
        roster = z.read("speaker-info.txt")
        with (ROOT / "speaker-info.snapshot.txt").open("xb") as f:
            f.write(roster)
        roster_ids = {
            line.split()[0] for line in roster.decode("utf8").splitlines()[1:] if line.split()
        }
        bad = []
        microphones = Counter()
        speaker_mics = defaultdict(set)
        events = defaultdict(set)
        for info in infos:
            try:
                recording_identity(info.filename)
            except ValueError as e:
                bad.append(dict(member=info.filename, error=str(e)))
            match = re.fullmatch(r"((?:p\d+|s5))_(\d+)_(mic[12])\.flac", Path(info.filename).name)
            assert match is not None, "Further unclassified namespace;stop diagnostic"
            speaker, utterance, mic = match.groups()
            assert Path(info.filename).parent.name == speaker
            assert speaker in roster_ids, "Unrostered speaker;cannot assert identity support"
            microphones[mic] += 1
            speaker_mics[speaker].add(mic)
            events[f"{speaker}_{utterance}"].add(mic)
    assert len(bad) == 796 and all(Path(row["member"]).parent.name == "s5" for row in bad)
    write(ROOT / "unrecognized_members.json", bad)
    write(
        ROOT / "metadata_summary.json",
        dict(
            total_FLAC_members=len(infos),
            speaker_directories=len(speaker_mics),
            rostered_speaker_ids=len(roster_ids),
            microphone_files=dict(microphones),
            recording_events=len(events),
            events_without_mic2=sum("mic2" not in m for m in events.values()),
            speakers_without_mic2=sorted(s for s, m in speaker_mics.items() if "mic2" not in m),
            s5_explicitly_rostered=True,
            unsupported_names=len(bad),
            evidence_scope="ZIP central directory and speaker-info text only;not acoustic eligibility",
        ),
    )
    e = json.loads((EXEC / "execution_manifest.json").read_text())
    assert all(sha(p) == h for p, h in e["dependencies"].items())
    authorization = json.loads((EXEC / "support_authorization_record.json").read_text())
    assert (
        sha(EXEC / "SOURCE_AUTHORIZATION_ACQUISITION_ONLY.snapshot.json")
        == authorization["prior_sha256"]
        and sha(EXEC / "SOURCE_AUTHORIZATION.json") == authorization["current_sha256"]
    )
    audit = dict(
        status="SUPPORT_AUDIT_INCOMPLETE_IDENTIFIER_SCHEMA_MISMATCH_LOCALISED",
        failed_attempt=str(EXEC / "support"),
        original_failure_sha256=sha(EXEC / "support" / "FAILURE.json"),
        offending_speaker="s5",
        offending_FLAC_members=796,
        archive_FLAC_members=len(infos),
        microphone_files=dict(microphones),
        speaker_directories=len(speaker_mics),
        speaker_roster_support=True,
        waveform_decodes=0,
        acoustic_feature_computations=0,
        model_inference=False,
        predictive_scoring=False,
        automatic_retry=False,
        frozen_runner_changed=False,
        real_acoustic_eligibility="UNKNOWN",
        scientific_transfer="UNEVALUATED",
    )
    write(ROOT / "closeout.json", audit)
    (
        ROOT / "RESULTS.md"
    ).write_text("""# VCTK source audit — INCOMPLETE; identifier mismatch localised

The authorised source-only support attempt passed archive-byte integrity checks but stopped in metadata identifier validation before any waveform decoding. Its pinned parser accepts speaker IDs matching p followed by digits. The actual official archive also contains **s5**, explicitly identified in speaker-info.txt. All **796** unrecognised FLAC names belong to s5 (398 utterance events with two microphone versions).

A separately recorded metadata-only diagnostic inspected the ZIP central directory and speaker-info.txt. It found **88,328 FLAC members**, **44,455 mic1 files**, **43,873 mic2 files** and **110 speaker directories**. These are archive metadata counts,not independent recordings retained after acoustic eligibility or duplicate checks. No mic2 waveform was decoded. No feature/origin eligibility counts are established. The complete roster text and offending member list are preserved as diagnostic evidence. Cross-corpus human identity remains unresolved.

The failed support attempt retains STARTED,archive receipt,FAILURE and console records. There is no waveform journal,final support summary or source-support-complete marker. The failure was not retried and the frozen parser/runner was not modified. This is an ingestion schema failure,not evidence that VCTK speech is invalid or a negative forecasting result.

The user authorised this source-only support stage in response to the explicit approval request. The prior acquisition-only authorisation bytes were preserved in SOURCE_AUTHORIZATION_ACQUISITION_ONLY.snapshot.json before the mutable control record was extended. support_authorization_record.json binds both states. The old acquisition manifest remains a historical snapshot whose authorisation-file digest refers to the preserved prior bytes;this intentional authorisation supersession is not a scientific-artifact change.

Next correction: prospectively version the identifier contract using the verified corpus roster,cover s5 in regression tests and preserve strict directory/event consistency. Bind a corrected support runner/manifest and new source-only authorisation before a fresh attempt. Do not silently exclude s5,relax all identifiers indiscriminately or overwrite/resume the failed attempt.

No real waveform decode,acoustic-feature extraction,forecasting-model access or predictive scoring occurred. RTPE-1/R1 remain closed and positive. Real VCTK acoustic eligibility is unknown;independent-source transfer remains unevaluated. P100/P960 and MACDE–CAVEM Stage 2 remain blocked.
""")
    files = [
        *ROOT.iterdir(),
        EXEC / "support" / "FAILURE.json",
        EXEC / "support" / "STARTED.json",
        EXEC / "support" / "archive_receipt.json",
        EXEC / "support_console.log",
        EXEC / "SOURCE_AUTHORIZATION.json",
        EXEC / "SOURCE_AUTHORIZATION_ACQUISITION_ONLY.snapshot.json",
        EXEC / "support_authorization_record.json",
    ]
    write(ROOT / "manifest.json", {str(p): sha(p) for p in files if p.is_file()})
    write(
        ROOT / "report_manifest.json",
        {
            str(Path(__file__)): sha(__file__),
            str(ROOT / "manifest.json"): sha(ROOT / "manifest.json"),
        },
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
