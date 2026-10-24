"""Build a read-only HTML word-diff report from archived ASR text predictions."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import zipfile
from bioacoustic_pipeline.diagnostics.rclat_transcript_report import (
    align_words,
    word_metrics,
    render_html,
)

ROOT = Path("outputs/rclat_transcript_comparison_v1")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def main():
    ROOT.mkdir(exist_ok=False)
    sources = {}
    records = []
    seen = set()

    def load(path):
        path = Path(path)
        sources[str(path)] = sha(path)
        return json.loads(path.read_text())

    def add(
        *,
        uid,
        speaker,
        reference,
        hypothesis,
        experiment,
        condition,
        condition_label,
        arm,
        split,
        source,
        location,
        saved_errors,
        saved_words,
        saved_wer=None,
        saved_cer=None,
        prefix=1.0,
    ):
        if not isinstance(reference, str) or not isinstance(hypothesis, str):
            raise ValueError("Actual source/reference and hypothesis strings required")
        alignment = align_words(reference, hypothesis)
        metrics = word_metrics(alignment)
        if not metrics["reference_words"]:
            raise ValueError("Reference has no supported words")
        if metrics["errors"] != saved_errors or metrics["reference_words"] != saved_words:
            raise ValueError(f"Archived word totals mismatch:{source}:{location}")
        if saved_wer is not None and not math.isclose(
            metrics["wer"], saved_wer, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("Archived WER mismatch")
        key = f"{experiment}/{arm}/{split}/{uid}/{condition}/{prefix}"
        if key in seen:
            raise ValueError("Duplicate saved comparison record")
        seen.add(key)
        records.append(
            dict(
                id=key,
                uid=str(uid),
                speaker=str(speaker),
                reference=reference,
                hypothesis=hypothesis,
                experiment=experiment,
                condition=condition,
                condition_label=condition_label,
                arm=arm,
                split=split,
                source=str(source),
                source_location=location,
                saved_cer=saved_cer,
                prefix_fraction=prefix,
                alignment=alignment,
                **metrics,
            )
        )

    ap = "outputs/rclat_study_a_preregistration.json"
    ar = "outputs/rclat_study_a_results.json"
    refs = {r["utterance_id"]: r["transcript"] for r in load(ap)["sources"]}
    labels = {
        "intact": "Intact",
        "lossless": "Lossless codec",
        "block_shuffle": "500 ms block shuffle (untapered)",
        "silence": "Silence",
    }
    for i, r in enumerate(load(ar)["rows"]):
        m = r["metrics"]
        add(
            uid=r["utterance_id"],
            speaker=r["speaker_id"],
            reference=refs[r["utterance_id"]],
            hypothesis=r["decoded_text"],
            experiment="Study A",
            condition=r["condition"],
            condition_label=labels[r["condition"]],
            arm="Frozen pretrained recogniser",
            split="pilot",
            source=ar,
            location=f"rows[{i}]",
            saved_errors=m["word_errors"],
            saved_words=m["reference_words"],
            saved_wer=m["wer"],
            saved_cer=m.get("cer"),
            prefix=r["fraction"],
        )
    tp = "outputs/rclat_temporal_control_preregistration.json"
    tr = "outputs/rclat_temporal_control_results.json"
    refs = {r["utterance_id"]: r["transcript"] for r in load(tp)["sources"]}
    labels = {
        "intact": "Intact",
        "lossless": "Lossless codec",
        "taper_sham": "Taper-only control",
        "global": "500 ms group reorder (tapered)",
        "local": "20 ms reorder within 500 ms groups",
    }
    for i, r in enumerate(load(tr)["rows"]):
        m = r["metrics"]
        add(
            uid=r["utterance_id"],
            speaker=r["speaker_id"],
            reference=refs[r["utterance_id"]],
            hypothesis=r["decoded_text"],
            experiment="Temporal control",
            condition=r["condition"],
            condition_label=labels[r["condition"]],
            arm="Frozen pretrained recogniser",
            split=r["cohort"],
            source=tr,
            location=f"rows[{i}]",
            saved_errors=m["word_errors"],
            saved_words=m["reference_words"],
            saved_wer=m["wer"],
            saved_cer=m.get("cer"),
        )
    bp = load("outputs/rclat_study_b_preregistration.json")
    for arm in ["B0", "B1", "B2", "B3"]:
        source = f"outputs/rclat_study_b_{arm}_results.json"
        data = load(source)
        features = bp["arms"][arm][0]
        label = f"{arm} · {features}" + (" · from scratch" if arm != "B3" else " encoder")
        for split, values in data["evaluations"].items():
            for i, r in enumerate(values["rows"]):
                m = r["metrics"]
                add(
                    uid=r["uid"],
                    speaker=r["speaker"],
                    reference=r["reference"],
                    hypothesis=r["prediction"],
                    experiment="Study B",
                    condition=arm,
                    condition_label=label,
                    arm=label,
                    split=split,
                    source=source,
                    location=f"evaluations.{split}.rows[{i}]",
                    saved_errors=m["word_errors"],
                    saved_words=m["reference_words"],
                    saved_wer=m["wer"],
                    saved_cer=m.get("cer"),
                )
    base = Path("outputs/rclat_bilstm_p10_v1/xpu_0")
    result = load(base / "result.json")
    for policy in ["selected", "final"]:
        epoch = result[policy]["epoch"]
        source = base / f"epoch_{epoch:02d}.json"
        data = load(source)
        for i, r in enumerate(data["predictions"]):
            add(
                uid=r["uid"],
                speaker=r["speaker"],
                reference=r["reference"],
                hypothesis=r["hypothesis"],
                experiment="BiLSTM-P10-v1",
                condition=f"{policy}_epoch_{epoch}",
                condition_label=f"{policy.capitalize()} checkpoint · epoch {epoch}",
                arm="Conv–BiLSTM · log-mel",
                split="validation",
                source=source,
                location=f"predictions[{i}]",
                saved_errors=r["word_errors"],
                saved_words=r["reference_words"],
                saved_cer=r["character_errors"] / max(1, r["reference_characters"]),
            )
    metadata = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        records=len(records),
        experiment_counts=dict(Counter(r["experiment"] for r in records)),
        distinct_utterance_ids=len({r["uid"] for r in records}),
        word_error_totals_reproduced=len(records),
        source_files=sources,
        normalisation="Archived English RCLAT rule:uppercase,A-Z/apostrophes/spaces only;collapse whitespace",
        alignment="Minimum-edit word alignment;ties S,D,I;total errors match archived metrics",
        prefix_policy="Prefix rows scored against full reference,hidden by default;not ordinary full-utteranceWER",
        device_replication="BiLSTM XPU0 only;XPU1same-model replication not duplicated",
        scope="Earlier saved ASR experiments only;RTPE predictions are acoustic features,not transcriptions",
        training_or_inference_performed=False,
        source_audio_read=False,
        scientific_study_outcomes_changed=False,
    )
    for path, h in sources.items():
        assert sha(path) == h
    write(ROOT / "metadata.json", metadata)
    with (ROOT / "records.json").open("x") as f:
        json.dump(records, f, ensure_ascii=True, separators=(",", ":"))
    with (ROOT / "index.html").open("x") as f:
        f.write(render_html(records, metadata))
    (ROOT / "README.md").write_text("""# RCLAT saved-ASR transcript comparison

Open index.html locally in a browser. It is self-contained and makes no network requests. Start with the paired temporal-controls button to view an exact utterance under intact,lossless,taper-only,500msgroup-reordered and20mswithin-group-reordered conditions. Use model/experiment/condition/speaker filters,word-length andWER ranges,sorting,pagination orfilteredJSONexport. Each card shows actual reference/hypothesis text,minimum-edit word alignment,S/D/I counts andper-utteranceWER. Empty model outputs remain visible.

Coverage includes StudyA,its saved prefixes,the separate temporal-control study,StudyB armsB0–B3 andBiLSTM-P10-v1selected/final checkpoints onXPU0. Exact utterance IDs are the only cross-record join. Different experiments can use different populations andprotocols;do not treat unlike utterances or historical models as matched superiority comparisons.

The displayed word edit totals were recomputed andmatched everyarchived row. Words follow the archived English normalisation. Alignment ties may change S/D/I assignments relative to another minimum-edit path without changing totalerrors. Filtered summaries are descriptive pooled row scores,not replacements for registered speaker-macro endpoints;repeated conditions/checkpoints share utterances.

Prefix rows are hidden bydefault because they compare partial audio with the FULL transcript,including future words. They cannot be interpreted as ordinary complete-utterance recognition WER or independent word-timing measurements. RTPE forecasting does not supply transcriptions andis excluded. No audio,model checkpoint,training ornewinference was used. Allclosedstudy outcomes andprogression restrictions remain unchanged.

metadata.json lists source-fileSHA256receipts;records.json contains the complete aligned text data. No saved predictions,references orstudy artifacts were modified.
""")
    files = [
        *ROOT.iterdir(),
        Path(__file__),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_transcript_report.py"),
        Path("scripts/assets/rclat_transcript_report.html"),
        Path("tests/test_rclat_transcript_report.py"),
    ]
    write(ROOT / "manifest.json", {str(p): sha(p) for p in files if p.is_file()})
    with zipfile.ZipFile(
        ROOT / "rclat_transcript_report.zip", "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as z:
        for name in ["index.html", "README.md", "metadata.json", "records.json", "manifest.json"]:
            z.write(ROOT / name, name)
    print(json.dumps({k: v for k, v in metadata.items() if k != "source_files"}, indent=2))


if __name__ == "__main__":
    main()
