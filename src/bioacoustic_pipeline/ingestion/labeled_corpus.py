"""Labeled-corpus ingestion: per-bout caller annotations + leakage-safe splits.

The Egyptian fruit bat corpus (Yovel lab; public at archive.org, details in
``egyptian_fruit_bats_202102``) annotates each WAV *bout* with emitter,
addressee, context and four pre/post-vocalization action codes. This module
parses that schema and produces the split machinery caller-ID evaluation
needs:

- **file-level splits** (a bout never spans splits — channel and session
  correlate within a recording);
- **caller-stratified** file assignment so every eligible caller trains;
- **held-out callers** for few-shot protocols, excluded from everything.

Split discipline is a statistical-honesty invariant, not a convenience: a
random row-level split of caller-ID data leaks recording conditions and
inflates every downstream number.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from bioacoustic_pipeline.pipeline.contracts import CorpusManifest

REQUIRED_COLUMNS = (
    "File Name",
    "Emitter",
    "Addressee",
    "Context",
    "Emitter pre-vocalization action",
    "Addressee pre-vocalization action",
    "Emitter post-vocalization action",
    "Addressee post-vocalization action",
)

_INT_FIELDS = (
    "Emitter",
    "Addressee",
    "Context",
    "Emitter pre-vocalization action",
    "Addressee pre-vocalization action",
    "Emitter post-vocalization action",
    "Addressee post-vocalization action",
)


@dataclass(frozen=True)
class CallAnnotation:
    """One annotated bout: emitter/addressee/context are integer codes.

    Interval contract (§45 addendum): the bat corpus is per-bout with no
    start/end columns (verified against the archive.org release), so
    ``start_ms``/``end_ms``/``recording_group`` are optional and default to
    whole-file / no-group. Interval-bearing corpora populate them; consumers
    resolve via :func:`interval_annotation`.
    """

    file_name: str
    emitter: int
    addressee: int
    context: int
    emit_pre: int
    addr_pre: int
    emit_post: int
    addr_post: int
    start_ms: float | None = None
    end_ms: float | None = None
    recording_group: str | None = None
    twin_id: str | None = None


@dataclass(frozen=True)
class IntervalView:
    """Resolved interval + recording group for window generation."""

    file_name: str
    start_ms: float
    end_ms: float
    recording_group: str


def interval_annotation(a: CallAnnotation, *, duration_ms: float) -> IntervalView:
    """Resolve an annotation to concrete bounds (defaults: whole file)."""
    return IntervalView(
        file_name=a.file_name,
        start_ms=0.0 if a.start_ms is None else float(a.start_ms),
        end_ms=duration_ms if a.end_ms is None else float(a.end_ms),
        recording_group=a.recording_group or a.file_name,
    )


def filename_block_group(file_name: str, *, block: int = 100) -> str:
    """Recording-group PROXY for the bat corpus: numeric file ids blocked.

    The corpus has no session column; consecutive numeric ids plausibly share
    recording conditions. This is a robustness-check grouping, not a claim —
    §45 reports block-grouped results alongside file-level ones.
    """
    stem = Path(file_name).stem
    return f"block_{int(stem) // block}" if stem.isdigit() else file_name


@dataclass
class HeldoutCallerSplit:
    """File split with whole callers reserved for few-shot evaluation."""

    heldout_callers: list[int]
    train_files: set = field(default_factory=set)
    val_files: set = field(default_factory=set)
    test_files: set = field(default_factory=set)


def parse_annotations_csv(path: str | Path) -> list[CallAnnotation]:
    """Parse the bat per-bout annotations CSV (strict column schema)."""
    path = Path(path)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty annotations CSV")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"{path}: missing required columns {missing}; found "
                f"{reader.fieldnames} — refusing to guess the schema"
            )
        out: list[CallAnnotation] = []
        for lineno, row in enumerate(reader, start=2):
            try:
                ints = {c: int(row[c]) for c in _INT_FIELDS}
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{lineno}: non-integer label in {exc} — row={row.get('File Name')!r}"
                ) from exc
            out.append(
                CallAnnotation(
                    file_name=row["File Name"].strip(),
                    emitter=ints["Emitter"],
                    addressee=ints["Addressee"],
                    context=ints["Context"],
                    emit_pre=ints["Emitter pre-vocalization action"],
                    addr_pre=ints["Addressee pre-vocalization action"],
                    emit_post=ints["Emitter post-vocalization action"],
                    addr_post=ints["Addressee post-vocalization action"],
                )
            )
    if not out:
        raise ValueError(f"{path}: no annotation rows")
    return out


def summarize_callers(annotations: list[CallAnnotation]) -> dict:
    """Per-caller file counts + context inventory (reported, never assumed)."""
    per_caller: dict[int, set] = {}
    contexts: set = set()
    for a in annotations:
        per_caller.setdefault(a.emitter, set()).add(a.file_name)
        contexts.add(a.context)
    counts = sorted(len(v) for v in per_caller.values())
    return {
        "n_files": len({a.file_name for a in annotations}),
        "n_callers": len(per_caller),
        "files_per_caller": {
            "min": counts[0],
            "median": counts[len(counts) // 2],
            "max": counts[-1],
        },
        "n_contexts": len(contexts),
    }


def split_files_by_caller(
    annotations: list[CallAnnotation],
    *,
    seed: int = 0,
    train: float = 0.70,
    val: float = 0.15,
    min_files_per_caller: int = 5,
) -> dict[str, set[str]]:
    """Caller-stratified file split: train/val/test keys -> file-name sets.

    Each caller's files are shuffled once and cut into the requested
    proportions, so every caller with enough files contributes to all three
    splits and no file lands in two of them.
    """
    import random

    if train + val >= 1.0:
        raise ValueError("train + val must be below 1.0")
    per_caller: dict[int, list[str]] = {}
    for a in annotations:
        files = per_caller.setdefault(a.emitter, [])
        if a.file_name not in files:
            files.append(a.file_name)
    rng = random.Random(seed)
    out: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for caller, files in sorted(per_caller.items()):
        if len(files) < min_files_per_caller:
            # too thin to stratify: everything into test (never training)
            out["test"].update(files)
            continue
        files = list(files)
        rng.shuffle(files)
        n = len(files)
        n_train = max(1, round(n * train))
        n_val = max(1, round(n * val))
        out["train"].update(files[:n_train])
        out["val"].update(files[n_train : n_train + n_val])
        out["test"].update(files[n_train + n_val :])
    return out


def holdout_callers(
    annotations: list[CallAnnotation],
    *,
    n_heldout: int = 10,
    min_files: int = 20,
    seed: int = 0,
) -> HeldoutCallerSplit:
    """Reserve whole callers (>=``min_files`` bouts) for few-shot evaluation."""
    import random

    per_caller: dict[int, list[str]] = {}
    for a in annotations:
        files = per_caller.setdefault(a.emitter, [])
        if a.file_name not in files:
            files.append(a.file_name)
    eligible = sorted(c for c, fs in per_caller.items() if len(fs) >= min_files)
    if len(eligible) < n_heldout + 2:
        raise ValueError(
            f"holdout_callers needs >= n_heldout+2 callers with >= {min_files} "
            f"files; eligible={len(eligible)}"
        )
    rng = random.Random(seed)
    rng.shuffle(eligible)
    heldout = sorted(eligible[:n_heldout])
    remaining = [a for a in annotations if a.emitter not in set(heldout)]
    strat = split_files_by_caller(remaining, seed=seed)
    return HeldoutCallerSplit(
        heldout_callers=heldout,
        train_files=strat["train"],
        val_files=strat["val"],
        test_files=strat["test"],
    )


def build_labeled_manifest(
    audio_dir: str | Path,
    annotations_csv: str | Path,
    species: str = "bat",
    dataset: str = "egyptian_fruit_bats",
) -> CorpusManifest:
    """Populate the (previously vacant) CorpusManifest from a labeled corpus.

    Keeps only annotation rows whose audio exists on disk, so manifest counts
    are honest about what is actually loadable.
    """
    audio_dir = Path(audio_dir)
    annotations = parse_annotations_csv(annotations_csv)
    paths = []
    for a in annotations:
        p = audio_dir / a.file_name
        if p.exists():
            paths.append(str(p))
    return CorpusManifest(
        dataset=dataset,
        dataset_role=f"labeled_per_bout[{species}]",
        audio_paths=paths,
        annotations=str(annotations_csv),
    )


def annotation_metadata_fn(annotations: list[CallAnnotation]):
    """``file basename -> {emitter, addressee, context}`` closure for the
    orchestrator's metadata seam; unknown files return None."""

    lookup = {a.file_name: a for a in annotations}

    def _fn(file_name: str):
        a = lookup.get(Path(file_name).name)
        if a is None:
            return None
        return {"emitter": a.emitter, "addressee": a.addressee, "context": a.context}

    return _fn


def split_recording_groups_by_caller(
    annotations: list[CallAnnotation],
    *,
    seed: int = 0,
    train: float = 0.70,
    val: float = 0.15,
) -> dict[str, list[CallAnnotation]]:
    """Caller-stratified split at the RECORDING-GROUP level.

    Same shape as :func:`split_files_by_caller` but the unit of assignment is
    the recording group (session/channel), so bouts recorded together never
    land in two splits — the channel-condition leakage guard. Callers with
    too few groups to stratify go wholly to test (never train).
    """
    import random

    if train + val >= 1.0:
        raise ValueError("train + val must be below 1.0")
    groups_by_caller: dict[int, list[str]] = {}
    for a in annotations:
        g = a.recording_group or a.file_name
        if g not in groups_by_caller.setdefault(a.emitter, []):
            groups_by_caller[a.emitter].append(g)
    rng = random.Random(seed)
    assign: dict[str, str] = {}
    for caller, groups in sorted(groups_by_caller.items()):
        groups = list(groups)
        rng.shuffle(groups)
        n = len(groups)
        n_train = max(1, round(n * train))
        n_val = max(1, round(n * val)) if n >= 3 else 0
        if n < 3:  # too few groups to stratify: everything to test
            assign.update({g: "test" for g in groups})
            continue
        assign.update({g: "train" for g in groups[:n_train]})
        assign.update({g: "val" for g in groups[n_train : n_train + n_val]})
        assign.update({g: "test" for g in groups[n_train + n_val :]})
    out: dict[str, list[CallAnnotation]] = {"train": [], "val": [], "test": []}
    for a in annotations:
        out[assign[a.recording_group or a.file_name]].append(a)
    return out


def parse_marmoset_labels_csv(
    path: str | Path, *, audio_prefix: str = "data/"
) -> list[CallAnnotation]:
    """Parse the InfantMarmosetsVox labels.csv (§46 P1).

    Schema: ``path,start,end,duration,calltypeID,callerID,twinID,vocID`` with
    start/end in SECONDS and intervals from BOTH twins annotated inside one
    shared recording (one mic per file — so the recording, not the interval,
    is the leakage unit; see the dataset's file-id semantics). callerID is
    globally unique (0-9 over 5 twin pairs) and maps to ``emitter``;
    calltypeID maps to ``context``; twinID is preserved on ``twin_id``.
    """
    out: list[CallAnnotation] = []
    with open(Path(path), newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        required = {"path", "start", "end", "calltypeID", "callerID", "twinID"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for lineno, row in enumerate(reader, start=2):
            try:
                start_s, end_s = float(row["start"]), float(row["end"])
                rel = row["path"]
                if rel.startswith(audio_prefix):
                    rel = rel[len(audio_prefix):]
                out.append(
                    CallAnnotation(
                        file_name=rel,
                        emitter=int(row["callerID"]),
                        addressee=0,
                        context=int(row["calltypeID"]),
                        emit_pre=0,
                        addr_pre=0,
                        emit_post=0,
                        addr_post=0,
                        start_ms=start_s * 1000.0,
                        end_ms=end_s * 1000.0,
                        recording_group=rel,
                        twin_id=str(row["twinID"]),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: bad row ({exc})") from exc
    if not out:
        raise ValueError(f"{path}: no rows")
    return out
