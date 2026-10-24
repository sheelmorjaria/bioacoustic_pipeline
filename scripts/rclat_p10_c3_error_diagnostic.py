"""Bounded retrospective saved-artifact analysis; no model loading/inference."""

import ctypes
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from bioacoustic_pipeline.diagnostics.rclat import normalize_text

OUT = Path("outputs/rclat_p10_c3_error_diagnostic_v1")
SOURCE = Path("outputs/rclat_p10_corrected_v3")


def main():
    plan = json.loads((OUT / "plan.json").read_text())
    for path, digest in plan["dependencies"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("Diagnostic dependency changed: " + path)
    if (OUT / "analysis.json").exists():
        raise RuntimeError("Existing analysis; no overwrite")
    native = ctypes.CDLL(str((OUT / "edit_decomposition.so").resolve()))
    ptr = ctypes.POINTER(ctypes.c_int)
    native.decompose.argtypes = [ptr, ctypes.c_int, ptr, ctypes.c_int, ptr]
    native.decompose.restype = None

    def align(ref, hyp):
        vocab = {v: i for i, v in enumerate(dict.fromkeys([*ref, *hyp]))}
        a = np.array([vocab[v] for v in ref], dtype=np.int32)
        b = np.array([vocab[v] for v in hyp], dtype=np.int32)
        result = np.empty(5, dtype=np.int32)
        native.decompose(
            a.ctypes.data_as(ptr), len(a), b.ctypes.data_as(ptr), len(b), result.ctypes.data_as(ptr)
        )
        distance, s, de, ins, match = map(int, result)
        assert (
            distance == s + de + ins and match + s + de == len(ref) and match + s + ins == len(hyp)
        )
        return dict(
            errors=distance,
            substitutions=s,
            deletions=de,
            insertions=ins,
            matches=match,
            reference_length=len(ref),
            hypothesis_length=len(hyp),
        )

    for ref, hyp, expected in [
        ("AB", "AB", [0, 0, 0]),
        ("AB", "AC", [1, 0, 0]),
        ("AB", "A", [0, 1, 0]),
        ("A", "AB", [0, 0, 1]),
        ("", "A", [0, 0, 1]),
        ("A", "", [0, 1, 0]),
        ("AB", "BA", [2, 0, 0]),
    ]:
        v = align(ref, hyp)
        assert [v[k] for k in ["substitutions", "deletions", "insertions"]] == expected
    cohort = json.loads(Path("outputs/rclat_p10/cohort.json").read_text())
    refs = {r["uid"]: r for r in cohort["selected_validation"]}
    started = time.monotonic()
    result = dict(
        retrospective=True, new_inference=False, new_training=False, alignment_unit_tests=7, arms={}
    )
    for arm in ["mel", "combined"]:
        final = json.loads((SOURCE / f"{arm}_result.json").read_text())
        epochs = [json.loads(f.read_text()) for f in sorted(SOURCE.glob(f"{arm}_epoch_*.json"))]
        assert len(epochs) == 30
        selected = min(epochs, key=lambda r: (r["speaker_macro_cer"], r["epoch"]))
        assert (
            selected["epoch"] == final["selected"]["epoch"]
            and selected["speaker_macro_cer"] == final["selected"]["speaker_macro_cer"]
        )
        trajectory = []
        for epoch in epochs:
            preds = epoch["predictions"]
            assert len(preds) == len(refs) and {r["uid"] for r in preds} == set(refs)
            hist = Counter("".join(normalize_text(r["hypothesis"]).replace(" ", "") for r in preds))
            words = Counter(w for r in preds for w in normalize_text(r["hypothesis"]).split())
            trajectory.append(
                {
                    **{
                        k: epoch[k]
                        for k in [
                            "epoch",
                            "mean_batch_training_loss",
                            "speaker_macro_cer",
                            "speaker_macro_wer",
                            "pooled_cer",
                            "pooled_wer",
                            "nonempty_fraction",
                        ]
                    },
                    "top_characters": hist.most_common(5),
                    "distinct_hypothesis_characters": len(hist),
                    "single_character_word_fraction": sum(
                        v for k, v in words.items() if len(k) == 1
                    )
                    / max(1, sum(words.values())),
                }
            )
        checkpoints = {}
        for label, epoch in [("selected", selected), ("final", epochs[-1])]:
            char_total = Counter()
            word_total = Counter()
            speakers = defaultdict(lambda: dict(character=Counter(), word=Counter()))
            character_hist = Counter()
            reference_hist = Counter()
            word_hist = Counter()
            length_ratios = []
            word_lengths = []
            unique_chars = []
            lexical_overlap = 0
            reference_words = 0
            exact = 0
            utterances = []
            for pred in epoch["predictions"]:
                if time.monotonic() - started > plan["seconds_bound"]:
                    raise TimeoutError("Bounded retrospective analysis")
                source = refs[pred["uid"]]
                reference = normalize_text(source["transcript"])
                hypothesis = normalize_text(pred["hypothesis"])
                char = align(reference.replace(" ", ""), hypothesis.replace(" ", ""))
                word = align(reference.split(), hypothesis.split())
                assert (
                    char["errors"] == pred["character_errors"]
                    and char["reference_length"] == pred["reference_characters"]
                )
                assert (
                    word["errors"] == pred["word_errors"]
                    and word["reference_length"] == pred["reference_words"]
                )
                char_total.update(char)
                word_total.update(word)
                speaker = speakers[pred["speaker"]]
                speaker["character"].update(char)
                speaker["word"].update(word)
                character_hist.update(hypothesis.replace(" ", ""))
                reference_hist.update(reference.replace(" ", ""))
                word_hist.update(hypothesis.split())
                word_lengths.extend(map(len, hypothesis.split()))
                unique_chars.append(len(set(hypothesis.replace(" ", ""))))
                length_ratios.append(char["hypothesis_length"] / char["reference_length"])
                exact += reference == hypothesis
                lexical_overlap += pred["bag_word_overlap"]
                reference_words += pred["reference_words"]
                utterances.append(
                    dict(uid=pred["uid"], speaker=pred["speaker"], character=char, word=word)
                )
            by_speaker = []
            for speaker, units in sorted(speakers.items()):
                row = dict(speaker=speaker)
                for unit, counts in units.items():
                    row[unit] = {
                        **dict(counts),
                        **{
                            k + "_rate": counts[k] / counts["reference_length"]
                            for k in ["errors", "substitutions", "deletions", "insertions"]
                        },
                    }
                by_speaker.append(row)
            macro = {
                unit: {
                    k: float(np.mean([r[unit][k + "_rate"] for r in by_speaker]))
                    for k in ["errors", "substitutions", "deletions", "insertions"]
                }
                for unit in ["character", "word"]
            }
            assert abs(macro["character"]["errors"] - epoch["speaker_macro_cer"]) < 1e-12
            assert abs(macro["word"]["errors"] - epoch["speaker_macro_wer"]) < 1e-12
            total = sum(character_hist.values())
            entropy = -sum((v / total) * math.log2(v / total) for v in character_hist.values())
            checkpoints[label] = dict(
                epoch=epoch["epoch"],
                nonempty_fraction=epoch["nonempty_fraction"],
                character_counts=dict(char_total),
                word_counts=dict(word_total),
                speaker_macro=macro,
                speaker_results=by_speaker,
                character_distribution=character_hist.most_common(),
                reference_character_distribution=reference_hist.most_common(),
                top_words=word_hist.most_common(20),
                hypothesis_character_entropy_bits=entropy,
                top_character_fraction=max(character_hist.values()) / total,
                single_character_word_fraction=sum(v for k, v in word_hist.items() if len(k) == 1)
                / sum(word_hist.values()),
                hypothesis_character_length_ratio_quantiles=np.quantile(
                    length_ratios, [0, 0.25, 0.5, 0.75, 1]
                ).tolist(),
                hypothesis_unique_characters_median=float(np.median(unique_chars)),
                hypothesis_word_length_median=float(np.median(word_lengths)),
                word_inventory_recall=lexical_overlap / reference_words,
                exact_utterances=exact,
                utterance_count=len(utterances),
                examples=[
                    dict(
                        uid=r["uid"],
                        reference=refs[r["uid"]]["transcript"],
                        hypothesis=r["hypothesis"],
                    )
                    for r in sorted(epoch["predictions"], key=lambda r: r["uid"])[:3]
                ],
            )
            (OUT / f"{arm}_{label}_utterance_decomposition.json").write_text(
                json.dumps(utterances, indent=2) + "\n"
            )
        result["arms"][arm] = dict(
            learning_gate_passed=final["learning_gate_passed"],
            trajectory=trajectory,
            checkpoints=checkpoints,
        )
    result.update(
        seconds=time.monotonic() - started,
        training_CER_available=False,
        frame_blank_probabilities_available=False,
        paired_learning_gate_passed=False,
        stage2_ready=False,
    )
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "Saved retrospective diagnostic; all decompositions match stored edit totals; no inference/training."
    )


if __name__ == "__main__":
    main()
