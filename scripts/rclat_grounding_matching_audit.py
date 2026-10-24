"""Metadata-only support audit; never reads predictions or acoustic outputs."""

from collections import Counter
from itertools import combinations
import hashlib
import json
from pathlib import Path

from bioacoustic_pipeline.diagnostics.rclat import normalize_text

ROOT = Path("outputs/rclat_grounding_matching_support_v1")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(a, b, duration_limit=1.10, character_limit=1.10):
    ta, tb = normalize_text(a["transcript"]), normalize_text(b["transcript"])
    ca, cb = ta.replace(" ", ""), tb.replace(" ", "")
    durations = [a["duration_seconds"], b["duration_seconds"]]
    if min(durations) <= 0 or not ca or not cb:
        raise ValueError("Invalid matching metadata")
    dr = max(durations) / min(durations)
    cr = max(len(ca), len(cb)) / min(len(ca), len(cb))
    wa, wb = len(ta.split()), len(tb.split())
    ha, hb = Counter(ca), Counter(cb)
    tv = 0.5 * sum(abs(ha[c] / len(ca) - hb[c] / len(cb)) for c in ha.keys() | hb.keys())
    return {
        "uids": [a["uid"], b["uid"]],
        "speaker": a["speaker"],
        "same_speaker": a["speaker"] == b["speaker"],
        "distinct_transcripts": ta != tb,
        "duration_ratio": dr,
        "character_count_ratio": cr,
        "word_count_ratio": max(wa, wb) / min(wa, wb),
        "character_composition_total_variation": tv,
        "eligible": a["speaker"] == b["speaker"]
        and ta != tb
        and dr <= duration_limit
        and cr <= character_limit,
        "cost": dr - 1 + cr - 1,
    }


def best_matching(uids, edges):
    if not uids:
        return []
    a = uids[0]
    options = []
    for b in uids[1:]:
        key = tuple(sorted([a, b]))
        if key not in edges:
            continue
        rest = best_matching([u for u in uids if u not in [a, b]], edges)
        if rest is not None:
            options.append([key, *rest])
    if not options:
        return None
    return min(options, key=lambda pairs: (sum(edges[p]["cost"] for p in pairs), pairs))


def main():
    plan = json.loads((ROOT / "protocol.json").read_text())
    for p, digest in plan["dependencies"].items():
        if sha(p) != digest:
            raise ValueError("Changed dependency: " + p)
    cohort = json.loads(Path(plan["cohort"]).read_text())
    result = {}
    for split in ["training", "validation"]:
        rows = cohort[split]
        by_uid = {r["uid"]: r for r in rows}
        pairs = []
        speakers = []
        mapping = {}
        for speaker in sorted({r["speaker"] for r in rows}):
            group = sorted([r for r in rows if r["speaker"] == speaker], key=lambda r: r["uid"])
            group_pairs = [
                describe(a, b, plan["duration_ratio_limit"], plan["character_ratio_limit"])
                for a, b in combinations(group, 2)
            ]
            pairs.extend(group_pairs)
            edges = {tuple(sorted(p["uids"])): p for p in group_pairs if p["eligible"]}
            match = best_matching([r["uid"] for r in group], edges)
            if match is not None:
                for a, b in match:
                    mapping[a] = b
                    mapping[b] = a
            speakers.append(
                {
                    "speaker": speaker,
                    "utterances": len(group),
                    "eligible_pairs": len(edges),
                    "full_matching_supported": match is not None,
                    "selected_full_matching": match,
                }
            )
        eligible = [p for p in pairs if p["eligible"]]
        unsupported = [u for u in by_uid if not any(u in p["uids"] for p in eligible)]
        full = all(r["full_matching_supported"] for r in speakers)
        result[split] = {
            "utterances": len(rows),
            "speakers": len(speakers),
            "candidate_pairs": len(pairs),
            "eligible_pairs": len(eligible),
            "utterances_with_eligible_alternative": len(rows) - len(unsupported),
            "unsupported_uids": unsupported,
            "speakers_with_full_matching": sum(r["full_matching_supported"] for r in speakers),
            "full_population_supported": full,
            "selected_mapping_supported_speakers_only": mapping,
            "speaker_details": speakers,
            "candidate_pair_details": pairs,
        }
    certified = result["validation"]["full_population_supported"]
    summary = {
        "completed": True,
        "metadata_only": True,
        "predictions_accessed": False,
        "model_inference_performed": False,
        "validation_matching_certified": certified,
        "scoring_authorized_by_this_audit": False,
        "decision": "matching support available; separate scoring freeze required"
        if certified
        else "ABSTAIN: full validation population lacks prospective matching support",
        "results": result,
        "P100_authorized": False,
        "P960_authorized": False,
        "stage2_ready": False,
    }
    with (ROOT / "summary.json").open("x") as f:
        json.dump(summary, f, indent=2)
    lines = [
        "# RCLAT duration/length matching-support audit",
        "",
        summary["decision"],
        "",
        "Metadata-only rules: distinct same-speaker transcripts; duration ratio <=1.10 and normalised character-count ratio <=1.10. Both bounds are max/min, applied jointly. Every validation utterance and speaker must support a disjoint reciprocal pairing. Word-count ratio and character-composition total variation are reported descriptively, not used to relax or redefine eligibility.",
        "",
        "| Population | Candidate pairs | Eligible pairs | Utterances with an alternative | Speakers with full pairing |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, r in result.items():
        lines.append(
            f"| {split} | {r['candidate_pairs']} | {r['eligible_pairs']} | {r['utterances_with_eligible_alternative']}/{r['utterances']} | {r['speakers_with_full_matching']}/{r['speakers']} |"
        )
    lines += [
        "",
        "The thresholds are prospective operational matching choices, not validated equivalence of phonetic difficulty. Character composition is not a phonetic or semantic reference. Duration-quartile sampling in the source diagnostic can limit close within-speaker matching. No prediction errors, logits, checkpoints or audio were read. No model was fitted or inferred, and no grounding contrast was scored. Unsupported populations were not replaced and criteria were not loosened. Prior study results and P100/P960 / MACDE–CAVEM Stage 2 blocks remain unchanged.",
        "",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
