"""Metadata-only audit of acquired dev-other matching support."""

from itertools import combinations
import json
from pathlib import Path

import networkx as nx
from rclat_grounding_matching_audit import describe, sha

ROOT = Path("outputs/rclat_grounding_source_support_v1")


def disjoint_pairs(uids, candidates):
    graph = nx.Graph()
    graph.add_nodes_from(sorted(uids))
    edges = sorted(candidates, key=lambda r: tuple(r["uids"]))
    # Cardinality first; integer cost then stable edge rank as a tie preference.
    multiplier = (len(edges) + 1) * (len(uids) + 1)
    for rank, r in enumerate(edges):
        graph.add_edge(*r["uids"], weight=-(round(r["cost"] * 1e9) * multiplier + rank))
    matching = nx.max_weight_matching(graph, maxcardinality=True, weight="weight")
    return sorted(tuple(sorted(pair)) for pair in matching)


def main():
    plan = json.loads((ROOT / "protocol.json").read_text())
    for path, digest in plan["dependencies"].items():
        if sha(path) != digest:
            raise ValueError("Changed dependency: " + path)
    if nx.__version__ != plan["networkx_version"]:
        raise ValueError("Matching library version changed")
    cohort = json.loads(Path(plan["cohort"]).read_text())
    diagnostic = json.loads(Path(plan["diagnostic_cohort"]).read_text())
    excluded = {r["uid"] for r in diagnostic["validation"]}
    pool = [r for r in cohort["selected_validation"] if r["uid"] not in excluded]
    assert len({r["uid"] for r in pool}) == len(pool)
    assert not {r["speaker"] for r in pool} & {r["speaker"] for r in diagnostic["training"]}
    if not all("/dev-other/" in r["path"] for r in pool):
        raise ValueError("Unexpected source partition")
    results = []
    candidates_total = eligible_total = 0
    for speaker in sorted({r["speaker"] for r in pool}):
        rows = sorted([r for r in pool if r["speaker"] == speaker], key=lambda r: r["uid"])
        eligible = []
        n = 0
        for a, b in combinations(rows, 2):
            n += 1
            edge = describe(a, b, plan["duration_ratio_limit"], plan["character_ratio_limit"])
            if edge["eligible"]:
                eligible.append(edge)
        candidates_total += n
        eligible_total += len(eligible)
        matching = disjoint_pairs([r["uid"] for r in rows], eligible)
        lookup = {tuple(r["uids"]): r for r in eligible}
        ranked = sorted(matching, key=lambda p: (lookup[p]["cost"], p))
        results.append(
            {
                "speaker": speaker,
                "utterances": len(rows),
                "candidate_pairs": n,
                "eligible_pairs": len(eligible),
                "maximum_disjoint_pairs": len(matching),
                "supported": len(matching) >= plan["pairs_per_speaker"],
                "selected_if_speaker_retained": ranked[: plan["pairs_per_speaker"]],
                "eligible_pair_details": eligible,
            }
        )
    supported = [r for r in results if r["supported"]]
    certified = len(supported) >= plan["minimum_speakers"]
    selected = supported[: plan["minimum_speakers"]] if certified else []
    pairing = {}
    for r in selected:
        for a, b in r["selected_if_speaker_retained"]:
            pairing[a] = b
            pairing[b] = a
    records = [r for r in pool if r["uid"] in pairing]
    summary = {
        "completed": True,
        "decision": "SUPPORT CERTIFIED; evaluation protocol and inference not authorised by this audit"
        if certified
        else "ABSTAIN: minimum speaker/pair support unmet",
        "support_certified": certified,
        "original_population": len(cohort["selected_validation"]),
        "excluded_previous_diagnostic_utterances": len(excluded),
        "eligible_pool_utterances": len(pool),
        "pool_speakers": len(results),
        "candidate_pairs": candidates_total,
        "eligible_pairs": eligible_total,
        "supported_speakers": len(supported),
        "selected_speakers": [r["speaker"] for r in selected],
        "selected_pairs": len(pairing) // 2,
        "selected_utterances": len(pairing),
        "speaker_details": results,
        "predictions_accessed": False,
        "audio_read": False,
        "model_inference_performed": False,
        "grounding_contrast_scored": False,
        "P100_authorized": False,
        "P960_authorized": False,
        "stage2_ready": False,
    }
    with (ROOT / "summary.json").open("x") as f:
        json.dump(summary, f, indent=2)
    with (ROOT / "selected_metadata.json").open("x") as f:
        json.dump(
            {
                "selected_only_if_support_certified": True,
                "pairing": pairing,
                "records": records,
                "prediction_or_inference_performed": False,
            },
            f,
            indent=2,
        )
    lines = [
        "# RCLAT dev-other source-support audit",
        "",
        summary["decision"],
        "",
        f"Prospective requirement: at least {plan['minimum_speakers']} same-speaker groups, each with {plan['pairs_per_speaker']} disjoint reciprocal pairs. Joint duration and normalised character-count max/min ratios <=1.10; distinct transcripts required. Previous 32-utterance diagnostic validation subset excluded before support calculation.",
        "",
        f"- Audited metadata: {len(pool)} utterances from {len(results)} speakers.",
        f"- Candidate within-speaker pairs: {candidates_total}.",
        f"- Eligible pairs: {eligible_total}.",
        f"- Speakers supporting the required disjoint pairing count: {len(supported)}.",
        f"- Selected support inventory: {len(selected)} speakers, {len(pairing) // 2} pairs, {len(pairing)} utterances.",
        "",
        "Selection maximises disjoint-pair cardinality within each speaker, then minimises fixed metadata cost with deterministic ordering under pinned NetworkX. The first 24 qualifying speaker IDs and first eight cost-ranked pairs per selected speaker are retained. All speaker support counts and eligible edges are preserved in summary.json. No thresholds or population rules were changed after calculating support.",
        "",
        "This is an already exposed development partition, not untouched confirmatory data. Prior model-development exposure is not removed by excluding the small diagnostic subset. Metadata support does not certify equal phonetic difficulty, acoustic grounding, model performance. Character composition and word-count differences are descriptive, not grounds for a stronger phonetic claim.",
        "",
        "No audio, prediction, logit or checkpoint file was read. No inference, fitting or contrast scoring occurred. A new separately frozen evaluation protocol is required before using the supported inventory for model assessment. Prior abstention and scientific results remain unchanged; P100/P960 and MACDE–CAVEM Stage 2 remain blocked.",
        "",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
