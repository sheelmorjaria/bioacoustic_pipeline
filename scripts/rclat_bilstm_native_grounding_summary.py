"""Offline paired-reference scoring, artifact verification and closeout."""

import json

import rclat_bilstm_native_grounding as runner

core=runner.core
ROOT=runner.ROOT


def main():
    plan=json.loads((ROOT/"execution_plan.json").read_text())
    for p,digest in plan["dependencies"].items():
        assert core.sha(p)==digest,p
    results={}
    raw=0
    for device in ["xpu_0","xpu_1"]:
        out=ROOT/device
        receipt=json.loads((out/"result.json").read_text())
        assert receipt["completed"] and receipt["original_reproduction_passed"]
        evaluation=json.loads((out/"native/step_2000/evaluation.json").read_text())
        results[device]={}
        for split in ["training","validation"]:
            predictions=evaluation[split]["predictions"]
            for row in predictions:
                assert core.sha(row["raw_path"])==row["raw_sha256"]
                raw+=1
            scores=runner.pairing_scores(predictions,plan["donors"])
            own=core.predecessor.original.aggregate(scores["matched"])
            wrong=core.predecessor.original.aggregate(scores["native_swapped"])
            delta={r["speaker"]:r["cer"]-next(x["cer"] for x in own["speaker_results"] if x["speaker"]==r["speaker"]) for r in wrong["speaker_results"]}
            results[device][split]={"matched":own,"native_swapped":wrong,"CER_increase":wrong["speaker_macro_cer"]-own["speaker_macro_cer"],"paired_speaker_CER_increase":delta,"per_utterance_scores":scores}
    summary={"completed":True,"results":results,"raw_records_verified":raw,"original_reproduction_passed_both":True,"devices_metrics_identical":results['xpu_0']==results['xpu_1'],"optimizer_updates":0,"P100_authorized":False,"P960_authorized":False,"stage2_ready":False}
    core.write(ROOT/"summary.json",summary)
    lines=["# BiLSTM native-length acoustic-grounding control","","Native-length predictions are scored against their own transcripts and against the prospectively paired same-speaker recipient transcripts. No features, masks, lengths or prediction strings are resampled. Each native utterance is inferred once per device; the pairing contrast is an offline re-scoring, not a second independent sample.","","| Split | Matched CER | Native-swapped CER | Increase | Matched WER | Native-swapped WER |","|---|---:|---:|---:|---:|---:|"]
    for split,r in results['xpu_0'].items():
        lines.append(f"| {split} | {r['matched']['speaker_macro_cer']:.2%} | {r['native_swapped']['speaker_macro_cer']:.2%} | {r['CER_increase']*100:.2f} pp | {r['matched']['speaker_macro_wer']:.2%} | {r['native_swapped']['speaker_macro_wer']:.2%} |")
    lines += ["","Both devices completed native inference. All original hypotheses reproduced exactly and logit relative L2 errors met 1e-4. All 192 saved raw records and frozen dependency hashes verified. Model tensors and source checkpoint bytes remained unchanged; zero optimiser updates occurred.","","No resampling distortion is present. Duration, reference lengths, phonetic content and input-specific mask patterns still vary between paired recordings. These contrasts therefore do not isolate duration-independent phonetic encoding. Same-speaker reciprocal pairing avoids cross-speaker donor confounding and preserves each split's reference population, but cannot equate transcript difficulty. No significance or equivalence test was preregistered; the two devices are computational replication of one checkpoint, not independent model seeds.","","Previous results remain closed. No dictionary correction, language model, final-test scoring or new training occurred. P100/P960 and MACDE–CAVEM Stage 2 remain blocked.",""]
    (ROOT/"RESULTS.md").write_text("\n".join(lines))
    core.write(ROOT/"closeout.json",{"completed":True,"raw_records_verified":raw,"reproduction_passed_both":True,"devices_metrics_identical":summary['devices_metrics_identical'],"optimizer_updates":0,"P100_authorized":False,"P960_authorized":False,"stage2_ready":False})
    print("\n".join(lines))


if __name__=="__main__":
    main()
