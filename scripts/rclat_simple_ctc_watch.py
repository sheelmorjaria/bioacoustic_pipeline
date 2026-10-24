"""Observe frozen jobs and write a terminal closeout without changing execution."""

import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("outputs/rclat_simple_ctc_diagnostic_v1")
while True:
    states = {}
    terminal = True
    for device in ["xpu_0", "xpu_1"]:
        out = ROOT / device
        failures = list(out.glob("*FAILURE.json"))
        result = out / "D1_result.json"
        if not result.exists() and not failures:
            terminal = False
        journal = out / "D1_journal.jsonl"
        updates = []
        if journal.exists():
            updates = [json.loads(x) for x in journal.read_text().splitlines() if '"step_complete"' in x]
        states[device] = {"completed":result.exists(),"successful_updates":updates[-1]["step"] if updates else 0,"failure_records":[str(p) for p in failures]}
    if terminal:
        break
    time.sleep(10)

if all(s["completed"] for s in states.values()):
    subprocess.run([sys.executable,"scripts/rclat_simple_ctc_summary.py"],check=True)
else:
    lines = ["# RCLAT small-encoder diagnostic: incomplete execution", "", "The fixed 2,000-update endpoint was not completed on both devices. The training-fit criterion is therefore **not evaluated**; this is not a completed negative representation-learning result.", ""]
    for device, state in states.items():
        lines.append(f"- {device}: {state['successful_updates']} successful updates; completed={state['completed']}; failure records: {state['failure_records']}")
    lines += ["", "Failure records preserve the exception, including any frozen wall-time bound. No budget extension, automatic retry or checkpoint resumption occurred. Saved intermediate metrics remain descriptive and do not replace the fixed endpoint. Prior studies remain closed; P100/P960 and MACDE–CAVEM Stage 2 remain blocked.", ""]
    with (ROOT / "RESULTS.md").open("x") as stream:
        stream.write("\n".join(lines))
with (ROOT / "execution_closeout.json").open("x") as stream:
    json.dump({"terminal":True,"devices":states,"fixed_endpoint_completed_both":all(s['completed'] for s in states.values()),"fit_gate_evaluated":all(s['completed'] for s in states.values()),"P100_authorized":False,"P960_authorized":False,"stage2_ready":False},stream,indent=2)
print("Automatic closeout written.",flush=True)
