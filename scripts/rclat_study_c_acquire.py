"""Bounded public-source acquisition for a reference-validity audit; no model fitting."""

import hashlib
import json
from pathlib import Path

ROOT = Path("data/rclat_study_c_audit")


def fetch(url, name, limit=25_000_000):
    path = ROOT / name
    if path.exists():
        return path
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(dir="/tmp") as temporary:
        completed = subprocess.run(
            [
                "curl",
                "-fL",
                "--silent",
                "--show-error",
                "--max-time",
                "45",
                "--max-filesize",
                str(limit),
                "-w",
                "%{url_effective}",
                url,
                "-o",
                temporary.name,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        raw = Path(temporary.name).read_bytes()
        final = completed.stdout
    if len(raw) > limit:
        raise ValueError("Audit download exceeded fixed size cap")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    log = ROOT / "acquisition.jsonl"
    with log.open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "url": url,
                    "resolved_url": final,
                    "path": str(path),
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
            + "\n"
        )
    return path


def listing(url, name):
    return json.loads(fetch(url, name).read_text())


def followup():
    for item in json.loads((ROOT / "sinagra_data.json").read_text())["data"]:
        if item["attributes"]["name"] in (
            "Stimuli-guide.csv",
            "prosody-task.csv",
            "Rcode-Indiv-diff-prosody.R",
        ):
            fetch(item["links"]["download"], "sinagra_data/" + item["attributes"]["name"])
    fetch("https://osf.io/download/ekm5q/", "xie_analysis.Rmd")
    for source in ["villani_natural", "sinagra_stimuli"]:
        page = json.loads((ROOT / (source + ".json")).read_text())
        number = 2
        while page.get("links", {}).get("next"):
            page = listing(page["links"]["next"], source + f"_page{number}.json")
            number += 1
    tree = json.loads((ROOT / "pros_scaled_tree.json").read_text())
    paths = [
        "README.md",
        "data/processed/behavioral/pros_scaled_anonymized.csv",
        "data/processed/audio/audio_features_scaled_stimuli.csv",
        "data/processed/norming/item_means.csv",
        "src/analysis/behavioral/behavioral.Rmd",
        "src/experiment/generate_experiment.py",
        "experiment/prosody_exp_full.html",
    ]
    for path in paths:
        fetch(
            "https://raw.githubusercontent.com/seantrott/pros_scaled/" + tree["sha"] + "/" + path,
            "pros_scaled/" + path,
        )
    print("Follow-up reference tables and complete selected audio inventories acquired.")


def audio_audit_population():
    import csv
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tree = json.loads((ROOT / "pros_scaled_tree.json").read_text())
    inventory = {r["path"]: r for r in tree["tree"] if r["type"] == "blob"}
    rows = list(
        csv.DictReader(
            (ROOT / "pros_scaled/data/processed/behavioral/pros_scaled_anonymized.csv").open()
        )
    )
    keys = sorted(
        {
            (r["speaker"], r["condition"], r["topic"])
            for r in rows
            if r["response"] in ("Request", "Non-Request") and r["speaker"] and r["condition"]
        }
    )
    paths = [
        f"data/raw/scaled_stimuli_normed/{speaker}_{condition}_{topic}.wav"
        for speaker, condition, topic in keys
    ]
    if sum(inventory[path]["size"] for path in paths) > 200_000_000:
        raise ValueError("Fixed audit audio-size cap exceeded")

    def download(path):
        local = fetch(
            "https://raw.githubusercontent.com/seantrott/pros_scaled/" + tree["sha"] + "/" + path,
            "pros_scaled/" + path,
            limit=2_000_000,
        )
        raw = local.read_bytes()
        if (
            len(raw) != inventory[path]["size"]
            or hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            != inventory[path]["sha"]
        ):
            raise ValueError("Pinned Git blob integrity failed")
        return path

    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, future in enumerate(
            as_completed([pool.submit(download, path) for path in paths]), 1
        ):
            future.result()
            if i % 50 == 0:
                print("Verified", i, "played-stimulus WAVs", flush=True)
    print("All", len(paths), "listener-linked normalized WAVs verified against pinned Git blobs.")


def main():
    for source in ["xie_data", "villani_data"]:
        for item in json.loads((ROOT / (source + ".json")).read_text())["data"]:
            attr = item["attributes"]
            if attr["kind"] == "file" and attr["name"].endswith((".csv", ".Rmd")):
                fetch(item["links"]["download"], source + "/" + attr["name"])
    fetch("https://osf.io/download/cm4x6/", "xie_README.txt")
    targets = [
        ("xie_all", "kr7y6", "5d07a7d80bea13001ab715a6"),
        ("xie_prepost", "kr7y6", "5d07a4ab0bea13001ab712fc"),
        ("xie_nonambiguous", "kr7y6", "5d07a4e0a844d4001abda293"),
        ("villani_natural", "c83ep", "6665d3f365e1de5370893aff"),
    ]
    for name, node, folder in targets:
        listing(
            f"https://api.osf.io/v2/nodes/{node}/files/osfstorage/{folder}/?page[size]=100",
            name + ".json",
        )
    for name, node in [("sinagra_stimuli", "a7y5v"), ("sinagra_data", "2haev")]:
        listing(
            f"https://api.osf.io/v2/nodes/{node}/files/osfstorage/?page[size]=100", name + ".json"
        )
    print("Bounded data tables and relevant audio inventories acquired; no labels inferred.")


if __name__ == "__main__":
    import sys

    if "--audio" in sys.argv:
        audio_audit_population()
    elif "--followup" in sys.argv:
        followup()
    else:
        main()
