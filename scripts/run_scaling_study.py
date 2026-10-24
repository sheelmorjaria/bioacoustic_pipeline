#!/usr/bin/env python3
"""Time–frequency scaling study driver (speech-like acoustic organisation).

Prespecified contract: bioacoustic_pipeline.diagnostics.scaling_organisation.
scaling_study_spec(). Stages:

  1. speech reference from LibriSpeech dev-clean (speaker-grouped; reference
     speakers disjoint from speech-eval speakers);
  2. DEV animal recordings choose alpha* = argmin mean d_H over the frozen
     ALPHA_GRID (identical grid/budget for every class); bootstrap selection
     stability reported; an AM-noise reference is frozen from DEV renders;
     nothing on EVAL is scored yet;
  3. EVAL (caller-disjoint files): paired per-file Delta(animal) vs
     Delta(control) for each control class — file-clustered bootstrap CI +
     sign-flip permutation p; specificity vs the AM-noise reference;
  4. positive control: speech compressed by alpha* then expanded by alpha*
     must recover d_H (>= 80% of eval clips with Delta >= 0).

Framing: acoustic-organisation measurement under a declared metric — NOT a
decoder, NOT linguistic-content evidence (see spec["explicitly_excluded"]).

Usage:
    python scripts/run_scaling_study.py --self-test
    python scripts/run_scaling_study.py                     # full study
    python scripts/run_scaling_study.py --n-dev 60 --n-eval 90   # reduced

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BAT_SR = 250000
N_REF_CLIPS = 120
N_SPEECH_EVAL_CLIPS = 30
BOOT_B = 2000
PERM_B = 2000
STAB_B = 200
SEED = 0
CONTROL_CLASSES = ("phase_surrogate", "unit_reorder", "spec_noise", "am_noise")


# ------------------------------------------------------------------ workers


def _render(cls, x, sr):

    from bioacoustic_pipeline.diagnostics.scaling_organisation import (
        am_modulated,
        phase_randomized_surrogate,
        spec_matched_noise,
        unit_reorder_surrogate,
    )

    if cls == "animal":
        return x
    if cls == "phase_surrogate":
        return phase_randomized_surrogate(x, seed=SEED)
    if cls == "unit_reorder":
        return unit_reorder_surrogate(x, sr, seed=SEED)
    if cls == "spec_noise":
        return spec_matched_noise(x, seed=SEED, sr=sr)
    if cls == "am_noise":
        return am_modulated(spec_matched_noise(x, seed=SEED, sr=sr), sr, rate_hz=4.0)
    raise ValueError(cls)


def _distances_for(audio, sr, alphas, refs):
    """{alpha: {ref_name: d or None}} for one signal under g_alpha."""
    from bioacoustic_pipeline.diagnostics.scaling_organisation import (
        standardized_distance,
        transform_and_describe,
    )

    out = {}
    for alpha in alphas:
        desc = transform_and_describe(audio, sr, alpha)
        out[alpha] = (
            None
            if desc is None
            else {k: standardized_distance(desc["vector"], r) for k, r in refs.items()}
        )
    return out


def _load_ref(path):
    import numpy as np

    j = json.loads(Path(path).read_text())
    return {
        "names": tuple(j["names"]),
        "mean": np.asarray(j["mean"]),
        "var": np.asarray(j["var"]),
        "n": j["n"],
    }


def _grid_worker(args):
    """Animal DEV file: speech-reference distances over the full alpha grid."""
    from bioacoustic_pipeline.utils.audio import load_audio

    path, name, caller, speech_ref_path = args
    audio, sr = load_audio(str(path), target_sr=BAT_SR)
    refs = {"speech": _load_ref(speech_ref_path)}
    return {
        "file": name,
        "caller": caller,
        "d": {
            str(a): (v["speech"] if v is not None else None)
            for a, v in _distances_for(audio, sr, _alphas(), refs).items()
        },
    }


def _alphas():
    from bioacoustic_pipeline.diagnostics.scaling_organisation import ALPHA_GRID

    return ALPHA_GRID


def _noise_ref_worker(args):
    """DEV am_noise render at alpha*: vectors for the noise reference."""
    from bioacoustic_pipeline.diagnostics.scaling_organisation import (
        transform_and_describe,
    )
    from bioacoustic_pipeline.utils.audio import load_audio

    path, alpha = args
    audio, sr = load_audio(str(path), target_sr=BAT_SR)
    x = _render("am_noise", audio, sr)
    desc = transform_and_describe(x, sr, alpha)
    return None if desc is None else desc["vector"].tolist()


def _eval_worker(args):
    """EVAL bat file: animal + every control at alpha=1 and alpha*.

    Rendering happens ONCE per class; the same rendered signal is measured
    at both alphas, so animal and controls receive the identical transform.
    """
    from bioacoustic_pipeline.utils.audio import load_audio

    path, name, caller, alpha, speech_ref_path, noise_ref_path = args
    refs = {"speech": _load_ref(speech_ref_path)}
    if noise_ref_path:
        refs["noise"] = _load_ref(noise_ref_path)
    audio, sr = load_audio(str(path), target_sr=BAT_SR)
    row = {"file": name, "caller": caller, "classes": {}}
    for cls in ("animal", *CONTROL_CLASSES):
        x = _render(cls, audio, sr)
        if x is None:
            row["classes"][cls] = None  # control unavailable (e.g. too few units)
            continue
        row["classes"][cls] = _distances_for(x, sr, (1, alpha), refs)
    return row


def _positive_worker(args):
    """Speech clip: original vs compress-by-beta then expand-by-beta.

    beta = alpha* when compression leaves a measurable clip for the median
    recording; otherwise the largest grid value that does (documented in the
    report — the control verifies recovery of a KNOWN compression, so it must
    stay measurable rather than silently evaporate).
    """
    import soundfile as sf

    from bioacoustic_pipeline.diagnostics.scaling_organisation import (
        compress_playback,
        expand_playback,
        long_window_descriptor,
        standardized_distance,
        to_analysis_rate,
    )

    path, alpha, speech_ref_path = args
    ref = _load_ref(speech_ref_path)
    x, sr = sf.read(str(path), dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x[: int(60.0 * sr)]
    comp = compress_playback(x, sr, alpha)
    rec = expand_playback(comp, sr, alpha)

    def d(sig):
        z = to_analysis_rate(sig, sr) if sr != 16000 else sig
        if len(z) < 16000:
            return None
        return standardized_distance(long_window_descriptor(z)["vector"], ref)

    return {"orig": d(x), "recovered": d(rec), "compressed": d(comp)}


def _positive_beta(speech_eval_pool, alpha_star: int) -> int:
    """Largest grid alpha <= alpha* keeping the MEDIAN clip measurable (>1.2 s)."""
    import numpy as np
    import soundfile as sf

    from bioacoustic_pipeline.diagnostics.scaling_organisation import ALPHA_GRID

    durs = []
    for p in speech_eval_pool:
        try:
            durs.append(sf.info(str(p)).duration)
        except Exception:  # noqa: BLE001 (corrupt clip: excluded, counted)
            continue
    med = float(np.median(durs)) if durs else 0.0
    for a in reversed([a for a in ALPHA_GRID if a <= alpha_star]):
        if med / a >= 1.2:
            return int(a)
    return 1


def _speech_ref_worker(path):
    import numpy as np
    import soundfile as sf

    from bioacoustic_pipeline.diagnostics.scaling_organisation import (
        long_window_descriptor,
    )

    x, sr = sf.read(str(path), dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        return None
    v = long_window_descriptor(x[: int(20.0 * sr)])["vector"]
    return None if not np.isfinite(v).all() else v.tolist()


# ------------------------------------------------------------------ pipeline


def _boot_ci_pairs(delta_a, delta_b, b=BOOT_B, seed=SEED):
    """Paired file bootstrap on mean(delta_a - delta_b) + sign-flip permutation."""
    import numpy as np

    d = np.asarray(delta_a) - np.asarray(delta_b)
    n = len(d)
    obs = float(d.mean())
    rng = np.random.RandomState(seed)
    picks = rng.randint(0, n, size=(b, n))
    reps = d[picks].mean(axis=1)
    ci = [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
    signs = rng.choice((-1, 1), size=(PERM_B, n))
    null = (d[None, :] * signs).mean(axis=1)
    p = float((1 + np.sum(np.abs(null) >= abs(obs))) / (PERM_B + 1))
    return {"q": obs, "ci95": ci, "perm_p": p, "n": int(n)}


def run_study(libri: Path, n_dev: int, n_eval: int, jobs: int) -> dict:
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    from bioacoustic_pipeline.diagnostics.scaling_organisation import (
        ALPHA_GRID,
        build_reference,
        scaling_study_spec,
    )
    from bioacoustic_pipeline.ingestion.labeled_corpus import (
        holdout_callers,
        parse_annotations_csv,
    )

    spec = scaling_study_spec()
    rng = np.random.RandomState(SEED)

    # ---- 1. speech reference (eval speakers held out, grouping preserved)
    clips = []
    for flac in sorted(libri.rglob("*.flac")):
        clips.append((flac, flac.parts[-3]))
    speakers = sorted({s for _, s in clips})
    eval_speakers = set(speakers[::4])
    ref_pool = [p for p, s in clips if s not in eval_speakers]
    speech_eval_pool = [p for p, s in clips if s in eval_speakers]
    rng.shuffle(ref_pool)
    rng.shuffle(speech_eval_pool)
    ref_pool = ref_pool[:N_REF_CLIPS]
    speech_eval_pool = speech_eval_pool[:N_SPEECH_EVAL_CLIPS]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        rows = [r for r in ex.map(_speech_ref_worker, ref_pool, chunksize=2) if r]
    ref = build_reference(np.asarray(rows))
    speech_ref_path = ROOT / "outputs" / "scaling_speech_reference.json"
    speech_ref_path.write_text(
        json.dumps(
            {
                "names": list(ref["names"]),
                "mean": ref["mean"].tolist(),
                "var": ref["var"].tolist(),
                "n": ref["n"],
                "eval_speakers_held_out": sorted(eval_speakers),
                "source": str(libri),
            }
        )
    )
    print(
        f"speech reference: {ref['n']} clips "
        f"({len(speakers) - len(eval_speakers)} ref speakers, "
        f"{len(eval_speakers)} eval speakers held out)",
        flush=True,
    )

    # ---- 2. caller-disjoint DEV/EVAL + frozen alpha*
    annos = [
        a
        for a in parse_annotations_csv(ROOT / "data/egyptian_fruit_bats/annotations.csv")
        if a.emitter > 0
    ]
    split = holdout_callers(annos, n_heldout=6, min_files=8, seed=SEED)
    eval_pool = sorted(split.test_files | split.val_files)
    dev_pool = sorted(split.train_files)
    rng.shuffle(dev_pool)
    rng.shuffle(eval_pool)
    dev_files, eval_files = dev_pool[:n_dev], eval_pool[:n_eval]
    audio_dir = ROOT / "data/egyptian_fruit_bats/audio"
    by_file = {a.file_name: a.emitter for a in annos}

    dev_args = [
        (audio_dir / f, f, by_file[f], str(speech_ref_path)) for f in dev_files if f in by_file
    ]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        dev_rows = [r for r in ex.map(_grid_worker, dev_args, chunksize=2)]
    dev_rows = [r for r in dev_rows if r["d"].get("1") is not None]

    def _mean_alpha(rows, a):
        v = [r["d"][str(a)] for r in rows if r["d"].get(str(a)) is not None]
        return float(np.mean(v)) if v else float("nan")

    curve = {str(a): round(_mean_alpha(dev_rows, a), 4) for a in ALPHA_GRID}
    alpha_star = int(min(ALPHA_GRID, key=lambda a: _mean_alpha(dev_rows, a)))
    print("DEV d_H curve: " + " ".join(f"a={a}:{curve[str(a)]}" for a in ALPHA_GRID), flush=True)
    print(f"alpha* frozen at {alpha_star} over {len(dev_rows)} dev recordings", flush=True)

    stab = {str(a): 0 for a in ALPHA_GRID}
    brng = np.random.RandomState(SEED)
    for _ in range(STAB_B):
        pick = brng.randint(0, len(dev_rows), len(dev_rows))
        sub = [dev_rows[i] for i in pick]
        stab[str(min(ALPHA_GRID, key=lambda a: _mean_alpha(sub, a)))] += 1

    # noise reference frozen from DEV am_noise renders at alpha*
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        nv = [
            v
            for v in ex.map(
                _noise_ref_worker,
                [(audio_dir / r["file"], alpha_star) for r in dev_rows],
                chunksize=2,
            )
            if v
        ]
    noise_ref = build_reference(np.asarray(nv))
    noise_ref_path = ROOT / "outputs" / "scaling_noise_reference.json"
    noise_ref_path.write_text(
        json.dumps(
            {
                "names": list(noise_ref["names"]),
                "mean": noise_ref["mean"].tolist(),
                "var": noise_ref["var"].tolist(),
                "n": noise_ref["n"],
                "class": "am_noise at frozen alpha* on DEV recordings",
            }
        )
    )
    print(f"noise reference: {noise_ref['n']} dev renders", flush=True)

    # ---- 3. EVAL: paired animal-vs-control deltas at frozen alpha*
    eval_args = [
        (audio_dir / f, f, by_file[f], alpha_star, str(speech_ref_path), str(noise_ref_path))
        for f in eval_files
        if f in by_file
    ]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        rows = [r for r in ex.map(_eval_worker, eval_args, chunksize=2)]

    def _delta(row, cls, ref_name):
        c = row["classes"].get(cls)
        if not c or c.get(1) is None or c.get(alpha_star) is None:
            return None
        if c[1].get(ref_name) is None or c[alpha_star].get(ref_name) is None:
            return None
        return c[1][ref_name] - c[alpha_star][ref_name]

    scored = [r for r in rows if _delta(r, "animal", "speech") is not None]
    report = {
        "spec": spec,
        "alpha_star": alpha_star,
        "dev_curve": curve,
        "selection_stability": stab,
        "n_dev_scored": len(dev_rows),
        "n_eval_scored": len(scored),
        "eval_callers": sorted({r["caller"] for r in scored}),
        "controls": {},
        "specificity": {},
        "animal_delta_mean": float(np.mean([_delta(r, "animal", "speech") for r in scored])),
        "unavailable_counts": {
            cls: sum(1 for r in rows if r["classes"].get(cls) is None) for cls in CONTROL_CLASSES
        },
    }
    animal_d = [_delta(r, "animal", "speech") for r in scored]
    report["per_file"] = [
        {
            "file": r["file"],
            "caller": r["caller"],
            "delta_animal": round(a, 4),
            "d_speech_a1": round(r["classes"]["animal"][1]["speech"], 4),
            "d_speech_astar": round(r["classes"]["animal"][alpha_star]["speech"], 4),
            "d_noise_astar": round(r["classes"]["animal"][alpha_star]["noise"], 4),
            **{
                f"delta_{cls}": (
                    None if _delta(r, cls, "speech") is None else round(_delta(r, cls, "speech"), 4)
                )
                for cls in CONTROL_CLASSES
            },
        }
        for r, a in zip(scored, animal_d)
    ]
    # absolute closeness at alpha*: which reference does the transformed
    # animal actually sit nearer? (baseline-scale-free specificity statement)
    report["absolute_at_alpha_star"] = {
        "mean_d_speech": round(
            float(np.mean([p["d_speech_astar"] for p in report["per_file"]])), 4
        ),
        "mean_d_noise": round(float(np.mean([p["d_noise_astar"] for p in report["per_file"]])), 4),
        "frac_closer_to_speech": round(
            float(np.mean([p["d_speech_astar"] < p["d_noise_astar"] for p in report["per_file"]])),
            4,
        ),
        "note": "d_noise reference is derived from DEV am_noise renders of the "
        "same species — closeness to it is partially self-referential and is "
        "reported for transparency, not as an independent comparison class",
    }
    for cls in CONTROL_CLASSES:
        pairs = [(a, _delta(r, cls, "speech")) for r, a in zip(scored, animal_d)]
        pairs = [(a, c) for a, c in pairs if c is not None]
        if not pairs:
            report["controls"][cls] = {"q": None, "note": "no scored pairs"}
            continue
        a, c = zip(*pairs)
        report["controls"][cls] = _boot_ci_pairs(a, c)
        if cls == "unit_reorder":
            report["controls"][cls]["scope_note"] = (
                "scored only on multi-unit recordings (single-bout files "
                "cannot be reordered): coverage reported in n"
            )
    # specificity: animal approach to speech vs approach to the noise class
    sp = [(_delta(r, "animal", "speech"), _delta(r, "animal", "noise")) for r in scored]
    sp = [(s, nn) for s, nn in sp if s is not None and nn is not None]
    if sp:
        s, nn = zip(*sp)
        report["specificity"] = _boot_ci_pairs(s, nn)

    # ---- 4. positive control on held-out speech
    beta = _positive_beta(speech_eval_pool, alpha_star)
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        pos = [
            r
            for r in ex.map(
                _positive_worker,
                [(p, beta, str(speech_ref_path)) for p in speech_eval_pool],
                chunksize=1,
            )
            if r
        ]
    ok = [r for r in pos if r["orig"] is not None and r["recovered"] is not None]
    rec = [r["orig"] - r["recovered"] for r in ok]  # >0: recovered is closer
    report["positive_control"] = {
        "beta": beta,
        "beta_note": "compression factor actually tested (largest grid value "
        "<= alpha* keeping the median clip above the 1.2 s descriptor minimum)",
        "n_clips": len(ok),
        "excluded": len(pos) - len(ok),
        "frac_recovered": float(np.mean([x >= 0 for x in rec])) if rec else None,
        "mean_delta": float(np.mean(rec)) if rec else None,
    }

    # ---- prespecified decision
    ctrl_ok = all(
        (v.get("q") or 0) > 0 and v["ci95"][0] > 0
        for v in report["controls"].values()
        if v.get("q") is not None
    )
    spec_ok = (report["specificity"].get("q") or 0) > 0 and (report["specificity"]["ci95"][0] > 0)
    pos_ok = (report["positive_control"]["frac_recovered"] or 0) >= 0.8
    report["decision"] = {
        "study_positive": bool(ctrl_ok and spec_ok and pos_ok),
        "controls_all_positive": ctrl_ok,
        "specificity_holds": spec_ok,
        "positive_control_passes": pos_ok,
        "coverage_caveat": "unit_reorder is evaluable only on multi-unit "
        "recordings (~1/3 of files); its Q is scoped to that subset — "
        "coverage is reported per class in controls[cls].n",
        "framing": "a positive supports the RESTRICTED claim: constrained "
        "rescaling moves these recordings closer to a speech reference under "
        "a declared acoustic metric, beyond matched controls. It is NOT "
        "linguistic-content, decoding, or 'compressed message' evidence",
    }
    out = ROOT / "outputs" / "scaling_study.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--libri", type=Path, default=ROOT / "data/LibriSpeech")
    ap.add_argument("--n-dev", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        import numpy as np

        from bioacoustic_pipeline.diagnostics.scaling_organisation import (
            build_reference,
            long_window_descriptor,
            standardized_distance,
            transform_and_describe,
        )

        sr = 48000
        t = np.arange(int(2.0 * sr)) / sr
        ref_rows = []
        for k in range(10):
            x = (
                0.3
                * (1 + 0.6 * np.sin(2 * np.pi * 4 * t + k))
                * np.sin(2 * np.pi * 300 * t + 0.1 * k)
            )
            ref_rows.append(long_window_descriptor(x)["vector"])
        ref = build_reference(ref_rows)
        animal = 0.3 * (1 + 0.6 * np.sin(2 * np.pi * 128 * t)) * np.sin(2 * np.pi * 9600 * t)
        d1 = standardized_distance(transform_and_describe(animal, sr, 1)["vector"], ref)
        d32 = standardized_distance(transform_and_describe(animal, sr, 32)["vector"], ref)
        assert d32 < d1, (d1, d32)
        # paired bootstrap machinery sanity
        r = _boot_ci_pairs([1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.0, 2.0], b=200)
        assert r["q"] > 0 and 0 <= r["perm_p"] <= 1
        print(f"SELF-TEST OK d(a=1)={d1:.3f} > d(a=32)={d32:.3f}", flush=True)
        return

    run_study(args.libri, args.n_dev, args.n_eval, args.jobs)


if __name__ == "__main__":
    main()
