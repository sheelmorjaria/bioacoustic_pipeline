"""Prospective finite-five-seed RTPE replication; no joint seed/speaker CI claim."""

import copy

import numpy as np

from .rclat_rtpe_protocol import paired_speaker_summary


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("Explicit nonnegative integer random seed required")
    return value


def _pairs(pairs):
    pairs = list(pairs)
    if len(pairs) != 5 or any(len(p) != 2 for p in pairs):
        raise ValueError("Exactly five prospective initialisation/sampler seed pairs required")
    for a, b in pairs:
        _seed(a)
        _seed(b)
    if (
        len({a for a, _ in pairs}) != 5
        or len({b for _, b in pairs}) != 5
        or any(a == 17 for a, _ in pairs)
    ):
        raise ValueError("Five distinct new streams required; original seed17 is historical only")
    return pairs


def seed_spec(base, *, initialization_seed, sampler_seed):
    result = copy.deepcopy(base)
    result["gru"]["seed"] = _seed(initialization_seed)
    result["gru"]["sampler_seed"] = _seed(sampler_seed)
    result["gru"]["sampling"] = (
        f"Uniform frozen training-origin indices with replacement;dedicated CPU generator seed{sampler_seed}"
    )
    return result


def unlock_replication(selections, *, pairs, updates, interval):
    pairs = _pairs(pairs)
    if (
        set(selections) != {str(s) for s, _ in pairs}
        or updates <= 0
        or interval <= 0
        or updates % interval
    ):
        raise ValueError("All five declared seed selections and fixed update schedule required")
    expected = list(range(interval, updates + 1, interval))
    for selection in selections.values():
        if (
            selection.get("completed_updates") != updates
            or selection.get("validation_steps") != expected
            or selection.get("gru_step") not in expected
        ):
            raise ValueError(
                "Evaluation sealed until every full seed schedule and selection completes"
            )
    return True


def replication_summary(ridge, seed_losses, *, pairs, bootstrap_seed, replicates=10000):
    """Seed-specific original criteria plus descriptive crossed-matrix summaries.

    Ridge is one shared deterministic comparator, not five independent fits.
    Each row's percentile interval resamples speakers conditional on that model
    and frozen origins. No bootstrap resampling of seeds or joint95% claim is
    introduced. The finite-five-seed rule requires all five original criteria.
    """
    pairs = _pairs(pairs)
    if set(seed_losses) != {str(s) for s, _ in pairs}:
        raise ValueError("No missing, additional or dropped seeds allowed")
    per_seed = {}
    for seed, _ in pairs:
        per_seed[str(seed)] = paired_speaker_summary(
            ridge, seed_losses[str(seed)], seed=bootstrap_seed, replicates=replicates
        )
    speakers = sorted(ridge)
    differences = np.array(
        [[per_seed[str(seed)]["speaker_differences"][s] for s in speakers] for seed, _ in pairs]
    )
    seed_mean = differences.mean(1)
    speaker_mean = differences.mean(0)
    mean = float(differences.mean())
    interaction = differences - seed_mean[:, None] - speaker_mean[None, :] + mean
    passed = sum(r["development_utility_passed"] for r in per_seed.values())
    return dict(
        seed_order=[s for s, _ in pairs],
        speaker_order=speakers,
        per_seed=per_seed,
        mean_difference=mean,
        between_seed_macro_sample_SD=float(seed_mean.std(ddof=1)),
        seed_macro_difference_range=[float(seed_mean.min()), float(seed_mean.max())],
        per_speaker_mean_difference=dict(zip(speakers, speaker_mean.tolist())),
        between_speaker_seed_mean_sample_SD=float(speaker_mean.std(ddof=1)),
        seed_speaker_interaction_RMS=float(np.sqrt(np.mean(interaction**2))),
        seed_speaker_difference_matrix=differences.tolist(),
        seeds_passed_original_criterion=passed,
        finite_seed_replication_gate="PASS" if passed == 5 else "FAIL",
        uncertainty_scope="Per-seed fixed-model speaker bootstrap only;seed spread descriptive;no joint or training-seed-population interval",
        P100_authorized=False,
        P960_authorized=False,
        stage2_ready=False,
    )
