"""Caller-identity disentanglement for the 44-D acoustic fingerprint.

Three firewall-compliant, deterministic methods to scrub caller identity from the
fingerprint while preserving call-form structure. All operate purely on the
all-species acoustic fingerprint (no cross-species model weights, no torch).

Context: §26 proved animal calls are non-additively caller-bound (raw distance
ratio ≈ 1). §25 proved per-speaker mean-subtraction (centering) fails for human
speech (Δ−0.003). These methods attempt stronger disentanglement via structural
(delta), algebraic (orthogonal projection), and affine (Z-normalization) transforms.

Success criterion: caller Cramér's V drops toward chance while call-type V
increases or remains stable. If BOTH drop, the speaker and form subspaces overlap
(non-linear entanglement — §26's finding holds, and linear methods are insufficient).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np


def disentangle_delta(fps_per_call: list[np.ndarray]) -> list[np.ndarray]:
    """Method 1: Delta-trajectory — structural disentanglement.

    Replaces absolute sub-call fingerprints with inter-unit differences:
    Δ_i = fp_{i+1} - fp_i. The static caller baseline (vocal tract length,
    larynx size) cancels in differencing; the dynamic motor pattern (form) remains.

    No caller labels required. Non-linear by construction. Best for DUSTED
    motif discovery (DTW on delta trajectories), not for k-means call-type
    recovery (loses absolute spectral shape).

    Args:
        fps_per_call: list of (n_units_i, 44) arrays — one per call.

    Returns: list of (n_units_i - 1, 44) delta arrays.
    """
    deltas = []
    for fps in fps_per_call:
        if len(fps) >= 2:
            deltas.append(np.diff(fps, axis=0).astype(np.float32))
        else:
            deltas.append(np.zeros((0, fps.shape[1] if fps.ndim > 1 else 1), dtype=np.float32))
    return deltas


def disentangle_orthogonal(
    fingerprints: np.ndarray,
    caller_labels: np.ndarray,
    *,
    method: str = "lda",
    n_components: int | None = None,
    random_state: int = 0,
) -> np.ndarray:
    """Method 2: Orthogonal subspace projection — algebraic disentanglement.

    Learns the caller-discriminating subspace from the data, then projects the
    fingerprints orthogonally to it. This forces the 44-D space to contain zero
    linear information about the caller. If form and identity are linearly separable,
    call-type structure survives; if they overlap (§26), both degrade.

    Args:
        fingerprints: (n, 44) acoustic fingerprint matrix.
        caller_labels: (n,) caller/emitter/individual IDs.
        method: "lda" (Linear Discriminant Analysis — principled between/within
            variance maximization) or "logreg" (Logistic Regression weight matrix).
        n_components: number of speaker-discriminating directions to project out.
            If None, uses min(n_callers - 1, n_features) for LDA or the rank of
            the LogReg weight matrix.
        random_state: RNG seed for LogReg (LDA is deterministic).

    Returns: (n, 44) disentangled fingerprints (caller subspace removed).
    """
    X = np.asarray(fingerprints, dtype=np.float64)
    unique_callers = np.unique(caller_labels)
    if len(unique_callers) < 2:
        return X.astype(np.float32)  # can't disentangle without ≥2 callers

    if method == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: PLC0415

        lda = LinearDiscriminantAnalysis()
        lda.fit(X, caller_labels)
        # lda.scalings_ has shape (n_features, n_dirs). Columns are speaker directions
        # in feature space. We need the COLUMN space → use U from SVD (not Vt).
        scalings = lda.scalings_
        U, S, _Vt = np.linalg.svd(scalings, full_matrices=False)
        rank = np.sum(S > 1e-10)
        if n_components is not None:
            rank = min(rank, n_components)
        speaker_basis = U[:, :rank].T  # (rank, n_features) orthonormal rows
    elif method == "logreg":
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
        clf.fit(X, caller_labels)
        W = clf.coef_  # (n_classes or n_classes-1, 44)
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        rank = np.sum(S > 1e-10)
        if n_components is not None:
            rank = min(rank, n_components)
        speaker_basis = Vt[:rank]
    else:
        raise ValueError(f"unknown method: {method!r} (use 'lda' or 'logreg')")

    # Project orthogonally to the speaker subspace:
    # FP_disentangled = FP - FP @ speaker_basis^T @ speaker_basis
    projection = X @ speaker_basis.T @ speaker_basis
    disentangled = X - projection
    return disentangled.astype(np.float32)


def disentangle_znorm(
    fingerprints: np.ndarray,
    caller_labels: np.ndarray,
    *,
    min_calls_per_caller: int = 5,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Method 3: Per-speaker Z-normalization — VTLN proxy.

    For each caller, center (subtract mean) and scale (divide by std) across all
    their calls. This forces every animal's acoustic distribution to mean=0, std=1,
    neutralizing gross physical differences while preserving relative patterns.

    Stronger than pure centering (which failed in §25) because the σ-scaling
    normalizes per-dimension dynamic range. May still fail if the entanglement is
    non-linear (§26's "non-additive" finding).

    Args:
        fingerprints: (n, 44) acoustic fingerprint matrix.
        caller_labels: (n,) caller/emitter IDs.
        min_calls_per_caller: callers with fewer calls are not normalized (kept raw).
        epsilon: numerical stability for division by σ.

    Returns: (n, 44) Z-normalized fingerprints.
    """
    X = np.asarray(fingerprints, dtype=np.float64)
    normalized = X.copy()
    for caller in np.unique(caller_labels):
        mask = caller_labels == caller
        if mask.sum() < min_calls_per_caller:
            continue  # too few calls — don't normalize (avoid overfitting)
        mu = X[mask].mean(axis=0)
        sigma = X[mask].std(axis=0) + epsilon
        normalized[mask] = (X[mask] - mu) / sigma
    return normalized.astype(np.float32)


def evaluate_disentanglement(
    fingerprints: np.ndarray,
    caller_labels: np.ndarray,
    form_labels: np.ndarray,
    k: int = 24,
    n_permutations: int = 200,
    seed: int = 0,
) -> dict:
    """Evaluate how well a fingerprint space separates form from identity.

    Runs k-means on the fingerprints, then computes:
    - caller_v: Cramér's V between k-means tokens and caller labels (should be LOW)
    - form_v: Cramér's V between k-means tokens and form (call-type) labels (should be HIGH)
    - disentangle_ratio: form_v / caller_v (higher = better disentanglement)

    Args:
        fingerprints: (n, 44) fingerprint matrix (raw or disentangled).
        caller_labels: (n,) caller IDs.
        form_labels: (n,) call-type / context labels.
        k: number of k-means clusters.
        n_permutations: permutation null iterations.
        seed: RNG seed.

    Returns: {caller_v, caller_p, form_v, form_p, disentangle_ratio}.
    """
    from bioacoustic_pipeline.diagnostics.context_binding import association_test  # noqa: PLC0415
    from sklearn.cluster import MiniBatchKMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    X = np.asarray(fingerprints, dtype=np.float32)
    scaler = StandardScaler().fit(X)
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init="auto", batch_size=4096)
    tokens = km.fit_predict(scaler.transform(X))

    caller_result = association_test(
        [[int(t)] for t in tokens],
        list(caller_labels),
        statistic="nmi",
        n_permutations=n_permutations,
        seed=seed,
    ).as_dict()
    form_result = association_test(
        [[int(t)] for t in tokens],
        list(form_labels),
        statistic="nmi",
        n_permutations=n_permutations,
        seed=seed,
    ).as_dict()

    caller_v = caller_result["cramers_v"]
    form_v = form_result["cramers_v"]
    return {
        "caller_v": round(caller_v, 3),
        "caller_p": caller_result["p_value"],
        "form_v": round(form_v, 3),
        "form_p": form_result["p_value"],
        "disentangle_ratio": round(form_v / max(caller_v, 1e-6), 2),
    }
