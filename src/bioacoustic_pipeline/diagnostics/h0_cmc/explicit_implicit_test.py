"""Explicit-vs-implicit communication test — is the signal acoustic or socially grounded?

H0-CMC Step 7. The test that classifies a signal into one of three communication modes:

  EXPLICIT_ACOUSTIC          — A(acoustic) ≈ G(full): the signal is self-contained in acoustics.
  SOCIALLY_GROUNDED_IMPLICIT — D(acoustic+social) ≫ A, D ≈ G: social context is needed.
  INSUFFICIENT               — A and D both low: the acoustic structure is insufficient.

``A`` = LogReg(acoustic_features) → context-label macro-F1 (what the acoustics alone recover).
``D`` = LogReg(acoustic + social features) → context-label macro-F1 (what adding social
covariates recovers). ``G`` = LogReg(full features) → theoretical upper bound.

The ``social_lift = D / A`` quantifies how much social context adds beyond acoustics alone.
On the bat (§9): A=2.7×, D=4.3×, G=4.7× → D/A=1.59 → SOCIALLY_GROUNDED_IMPLICIT.

This is the formal test behind the architecture's three-bucket thesis: "what acoustics encode"
(A high), "what acoustics do not encode" (A low, D low), and "what needs social validation"
(A low, D high). Pure-stdlib dataclass + lazy sklearn for the LogReg. NOT validated meaning.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np


class CommunicationMode:
    """The three H0-CMC communication modes (string constants, codebase idiom)."""

    EXPLICIT_ACOUSTIC = "explicit_acoustic_code"
    SOCIALLY_GROUNDED_IMPLICIT = "socially_grounded_implicit"
    INSUFFICIENT = "acoustic_structure_insufficient"


def explicit_vs_implicit_test(
    acoustic_features: np.ndarray,
    social_features: np.ndarray,
    context_labels: np.ndarray,
    full_features: np.ndarray | None = None,
    *,
    cv_folds: int = 5,
    random_state: int = 0,
) -> dict:
    """Classify a signal's communication mode via acoustic vs acoustic+social recovery.

    Args:
        acoustic_features: (n, D_ac) whole-call acoustic descriptors (e.g. fingerprint).
        social_features: (n, D_soc) social covariates (emitter/addressee/relationship one-hot).
        context_labels: (n,) context classes to predict.
        full_features: (n, D_full) ground-truth upper bound (all available features). If None,
            G is estimated as 1.0 (theoretical ceiling).
        cv_folds: cross-validation folds.
        random_state: RNG seed.

    Returns: {mode, accuracy_acoustic (A), accuracy_acoustic_social (D), accuracy_full (G),
        social_lift (D/A), rationale}.
    """
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.model_selection import cross_val_score  # noqa: PLC0415

    clf_kwargs = dict(max_iter=1000, class_weight="balanced", random_state=random_state)
    scoring = "f1_macro"

    # A: acoustic only
    acc_a = float(
        cross_val_score(
            LogisticRegression(**clf_kwargs),
            acoustic_features,
            context_labels,
            cv=min(cv_folds, _n_folds(context_labels)),
            scoring=scoring,
        ).mean()
    )

    # D: acoustic + social
    combined = np.hstack([acoustic_features, social_features])
    acc_d = float(
        cross_val_score(
            LogisticRegression(**clf_kwargs),
            combined,
            context_labels,
            cv=min(cv_folds, _n_folds(context_labels)),
            scoring=scoring,
        ).mean()
    )

    # G: full (upper bound)
    if full_features is not None:
        acc_g = float(
            cross_val_score(
                LogisticRegression(**clf_kwargs),
                full_features,
                context_labels,
                cv=min(cv_folds, _n_folds(context_labels)),
                scoring=scoring,
            ).mean()
        )
    else:
        acc_g = 1.0  # theoretical ceiling

    social_lift = acc_d / acc_a if acc_a > 0.01 else float("inf")

    # Classify the communication mode.
    if acc_a >= 0.85 * acc_g:
        mode = CommunicationMode.EXPLICIT_ACOUSTIC
        rationale = f"A={acc_a:.3f} ≈ G={acc_g:.3f} → signal self-contained in acoustics"
    elif acc_d > acc_a * 1.15 and acc_d >= 0.85 * acc_g:
        mode = CommunicationMode.SOCIALLY_GROUNDED_IMPLICIT
        rationale = (
            f"D={acc_d:.3f} ≫ A={acc_a:.3f} (lift {social_lift:.2f}×), D ≈ G={acc_g:.3f} "
            f"→ social context needed"
        )
    else:
        mode = CommunicationMode.INSUFFICIENT
        rationale = f"A={acc_a:.3f}, D={acc_d:.3f}, G={acc_g:.3f} → acoustic structure insufficient"

    return {
        "mode": mode,
        "accuracy_acoustic": round(acc_a, 3),
        "accuracy_acoustic_social": round(acc_d, 3),
        "accuracy_full": round(acc_g, 3),
        "social_lift": round(social_lift, 2),
        "rationale": rationale,
        "n_samples": len(context_labels),
        "n_context_classes": len(set(context_labels)),
    }


def _n_folds(labels) -> int:
    """Min CV folds = min class count (sklearn requires ≥2 per class for stratified)."""
    from collections import Counter  # noqa: PLC0415

    counts = Counter(labels)
    min_cls = min(counts.values()) if counts else 1
    return max(2, min(min_cls, 5))
