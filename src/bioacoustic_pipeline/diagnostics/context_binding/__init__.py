"""Annotation-grounded context binding (token distribution ↔ behavioral label).

Functional-CONTEXT evidence from per-bout annotations (Emitter / Addressee /
Context). NOT semantic validation — a token class predictive of context is not a
token that MEANS the context (that needs Track C playback).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from bioacoustic_pipeline.diagnostics.context_binding.annotation_binding import (
    BindingResult,
    association_test,
    benjamini_hochberg,
    build_contingency,
    chi_square,
    mutual_information,
    shuffle_labels_global,
    shuffle_labels_within_group,
    test_token_annotations,
)

__all__ = [
    "BindingResult",
    "association_test",
    "benjamini_hochberg",
    "build_contingency",
    "chi_square",
    "mutual_information",
    "shuffle_labels_global",
    "shuffle_labels_within_group",
    "test_token_annotations",
]
