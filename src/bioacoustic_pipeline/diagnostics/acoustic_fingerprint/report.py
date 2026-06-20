"""Text report for the acoustic-fingerprint evidence channel (verdict-facing).

This feeds the verdict engine's evidence/falsification layer. The fingerprint itself is
NOT a semantic claim — it audits learned tokens and compares segment acoustics.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np


def format_fingerprint_report(species, token_coherence=None, alias_divergence=None,
                              token_null=None, alias_null=None) -> str:
    lines = ["=" * 60, "ACOUSTIC FINGERPRINT EVIDENCE", "=" * 60]
    lines.append(f"Species: {species}")
    lines.append("Framing: physics-grounded descriptor (not a meaning claim).")
    if token_coherence:
        cohs = [v["cohesion"] for v in token_coherence.values()]
        mean = float(np.mean(cohs)) if cohs else 0.0
        lines.append(f"Token-fingerprint coherence: mean={mean:.3f} "
                     f"(high => VQ tokens are acoustically coherent)")
        if token_null is not None:
            lines.append(f"  null: p={token_null.p_value:.3f} ({token_null.null_type})")
    if alias_divergence:
        divs = [v["divergence"] for v in alias_divergence.values()]
        mean = float(np.mean(divs)) if divs else 0.0
        lines.append(f"Alias-acoustic divergence: mean={mean:.3f} "
                     f"(high => aliases are different sounds, same role)")
        if alias_null is not None:
            lines.append(f"  null: p={alias_null.p_value:.3f} ({alias_null.null_type})")
    lines.append("=" * 60)
    return "\n".join(lines)
