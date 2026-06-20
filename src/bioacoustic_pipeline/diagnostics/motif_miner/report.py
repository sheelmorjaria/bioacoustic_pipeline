"""Motif report formatting (verdict-facing text).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations


def format_motif_report(report) -> str:
    lines = ["=" * 60, "MOTIF MINER REPORT", "=" * 60]
    lines.append(f"Segments analyzed: {report.segment_count}")
    lines.append(f"Token motifs: {len(report.token_motifs)}")
    lines.append(f"Alias motifs: {len(report.alias_motifs)}")
    lines.append(f"Acoustic motifs: {len(report.acoustic_motifs)}")
    lines.append(f"Turn motifs: {len(report.turn_motifs)}")
    if report.token_motifs:
        lines.append("\nTop token motifs:")
        for m in report.token_motifs[:3]:
            lines.append(f"  {list(m.pattern)} support={m.support} npmi={m.npmi:.2f} score={m.score:.3f}")
    if report.alias_motifs:
        lines.append("\nTop alias motifs:")
        for m in report.alias_motifs[:3]:
            ag = f" gain={m.alias_gain:.2f}" if m.alias_gain is not None else ""
            lines.append(f"  {list(m.pattern)} support={m.support}{ag}")
    if report.summary:
        lines.append(f"\nSummary: {report.summary}")
    lines.append("=" * 60)
    return "\n".join(lines)
