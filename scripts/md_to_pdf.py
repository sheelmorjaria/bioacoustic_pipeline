#!/usr/bin/env python3
"""Simple markdown-to-PDF converter using fpdf2 + DejaVu Unicode fonts."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from fpdf import FPDF

DV = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DVB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
DVI = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"
DVM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
DVS = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
DVSB = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"


def md_to_pdf(md_path: Path, pdf_path: Path):
    lines = md_path.read_text(encoding="utf-8").split("\n")
    pdf = FPDF("P", "mm", "A4")
    pdf.set_auto_page_break(True, 20)
    pdf.set_margins(18, 15, 18)
    pdf.add_page()
    pdf.add_font("DV", "", DV)
    pdf.add_font("DV", "B", DVB)
    pdf.add_font("DV", "I", DVI)
    pdf.add_font("DVM", "", DVM)
    pdf.add_font("DVS", "", DVS)
    pdf.add_font("DVS", "B", DVSB)

    W = 174  # usable width mm (A4=210, margins 18+18)

    for line in lines:
        s = line.rstrip()

        # Skip code fences
        if s.strip().startswith("```"):
            continue

        # Headings
        if s.startswith("# "):
            pdf.ln(4)
            pdf.set_font("DVS", "B", 16)
            pdf.set_text_color(26, 26, 46)
            pdf.multi_cell(W, 8, s[2:])
            pdf.ln(2)
            pdf.set_text_color(0, 0, 0)
        elif s.startswith("## "):
            pdf.ln(3)
            pdf.set_font("DVS", "B", 13)
            pdf.set_text_color(22, 33, 62)
            pdf.multi_cell(W, 6, s[3:])
            pdf.ln(1)
            pdf.set_text_color(0, 0, 0)
        elif s.startswith("### "):
            pdf.ln(2)
            pdf.set_font("DV", "B", 11)
            pdf.set_text_color(15, 52, 96)
            pdf.multi_cell(W, 5, s[4:])
            pdf.set_text_color(0, 0, 0)
        elif s.startswith("#### "):
            pdf.ln(1)
            pdf.set_font("DV", "B", 10)
            pdf.multi_cell(W, 5, s[5:])

        # Horizontal rule
        elif s.strip() in ("---", "***"):
            pdf.ln(1)
            y = pdf.get_y()
            pdf.set_draw_color(180, 180, 180)
            pdf.line(18, y, 192, y)
            pdf.ln(2)

        # Empty line
        elif s.strip() == "":
            pdf.ln(2)

        # Table separator (skip)
        elif re.match(r"^\s*\|[\s\-:|]+\|\s*$", s):
            continue

        # Table row
        elif s.strip().startswith("|"):
            cells = [c.strip()[:40] for c in s.strip().strip("|").split("|")]
            n = max(len(cells), 1)
            cw = W / n
            for c in cells:
                pdf.set_font("DV", "", 8)
                try:
                    pdf.cell(cw, 5, c, border=1)
                except Exception:
                    pdf.cell(cw, 5, "", border=1)
            pdf.ln()

        # Bullet point
        elif s.lstrip().startswith("- ") or s.lstrip().startswith("* "):
            text = "  " + s.lstrip()[2:]
            pdf.set_font("DV", "", 10)
            pdf.multi_cell(W, 5, text)

        # Regular text (catch-all)
        else:
            pdf.set_font("DV", "", 10)
            if len(s) > 500:
                s = s[:500] + "..."
            pdf.multi_cell(W, 5, s)

    pdf.output(str(pdf_path))


if __name__ == "__main__":
    files = sys.argv[1:] if len(sys.argv) > 1 else [
        "MANUSCRIPT.md", "README.md", "ARCHITECTURE.md", "BENCHMARK.md"
    ]
    for f in files:
        p = Path(f)
        if not p.exists():
            print(f"SKIP {f}")
            continue
        try:
            md_to_pdf(p, p.with_suffix(".pdf"))
            kb = p.with_suffix(".pdf").stat().st_size / 1024
            print(f"OK {f} -> {p.with_suffix('.pdf').name} ({kb:.0f} KB)")
        except Exception as e:
            print(f"ERROR {f}: {e}")
