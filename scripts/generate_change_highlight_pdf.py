#!/usr/bin/env python3
"""Generate a dated PDF highlighting manuscript changes in red vs baseline."""

from __future__ import annotations

import difflib
import os
from datetime import date

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO, "paper_files", "bare_jrnl_original_baseline.tex")
CURR = os.path.join(REPO, "paper_files", "bare_jrnl_new_sample4.tex")
OUT = os.path.join(
    REPO,
    "paper_files",
    f"MMCTL_manuscript_changes_{date.today().isoformat()}.pdf",
)
TRACKER = os.path.join(REPO, "results", "REVIEW_COMMENTS_TRACKER.md")


def esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def main():
    with open(BASE, encoding="utf-8", errors="replace") as f:
        a = f.read().splitlines()
    with open(CURR, encoding="utf-8", errors="replace") as f:
        b = f.read().splitlines()

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "T",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=12,
        spaceBefore=12,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "B",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        spaceAfter=4,
    )
    red = ParagraphStyle(
        "Red",
        parent=body,
        textColor=colors.HexColor("#CC0000"),
        fontName="Courier",
        fontSize=7.5,
        leading=9,
    )
    green = ParagraphStyle(
        "GreenNote",
        parent=body,
        textColor=colors.HexColor("#006600"),
        fontName="Courier",
        fontSize=7.5,
        leading=9,
    )
    gray = ParagraphStyle(
        "Gray",
        parent=body,
        textColor=colors.HexColor("#555555"),
        fontName="Courier",
        fontSize=7,
        leading=8.5,
    )

    doc = SimpleDocTemplate(
        OUT,
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    story = []
    today = date.today().isoformat()
    story.append(Paragraph("MMCTL Manuscript Change Highlight", title))
    story.append(Paragraph(f"<b>Change date:</b> {today}", body))
    story.append(
        Paragraph(
            f"<b>Baseline:</b> {esc(os.path.basename(BASE))} (first committed draft)<br/>"
            f"<b>Current:</b> {esc(os.path.basename(CURR))} (revision with B1–B3 edits)<br/>"
            "<font color='#CC0000'><b>Red</b></font> = added / changed lines &nbsp;&nbsp; "
            "<font color='#006600'><b>Green</b></font> = removed baseline lines (context)",
            body,
        )
    )
    story.append(Spacer(1, 8))

    # Summary from tracker if present
    story.append(Paragraph("Reviewer-comment status snapshot", h2))
    if os.path.exists(TRACKER):
        with open(TRACKER, encoding="utf-8") as f:
            tracker_lines = f.read().splitlines()
        # pull status table rows only
        for line in tracker_lines:
            if line.startswith("| R") or line.startswith("| —") or (
                line.startswith("| ")
                and ("DONE" in line or "OPEN" in line or "PARTIAL" in line or "DEFERRED" in line)
            ):
                if line.startswith("| #") or line.startswith("|---") or line.startswith("| Edit"):
                    continue
                # color status words
                html = esc(line)
                html = html.replace("**DONE**", "<font color='#006600'><b>DONE</b></font>")
                html = html.replace("**PARTIAL**", "<font color='#CC6600'><b>PARTIAL</b></font>")
                html = html.replace("**OPEN**", "<font color='#CC0000'><b>OPEN</b></font>")
                html = html.replace("**DEFERRED**", "<font color='#CC0000'><b>DEFERRED</b></font>")
                html = html.replace("**OPEN / IN PROGRESS**", "<font color='#CC0000'><b>OPEN / IN PROGRESS</b></font>")
                story.append(Paragraph(html, body))
    else:
        story.append(Paragraph("Tracker file not found.", body))

    story.append(Paragraph("Line-level TeX diff (red = new/changed)", h2))
    story.append(
        Paragraph(
            "Only changed hunks are shown. Unchanged context lines are gray.",
            body,
        )
    )

    sm = difflib.SequenceMatcher(None, a, b)
    n_add = n_del = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            # show at most 1 context line around changes is handled by difflib hunks
            continue
        story.append(
            Paragraph(
                f"<b>Hunk</b> baseline L{i1+1}–{i2} → current L{j1+1}–{j2} ({tag})",
                body,
            )
        )
        if tag in ("replace", "delete"):
            for line in a[i1:i2]:
                n_del += 1
                story.append(Paragraph("− " + esc(line[:240]), green))
        if tag in ("replace", "insert"):
            for line in b[j1:j2]:
                n_add += 1
                story.append(Paragraph("+ " + esc(line[:240]), red))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            f"<b>Diff stats:</b> {n_add} added/changed lines shown in red; "
            f"{n_del} removed baseline lines shown in green.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Full comment checklist: <font face='Courier'>results/REVIEW_COMMENTS_TRACKER.md</font>. "
            "Numeric results: <font face='Courier'>results/RESULTS_TRACKER.md</font>.",
            body,
        )
    )

    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
