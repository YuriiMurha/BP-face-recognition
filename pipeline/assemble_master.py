"""Regenerate `tukethesis.tex` master from the legacy TUKE template.

Reads `LaTeX/tukethesis.tex` (the template inherited from the legacy "BP"
Overleaf project) and produces `thesis/build/tex/tukethesis.tex` with:

  * The 5-block `\\include` list replaced by 8 `\\include{chapters/chapter-NN}`.
  * `\\include{bibliography}` replaced by `\\bibliography{references}`.
  * The dead `\\bibliographystyle{dcu}` line (active before the body) removed.
  * The dead `\\bibliographystyle{plain}` line (after `\\end{document}`) removed.
  * A canonical `\\bibliographystyle{plain}` inserted just before
    `\\bibliography{references}`.
  * Preamble additions for unicode and pandoc-friendly packages.
  * `\\dateofsubmission` updated to the value in PROGRESS.md (best-effort
    extraction; defaults to "May 2026" if not parseable).

The script is deterministic and idempotent: re-running on its own output
yields the same output (modulo the dateofsubmission value).

Usage:
    python pipeline/assemble_master.py
    python pipeline/assemble_master.py --template path/to/source.tex
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = REPO_ROOT / "LaTeX" / "tukethesis.tex"
DEFAULT_OUTPUT = REPO_ROOT / "thesis" / "build" / "tex" / "tukethesis.tex"
PROGRESS_MD = REPO_ROOT / "thesis" / "PROGRESS.md"

def _build_includes(main_chapters: range, appendix_chapters: range) -> str:
    """Emit \\include lines for main chapters, then \\appendix, then appendices.

    Between the last main chapter and the first appendix we also emit a
    "List of Appendices" section (unnumbered, but added to the table of
    contents) that mirrors the Zoznam priloh pattern from the legacy
    template -- it gives the reader a one-line description of each
    appendix before they encounter the appendix bodies.

    The \\appendix command must appear exactly once and BEFORE the first
    appendix \\include. After it fires, LaTeX numbers subsequent
    \\section commands with letters (A, B, C, ...) instead of digits.
    """
    lines = [f"\\include{{chapters/chapter-{n:02d}}}" for n in main_chapters]
    lines.append(LIST_OF_APPENDICES)
    lines.append("\\appendix")
    lines.extend(f"\\include{{chapters/chapter-{n:02d}}}" for n in appendix_chapters)
    return "\n".join(lines)


LIST_OF_APPENDICES = r"""%
%% List of Appendices -- mirrors the legacy Zoznam priloh pattern.
\section*{List of Appendices}
\addcontentsline{toc}{section}{List of Appendices}
\begin{description}
    \item[Appendix A] User Manual --- installation, configuration, and
        day-to-day operation of the face recognition system for the
        end user.
    \item[Appendix B] System Manual --- technical reference for the
        repository layout, component architecture, configuration files,
        training workflow, and reproducibility of the experimental
        results.
\end{description}
\newpage
%"""


NEW_INCLUDES = _build_includes(
    main_chapters=range(1, 9),       # chapters 01..08
    appendix_chapters=range(9, 11),  # chapters 09 (User Manual), 10 (System Manual)
)

PREAMBLE_ADDITIONS = r"""%
%% --- Preamble additions for pandoc-generated LaTeX ----------------------
\usepackage{listings}      % pandoc --listings code blocks
\usepackage{listingsutf8}  % accept arbitrary UTF-8 inside lstlisting
\usepackage{booktabs}      % pandoc table output (\toprule, \midrule)
\usepackage{longtable}     % multi-page pandoc tables
\usepackage{calc}          % pandoc proportional column widths (p{(\linewidth - ...) * \real{...}})
\usepackage{array}         % needed for >{...} column-spec prefixes used by pandoc
\usepackage[pdftex]{graphicx}% pre-load with template's option so tikz's graphicx load doesn't clash
\usepackage{tikz}          % vector diagrams (FaceNet architecture figure)
\usetikzlibrary{arrows.meta, positioning}
\usepackage{chngcntr}      % \counterwithin for section-scoped figure/table numbers
%% Reset figure and table counters at each section boundary, then format them
%% as "N.M" so figures/tables in chapter 7 read as "Figure 7.1", "Table 7.1",
%% etc. matching the author-typed prefixes in the markdown source. This is
%% needed for \listoffigures / \listoftables to show book-style numbering.
\counterwithin{figure}{section}
\counterwithin{table}{section}
\renewcommand{\thefigure}{\thesection.\arabic{figure}}
\renewcommand{\thetable}{\thesection.\arabic{table}}
\usepackage{textcomp}
\usepackage{amssymb}
\usepackage{newunicodechar}
\newunicodechar{≥}{$\geq$}
\newunicodechar{≤}{$\leq$}
\newunicodechar{≈}{$\approx$}
\newunicodechar{→}{$\rightarrow$}
\newunicodechar{×}{$\times$}
\newunicodechar{−}{$-$}
%% Cyrillic homoglyph fixes (typos in legacy appendix files).
\newunicodechar{с}{c}    % Cyrillic small letter es U+0441 -> Latin c
\newunicodechar{а}{a}    % Cyrillic small letter a  U+0430 -> Latin a
\newunicodechar{е}{e}    % Cyrillic small letter ie U+0435 -> Latin e
\newunicodechar{о}{o}    % Cyrillic small letter o  U+043E -> Latin o
\newunicodechar{р}{p}    % Cyrillic small letter er U+0440 -> Latin p
%% Pandoc emits several commands that are normally defined by its default
%% LaTeX template (which we don't include because we're generating
%% fragments). Provide fallbacks so the master compiles cleanly.
\providecommand{\pandocbounded}[1]{#1}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\passthrough}[1]{#1}
\providecommand{\real}[1]{#1}
%% -----------------------------------------------------------------------
"""

GENERATED_HEADER = (
    "%% =====================================================================\n"
    "%% AUTO-GENERATED by pipeline/assemble_master.py\n"
    "%% Source: LaTeX/tukethesis.tex (TUKE template, legacy BP Overleaf)\n"
    "%% Chapter bodies are generated from thesis/chapters/*.md by pandoc.\n"
    "%% Manual edits here will be overwritten. Edit the SOURCES:\n"
    "%%   - thesis/chapters/NN-name.md  (chapter content + figures + cites)\n"
    "%%   - LaTeX/tukethesis.tex        (title, abstracts, acknowledgement,\n"
    "%%                                  preface, glossary, declaration)\n"
    "%% =====================================================================\n"
)


DEFAULT_SUBMISSION_DATE = "May. 29. 2026"  # TUKE class expects Month.Day.Year


def _extract_submission_date(progress: str) -> str | None:
    """Best-effort: find a submission date in PROGRESS.md.

    The TUKE class macro `\\vyberrok` parses the date by splitting on
    literal `.` -- it needs `Month. Day. Year` format. Any other format
    causes a fatal LaTeX error at `\\firstpage`. So:

      * If we find "Month Day, Year" or "Month Day Year" -> reformat.
      * If we find just "Month Year" -> use day 30 as a placeholder.
      * Otherwise return None and let the caller use the default.
    """
    full = re.search(
        r"\b([A-Z][a-z]+)\s+(\d{1,2})(?:,|\s)\s*(\d{4})\b", progress
    )
    if full:
        return f"{full.group(1)}. {full.group(2)}. {full.group(3)}"
    short = re.search(
        r"due\s*[:\-]?\s*([A-Z][a-z]+)\s+(\d{4})", progress, re.IGNORECASE
    )
    if short:
        return f"{short.group(1)}. 30. {short.group(2)}"
    return None


def assemble(
    template_text: str,
    submission_date: str | None = None,
) -> str:
    """Apply all transformations to the template and return the new master."""
    text = template_text

    # Replace the legacy include list with the new 8-chapter list. Match
    # from the first `\include{introduction}` through `\include{conclusion}`.
    # The legacy file has each on its own line separated by `%` blank lines;
    # we collapse those into a contiguous block.
    legacy_includes_re = re.compile(
        r"\\include\{introduction\}.*?\\include\{conclusion\}",
        re.DOTALL,
    )
    if not legacy_includes_re.search(text):
        raise SystemExit(
            "ERROR: could not find legacy include list (\\include{introduction} ... "
            "\\include{conclusion}) in the template."
        )
    text = legacy_includes_re.sub(lambda m: NEW_INCLUDES, text, count=1)

    # Remove `\include{appendices}` -- the appendices.tex file itself contains
    # \include{appendixa}, \include{appendixb}, \include{appendixc} which
    # nests inside the outer include and triggers
    # "LaTeX Error: \include cannot be nested". The master already includes
    # each appendix file directly, so the wrapper is redundant.
    text = re.sub(r"\\include\{appendices\}\s*\n?", "", text, count=1)

    # Drop legacy appendix includes. appendixa.tex carries placeholder text
    # describing CD/User Manual/System Manual; appendixb.tex and appendixc.tex
    # are empty. They were inherited from the BP project and aren't relevant
    # to the new thesis -- the user can add real appendices later, and the
    # bibliography \include is the only one we keep in the back matter.
    text = re.sub(r"\\include\{appendix[a-c]\}[^\n]*\n?", "", text)

    # Replace `\include{bibliography}` with `\bibliography{references}`,
    # preceded by `\bibliographystyle{plain}`.
    text = re.sub(
        r"\\include\{bibliography\}",
        lambda m: "\\bibliographystyle{plain}\n\\bibliography{references}",
        text,
        count=1,
    )

    # Remove pre-body `\bibliographystyle{dcu}` (was line ~99 in legacy).
    text = re.sub(r"^\\bibliographystyle\{dcu\}\s*\n", "", text, flags=re.MULTILINE)

    # Remove the dead `\bibliographystyle{plain}` that sits right before
    # \end{document} in the legacy template (it's after all \include calls,
    # so it's redundant -- the active style is the one before
    # \bibliography{references}).
    text = re.sub(
        r"\\bibliographystyle\{plain\}\s*\n(?=\\end\{document\})",
        lambda m: "",
        text,
        count=1,
    )

    # Insert preamble additions after `\usepackage{parskip}%`.
    insert_anchor = re.compile(r"^\\usepackage\{parskip\}%\s*\n", re.MULTILINE)
    if not insert_anchor.search(text):
        raise SystemExit(
            "ERROR: could not find \\usepackage{parskip}% anchor for preamble "
            "additions."
        )
    text = insert_anchor.sub(
        lambda m: m.group(0) + PREAMBLE_ADDITIONS, text, count=1
    )

    # Update \dateofsubmission if a submission date is provided.
    if submission_date:
        text = re.sub(
            r"\\dateofsubmission\{[^}]*\}",
            lambda m: f"\\dateofsubmission{{{submission_date}}}",
            text,
            count=1,
        )

    # Stick a generated-header banner at the very top (idempotent: only
    # one banner).
    text = re.sub(
        r"^%% =+\n%% AUTO-GENERATED.*?%% =+\n",
        "",
        text,
        flags=re.DOTALL | re.MULTILINE,
        count=1,
    )
    text = GENERATED_HEADER + text

    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help=f"Source template (default: {DEFAULT_TEMPLATE.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--submission-date",
        type=str,
        default=None,
        help="Override \\dateofsubmission value (e.g., 'May 2026').",
    )
    args = parser.parse_args()

    if not args.template.exists():
        print(f"ERROR: template not found: {args.template}", file=sys.stderr)
        return 2

    template_text = args.template.read_text(encoding="utf-8")

    submission_date = args.submission_date
    if submission_date is None and PROGRESS_MD.exists():
        submission_date = _extract_submission_date(
            PROGRESS_MD.read_text(encoding="utf-8")
        )
    submission_date = submission_date or DEFAULT_SUBMISSION_DATE

    new_master = assemble(template_text, submission_date=submission_date)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(new_master, encoding="utf-8")

    print(f"Wrote {args.output.relative_to(REPO_ROOT)}")
    print(f"\\dateofsubmission = {submission_date}")
    print(f"\\include list      = 8 chapters (chapter-01 ... chapter-08)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
