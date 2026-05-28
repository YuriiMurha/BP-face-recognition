"""Bundle the 8 preprocessed chapter `.md` files into one `thesis.md`.

The bundle is for human review and sharing only -- pandoc still operates
on the per-chapter files when producing LaTeX. This is the "final
markdown file" deliverable described in the plan.

Output: `thesis/build/thesis.md`

Layout:

    ---
    title: <from PROGRESS.md or default>
    author: Yurii Murha
    date: <from PROGRESS.md or default>
    ---

    [chapter-01 contents]

    \\newpage

    [chapter-02 contents]

    ...

A YAML front-matter block at the top makes the bundle render correctly
through any pandoc-aware previewer (VS Code, GitHub, Obsidian).

Usage:
    python pipeline/concat_chapters.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_MD_DIR = REPO_ROOT / "thesis" / "build" / "md"
DEFAULT_OUTPUT = REPO_ROOT / "thesis" / "build" / "thesis.md"
PROGRESS_MD = REPO_ROOT / "thesis" / "PROGRESS.md"

DEFAULT_TITLE = "Face Recognition in Camera Footage"
DEFAULT_AUTHOR = "Yurii Murha"
DEFAULT_DATE = "May 2026"


def _front_matter(title: str, author: str, date: str) -> str:
    return (
        "---\n"
        f"title: {title}\n"
        f"author: {author}\n"
        f"date: {date}\n"
        "---\n\n"
    )


def _extract_date(progress: str) -> str:
    for line in progress.splitlines():
        m = re.search(r"due\s*[:\-]?\s*([A-Z][a-z]+\s+\d{4})", line, re.IGNORECASE)
        if m:
            return m.group(1)
    return DEFAULT_DATE


def concat(chapter_files: list[Path], title: str, author: str, date: str) -> str:
    parts: list[str] = [_front_matter(title, author, date)]
    for i, ch in enumerate(chapter_files):
        if i > 0:
            parts.append("\n\\newpage\n\n")
        parts.append(ch.read_text(encoding="utf-8").rstrip())
        parts.append("\n")
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=BUILD_MD_DIR,
        help=f"Source dir of preprocessed chapter .md (default: {BUILD_MD_DIR.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})",
    )
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--author", default=DEFAULT_AUTHOR)
    parser.add_argument("--date", default=None)
    args = parser.parse_args()

    chapter_files = sorted(args.input_dir.glob("chapter-*.md"))
    if not chapter_files:
        print(
            f"ERROR: no chapter-*.md under {args.input_dir}. "
            "Run preprocess_citations.py first.",
            file=sys.stderr,
        )
        return 2

    date = args.date
    if date is None and PROGRESS_MD.exists():
        date = _extract_date(PROGRESS_MD.read_text(encoding="utf-8"))
    date = date or DEFAULT_DATE

    bundle = concat(chapter_files, args.title, args.author, date)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(bundle, encoding="utf-8")

    n_lines = bundle.count("\n")
    n_chapters = len(chapter_files)
    print(
        f"Wrote {args.output.relative_to(REPO_ROOT)} "
        f"({n_chapters} chapters, {n_lines} lines, {len(bundle)} chars)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
