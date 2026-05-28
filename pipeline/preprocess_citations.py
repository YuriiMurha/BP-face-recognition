"""Convert `[CITE: ...]` markers and chapter-04 numeric refs into `\\cite{key}`.

Three citation flavors are handled:

  A. Clean bib keys    -- `[CITE: schroff2015facenet]`, `[CITE: a, b; c]`
  B. Author-year prose -- `[CITE: Schroff et al. 2015]` (via alias map)
  C. Numeric inline    -- `[1]`, `[2]` ... in chapter 04 only (via numref map)

Chapter 04 has its own trailing `## References` block; the preprocessor drops it
because the bibliography is rendered from `references.bib` at the LaTeX level.

The transformation is text-in / text-out so it can be unit-tested without
filesystem dependencies. The CLI wraps it for batch use against the 8
chapter files.

Usage as CLI:
    python pipeline/preprocess_citations.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHAPTERS_DIR = REPO_ROOT / "thesis" / "chapters"
BUILD_DIR = REPO_ROOT / "thesis" / "build"
MD_OUT_DIR = BUILD_DIR / "md"
ALIASES_PATH = BUILD_DIR / "citation_aliases.json"
BIBFILE = REPO_ROOT / "thesis" / "references.bib"


def _numref_path(chapter_num: str) -> Path:
    """Per-chapter numeric-citation map: thesis/build/chapter{NN}_numref_map.json.

    Any chapter whose source markdown ends with a `## References` heading is
    expected to use Flavor-C numeric citations (`[1]`, `[2]`, ...) in the
    body. The trailing References block is dropped during preprocessing and
    each `[N]` is replaced with `\\cite{<bibkey>}` where the mapping lives in
    a per-chapter JSON file named by convention.
    """
    return BUILD_DIR / f"chapter{chapter_num}_numref_map.json"

# Same regexes as build_citation_aliases.py - keep them DRY conceptually but
# duplicated here so each script is self-contained for the user.
CITE_RE = re.compile(r"\[CITE:\s*([^\]]*?)\s*\]")
BIBKEY_RE = re.compile(r"^[a-z][a-z0-9_-]*\d{4}[a-z0-9]*$")
SPLIT_RE = re.compile(r"[;,]")

# Numeric inline citation: `[1]`, `[12]`. Used only in chapter 04 body.
NUMREF_RE = re.compile(r"\[(\d+)\]")

# Markdown link `[label](url)`: masked before NUMREF processing.
MD_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")

# Strip "Chapter N:" prefix from a level-1 heading so LaTeX owns the chapter
# number. `# Chapter 1: Introduction` -> `# Introduction`.
HEADING_CHAPTER_RE = re.compile(
    r"^(#)[^\S\n]+Chapter[^\S\n]+\d+[^\S\n]*:[^\S\n]*(.+?)[^\S\n]*$",
    re.MULTILINE,
)

# Strip dotted numeric/letter prefix from sub-headings so LaTeX owns the
# section number. `## 1.1 Background` -> `## Background`,
# `## A.1 Overview` -> `## Overview`. Matches level 2-6 headings whose body
# starts with a dotted index like `1.1`, `7.1.1`, `A.1`, `B.10`.
# `[^\S\n]` matches whitespace EXCEPT newlines so we don't eat the line break.
HEADING_SECTION_RE = re.compile(
    r"^(#{2,6})[^\S\n]+(?:[A-Z]|\d+)(?:\.\d+)+[^\S\n]+(.+?)[^\S\n]*$",
    re.MULTILINE,
)


def _split_payload(payload: str) -> list[str]:
    return [t.strip() for t in SPLIT_RE.split(payload) if t.strip()]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "EMPTY"


def _resolve_token(
    token: str,
    aliases: dict[str, str | None],
    bib_keys: frozenset[str] | None,
) -> tuple[list[str], bool]:
    """Resolve a single token to a list of bib keys.

    Returns (keys, ok). `ok` is False when the token is unresolved.
    """
    # Bib-key shape, or known bib key when the bib is loaded.
    if BIBKEY_RE.match(token) or (bib_keys is not None and token in bib_keys):
        return [token], True

    mapped = aliases.get(token)
    if mapped is None:
        return [f"UNRESOLVED_{_slug(token)}"], False
    # An alias value may itself be comma-separated (multi-cite shortcut).
    keys = [k.strip() for k in mapped.split(",") if k.strip()]
    return keys, True


def _replace_cite(
    text: str,
    aliases: dict[str, str | None],
    bib_keys: frozenset[str] | None,
    warnings: list[str],
) -> str:
    """Replace every `[CITE: ...]` with `\\cite{key1,key2,...}`."""

    def _sub(m: re.Match[str]) -> str:
        payload = m.group(1)
        tokens = _split_payload(payload)
        if not tokens:
            warnings.append("empty CITE payload")
            return "\\cite{UNRESOLVED_EMPTY}"
        resolved: list[str] = []
        for tok in tokens:
            keys, ok = _resolve_token(tok, aliases, bib_keys)
            if not ok:
                warnings.append(f"unresolved citation token: {tok!r}")
            resolved.extend(keys)
        return "\\cite{" + ",".join(resolved) + "}"

    return CITE_RE.sub(_sub, text)


def _replace_numref(
    text: str,
    numref_map: dict[str, str | None],
    warnings: list[str],
) -> str:
    """Replace `[N]` markers (chapter 04 only) with `\\cite{key}`.

    Markdown links are masked out first so labels like `[Section 4.1](#4-1)`
    are not matched. The masked text is reconstructed afterwards.
    """
    # Tokenise: split on markdown-link spans, only touch non-link spans.
    pieces: list[str] = []
    last = 0
    for m in MD_LINK_RE.finditer(text):
        pieces.append(_sub_numref(text[last : m.start()], numref_map, warnings))
        pieces.append(m.group(0))  # markdown link, unchanged
        last = m.end()
    pieces.append(_sub_numref(text[last:], numref_map, warnings))
    return "".join(pieces)


def _sub_numref(
    segment: str,
    numref_map: dict[str, str | None],
    warnings: list[str],
) -> str:
    def _sub(m: re.Match[str]) -> str:
        n = m.group(1)
        key = numref_map.get(n)
        if key is None:
            warnings.append(f"unresolved chapter-04 numref [{n}]")
            return f"\\cite{{UNRESOLVED_NUM{n}}}"
        return f"\\cite{{{key}}}"

    return NUMREF_RE.sub(_sub, segment)


_TABLE_SEP_RE = re.compile(r"^[ \t]*\|(?:[ \t]*:?-+:?[ \t]*\|)+[ \t]*$")


def _normalise_pipe_table_separators(text: str) -> str:
    """Equalise dash widths in pipe-table separator rows.

    Pandoc treats *unequal* dash counts in a separator row as proportional
    column widths. The author of these chapters wrote ~58 dashes under
    "Individual Test Accuracy (seeds 42, 123, 456, 789, 1024)" because the
    header is long, but only 6 dashes under "Mean" -- which pandoc then
    interpreted as "give Mean only 5.7% of textwidth", so "96.52%" overflows
    the cell.

    When every column's separator has the same dash count, pandoc instead
    auto-sizes each column to fit its content -- the desired behaviour.
    We rewrite each separator row in place, preserving the original
    alignment markers (`:---:`, `:---`, `---:`).
    """
    out_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        if not _TABLE_SEP_RE.match(line):
            out_lines.append(line)
            continue
        # Split on `|`, drop empty leading/trailing parts caused by the
        # outer pipes, then rebuild with uniform 4-dash cells that keep
        # the original alignment marker positions.
        parts = line.strip().strip("|").split("|")
        rebuilt: list[str] = []
        for cell in parts:
            cell = cell.strip()
            left = cell.startswith(":")
            right = cell.endswith(":")
            marker = ":" + ("-" * 4) + ":" if left and right else (
                ":" + ("-" * 4) if left else (
                    ("-" * 4) + ":" if right else "-" * 4
                )
            )
            rebuilt.append(marker)
        trailing_nl = "\n" if line.endswith("\n") else ""
        out_lines.append("|" + "|".join(rebuilt) + "|" + trailing_nl)
    return "".join(out_lines)


def _strip_heading_prefixes(text: str) -> str:
    """Strip author-typed numeric prefixes from headings.

    LaTeX section-numbering is what produces the visible "N", "N.M", etc. in
    the rendered PDF. When the author also types `1.1` or `Chapter 1:` into
    the heading, you get duplication like "1.1 1.1 Background" in the PDF.
    Strip the author-typed prefix so the LaTeX numbering is the only source
    of truth.
    """
    text = HEADING_CHAPTER_RE.sub(r"\1 \2", text)
    text = HEADING_SECTION_RE.sub(r"\1 \2", text)
    return text


def _has_trailing_references_block(text: str) -> bool:
    """Whether the text contains a `## References` heading anywhere.

    Author-typed local reference lists are a Flavor-C convention: every
    chapter that has one also uses numeric `[N]` markers in its body.
    """
    return any(
        re.match(r"^##\s+References\s*$", line.strip(), re.IGNORECASE)
        for line in text.splitlines()
    )


def _drop_references_block(text: str) -> str:
    """Drop a chapter's trailing `## References` block.

    Removes from the first line matching `## References` (any case) to the
    end of the file. Safe to call on text that has no such block.
    """
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if re.match(r"^##\s+References\s*$", line.strip(), re.IGNORECASE):
            return "".join(lines[:i])
    return text


def preprocess(
    text: str,
    filename: str,
    *,
    aliases: dict[str, str | None] | None = None,
    numref_map: dict[str, str | None] | None = None,
    bib_keys: frozenset[str] | None = None,
    warnings: list[str] | None = None,
) -> str:
    """Apply all citation transforms to a single chapter's markdown text.

    Args:
        text: Source markdown.
        filename: Determines whether the chapter has a trailing References
            block (detected by presence, not by chapter number).
        aliases: Flavor B prose -> bib key (or None for unresolved).
        numref_map: Numeric "1".."N" -> bib key for Flavor-C chapters.
        bib_keys: Known bib keys, used to recognise non-year-shaped keys
            (e.g., `astralruff`). Optional; falls back to regex shape only.
        warnings: If provided, unresolved tokens are appended here as strings.

    Returns the transformed markdown.
    """
    aliases = aliases or {}
    numref_map = numref_map or {}
    warnings = warnings if warnings is not None else []

    # Flavor-C handling: any chapter with a trailing `## References` block is
    # assumed to use numeric `[N]` markers. Drop the block and remap markers.
    if _has_trailing_references_block(text):
        text = _drop_references_block(text)
        text = _replace_numref(text, numref_map, warnings)

    text = _replace_cite(text, aliases, bib_keys, warnings)
    text = _strip_heading_prefixes(text)
    text = _normalise_pipe_table_separators(text)
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_bib_keys() -> frozenset[str]:
    if not BIBFILE.exists():
        return frozenset()
    keys: set[str] = set()
    for line in BIBFILE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*@\w+\s*\{\s*([^,\s]+)\s*,", line)
        if m:
            keys.add(m.group(1))
    return frozenset(keys)


def _load_json(path: Path) -> dict[str, str | None]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any unresolved citations remain.",
    )
    args = parser.parse_args()

    aliases = _load_json(ALIASES_PATH)
    bib_keys = _load_bib_keys()

    MD_OUT_DIR.mkdir(parents=True, exist_ok=True)

    chapters = sorted(CHAPTERS_DIR.glob("*.md"))
    total_warnings = 0
    for ch in chapters:
        warnings: list[str] = []
        src = ch.read_text(encoding="utf-8")
        # Load this chapter's per-chapter numref map (empty if it doesn't
        # exist; that's fine for chapters that use only Flavor A or B).
        chapter_num = ch.name.split("-")[0]
        numref_map = _load_json(_numref_path(chapter_num))
        out = preprocess(
            src,
            ch.name,
            aliases=aliases,
            numref_map=numref_map,
            bib_keys=bib_keys,
            warnings=warnings,
        )
        out_path = MD_OUT_DIR / f"chapter-{ch.name.split('-')[0]}.md"
        out_path.write_text(out, encoding="utf-8")
        if warnings:
            print(f"{ch.name}: {len(warnings)} warning(s)")
            for w in warnings:
                print(f"  - {w}")
            total_warnings += len(warnings)
        else:
            print(f"{ch.name}: ok")

    print(f"\nWrote {len(chapters)} files to {MD_OUT_DIR.relative_to(REPO_ROOT)}/")
    if args.strict and total_warnings:
        print(f"FAIL: {total_warnings} unresolved citation(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
