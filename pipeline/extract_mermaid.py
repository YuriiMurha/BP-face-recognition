"""Extract Mermaid fenced blocks to standalone `.mmd` files.

Each ```` ```mermaid ... ``` ```` block in a chapter is replaced inline
with a markdown image reference, and the Mermaid source is written to
`thesis/figures/diagrams/chap{NN}-fig{MM}.mmd`. The user then opens each
`.mmd` via Excalidraw's "Mermaid to Excalidraw" importer, polishes the
diagram, and exports to PNG (or PDF) with the matching filename.

Behaviour:

* The transformation is text-in / text-out, returning the rewritten
  markdown and a list of (slug, mermaid_source) pairs.
* The slug follows `chap{NN}-fig{MM}` where NN is the chapter prefix in
  the filename (`04-methods.md` -> 04) and MM is the 1-based index of
  the block within the chapter.
* The image's alt text is the most recent section heading text. Fall back
  to a capitalised filename stem when no heading precedes the block.
* Existing `.mmd` files are NOT overwritten (the user may have polished
  the source); only new diagrams are written.
* Already-extracted chapters (no ``` ```mermaid `` blocks remain) pass
  through unchanged.

Usage:
    python pipeline/extract_mermaid.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_MD_DIR = REPO_ROOT / "thesis" / "build" / "md"
DIAGRAMS_DIR = REPO_ROOT / "thesis" / "figures" / "diagrams"
INDEX_PATH = DIAGRAMS_DIR / "mermaid_index.json"

# Fenced ```mermaid block detection. DOTALL so newlines inside the body
# are part of the capture. We allow leading whitespace on the opening
# fence so indented blocks (inside lists) are also caught. The closing
# fence must match the leading-whitespace indent.
MERMAID_FENCE_RE = re.compile(
    r"^(?P<indent>[ \t]*)```mermaid[ \t]*\r?\n"  # opening fence
    r"(?P<body>.*?)"                                # mermaid source (lazy)
    r"\r?\n(?P=indent)```[ \t]*$",                  # closing fence at same indent
    re.MULTILINE | re.DOTALL,
)

HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def _slug_for_chapter(filename: str) -> str:
    """Return the 'chap{NN}' prefix for a chapter filename.

    Matches both source layout `04-methods.md` and build-dir layout
    `chapter-04.md`. Falls back to `chapXX` for filenames that don't
    contain a 2-digit chapter index.
    """
    name = Path(filename).name
    m = re.match(r"^(\d{2})-", name) or re.match(r"^chapter-(\d{2})", name)
    return f"chap{m.group(1)}" if m else "chapXX"


def _alt_for_block(text_before: str, filename: str) -> str:
    """Best-effort caption from the nearest preceding heading."""
    headings = HEADING_RE.findall(text_before)
    if headings:
        last = headings[-1].strip()
        # Strip numeric prefix like "4.4.2 FaceNet" -> "FaceNet".
        last = re.sub(r"^([0-9]+\.)+[0-9]+\s+", "", last)
        # Strip "Chapter N:" prefixes.
        last = re.sub(r"^Chapter\s+\d+:\s*", "", last, flags=re.IGNORECASE)
        if last:
            return last
    stem = Path(filename).stem
    stem = re.sub(r"^\d+-", "", stem).replace("-", " ")
    return f"{stem.capitalize()} diagram" if stem else "Diagram"


def _strip_indent(text: str, indent: str) -> str:
    """Remove `indent` from every line of `text` (best-effort)."""
    if not indent:
        return text
    lines = text.splitlines()
    stripped: list[str] = []
    for ln in lines:
        if ln.startswith(indent):
            stripped.append(ln[len(indent):])
        else:
            stripped.append(ln)
    return "\n".join(stripped)


def extract(text: str, filename: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (rewritten_text, [(slug, mermaid_source), ...]).

    `slug` is the basename without extension (e.g., `chap04-fig01`).
    """
    chapter_prefix = _slug_for_chapter(filename)
    blocks: list[tuple[str, str]] = []
    out_parts: list[str] = []
    last_end = 0
    counter = 0

    for m in MERMAID_FENCE_RE.finditer(text):
        counter += 1
        slug = f"{chapter_prefix}-fig{counter:02d}"
        indent = m.group("indent")
        body = _strip_indent(m.group("body"), indent).rstrip("\n")
        # The "before" segment is everything from the previous block end
        # up to the start of this match -- used to choose the alt text.
        before = text[:m.start()]
        alt = _alt_for_block(before, filename)
        replacement = f"{indent}![{alt}](../figures/diagrams/{slug}.png)"

        out_parts.append(text[last_end : m.start()])
        out_parts.append(replacement)
        blocks.append((slug, body))
        last_end = m.end()

    out_parts.append(text[last_end:])
    return "".join(out_parts), blocks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_mmd(slug: str, mermaid_source: str) -> bool:
    """Write the .mmd file unless it already exists. Returns True on write."""
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    path = DIAGRAMS_DIR / f"{slug}.mmd"
    if path.exists():
        return False
    path.write_text(mermaid_source + "\n", encoding="utf-8")
    return True


def _load_index() -> dict[str, dict[str, str]]:
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    return {}


def _save_index(index: dict[str, dict[str, str]]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(
        json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    if not BUILD_MD_DIR.exists():
        print(
            f"ERROR: {BUILD_MD_DIR.relative_to(REPO_ROOT)} does not exist. "
            "Run preprocess_citations.py first.",
            file=sys.stderr,
        )
        return 2

    index = _load_index()
    total_blocks = 0
    total_new_files = 0

    for ch in sorted(BUILD_MD_DIR.glob("chapter-*.md")):
        src = ch.read_text(encoding="utf-8")
        out, blocks = extract(src, ch.name)
        if blocks:
            ch.write_text(out, encoding="utf-8")
            for slug, body in blocks:
                wrote = _write_mmd(slug, body)
                index.setdefault(ch.name, {})[slug] = (
                    "wrote" if wrote else "kept-existing"
                )
                if wrote:
                    total_new_files += 1
            total_blocks += len(blocks)
            print(f"{ch.name}: {len(blocks)} mermaid block(s)")
        else:
            print(f"{ch.name}: no mermaid blocks")

    if total_blocks:
        _save_index(index)
    print(
        f"\n{total_blocks} mermaid block(s); "
        f"{total_new_files} new .mmd file(s) written to "
        f"{DIAGRAMS_DIR.relative_to(REPO_ROOT)}/"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
