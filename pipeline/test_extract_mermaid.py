"""Tests for extract_mermaid.

Run directly with the system Python:

    python pipeline/test_extract_mermaid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_mermaid import extract  # noqa: E402


def test_no_mermaid_unchanged() -> None:
    src = "# Chapter 1\n\nNo diagrams here.\n"
    out, blocks = extract(src, "01-introduction.md")
    assert out == src, f"expected no change, got: {out!r}"
    assert blocks == [], f"expected zero blocks, got {blocks}"
    print("PASS  test_no_mermaid_unchanged")


def test_single_mermaid_replaced() -> None:
    src = (
        "## 4.4.2 FaceNet\n"
        "\n"
        "Some text before.\n"
        "\n"
        "```mermaid\n"
        "flowchart TB\n"
        "    A --> B\n"
        "```\n"
        "\n"
        "Some text after.\n"
    )
    out, blocks = extract(src, "04-methods.md")
    assert "```mermaid" not in out
    assert "![FaceNet](../figures/diagrams/chap04-fig01.png)" in out, out
    assert len(blocks) == 1
    name, mermaid = blocks[0]
    assert name == "chap04-fig01"
    assert mermaid == "flowchart TB\n    A --> B"
    print("PASS  test_single_mermaid_replaced")


def test_multiple_mermaid_numbered_sequentially() -> None:
    src = (
        "### 4.4.2 FaceNet\n\n"
        "```mermaid\nflowchart TB\nA-->B\n```\n\n"
        "Body text.\n\n"
        "### 4.5.1 Strategy A\n\n"
        "```mermaid\nflowchart TB\nC-->D\n```\n"
    )
    out, blocks = extract(src, "04-methods.md")
    assert "chap04-fig01" in out
    assert "chap04-fig02" in out
    assert len(blocks) == 2
    assert blocks[0][0] == "chap04-fig01"
    assert blocks[1][0] == "chap04-fig02"
    print("PASS  test_multiple_mermaid_numbered_sequentially")


def test_alt_text_derived_from_nearest_section_heading() -> None:
    src = (
        "## 4.4.2 FaceNet\n\n"
        "Body before diagram.\n\n"
        "```mermaid\nA-->B\n```\n"
    )
    out, blocks = extract(src, "04-methods.md")
    assert "![FaceNet]" in out, out
    print("PASS  test_alt_text_derived_from_nearest_section_heading")


def test_alt_text_falls_back_when_no_heading() -> None:
    src = "Body without any heading.\n\n```mermaid\nA-->B\n```\n"
    out, blocks = extract(src, "06-implementation.md")
    # Falls back to capitalised filename stem.
    assert "![Implementation diagram]" in out or "![Diagram]" in out, out
    print("PASS  test_alt_text_falls_back_when_no_heading")


def test_non_chapter_filename_uses_chapter_prefix_from_basename() -> None:
    src = "## Heading\n\n```mermaid\nA-->B\n```\n"
    out, blocks = extract(src, "06-implementation.md")
    assert blocks[0][0] == "chap06-fig01"
    print("PASS  test_non_chapter_filename_uses_chapter_prefix_from_basename")


def test_build_dir_filename_chapter_NN_md_recognised() -> None:
    """Build output uses `chapter-04.md` (not `04-methods.md`) -- must work."""
    src = "## A\n\n```mermaid\nA-->B\n```\n"
    out, blocks = extract(src, "chapter-04.md")
    assert blocks[0][0] == "chap04-fig01", f"got {blocks[0][0]!r}"
    print("PASS  test_build_dir_filename_chapter_NN_md_recognised")


def test_image_path_uses_forward_slashes() -> None:
    """Markdown paths must use forward slashes even on Windows."""
    src = "## A\n\n```mermaid\nA-->B\n```\n"
    out, _ = extract(src, "04-methods.md")
    assert "../figures/diagrams/" in out, out
    print("PASS  test_image_path_uses_forward_slashes")


def test_indented_mermaid_block_handled() -> None:
    """A Mermaid block indented inside a list still gets extracted."""
    src = (
        "- item\n"
        "  ```mermaid\n"
        "  flowchart TB\n"
        "    A-->B\n"
        "  ```\n"
    )
    out, blocks = extract(src, "04-methods.md")
    assert "```mermaid" not in out
    assert len(blocks) == 1
    print("PASS  test_indented_mermaid_block_handled")


def test_idempotent_already_extracted() -> None:
    """Markdown without any Mermaid blocks (already extracted) is unchanged."""
    src = (
        "## A\n\n"
        "![A](../figures/diagrams/chap04-fig01.png)\n\n"
        "more text\n"
    )
    out, blocks = extract(src, "04-methods.md")
    assert out == src
    assert blocks == []
    print("PASS  test_idempotent_already_extracted")


def run() -> int:
    tests = [
        test_no_mermaid_unchanged,
        test_single_mermaid_replaced,
        test_multiple_mermaid_numbered_sequentially,
        test_alt_text_derived_from_nearest_section_heading,
        test_alt_text_falls_back_when_no_heading,
        test_non_chapter_filename_uses_chapter_prefix_from_basename,
        test_build_dir_filename_chapter_NN_md_recognised,
        test_image_path_uses_forward_slashes,
        test_indented_mermaid_block_handled,
        test_idempotent_already_extracted,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failures += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failures += 1
    print(f"\n{len(tests) - failures}/{len(tests)} passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
