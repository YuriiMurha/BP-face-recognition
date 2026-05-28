"""Tests for preprocess_citations.

Run directly with the system Python; no pytest/uv needed:

    python pipeline/test_preprocess_citations.py

Tests use literal markdown excerpts from the actual chapter files so
regressions on the real source are caught immediately.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess_citations import preprocess  # noqa: E402

# A minimal alias map used across tests. Real values mirror what is in
# thesis/build/citation_aliases.json.
ALIASES = {
    "Schroff et al. 2015": "schroff2015facenet",
    "Bazarevsky et al. 2019": "bazarevsky2019blazeface",
    "catastrophic forgetting in fine-tuning": "mccloskey1989catastrophic,goodfellow2013catastrophic",
    "albumentations": "buslaev2020albumentations",
    "mediapipe": "lugaresi2019mediapipe",
    "uv package manager": "astraluv",
    "python packaging src-layout": None,  # genuinely unresolved
}

# Chapter-04 numeric reference map (matches the trailing References block).
NUMREF = {
    "1": "bazarevsky2019blazeface",
    "2": "viola2001robust",
    "6": "schroff2015facenet",
    "7": "howard2018ulmfit",
}

CASES: list[tuple[str, str, str, str]] = []
# (test_name, chapter_filename, input_markdown, expected_output)


def case(name: str, filename: str, src: str, expected: str) -> None:
    CASES.append((name, filename, src, expected))


# ----- Flavor A: clean bib keys ------------------------------------------

case(
    "flavor_a_single_key",
    "01-introduction.md",
    "Convolutional networks [CITE: schroff2015facenet] dominate.",
    "Convolutional networks \\cite{schroff2015facenet} dominate.",
)

case(
    "flavor_a_multi_key_comma",
    "01-introduction.md",
    "models [CITE: wang2021deepface, schroff2015facenet] now achieve",
    "models \\cite{wang2021deepface,schroff2015facenet} now achieve",
)

case(
    "flavor_a_multi_key_semicolon",
    "03-tools-and-libraries.md",
    "are written in [CITE: abadi2016tensorflow; chollet2015keras] backed by",
    "are written in \\cite{abadi2016tensorflow,chollet2015keras} backed by",
)

case(
    "flavor_a_inner_whitespace_tolerated",
    "01-introduction.md",
    "ref [CITE:  schroff2015facenet ] here",
    "ref \\cite{schroff2015facenet} here",
)

case(
    "flavor_a_payload_whitespace_around_comma",
    "01-introduction.md",
    "[CITE: schroff2015facenet , wang2021deepface]",
    "\\cite{schroff2015facenet,wang2021deepface}",
)


# ----- Flavor B: author-year prose, resolved via alias --------------------

case(
    "flavor_b_resolved",
    "07-results.md",
    "MediaPipe BlazeFace [CITE: Bazarevsky et al. 2019] is fast.",
    "MediaPipe BlazeFace \\cite{bazarevsky2019blazeface} is fast.",
)

case(
    "flavor_b_alias_expands_to_multi_key",
    "06-implementation.md",
    "addresses [CITE: catastrophic forgetting in fine-tuning] by ...",
    "addresses \\cite{mccloskey1989catastrophic,goodfellow2013catastrophic} by ...",
)

case(
    "flavor_b_mixed_with_flavor_a_in_same_cite",
    "07-results.md",
    "FaceNet [CITE: Schroff et al. 2015, hermans2017triplet] uses triplet loss.",
    "FaceNet \\cite{schroff2015facenet,hermans2017triplet} uses triplet loss.",
)


# ----- Flavor B: unresolved (no alias mapping) ----------------------------

case(
    "flavor_b_unresolved_emits_placeholder",
    "06-implementation.md",
    "follows the [CITE: python packaging src-layout] convention",
    "follows the \\cite{UNRESOLVED_python_packaging_src_layout} convention",
)


# ----- Flavor C: chapter 04 numeric markers -------------------------------

case(
    "flavor_c_numeric_marker_chapter_04",
    "04-methods.md",
    "MediaPipe BlazeFace [1] runs at 200 FPS.\n\n## References\n\n[1] Bazarevsky.\n",
    "MediaPipe BlazeFace \\cite{bazarevsky2019blazeface} runs at 200 FPS.\n\n",
)

case(
    "flavor_c_numeric_marker_NOT_in_other_chapters",
    "01-introduction.md",
    "Section [1] of the introduction stays as-is.",
    "Section [1] of the introduction stays as-is.",
)

case(
    "flavor_c_markdown_link_not_mistaken",
    "04-methods.md",
    "see [Section 4.1](#4-1) for details",
    "see [Section 4.1](#4-1) for details",
)

case(
    "flavor_c_drops_trailing_references_block_in_chapter_04",
    "04-methods.md",
    """# Chapter 4: Methods

The detector [1] works well.

## References

[1] Bazarevsky, V. et al. (2019). BlazeFace.

[2] Viola, P. & Jones, M. (2001). Rapid object detection.
""",
    """# Methods

The detector \\cite{bazarevsky2019blazeface} works well.

""",
)

case(
    "flavor_c_drops_references_block_in_any_chapter",
    "02-literature-review.md",
    """# Chapter 2

Body text.

## References

[1] Dropped because the new generalised logic detects the block by presence, not by chapter number.
""",
    """# Chapter 2

Body text.

""",
)


# ----- Idempotency --------------------------------------------------------

case(
    "idempotent_already_converted_text_unchanged",
    "01-introduction.md",
    "Already converted \\cite{schroff2015facenet} here.",
    "Already converted \\cite{schroff2015facenet} here.",
)


# ----- Edge cases ---------------------------------------------------------

case(
    "empty_cite_payload_emits_warning_marker",
    "01-introduction.md",
    "broken [CITE: ] citation",
    "broken \\cite{UNRESOLVED_EMPTY} citation",
)


# ----- Heading prefix stripping (LaTeX owns numbering) --------------------

case(
    "heading_strips_chapter_prefix_from_h1",
    "01-introduction.md",
    "# Chapter 1: Introduction\n",
    "# Introduction\n",
)

case(
    "heading_strips_dotted_prefix_from_h2",
    "01-introduction.md",
    "## 1.1 Background\n",
    "# Background\n",  # NOTE: also no chapter prefix, but h2 is what we test
) if False else None

case(
    "heading_strips_dotted_prefix_from_h2_correct",
    "01-introduction.md",
    "## 1.1 Background\n",
    "## Background\n",
)

case(
    "heading_strips_three_level_dotted_prefix_from_h3",
    "07-results.md",
    "### 7.1.1 Benchmark Methodology\n",
    "### Benchmark Methodology\n",
)

case(
    "heading_leaves_unprefixed_heading_alone",
    "02-literature-review.md",
    "## Open Questions\n",
    "## Open Questions\n",
)

case(
    "heading_strips_h1_chapter_and_keeps_body_intact",
    "07-results.md",
    "# Chapter 7: Experimental Results\n\nBody text remains.\n",
    "# Experimental Results\n\nBody text remains.\n",
)

case(
    "heading_strips_letter_prefix_for_appendix_sections",
    "09-user-manual.md",
    "## A.1 Overview\n\nText.\n",
    "## Overview\n\nText.\n",
)

case(
    "heading_strips_two_level_letter_prefix",
    "10-system-manual.md",
    "### B.10 Reproducing Results\n",
    "### Reproducing Results\n",
)


# ----- Pipe-table separator normalisation ---------------------------------

case(
    "table_separator_rebalanced_to_equal_dashes",
    "07-results.md",
    "| A | Long Header With Many Words | M | S | Note |\n"
    "|---|-----------------------------|---|---|------|\n"
    "| x | y                           | 1 | 2 | foo  |\n",
    "| A | Long Header With Many Words | M | S | Note |\n"
    "|----|----|----|----|----|\n"
    "| x | y                           | 1 | 2 | foo  |\n",
)

case(
    "table_separator_preserves_alignment_markers",
    "07-results.md",
    "| A | B | C |\n|:---|:---:|---:|\n| x | y | z |\n",
    "| A | B | C |\n|:----|:----:|----:|\n| x | y | z |\n",
)

case(
    "non_table_line_with_pipes_unchanged",
    "07-results.md",
    "Some prose | with a | pipe character is unchanged.\n",
    "Some prose | with a | pipe character is unchanged.\n",
)


def run() -> int:
    failures: list[tuple[str, str, str]] = []
    for name, filename, src, expected in CASES:
        actual = preprocess(src, filename, aliases=ALIASES, numref_map=NUMREF)
        if actual == expected:
            print(f"PASS  {name}")
        else:
            print(f"FAIL  {name}")
            failures.append((name, expected, actual))

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for name, expected, actual in failures:
            print(f"\n--- {name} ---")
            print(f"expected: {expected!r}")
            print(f"actual:   {actual!r}")
        return 1

    print(f"\nAll {len(CASES)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
