"""Validate every `\\cite{...}` in the preprocessed markdown against the bib.

Run after `preprocess_citations.py`. Emits a human-readable report:

    thesis/build/audit_report.txt

Exit codes:
    0 -- all cites resolve, no UNRESOLVED markers
    1 -- one or more cites do NOT exist in references.bib, OR
         one or more UNRESOLVED_* markers remain
    2 -- nothing to audit (build/md/ is empty)

The audit is conservative: bib keys are read line-by-line from the
`@entry{key, ... }` headers. Multi-line entries are still detected
because the regex anchors on the `@type{key,` opening pattern.

Usage:
    python pipeline/audit_citations.py
    python pipeline/audit_citations.py --strict   # exit 1 on warnings too
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_MD_DIR = REPO_ROOT / "thesis" / "build" / "md"
BIBFILE = REPO_ROOT / "thesis" / "references.bib"
REPORT_PATH = REPO_ROOT / "thesis" / "build" / "audit_report.txt"

CITE_RE = re.compile(r"\\cite\{([^}]+)\}")
BIB_ENTRY_RE = re.compile(r"^\s*@\w+\s*\{\s*([^,\s]+)\s*,")
UNRESOLVED_RE = re.compile(r"^UNRESOLVED_")


def _load_bib_keys() -> set[str]:
    keys: set[str] = set()
    if not BIBFILE.exists():
        return keys
    for line in BIBFILE.read_text(encoding="utf-8").splitlines():
        m = BIB_ENTRY_RE.match(line)
        if m:
            keys.add(m.group(1))
    return keys


def _collect_cites() -> dict[str, list[str]]:
    """Return mapping key -> list of "filename:lineno" occurrences."""
    cites: dict[str, list[str]] = defaultdict(list)
    for md in sorted(BUILD_MD_DIR.glob("chapter-*.md")):
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for m in CITE_RE.finditer(line):
                # `\cite{a,b,c}` may contain multiple keys.
                for key in m.group(1).split(","):
                    key = key.strip()
                    if key:
                        cites[key].append(f"{md.name}:{lineno}")
    return dict(cites)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat unused bib entries as a failure too (not just undefined cites).",
    )
    args = parser.parse_args()

    if not BUILD_MD_DIR.exists() or not any(BUILD_MD_DIR.glob("chapter-*.md")):
        print(
            f"ERROR: no preprocessed chapters under "
            f"{BUILD_MD_DIR.relative_to(REPO_ROOT)}. "
            "Run preprocess_citations.py first.",
            file=sys.stderr,
        )
        return 2

    bib_keys = _load_bib_keys()
    cites = _collect_cites()

    cited_keys = set(cites.keys())
    undefined = sorted(
        {k for k in cited_keys if not UNRESOLVED_RE.match(k) and k not in bib_keys}
    )
    unresolved = sorted({k for k in cited_keys if UNRESOLVED_RE.match(k)})
    unused = sorted(bib_keys - cited_keys)

    lines: list[str] = []
    lines.append("Citation audit report")
    lines.append("=" * 60)
    lines.append(f"Bib keys defined : {len(bib_keys)}  ({BIBFILE.relative_to(REPO_ROOT)})")
    lines.append(f"Cite keys used   : {len(cited_keys)}")
    lines.append(f"Undefined cites  : {len(undefined)}  (in markdown but not in bib)")
    lines.append(f"UNRESOLVED_* mark: {len(unresolved)}  (preprocess couldn't map)")
    lines.append(f"Unused bib entry : {len(unused)}  (in bib but never cited)")
    lines.append("")

    if undefined:
        lines.append("== Undefined cites ==")
        for k in undefined:
            locs = ", ".join(cites[k][:5])
            lines.append(f"  {k}\n      cited at: {locs}")
        lines.append("")

    if unresolved:
        lines.append("== UNRESOLVED markers ==")
        for k in unresolved:
            locs = ", ".join(cites[k][:5])
            lines.append(f"  {k}\n      cited at: {locs}")
        lines.append("")

    if unused:
        lines.append("== Unused bib entries (informational) ==")
        for k in unused:
            lines.append(f"  {k}")
        lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:8]))
    print(f"\nFull report: {REPORT_PATH.relative_to(REPO_ROOT)}")

    if undefined or unresolved:
        print(
            f"FAIL: {len(undefined)} undefined + {len(unresolved)} unresolved.",
            file=sys.stderr,
        )
        return 1
    if args.strict and unused:
        print(f"FAIL (strict): {len(unused)} unused bib entries.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
