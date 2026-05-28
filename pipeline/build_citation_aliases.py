"""Scan thesis chapters and emit a citation-alias JSON skeleton.

The chapter Markdown uses three citation flavors:

  A. Clean bib keys     -> `[CITE: schroff2015facenet]`, `[CITE: a, b]`
  B. Author-year prose  -> `[CITE: Schroff et al. 2015]`
  C. Numeric inline     -> `[1]`, `[2]` (chapter 04 only, with local ref list)

Flavor A keys map directly to references.bib. Flavor B and Flavor C need
human-curated maps. This script identifies every unique token across the 8
chapters and emits two skeleton JSON files:

  thesis/build/citation_aliases.json      <- Flavor B prose -> bib key
  thesis/build/chapter04_numref_map.json  <- "1" .. "N"     -> bib key

The user fills in the right-hand sides; preprocess_citations.py consumes both.

Idempotent: existing entries in the JSON files are preserved across re-runs.
New unresolved tokens are added with a null value (or "TODO" comment) so the
user knows what's still missing.

Usage:
    uv run python pipeline/build_citation_aliases.py
    uv run python pipeline/build_citation_aliases.py --check  # exit 1 if unresolved
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
ALIASES_PATH = BUILD_DIR / "citation_aliases.json"
NUMREF_PATH = BUILD_DIR / "chapter04_numref_map.json"
BIBFILE = REPO_ROOT / "thesis" / "references.bib"

# Matches `[CITE: <payload>]` where payload may contain commas (multi-cite),
# spaces, author-year prose, or a single bib key. Non-greedy on the payload
# to avoid swallowing trailing markdown brackets.
CITE_RE = re.compile(r"\[CITE:\s*([^\]]+?)\s*\]")

# A bib-key-shaped token: starts with a lowercase letter, contains lowercase
# alphanumerics/hyphens/underscores, includes a 4-digit year, optional suffix.
BIBKEY_RE = re.compile(r"^[a-z][a-z0-9_-]*\d{4}[a-z0-9]*$")

# Numeric inline citation, ONLY relevant inside chapter 04 body text.
# Matches `[1]`, `[12]` but NOT inside markdown links `[text](url)` because
# we strip those first.
NUMREF_RE = re.compile(r"\[(\d+)\]")

# Markdown link `[label](url)` -- masked before NUMREF_RE runs so e.g.
# `[Section 2.1](#2-1)` doesn't get treated as a numeric citation.
MD_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")


SPLIT_RE = re.compile(r"[;,]")


def split_payload(payload: str) -> list[str]:
    """Split a multi-cite payload on either ',' or ';' and strip whitespace."""
    return [t.strip() for t in SPLIT_RE.split(payload) if t.strip()]


def collect_cite_tokens(text: str) -> set[str]:
    """Return the set of distinct citation tokens."""
    tokens: set[str] = set()
    for m in CITE_RE.finditer(text):
        for tok in split_payload(m.group(1)):
            tokens.add(tok)
    return tokens


def collect_cite_contexts(text: str, source: str) -> dict[str, list[str]]:
    """Return per-token list of `source:line` occurrences (for the audit log)."""
    out: dict[str, list[str]] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        for m in CITE_RE.finditer(line):
            for tok in split_payload(m.group(1)):
                out.setdefault(tok, []).append(f"{source}:{lineno}")
    return out


def collect_numref_tokens(text: str) -> set[str]:
    """Return distinct numeric tokens from `[N]` markers in chapter 04 body.

    Markdown links are masked first so labels like `[Section 4.1](#4-1)`
    don't get mistaken for numeric citations.
    """
    body = MD_LINK_RE.sub("", text)
    return {m.group(1) for m in NUMREF_RE.finditer(body)}


def load_bib_keys(bibfile: Path) -> set[str]:
    """Parse @entry{key, ... }`-style headers from a BibTeX file."""
    keys: set[str] = set()
    if not bibfile.exists():
        return keys
    for line in bibfile.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*@\w+\s*\{\s*([^,\s]+)\s*,", line)
        if m:
            keys.add(m.group(1))
    return keys


def load_existing(path: Path) -> dict[str, str | None]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if any token is still unresolved (null value).",
    )
    args = parser.parse_args()

    chapters = sorted(CHAPTERS_DIR.glob("*.md"))
    if not chapters:
        print(f"ERROR: no chapter .md files under {CHAPTERS_DIR}", file=sys.stderr)
        return 2

    bib_keys = load_bib_keys(BIBFILE)
    aliases = load_existing(ALIASES_PATH)
    numref = load_existing(NUMREF_PATH)

    # Per-chapter scan with context tracking.
    all_tokens: set[str] = set()
    contexts: dict[str, list[str]] = {}
    for ch in chapters:
        text = ch.read_text(encoding="utf-8")
        all_tokens |= collect_cite_tokens(text)
        for tok, locs in collect_cite_contexts(text, ch.name).items():
            contexts.setdefault(tok, []).extend(locs)
        if ch.name.startswith("04-"):
            for n in collect_numref_tokens(text):
                # Cap at 99 -- no chapter has more numbered refs than that.
                if int(n) <= 99:
                    numref.setdefault(n, None)

    # Classification: prefer "token is a known bib key" over the regex shape
    # test, because some bib keys (astralruff, kerasfacenet, noxproject)
    # legitimately omit the year-digit pattern.
    def is_flavor_a(tok: str) -> bool:
        return tok in bib_keys or bool(BIBKEY_RE.match(tok))

    flavor_b_tokens = {tok for tok in all_tokens if not is_flavor_a(tok)}

    for tok in flavor_b_tokens:
        aliases.setdefault(tok, None)

    # Validate existing aliases against bib (warn but don't drop).
    invalid_aliases = {
        k: v for k, v in aliases.items() if v is not None and v not in bib_keys
    }
    invalid_numref = {
        k: v for k, v in numref.items() if v is not None and v not in bib_keys
    }

    save_json(ALIASES_PATH, aliases)
    save_json(NUMREF_PATH, numref)

    flavor_a_count = sum(1 for t in all_tokens if is_flavor_a(t))
    unresolved_aliases = sum(1 for v in aliases.values() if v is None)
    unresolved_numref = sum(1 for v in numref.values() if v is None)

    print(f"Chapters scanned : {len(chapters)}")
    print(f"Bib keys loaded  : {len(bib_keys)} from {BIBFILE.relative_to(REPO_ROOT)}")
    print(f"Flavor A (bibkey): {flavor_a_count}")
    print(f"Flavor B (prose) : {len(flavor_b_tokens)} distinct  ({unresolved_aliases} unresolved)")
    print(f"Flavor C (numref): {len(numref)} distinct  ({unresolved_numref} unresolved)")

    if unresolved_aliases:
        print("\nUnresolved Flavor B tokens (and where they appear):")
        for tok, val in sorted(aliases.items()):
            if val is None:
                locs = ", ".join(sorted(set(contexts.get(tok, [])))[:3])
                print(f"  {tok!r}\n      seen: {locs}")
    if invalid_aliases:
        print(f"WARN  {len(invalid_aliases)} alias(es) point to keys NOT in references.bib:")
        for k, v in sorted(invalid_aliases.items()):
            print(f"      {k!r} -> {v!r}")
    if invalid_numref:
        print(f"WARN  {len(invalid_numref)} numref(s) point to keys NOT in references.bib:")
        for k, v in sorted(invalid_numref.items()):
            print(f"      [{k}] -> {v!r}")
    print(f"Wrote {ALIASES_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {NUMREF_PATH.relative_to(REPO_ROOT)}")

    if args.check and (unresolved_aliases or unresolved_numref):
        print("FAIL: unresolved citation tokens remain.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
