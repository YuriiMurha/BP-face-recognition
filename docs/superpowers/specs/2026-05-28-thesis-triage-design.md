# Thesis Triage to Supervisor Spec — Design

- **Date:** 2026-05-28
- **Deadline:** 2026-05-29 (submission, per zadávací list) — ~1 day
- **Author:** Yurii Murha
- **Scope chosen:** Structure + big length cuts. Skip line-level voice rewrite.

## Context

Bachelor thesis "Face Recognition in Camera Footage" (TUKE FEI). Source of truth is
markdown in `thesis/chapters/*.md`; the `pipeline/` scripts convert it via pandoc to
LaTeX, copy into the `overleaf-bp-new/` git-bridge clone, compile locally with
`latexmk`, then push to Overleaf. LaTeX auto-numbers sections (the pipeline strips
literal heading numbers); in-prose cross-references ("Chapter 4", "Figure 7.1") are
hand-typed literals and must be fixed by hand where they survive a cut.

The supervisor's review asks for a radical shortening to **30–40 pp of clean text**,
a return to the canonical TUKE structure, and less "over-generated" prose.

## Locked decisions

1. **Ch3 Tools & Libraries** — delete entirely.
2. **Manuals** — keep both in the PDF as appendices, reordered to the very end.
3. **Acknowledgement** — leave unchanged (author's explicit choice).
4. **Publishing** — push to BOTH repos (markdown → `BP-face-recognition`, LaTeX →
   `overleaf-bp-new`) only AFTER the author approves the locally compiled PDF.
5. **Implementation (Ch5) & Results (Ch6)** — **condense, do not delete**: rewrite
   each subsubsection down to a few sentences that retain the information. The author
   makes the final call on these two chapters during their phases (checkpoint).

## Target structure

Front matter (`LaTeX/tukethesis.tex`): title · abstract EN (≤200 w) · abstract SK
(≤200 w) · zadávací list · declaration · acknowledgement (unchanged) · ~~preface~~
deleted · TOC · list of figures/tables · symbols & abbreviations (trim unused).

Body (markdown → numbered chapters):

| New # | Chapter | From | Action | Budget |
|---|---|---|---|---|
| — | **Introduction** (úvod, unnumbered) | Ch1 | ≤2 pp, strip all citations (move cited background → Lit Review); keep context + thesis-structure paragraph | 2 pp |
| **1** | **Problem Formulation** (new) | — | 4 assignment goals × how each was met | 1–1.5 pp |
| **2** | Literature Review | Ch2 | Cut §2.4 loss functions (5 subsecs) + §2.9 privacy/ethics/law; collapse §2.5 transfer learning → 1 paragraph; keep detection/recognition history + FaceNet + metrics + gap | ~5 pp |
| **3** | Methods | Ch4 | Keep theory; shrink/cut full-page diagram figures | ~5 pp |
| **4** | Datasets | Ch5 | Light trim (supervisor: OK) | ~4 pp |
| **5** | Implementation | Ch6 | **Condense to global view** — each subsection → a few sentences (CHECKPOINT) | ~4 pp |
| **6** | Results | Ch7 | **Condense to essentials** — headline tables/figures kept full; deep-dive subsections → tight paragraphs (CHECKPOINT) | ~8 pp |
| — | **Conclusion** (záver, unnumbered) | Ch8 | ≤2 pp: summary · main outputs · shortcomings · future work | 2 pp |

Back matter: Bibliography → **List of Appendices (last numbered page)** →
Appendix A: User Manual → Appendix B: System Manual.

Body ≈ 31 pp; list-of-appendices well before p.60.

## Front-matter fixes (mechanical)

- `\dateofsubmission{May. 30. 2025}` → `May. 29. 2026`
- `\fieldofstudy{5.2.13 Electronics}` → Computer Science (per assignment)
- Delete `\preface{…}\endpreface`
- Rewrite both abstracts to ≤200 words, aligned to the real thesis (drop
  Eigenfaces/LBP/ethics claims)
- Trim now-unused glossary entries (AI Act, CCPA, CPRA, GDPR-only-for-ethics, etc.)

## Pipeline / file changes

- Delete `thesis/chapters/03-tools-and-libraries.md`.
- Add `thesis/chapters/01-problem-formulation.md`.
- Renumber files: `04-methods → 03`, `05-datasets → 04`, `06-implementation → 05`,
  `07-results → 06`; keep `introduction`/`conclusion` but mark their top headings
  unnumbered (`{-}`) so they render as úvod/záver (verify pandoc emits
  `\section*` + `\addcontentsline`).
- Update hardcoded chapter lists: `Makefile` `CHAPTERS`, `assemble_master.py`
  ranges + the appendix descriptions if affected.
- Fix surviving literal cross-refs after renumber/cut.

## Build → verify → push (gated)

`make thesis-tex` → `make thesis-build` (local latexmk; catches errors) → read
`overleaf-bp-new/tukethesis.pdf`, confirm clean compile + page count + structure →
show author → on approval: `make thesis-overleaf` (push LaTeX) AND commit markdown
to `BP-face-recognition`.

## Out of scope tonight

Line-level voice/AI-detector rewrite, new experiments, figure regeneration (reuse
existing PNGs), new citations/bibliography work.

## Risks

- Renumber breaks literal cross-refs → mitigate by grepping `(Chapter|Section|Figure|Table)\s+\d` after edits.
- Unnumbered úvod/záver TOC handling → verify in local compile before push.
- Condensing Ch5/Ch6 without author input → mitigated by checkpoints.
- Time: if the clock runs out, priority order is front-matter+structure (submittable)
  first, then Lit Review cut, then Results, then Implementation.
