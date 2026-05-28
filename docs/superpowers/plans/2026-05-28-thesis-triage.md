# Thesis Triage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline, with checkpoints) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure and radically shorten the thesis to the supervisor's spec (30–40 pp body, canonical TUKE structure) in one night, then compile-verify and push to both repos after author approval.

**Architecture:** Markdown chapters in `thesis/chapters/*.md` are the source. `make thesis-tex` (pandoc) → `make thesis-build` (copy to `overleaf-bp-new/`, local `latexmk`) → `make thesis-overleaf` (push). LaTeX owns all numbering; files are mapped to chapters by `NN-` prefix, so keeping a contiguous `01..10` set means **no Makefile / assemble_master.py changes**. Verification = clean compile + structure/page/cross-ref greps, not unit tests.

**Tech Stack:** pandoc, latexmk (MiKTeX), TUKE `tukethesis.cls`, Python pipeline scripts, git bridge to Overleaf.

**Checkpoints (author decides):** Task 10 (Implementation) and Task 11 (Results) — condense, do not delete; stop for author input before finalizing each.

**Priority if time runs out:** Tasks 0–6 make it submittable; 7 (Lit cut) → 11 (Results) → 10 (Impl) deliver the page target.

---

### Task 0: Safety baseline

**Files:** none (git + compile)

- [ ] **Step 1:** Confirm working tree state: `git -C D:/Coding/Personal/BP-face-recognition status`
- [ ] **Step 2:** Create safety branch (confirm cadence with author first): `git checkout -b thesis-triage-2026-05-28`
- [ ] **Step 3:** Baseline compile to prove the pipeline is green BEFORE edits: `make thesis-tex && make thesis-build`
  Expected: `latexmk` finishes, `overleaf-bp-new/tukethesis.pdf` updates, no fatal errors.
- [ ] **Step 4:** Record baseline page count from `overleaf-bp-new/tukethesis.toc` (note last appendix page) for before/after comparison.

---

### Task 1: Front-matter mechanical fixes

**Files:** Modify `LaTeX/tukethesis.tex`

- [ ] **Step 1:** `\dateofsubmission{May. 30. 2025}` → `\dateofsubmission{May. 29. 2026}` (line ~87).
- [ ] **Step 2:** `\fieldofstudy{5.2.13 Electronics}` → `\fieldofstudy{Computer Science}` (line ~82). Leave `\studyprogramme{Intelligent Systems}`.
- [ ] **Step 3:** Delete the `\preface … \endpreface` block (lines ~144–150).
- [ ] **Step 4:** In the "List of Symbols and Abbreviations" `description` block, remove entries that only existed for the cut ethics/law material: `AI Act`, `CCPA`, `CPRA`. Keep `GDPR` only if still referenced after Task 5/7 (re-grep before deciding).
- [ ] **Step 5:** Commit: `git add LaTeX/tukethesis.tex && git commit -m "thesis: fix submission date, field of study; drop preface + unused glossary"`

---

### Task 2: Rewrite both abstracts to ≤200 words

**Files:** Modify `LaTeX/tukethesis.tex` (`\abstrakte{…}` EN line ~89, `\abstrakt{…}` SK line ~92)

- [ ] **Step 1:** Replace `\abstrakte` with a ≤200-word abstract describing the ACTUAL thesis: open-source surveillance face-recognition pipeline; ground-truth detector benchmark (MediaPipe/MTCNN/Haar/dlib); three FaceNet fine-tuning strategies on a 14-class custom dataset (PU best, 99.15%); open-set verification; reproducible plugin architecture. Drop Eigenfaces/Fisherfaces/LBP and the ethics sentence.
- [ ] **Step 2:** Replace `\abstrakt` (Slovak) with a faithful translation of the new English abstract, same ≤200-word budget.
- [ ] **Step 3:** Verify word counts (≤200 each) before moving on.
- [ ] **Step 4:** Commit: `git add LaTeX/tukethesis.tex && git commit -m "thesis: shorten EN/SK abstracts to <=200 words, align to real content"`

---

### Task 3: Structural scaffolding (must keep compiling)

**Files:** Delete `thesis/chapters/03-tools-and-libraries.md`; rename `thesis/chapters/02-literature-review.md` → `thesis/chapters/03-literature-review.md`; create `thesis/chapters/02-problem-formulation.md` (stub); modify headings of `thesis/chapters/01-introduction.md` and `thesis/chapters/08-conclusion.md`.

- [ ] **Step 1:** `git rm thesis/chapters/03-tools-and-libraries.md`
- [ ] **Step 2:** `git mv thesis/chapters/02-literature-review.md thesis/chapters/03-literature-review.md`
- [ ] **Step 3:** Create `thesis/chapters/02-problem-formulation.md` with a one-line stub heading `# Problem Formulation` + placeholder sentence (real content in Task 4) so the skeleton compiles.
- [ ] **Step 4:** Mark intro/conclusion top headings unnumbered:
  `01-introduction.md` line 1 → `# Introduction {.unnumbered}`;
  `08-conclusion.md` line 1 → `# Conclusion {.unnumbered}`.
- [ ] **Step 5:** `make thesis-tex` — expect 10 chapter files written, no pandoc errors.
- [ ] **Step 6:** Inspect `thesis/build/tex/chapters/chapter-01.tex` and `chapter-08.tex` — confirm `\section*{Introduction}` / `\section*{Conclusion}`. If pandoc did NOT emit `\addcontentsline{toc}{section}{…}`, add a lua/sed fix or a manual `\addcontentsline` so they still appear in the TOC.
- [ ] **Step 7:** `make thesis-build` — confirm clean compile. Read `overleaf-bp-new/tukethesis.toc`: expect `Introduction` (no number), `1 Problem Formulation`, `2 Literature Review`, `3 Methods…`, `4 Datasets`, `5 …Implementation`, `6 …Results`, `Conclusion` (no number), then List of Appendices + appendices.
- [ ] **Step 8:** Commit: `git add -A && git commit -m "thesis: restructure — drop Tools ch, add Problem Formulation, unnumber intro/conclusion"`

---

### Task 4: Write Problem Formulation (Ch1)

**Files:** Modify `thesis/chapters/02-problem-formulation.md`

- [ ] **Step 1:** Write ~1–1.5 pp. Open with one sentence framing the assignment, then one short subsection or paragraph per official goal, each stating the goal and how it was met:
  1. *Overview of AI methods for detection & recognition* → delivered in the literature review and methods chapters (MediaPipe/MTCNN/Haar/dlib; FaceNet + fine-tuning).
  2. *Curate a real-life multi-camera dataset* → 14-class webcam + security-camera dataset, ~7,080 augmented images, manually annotated detection ground truth.
  3. *Train & evaluate multiple models, validate real-life deployment* → three FaceNet strategies compared; PU = 99.15%; CPU real-time pipeline; open-set verification.
  4. *Prepare documentation per supervisor* → User & System manuals (Appendices A/B), reproducible build.
- [ ] **Step 2:** No citations needed here (it reflects the assignment). Keep prose tight (supervisor: short and unambiguous).
- [ ] **Step 3:** `make thesis-tex` (no errors). Commit: `git add thesis/chapters/02-problem-formulation.md && git commit -m "thesis: add Problem Formulation chapter reflecting the assignment goals"`

---

### Task 5: Introduction → úvod (≤2 pp, no citations)

**Files:** Modify `thesis/chapters/01-introduction.md`

- [ ] **Step 1:** Cut to a short problem intro + thesis-structure paragraph. Remove every `\cite`/`[CITE:]` (move any genuinely needed cited background into Lit Review Task 7). Drop the RQ list and contributions list (those belong in Problem Formulation / body).
- [ ] **Step 2:** Update the "thesis structure" paragraph to the NEW structure (Problem Formulation, Lit Review, Methods, Datasets, Implementation, Results, Conclusion, Appendices A/B) — and renumber any "Chapter N" mentions.
- [ ] **Step 3:** `grep -nE "\\\\cite|\[CITE:" thesis/chapters/01-introduction.md` → expect no matches.
- [ ] **Step 4:** Commit: `git add thesis/chapters/01-introduction.md && git commit -m "thesis: trim introduction to <=2pp, remove citations, fix structure refs"`

---

### Task 6: Conclusion → záver (≤2 pp)

**Files:** Modify `thesis/chapters/08-conclusion.md`

- [ ] **Step 1:** Reduce to: short summary of the work · main outputs · shortcomings · future work. Remove the per-RQ walkthrough and "Practical Recommendations" if it pushes past ~2 pp; fold the essentials into the summary.
- [ ] **Step 2:** Commit: `git add thesis/chapters/08-conclusion.md && git commit -m "thesis: shorten conclusion to <=2pp"`

---

### Task 7: Literature Review cut (Ch2)

**Files:** Modify `thesis/chapters/03-literature-review.md`

- [ ] **Step 1:** Delete §2.4 "Loss functions" entirely (all 5 subsections).
- [ ] **Step 2:** Delete §2.9 "Privacy, ethics, and regulation" entirely.
- [ ] **Step 3:** Collapse §2.5 "Transfer learning and fine-tuning strategies" (4 subsections) to ONE paragraph (name PU/discriminative-LR; the rest belongs in Methods).
- [ ] **Step 4:** Fix the intro paragraph's "Section 2.x" roadmap to match the surviving sections; fix any "Chapter 4" → "Chapter 3" (methods renumbered).
- [ ] **Step 5:** `grep -nE "2\.4|2\.5|2\.9|Chapter 4" thesis/chapters/03-literature-review.md` → review each remaining hit is intentional.
- [ ] **Step 6:** `make thesis-tex` clean. Commit: `git add thesis/chapters/03-literature-review.md && git commit -m "thesis: cut loss-functions + ethics, collapse transfer-learning section"`

---

### Task 8: Methods (Ch3) — figures + global cross-ref fix

**Files:** Modify `thesis/chapters/04-methods.md`; sweep all chapters for "Chapter 4"/"derived in Chapter 4"

- [ ] **Step 1:** Inspect the 2 mermaid diagrams (`04-methods.md` ~lines 95, 123). For each: if decorative/large, cut it; if useful, keep but ensure the rendered figure is sized `< 0.8\linewidth` (check `fix_image_paths.lua` / caption width handling). Supervisor: no full-page drawings.
- [ ] **Step 2:** Light prose trim where verbose.
- [ ] **Step 3:** Global sweep for references to the renumbered methods chapter: `grep -rnE "Chapter 4|in Chapter 4|derived in Chapter" thesis/chapters/` → change to "Chapter 3" where it means methods.
- [ ] **Step 4:** `make thesis-tex` clean. Commit: `git add -A && git commit -m "thesis: shrink/cut methods diagrams, fix methods chapter cross-refs"`

---

### Task 9: Datasets (Ch4) — light trim

**Files:** Modify `thesis/chapters/05-datasets.md`

- [ ] **Step 1:** Light trim of verbose passages only (supervisor said this chapter is OK). Fix any stale chapter cross-refs.
- [ ] **Step 2:** Commit: `git add thesis/chapters/05-datasets.md && git commit -m "thesis: light trim of datasets chapter"`

---

### Task 10: Implementation (Ch5) — condense [CHECKPOINT]

**Files:** Modify `thesis/chapters/06-implementation.md`

- [ ] **Step 1:** Read the full chapter. Draft a condensed version: each of the 8 sections / subsections reduced to a few sentences that RETAIN the information (global architecture view), not deleted.
- [ ] **Step 2:** **STOP — present the condensed draft (section-by-section before/after) to the author and get their decisions before writing it.**
- [ ] **Step 3:** Apply approved condensation. Fix stale cross-refs.
- [ ] **Step 4:** `make thesis-tex` clean. Commit: `git add thesis/chapters/06-implementation.md && git commit -m "thesis: condense implementation chapter to global overview"`

---

### Task 11: Results (Ch6) — condense [CHECKPOINT]

**Files:** Modify `thesis/chapters/07-results.md`

- [ ] **Step 1:** Read the full chapter. Keep full: detector benchmark table, 3-strategy comparison table, PU confusion matrix figure, verification ROC/EER. Condense deep-dive subsections (per-class, class-imbalance, cross-validation, embedding-geometry) to tight paragraphs/sentences — retain numbers, drop walls of text.
- [ ] **Step 2:** **STOP — present the keep/condense plan and the surviving figure/table list to the author; get decisions before writing.**
- [ ] **Step 3:** Apply. Renumber surviving "Figure 7.x"/"Table 7.x" → "6.x" and any prose "Chapter 7" refs.
- [ ] **Step 4:** `grep -nE "7\.[0-9]|Chapter 7|Figure 7|Table 7" thesis/chapters/07-results.md` → confirm only intentional remain.
- [ ] **Step 5:** `make thesis-tex` clean. Commit: `git add thesis/chapters/07-results.md && git commit -m "thesis: condense results to essential findings"`

---

### Task 12: Global sweep — cross-refs, heading-on-heading, glossary

**Files:** all `thesis/chapters/*.md`, `LaTeX/tukethesis.tex`

- [ ] **Step 1:** Heading-on-heading: in each chapter, ensure no heading is immediately followed by a lower-level heading with no text between (supervisor rule). Add a 1–2 sentence lead-in where needed.
- [ ] **Step 2:** Stale-number sweep across all chapters: `grep -rnE "(Chapter|Section|Figure|Table)\s+[0-9]" thesis/chapters/` → verify every surviving literal matches the new numbering.
- [ ] **Step 3:** Re-grep for now-unused glossary acronyms and remove from `tukethesis.tex` if zero references remain.
- [ ] **Step 4:** Commit: `git add -A && git commit -m "thesis: fix heading-on-heading, finalize cross-refs and glossary"`

---

### Task 13: Full build + verify

**Files:** none (build + inspect)

- [ ] **Step 1:** `make thesis-build` — confirm `latexmk` clean compile (scan log for `! ` LaTeX errors and undefined references).
- [ ] **Step 2:** Read `overleaf-bp-new/tukethesis.toc`: confirm full target structure and that the List of Appendices lands before p.60; body ≈30–40 pp.
- [ ] **Step 3:** Open `overleaf-bp-new/tukethesis.pdf` (Read tool, spot pages) — sanity-check abstract length, úvod/záver unnumbered, no full-page diagrams, tables fit.
- [ ] **Step 4:** **Present results to author** (page count, structure, any warnings) and get the go for push.

---

### Task 14: Publish (gated on Task 13 approval)

**Files:** none (git push)

- [ ] **Step 1:** Push LaTeX to Overleaf: `make thesis-overleaf` (commits + pushes `overleaf-bp-new` to the git bridge).
- [ ] **Step 2:** Push markdown source: merge/commit the triage branch and push `BP-face-recognition` (include the spec + plan docs).
- [ ] **Step 3:** Confirm both remotes updated; report final page count + what changed.
