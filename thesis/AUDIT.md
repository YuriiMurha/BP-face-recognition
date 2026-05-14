# Thesis Content Audit

**Original audit date:** 2026-04-07
**Status update:** 2026-05-13 — see note below before reading the chapter-by-chapter analysis

## Status update — what changed since the audit was written

This audit was written when the thesis was still fragmented across draft directories. It is preserved as planning guidance for the chapters that still need to be written, but several of its premises are now out of date:

- **`docs/thesis/` has been deleted.** All references in this audit to `docs/thesis/chapters/detection_methods.md`, `docs/thesis/chapters/facenet_finetuning.md`, `docs/thesis/chapters/results_analysis.md`, `docs/thesis/chapters/system_architecture.md`, `docs/thesis/chapters/open_vs_closed_set.md`, and `docs/thesis/tables/facenet_results_tables.md` now point to nothing. The relevant content from those drafts has been integrated into the canonical Markdown chapters at `thesis/chapters/`.
- **Chapters 4–8 now exist as canonical Markdown** at `thesis/chapters/{04-methods, 05-datasets, 06-implementation, 07-results, 08-conclusion}.md`. The analysis below describes those chapters as if they still need to be written; that is no longer true. `07-results.md` in particular has been substantially extended since this audit (new ground-truth detection eval, new embedding-quality analysis, training-curve figures, Triplet-Loss per-class column).
- **What remains genuinely useful from this audit:** the chapter-by-chapter guidance for **Chapters 1 (Introduction), 2 (Literature Review), and 3 (Tools & Libraries)**, which have not yet been written as canonical Markdown. For Chapters 4–8, treat this audit as a historical record rather than a to-do list.

The original audit text follows unchanged below.

---

## Executive Summary (original, 2026-04-07)

The thesis "Face Recognition in Camera Footage" has substantial existing content across Markdown drafts, LaTeX files, and project reports. However, the material is fragmented across 18+ files with significant overlap and inconsistencies. The **LaTeX files represent an older codebase** (pre-FaceNet fine-tuning, pre-plugin system) and are largely stale. The **Markdown chapter drafts** (`docs/thesis/chapters/`) are high-quality and current, covering detection, recognition fine-tuning, open-vs-closed-set, results analysis, and system architecture. The old `Thesis.md` contains a complete first draft of the analytical/literature sections that is mostly reusable but needs updating.

**Overall status**: ~60% of thesis content exists in usable form. The analytical/literature chapters need moderate updates. The implementation and results chapters have excellent source material in Markdown but need consolidation. Introduction and Conclusion need rewriting to reflect the final system.

**Estimated total effort**: Medium-High. Most chapters have strong source material; the work is primarily consolidation, updating stale references, and gap-filling rather than writing from scratch.

---

## Chapter-by-Chapter Analysis

### Chapter 1: Introduction (Úvod)
- **Existing content sources**:
  - `Thesis.md` lines 63-75 (Background, Motivation, Problem Statement) — ~800 words
  - `LaTeX/introduction.tex` — ~800 words (same content, formatted for LaTeX)
- **Quality assessment**: **Needs rewrite**
- **Usable content**: The Background, Motivation, and Problem Statement sections are well-written and reusable with minor edits. The general framing of the surveillance/security motivation is solid.
- **Stale content**:
  - Thesis structure outline (line 9-17 of `LaTeX/introduction.tex`) references old chapter structure that no longer matches actual work
  - No mention of FaceNet fine-tuning, progressive unfreezing, open-set vs closed-set paradigms, or the plugin system — the key contributions of the thesis
  - Problem statement focuses generically on "investigating and comparing algorithms" without mentioning the specific contributions (transfer learning comparison, 99.15% accuracy achievement)
- **Missing content**:
  - Updated thesis structure overview reflecting actual chapters
  - Brief summary of key contributions (progressive unfreezing achieving 99.15%, dual-paradigm system, etc.)
  - Thesis objectives stated clearly (currently only implicit)
- **Estimated effort**: **Low-Medium** — existing text is solid foundation, needs updating not rewriting
- **Key sources for writing**: `Thesis.md` (Introduction section), `LaTeX/introduction.tex`, `ARCHITECTURE.md` (for system overview), `docs/thesis/chapters/facenet_finetuning.md` (for contributions summary)

---

### Chapter 2: Literature Review / Analytical Part (Prehľad literatúry)
- **Existing content sources**:
  - `Thesis.md` (Literature Review section) — ~3,500 words on AI in security, surveillance, HMI, ethics, privacy, FRS features
  - `LaTeX/analytical.tex` §2 "Methods and Algorithms" — ~2,000 words on CNNs, Eigenfaces, Fisherfaces, LBP, Haar Cascades, HOG, Deep Metric Learning, 3D FR, Viola-Jones
- **Quality assessment**: **Good with updates needed**
- **Usable content**:
  - Surveillance/security industry analysis (`Thesis.md` lines 80-156) — well-cited, thorough, directly usable
  - HMI discussion (`Thesis.md` line 154-157) — brief but adequate
  - Ethics and privacy sections — good coverage of GDPR, CCPA, FAR/FRR metrics
  - Methods/Algorithms descriptions (Eigenfaces, Fisherfaces, LBP, HOG, Haar, CNNs, Deep Metric Learning) — all usable as-is
- **Stale content**:
  - Amazon Rekognition section — not used in the project, should be reduced to a brief mention or removed
  - 3D Face Recognition section — not used in the project, should be brief mention only
  - `LaTeX/analytical.tex` includes Viola-Jones as separate section when it's the same as Haar Cascades — redundant
  - Citation format inconsistency: `Thesis.md` uses `[cite: ...]` notation while LaTeX uses `\cite{}`
- **Missing content**:
  - **FaceNet / InceptionResNetV1 architecture** — currently only superficial coverage. Needs deeper treatment as the backbone of the thesis
  - **Transfer learning theory** — not covered in existing literature review at all. Critical gap since 3 fine-tuning strategies are the main contribution
  - **Triplet loss and metric learning theory** — only briefly mentioned. Needs formal mathematical treatment
  - **Progressive unfreezing / discriminative fine-tuning** — not in lit review
  - **ArcFace, SphereFace, CosFace** — modern face recognition losses not discussed
  - **MediaPipe BlazeFace** — not covered despite being the default detector
- **Estimated effort**: **Medium** — strong existing base but needs significant additions for transfer learning and modern face recognition methods
- **Key sources for writing**: `Thesis.md` (Literature Review), `LaTeX/analytical.tex`, `docs/thesis/chapters/facenet_finetuning.md` §2 (Related Work — excellent material to incorporate), `docs/thesis/chapters/detection_methods.md` §2 (method descriptions)

---

### Chapter 3: Tools & Libraries (Nástroje a knižnice)
- **Existing content sources**:
  - `Thesis.md` (Tools and Libraries section) — ~2,000 words covering OpenCV, TensorFlow, Albumentations, Labelme, face_recognition, dlib, MTCNN, FaceNet, Amazon Rekognition
  - `LaTeX/analytical.tex` §1 — ~800 words (same content, LaTeX formatted)
- **Quality assessment**: **Needs rewrite**
- **Usable content**:
  - Tool descriptions for OpenCV, TensorFlow, dlib, face_recognition, MTCNN, Albumentations — all fundamentally correct
  - FaceNet description — correct but too shallow
- **Stale content**:
  - **Amazon Rekognition** — not used in the final system, takes up significant space
  - **Labelme** — no longer used for labeling (switched to flat file naming convention)
  - Missing **MediaPipe** — the default detector in the final system, not mentioned at all
  - Missing **uv** package manager, **nox** test runner, **Pydantic** settings — tools actually used
  - Missing **keras-facenet** package — how FaceNet is loaded
  - Missing **TensorFlow Lite** — used for quantization
  - Comparison section marked "doplniť" (to be completed) — never written
- **Missing content**:
  - MediaPipe BlazeFace description and rationale
  - TensorFlow Lite / model quantization tools
  - Updated comparison table of all tools actually used
  - Build tooling (uv, nox, Makefile) — brief mention
- **Estimated effort**: **Medium** — existing descriptions need updating, new tools need adding, irrelevant ones need trimming
- **Key sources for writing**: `Thesis.md` (Tools section), `CLAUDE.md` (tool list), `README.md` (technology stack), `config/models.yaml` (all models), `.maintenance/reports/IMPLEMENTATION_CONTEXT.md` (detailed tool inventory)

---

### Chapter 4: Methods & Algorithms (Metódy a algoritmy)
- **Existing content sources**:
  - `Thesis.md` (Methods and Algorithms section) — ~2,500 words
  - `LaTeX/analytical.tex` §2 — ~2,000 words
  - `docs/thesis/chapters/detection_methods.md` — ~2,500 words (excellent, production quality)
  - `docs/thesis/chapters/facenet_finetuning.md` §2-3 — ~2,000 words (methodology)
  - `docs/thesis/chapters/open_vs_closed_set.md` — ~1,800 words
- **Quality assessment**: **Good — strong source material**
- **Usable content**:
  - `detection_methods.md` — **production-ready chapter content** covering all 4 detection methods with benchmarks, tables, LaTeX code. Can be used nearly verbatim.
  - `facenet_finetuning.md` methodology section — comprehensive coverage of all 3 fine-tuning strategies with architecture diagrams, hyperparameters, formulas
  - `open_vs_closed_set.md` — complete comparison of paradigms with formal definitions and trade-off analysis
  - Eigenfaces, Fisherfaces, LBP descriptions from `Thesis.md` — usable for background
- **Stale content**:
  - `Thesis.md` and `LaTeX/analytical.tex` method descriptions are duplicated and don't cover the actual methods used (no MediaPipe, no FaceNet fine-tuning strategies)
  - Old detection timing data in `LaTeX/mainpart.tex` (Table: Haar 11ms, Dlib 80ms, FaceNet 105ms) contradicts new benchmarks (MediaPipe 3ms, Haar 31ms, Dlib 156ms, MTCNN 240ms) — different test conditions
- **Missing content**:
  - Unified narrative connecting detection methods → recognition methods → training strategies
  - Cosine similarity matching algorithm description (used in open-set pipeline)
  - Data augmentation methodology (Albumentations pipeline specifics)
- **Estimated effort**: **Low-Medium** — excellent source material exists, needs consolidation and narrative flow
- **Key sources for writing**: `docs/thesis/chapters/detection_methods.md`, `docs/thesis/chapters/facenet_finetuning.md` §3, `docs/thesis/chapters/open_vs_closed_set.md`, `src/bp_face_recognition/vision/interfaces.py`, `config/models.yaml`

---

### Chapter 5: Datasets (Datasety)
- **Existing content sources**:
  - `Thesis.md` (Datasets section) — ~1,200 words on dataset importance, creation, labeling, augmentation, splitting
  - `LaTeX/mainpart.tex` §1-2 (Dataset Creation) — ~500 words
  - `docs/thesis/chapters/facenet_finetuning.md` §3.1 — ~400 words (dataset for FaceNet experiments)
  - `docs/thesis/chapters/results_analysis.md` §4 — class imbalance analysis
- **Quality assessment**: **Needs rewrite**
- **Usable content**:
  - General discussion of dataset importance and creation methodology (`Thesis.md`) — reusable
  - FaceNet dataset description (7,080 images, 14 identities, splits) — accurate and current
  - Class distribution analysis from `results_analysis.md` §4 — excellent, directly usable
  - Augmentation pipeline description — correct
- **Stale content**:
  - `Thesis.md` references creating dataset with "30 images per dataset" and using Labelme for annotation — this was the early approach. The final system uses 7,080 images with flat file naming
  - `LaTeX/mainpart.tex` describes an older annotation workflow with Labelme JSON files — no longer used
  - References to `seccam` (first security camera) dataset which was superseded by `seccam_2`
  - Code snippet showing Albumentations augmentor with `RandomCrop(450, 450)` — actual pipeline uses `160x160`
- **Missing content**:
  - Description of **combined dataset** (webcam + seccam_2) — the actual dataset used for FaceNet experiments
  - **LFW dataset** usage for metric learning — mentioned in reports but not documented in thesis content
  - Updated **preprocessing pipeline** description (crop_faces.py → split_lfw.py → augmentation.py)
  - **Flat file structure** convention (`{label}_{uuid}.jpg`)
  - Ethics/privacy statement about the custom dataset (using own face + strangers)
- **Estimated effort**: **Medium** — significant rewriting needed to match actual dataset
- **Key sources for writing**: `docs/thesis/chapters/facenet_finetuning.md` §3.1, `docs/thesis/chapters/results_analysis.md` §4, `.maintenance/reports/FACENET_TRANSFER_LEARNING_REPORT.md` §3, `src/bp_face_recognition/preprocessing/`, `CLAUDE.md` (dataset conventions)

---

### Chapter 6: Implementation (Implementácia)
- **Existing content sources**:
  - `docs/thesis/chapters/system_architecture.md` — ~2,500 words (excellent, production quality)
  - `LaTeX/mainpart.tex` §5 (System Architecture) — ~800 words (old version)
  - `ARCHITECTURE.md` — ~1,500 words (concise current architecture)
  - `.maintenance/reports/IMPLEMENTATION_CONTEXT.md` — ~2,500 words (comprehensive file inventory)
  - `.maintenance/reports/PROJECT_COMPLETION_SUMMARY.md` — ~800 words
  - `.maintenance/reports/IMPLEMENTATION_COMPLETE.md` — ~800 words (GPU delegate)
- **Quality assessment**: **Good — needs consolidation**
- **Usable content**:
  - `system_architecture.md` — **production-ready** chapter covering layered architecture, plugin system, service layer, preprocessing pipeline, training infrastructure, cross-platform support, design decisions table. Essentially a complete implementation chapter.
  - `ARCHITECTURE.md` — clean diagrams and data flow descriptions
- **Stale content**:
  - `LaTeX/mainpart.tex` describes old architecture: `camera.py`, `model.py`, `database.py`, `main.py` — monolithic design that was replaced by the layered service architecture
  - `LaTeX/mainpart.tex` references MTCNN as the detector, `EfficientNetB0` as the primary model, `FaceTracker` class — partially outdated
  - `IMPLEMENTATION_CONTEXT.md` has mixed old/new file paths and some references to files that may have been reorganized
  - `PROJECT_COMPLETION_SUMMARY.md` focuses on Sessions 1-2 (MediaPipe + quantization) — doesn't cover FaceNet fine-tuning
- **Missing content**:
  - **Closed-set pipeline** implementation details (entry point, service, model loading)
  - **Registration workflow** (register_from_camera.py)
  - **Model switching** mechanism (scripts/switch_model.py)
  - Class diagrams / UML (only ASCII diagrams exist)
  - **Deployment instructions** brief summary
- **Estimated effort**: **Low-Medium** — `system_architecture.md` is nearly complete, just needs supplementation
- **Key sources for writing**: `docs/thesis/chapters/system_architecture.md`, `ARCHITECTURE.md`, `src/bp_face_recognition/services/`, `src/bp_face_recognition/vision/interfaces.py`, `config/models.yaml`

---

### Chapter 7: Results (Výsledky)
- **Existing content sources**:
  - `docs/thesis/chapters/detection_methods.md` §4-5 — ~1,000 words (detection benchmarks)
  - `docs/thesis/chapters/facenet_finetuning.md` §4-5 — ~2,000 words (recognition results)
  - `docs/thesis/chapters/results_analysis.md` — ~3,000 words (per-class analysis, confusion matrices, class imbalance)
  - `docs/thesis/chapters/open_vs_closed_set.md` §4 — ~500 words (paradigm comparison)
  - `LaTeX/mainpart.tex` §3-4 (Evaluation, Experimental Results) — ~800 words (old results)
  - `.maintenance/reports/FINAL_EVALUATION_REPORT.md` — ~300 words (old Jan 2026 results)
  - `.maintenance/reports/FACENET_TRANSFER_LEARNING_REPORT.md` §5-6 — ~2,000 words
  - `results/thesis/` directory — JSON data, benchmark reports
- **Quality assessment**: **Good — excellent source material, needs unification**
- **Usable content**:
  - Detection benchmark tables and analysis (`detection_methods.md`) — production-ready, includes LaTeX tables
  - FaceNet 3-way comparison table (TL: 92.84%, PU: 99.15%, TLoss: 94.63%) — definitive results
  - Per-class accuracy breakdown (`results_analysis.md` Table 1) — 14-class detailed analysis
  - Confusion matrix interpretation — thorough, well-written
  - Class imbalance impact analysis — includes correlation with training size
  - Training curves analysis — covers all 3 strategies
  - Model size vs accuracy trade-off table
  - Statistical significance analysis (p-values)
  - All LaTeX table code ready for inclusion
- **Stale content**:
  - `LaTeX/mainpart.tex` old results: references training on webcam/seccam/seccam_2 separately, shows different timing data, discusses overfitting on webcam dataset — all from pre-FaceNet era
  - `FINAL_EVALUATION_REPORT.md` (Jan 2026): MTCNN 72.55%, Dlib HOG 23.47%, Haar 2.92% — these are **recognition accuracy** numbers from old evaluate_methods.py, not current detection benchmarks. Very confusing if mixed with current data.
  - `FACENET_TRANSFER_LEARNING_REPORT.md` shows Option C as "TBD" / "In Progress" — it was later completed (94.63%)
  - EfficientNetB0 results (100% on seccam_2) noted with caveat "different dataset" — needs clear disclaimer
- **Missing content**:
  - **Unified results narrative** tying detection → recognition → full pipeline
  - **Open-set recognition accuracy** (with database matching) — no quantitative results for this paradigm
  - **Real-time FPS** of full pipeline (detection + recognition together)
  - **Comparison with state-of-the-art** (beyond brief mention of original FaceNet 99.63%)
  - **Quantization results** impact on accuracy (mentioned as 62% size reduction but no accuracy verification)
- **Estimated effort**: **Medium** — rich source material but needs careful unification and gap-filling
- **Key sources for writing**: `docs/thesis/chapters/facenet_finetuning.md` §4, `docs/thesis/chapters/results_analysis.md`, `docs/thesis/chapters/detection_methods.md` §4, `results/thesis/` (JSON data), `docs/thesis/tables/facenet_results_tables.md`

---

### Chapter 8: Conclusion (Záver)
- **Existing content sources**:
  - `LaTeX/conclusion.tex` — ~200 words
  - `docs/thesis/chapters/facenet_finetuning.md` §6 — ~500 words (FaceNet conclusions)
  - `docs/thesis/chapters/open_vs_closed_set.md` §7 — ~200 words
  - `docs/thesis/chapters/detection_methods.md` §5.4 — ~200 words (recommendations)
- **Quality assessment**: **Needs rewrite**
- **Usable content**:
  - `LaTeX/conclusion.tex` — generic but structurally sound. The framework (achievements, limitations, future directions) is reusable.
  - FaceNet conclusions — research questions answered, practical recommendations. Directly usable.
  - Detection recommendations (MediaPipe for real-time, MTCNN for batch) — directly usable.
- **Stale content**:
  - `LaTeX/conclusion.tex` references EfficientNetB0 as backbone and doesn't mention FaceNet, progressive unfreezing, or the 99.15% achievement
  - No mention of dual-paradigm (open-set/closed-set) as a contribution
  - Generic future directions ("lightweight architectures", "reducing bias") — should be specific to this work
- **Missing content**:
  - Specific thesis contributions enumerated
  - Key quantitative results summary (99.15% accuracy, 325 FPS detection, etc.)
  - Specific future work (semi-hard mining for triplet loss, larger datasets, FAISS integration, etc.)
  - Reflection on research questions answered
- **Estimated effort**: **Medium** — needs to be rewritten to summarize actual achievements
- **Key sources for writing**: `docs/thesis/chapters/facenet_finetuning.md` §6, `docs/thesis/chapters/detection_methods.md` §5.4, `docs/thesis/chapters/open_vs_closed_set.md` §7, `ARCHITECTURE.md`, `README.md` (benchmark results)

---

## Content Inventory

| Metric | Estimate |
|--------|----------|
| **Total existing words** | ~28,000 |
| **Usable as-is or with minor edits** | ~14,000 (50%) |
| **Needs major rewrite** | ~8,000 (29%) |
| **Needs to be written from scratch** | ~6,000 (21%) |

### Word count breakdown by source:
- `Thesis.md`: ~10,000 words — 50% usable, 30% stale, 20% redundant with newer sources
- `docs/thesis/chapters/*.md` (5 files): ~12,300 words — 85% usable, high quality
- `LaTeX/*.tex` (4 files): ~4,500 words — 30% usable (mostly lit review), 70% stale
- `.maintenance/reports/*` (5 files): ~6,500 words — reference material, not directly usable as thesis text
- `ARCHITECTURE.md`, `README.md`: ~3,500 words — reference material for implementation chapter

### Estimated final thesis length:
- Target: ~15,000-20,000 words (typical for Slovak/Czech bachelor thesis at TUKE)
- With existing usable material: ~14,000 words available
- New content needed: ~4,000-6,000 words

---

## Priority Order

Recommended writing order based on dependencies, effort, and available material:

### Phase 1: Core technical chapters (best source material, highest ROI)
1. **Chapter 6: Implementation** — `system_architecture.md` is 90% ready. Quick win.
2. **Chapter 7: Results** — Excellent source material in 3 Markdown chapter files + LaTeX tables. Needs consolidation.
3. **Chapter 4: Methods & Algorithms** — `detection_methods.md` + `facenet_finetuning.md` methodology provide most content.

### Phase 2: Foundation chapters (need updates to match final system)
4. **Chapter 5: Datasets** — Moderate rewrite, but FaceNet dataset description is solid.
5. **Chapter 3: Tools & Libraries** — Update tool inventory, add MediaPipe, remove Rekognition.
6. **Chapter 2: Literature Review** — Strong existing base, but needs transfer learning additions.

### Phase 3: Framing chapters (depend on knowing what's in the body)
7. **Chapter 1: Introduction** — Rewrite once body chapters are done to accurately reflect content.
8. **Chapter 8: Conclusion** — Write last, summarizing actual results and contributions.

### Rationale:
- Chapters 6-7 have the most complete source material and define the thesis's core contributions
- Chapters 4-5 provide necessary context that later chapters reference
- Literature review additions (transfer learning theory) are needed before Methods can be finalized
- Introduction and Conclusion should be written last to accurately frame the completed work

---

## Risks & Notes

### Inconsistencies Found

1. **Detection benchmark data mismatch**: `FINAL_EVALUATION_REPORT.md` (Jan 2026) shows MTCNN at 72.55% "accuracy" and Haar at 2.92% — these appear to be **recognition accuracy** from the old evaluate_methods.py (IoU-based detection + recognition combined), NOT the detection-only benchmarks in `detection_methods.md` (which measure detection count and speed). These must not be conflated.

2. **Detection timing discrepancy**: `LaTeX/mainpart.tex` Table (Haar: 11ms, Dlib: 80ms, FaceNet: 105ms) vs `detection_methods.md` (Haar: 31ms, Dlib: 156ms, MediaPipe: 3ms, MTCNN: 240ms). Different test conditions (different image sizes, different hardware?). The `detection_methods.md` data is authoritative (19 surveillance frames, 800px max).

3. **EfficientNetB0 100% accuracy caveat**: Reported in multiple files but on a **different dataset** (seccam_2, 15 classes) than the FaceNet models (combined webcam+seccam_2, 14 classes). Direct comparison is invalid — this is noted in `results_analysis.md` but could easily be misrepresented.

4. **Option C (Triplet Loss) status confusion**: `FACENET_TRANSFER_LEARNING_REPORT.md` and `THESIS_CHAPTER_STRUCTURE.md` show it as "TBD" / "In Progress", but `facenet_finetuning.md` and `results_analysis.md` have complete results (94.63%). The later files are authoritative.

5. **Hypothesis H3 was wrong**: The report predicted triplet loss would achieve highest accuracy (>97%), but it actually got 94.63% — lower than progressive unfreezing. This should be honestly discussed in the thesis.

### Missing Experimental Data

1. **Open-set recognition accuracy**: No quantitative evaluation of the open-set (embedding + database) pipeline exists. Only closed-set classification accuracy is measured. This is a gap — the thesis claims open-set recognition works but has no accuracy numbers for it.

2. **Full pipeline FPS**: Detection benchmark measures detection speed alone. No end-to-end FPS measurement (detection + recognition + database lookup) is documented.

3. **Cross-dataset generalization**: All FaceNet results are on the same dataset. No cross-dataset validation was performed. This is acknowledged as a limitation.

4. **Statistical significance methodology**: P-values mentioned (p < 0.001) but the statistical test used is not specified. Needs clarification or removal.

### Citations Needing Verification

1. `Thesis.md` uses `[cite: securityindustry_2025_transforming]` extensively (~20 times) — need to verify this maps to an actual reference entry
2. `[cite: kairos_secret_2018]`, `[cite: getfocal_biometric_2025]`, `[cite: transcend_ccpa_2025]` — verify these exist in bibliography
3. `[cite: wijaya_trends_2025]`, `[cite: researchgate_evaluation_2023]` — referenced in Viola-Jones section
4. `[cite: mdpi_systematic_2025]` — referenced in preprocessing section
5. FaceNet chapter cites Felbo et al. (2017) for progressive unfreezing — this is actually from the NLP domain (emoji sentiment). Verify if Howard & Ruder (2018) ULMFiT is the more standard citation for progressive unfreezing
6. Thesis references `docs/thesis/chapters/facenet_finetuning.md` as having references — need to ensure all 9 references are in the LaTeX bibliography

### Architecture / Structural Notes

1. **Thesis language**: The thesis will need to be in **Slovak** (TUKE requirement). All English Markdown content needs translation. This audit assumes English drafting first, then translation.

2. **LaTeX template**: Existing `LaTeX/` directory has a working template structure. Chapter files exist but content is outdated. The Markdown → LaTeX migration path is viable.

3. **Figures needed**: Confusion matrices, training curves, architecture diagrams, class distribution charts exist in `results/thesis/` as PNG files. Need to verify all figures referenced in chapter drafts actually exist.

4. **`THESIS_CHAPTER_STRUCTURE.md` is partially obsolete**: It was a planning document for the FaceNet chapter only, not the full thesis. Its timeline and checklist items are outdated. The actual chapter structure has evolved beyond this plan.

5. **Duplicate content risk**: Several topics (FaceNet architecture, progressive unfreezing, dataset description) appear in 3-4 files. When consolidating, care must be taken to use the **most recent and complete** version, which is generally in `docs/thesis/chapters/`.
