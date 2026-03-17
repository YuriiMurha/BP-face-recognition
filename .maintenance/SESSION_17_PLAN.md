# SESSION 17: FaceNet Runtime Testing & Standardization Plan

## Overview
**Goal:** Test all three FaceNet models in runtime, compare them, standardize on the best, and prepare thesis documentation.

**Status**: 🔄 **IN PROGRESS**

**Models to Test:**
1. **FaceNet PU** (Progressive Unfreezing) - 99.15% accuracy
2. **FaceNet TL** (Transfer Learning) - 92.84% accuracy  
3. **FaceNet TLoss** (Triplet Loss) - 94.63% accuracy

---

## PHASE 1: Infrastructure Setup (15 minutes)

### Tasks
- [x] Backup current 128D database
- [ ] Create model switching scripts
- [ ] Create testing checklist template
- [ ] Create results tracking spreadsheet

### Deliverables
- `scripts/switch_model.bat` - Switch between FaceNet variants
- `scripts/test_model.bat` - Run app with specific model
- `.maintenance/testing/TEST_CHECKLIST.md` - Testing checklist
- `.maintenance/testing/RESULTS_TEMPLATE.md` - Results template

---

## PHASE 2: Runtime Testing (1-2 hours)

### Test Protocol
For each model:
1. Clear database
2. Register face (10 samples from different angles)
3. Test recognition:
   - Frontal view
   - Side angles (left/right)
   - Different lighting
   - Different distances
   - With/without glasses
   - Test with another person (if available)
4. Document results

### Test 2.1: FaceNet PU (99.15%)
- [ ] Switch to facenet_pu
- [ ] Clear database
- [ ] Register 10 face samples
- [ ] Run recognition tests
- [ ] Document observations

### Test 2.2: FaceNet TL (92.84%)
- [ ] Switch to facenet_tl
- [ ] Clear database
- [ ] Register 10 face samples
- [ ] Run recognition tests
- [ ] Document observations

### Test 2.3: FaceNet TLoss (94.63%)
- [ ] Switch to facenet_tloss
- [ ] Clear database
- [ ] Register 10 face samples
- [ ] Run recognition tests
- [ ] Document observations

---

## PHASE 3: Analysis & Standardization (30 minutes)

### Tasks
- [ ] Compare all three models
- [ ] Choose winner based on:
  - Recognition accuracy in practice
  - Robustness to variations
  - False positive/negative rates
  - User experience
- [ ] Standardize on winner
- [ ] Update config to use winner as default

### Deliverables
- `.maintenance/testing/RUNTIME_COMPARISON_REPORT.md`
- `config/models.yaml` updated with default recognizer
- Clean database with winner's embeddings

---

## PHASE 4: Thesis Documentation (1 hour)

### Tasks
- [ ] Create comprehensive comparison document
- [ ] Include tables: accuracy, speed, robustness
- [ ] Add qualitative observations
- [ ] Create recommendations section
- [ ] Update thesis chapter

### Deliverables
- `docs/thesis/comparisons/facenet_runtime_comparison.md`
- Updated thesis chapter with findings
- Publication-ready tables

---

## Current Status

### Completed
- ✅ Backed up 128D database to `data/faces_backup_128d.csv`
- ✅ Identified available FaceNet models
- ✅ Created testing plan

### In Progress
- 🔄 Setting up testing infrastructure

### Next Steps
1. Create model switching scripts
2. Start FaceNet PU testing
3. Document results
4. Repeat for TL and TLoss
5. Compare and standardize

---

**Last Updated:** Session 17 - Phase 1 Setup
