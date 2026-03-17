# FaceNet Runtime Testing - Quick Start Guide

## Overview
Test all three FaceNet models in runtime and choose the best one for standardization.

**Models to Test:**
1. **FaceNet PU** (Progressive Unfreezing) - 99.15% accuracy ⭐ RECOMMENDED
2. **FaceNet TL** (Transfer Learning) - 92.84% accuracy
3. **FaceNet TLoss** (Triplet Loss) - 94.63% accuracy

---

## Prerequisites

- [ ] Camera working and accessible
- [ ] Good lighting in testing area
- [ ] At least 30 minutes available
- [ ] Optional: second person for false positive testing

---

## Quick Commands

### Switch Models
```bash
# On Windows Command Prompt:
switch.bat pu

# Or using Python directly (works everywhere):
uv run python scripts/switch_model.py pu

# Available commands:
#   pu     - FaceNet PU (Progressive Unfreezing) - 99.15% accuracy
#   tl     - FaceNet TL (Transfer Learning) - 92.84% accuracy
#   tloss  - FaceNet TLoss (Triplet Loss) - 94.63% accuracy
#   clear  - Clear database (backs up first)
```

### Run Application
```bash
make run
# or
uv run python -m bp_face_recognition.main
```

---

## Testing Protocol (Per Model)

### Step 1: Switch Model
```bash
switch.bat pu
```

### Step 2: Clear Database
```bash
switch.bat clear
```

### Step 3: Run App & Register
```bash
make run
```
- Press 'r' to register
- Enter name: "Yurii"
- Capture 10 samples from different angles
- Follow checklist in `.maintenance/testing/TEST_CHECKLIST.md`

### Step 4: Test Recognition
- Test various conditions (angles, lighting, distance)
- Use checklist to track results
- Note confidence scores

### Step 5: Document Results
- Fill out `.maintenance/testing/RESULTS_TEMPLATE.md`
- Save as `.maintenance/testing/results_facenet_pu.md`

### Step 6: Repeat for Other Models
Repeat steps 1-5 for:
- FaceNet TL: `switch.bat tl`
- FaceNet TLoss: `switch.bat tloss`

---

## Testing Checklist

For each model, test these conditions:

### Basic (10 tests)
- [ ] Frontal, good light
- [ ] Frontal, dim light
- [ ] Frontal, bright light
- [ ] Slight left angle
- [ ] Slight right angle
- [ ] Looking up
- [ ] Looking down
- [ ] Close distance
- [ ] Far distance
- [ ] With glasses/accessories

### Robustness (10 tests)
- [ ] Extreme left angle
- [ ] Extreme right angle
- [ ] Partial face
- [ ] Low light
- [ ] Backlighting
- [ ] Fast movement
- [ ] Slow movement
- [ ] Smiling expression
- [ ] Serious expression
- [ ] Different background

### False Positives (if second person available)
- [ ] Person 2 - frontal
- [ ] Person 2 - angle
- [ ] Person 2 - with glasses

---

## What to Measure

### For Each Test
- **Recognized?** Yes/No
- **Confidence Score:** ___% (shown in app)
- **Notes:** Any issues or observations

### Overall Metrics
- **Basic Recognition Rate:** ___/20
- **Robustness Score:** ___/20
- **False Positive Rate:** ___/3 (if tested)
- **Average Confidence:** ___%

---

## Expected Results

Based on offline evaluation:

| Model | Expected Accuracy | Expected Performance |
|-------|-------------------|---------------------|
| FaceNet PU | 99.15% | ⭐⭐⭐⭐⭐ Excellent |
| FaceNet TL | 92.84% | ⭐⭐⭐⭐ Good |
| FaceNet TLoss | 94.63% | ⭐⭐⭐⭐ Very Good |

---

## Decision Criteria

Choose the model that:
1. Has highest recognition accuracy
2. Is most robust to variations
3. Has lowest false positive rate
4. Feels responsive and reliable
5. Works consistently across conditions

---

## After Testing

1. Compare all three models
2. Fill out comparison report
3. Choose winner
4. Standardize on winner (clear other databases)
5. Update thesis documentation

---

## Troubleshooting

### "Model not found"
- Check that model files exist in `src/bp_face_recognition/models/finetuned/`
- Run training first if needed

### "Database error"
- Clear database: `switch.bat clear`
- Check file permissions

### "Recognition not working"
- Ensure config updated correctly
- Try different lighting
- Check camera settings

---

## Files & Templates

- **Testing Checklist:** `.maintenance/testing/TEST_CHECKLIST.md`
- **Results Template:** `.maintenance/testing/RESULTS_TEMPLATE.md`
- **Comparison Report:** `.maintenance/testing/RUNTIME_COMPARISON_REPORT.md` (to be created)

---

## Support

If you encounter issues:
1. Check `.maintenance/SESSION_17_PLAN.md` for detailed plan
2. Review this quick start guide
3. Document issues for troubleshooting

---

**Let's start testing! Run:**
```bash
switch.bat pu
switch.bat clear
make run
```

