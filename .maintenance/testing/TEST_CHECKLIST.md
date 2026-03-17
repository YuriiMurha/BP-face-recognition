# FaceNet Runtime Testing Checklist

## Pre-Test Setup

- [ ] Ensure camera is working
- [ ] Good lighting conditions
- [ ] Test subject (you) available
- [ ] Optional: second person for false positive testing
- [ ] Config/models.yaml updated with correct model
- [ ] Database cleared (fresh start)

---

## Model Under Test: _________________

**Model Info:**
- Name: 
- Type: FaceNet (PU / TL / TLoss)
- Expected Accuracy: ___%
- File: 

**Test Date:** ___________
**Tester:** ___________

---

## Phase 1: Registration (10 samples)

Register your face from different angles:

| Sample # | Angle | Distance | Lighting | Notes |
|----------|-------|----------|----------|-------|
| 1 | Frontal | Normal | Good | Baseline |
| 2 | Frontal | Close | Good | |
| 3 | Frontal | Far | Good | |
| 4 | Slight Left | Normal | Good | |
| 5 | Slight Right | Normal | Good | |
| 6 | Frontal | Normal | Dim | |
| 7 | Frontal | Normal | Bright | |
| 8 | Looking Up | Normal | Good | |
| 9 | Looking Down | Normal | Good | |
| 10 | Frontal | Normal | Good | With glasses/hat? |

**Registration Quality:** ___/10
- Were all samples captured clearly? [ ] Yes [ ] No
- Any issues during registration? _______________

---

## Phase 2: Recognition Testing

### Test 2.1: Basic Recognition

| Test # | Condition | Recognized? | Confidence | Notes |
|--------|-----------|-------------|------------|-------|
| 1 | Frontal, good light | [ ] Yes [ ] No | ___% | |
| 2 | Frontal, dim light | [ ] Yes [ ] No | ___% | |
| 3 | Frontal, bright light | [ ] Yes [ ] No | ___% | |
| 4 | Slight left angle | [ ] Yes [ ] No | ___% | |
| 5 | Slight right angle | [ ] Yes [ ] No | ___% | |
| 6 | Looking up | [ ] Yes [ ] No | ___% | |
| 7 | Looking down | [ ] Yes [ ] No | ___% | |
| 8 | Close distance | [ ] Yes [ ] No | ___% | |
| 9 | Far distance | [ ] Yes [ ] No | ___% | |
| 10 | With glasses/hat | [ ] Yes [ ] No | ___% | |

**Basic Recognition Score:** ___/10

### Test 2.2: Robustness Testing

| Test # | Condition | Recognized? | Confidence | Notes |
|--------|-----------|-------------|------------|-------|
| 11 | Extreme left angle | [ ] Yes [ ] No | ___% | |
| 12 | Extreme right angle | [ ] Yes [ ] No | ___% | |
| 13 | Partial face (turned) | [ ] Yes [ ] No | ___% | |
| 14 | Low light | [ ] Yes [ ] No | ___% | |
| 15 | Backlighting | [ ] Yes [ ] No | ___% | |
| 16 | Fast movement | [ ] Yes [ ] No | ___% | |
| 17 | Slow movement | [ ] Yes [ ] No | ___% | |
| 18 | Expression change (smile) | [ ] Yes [ ] No | ___% | |
| 19 | Expression change (serious) | [ ] Yes [ ] No | ___% | |
| 20 | Different background | [ ] Yes [ ] No | ___% | |

**Robustness Score:** ___/10

### Test 2.3: False Positive Testing (if second person available)

| Test # | Person | Recognized As You? | Notes |
|--------|--------|-------------------|-------|
| 1 | Person 2 | [ ] Yes [ ] No | |
| 2 | Person 2 (different angle) | [ ] Yes [ ] No | |
| 3 | Person 2 (with glasses) | [ ] Yes [ ] No | |

**False Positive Rate:** ___/3 (lower is better)

---

## Phase 3: Performance Metrics

### Speed Assessment
- Registration speed: [ ] Fast [ ] Medium [ ] Slow
- Recognition speed: [ ] Fast [ ] Medium [ ] Slow
- App responsiveness: [ ] Good [ ] Acceptable [ ] Laggy

### User Experience
- How did the recognition feel?
  - [ ] Very responsive and accurate
  - [ ] Good, occasional misses
  - [ ] Okay, some issues
  - [ ] Frustrating, many failures

### Overall Satisfaction
- Rate the model: ___/10
- Would you use this daily? [ ] Yes [ ] Maybe [ ] No

---

## Summary

**Total Recognition Score:** ___/20 (Basic + Robustness)

**Key Strengths:**
1. 
2. 
3. 

**Key Weaknesses:**
1. 
2. 
3. 

**Overall Assessment:**
_____________________________________________
_____________________________________________

**Recommendation:**
- [ ] Excellent - Use as primary model
- [ ] Good - Viable alternative
- [ ] Okay - Has limitations
- [ ] Poor - Not suitable

---

## Notes & Observations

Additional comments, issues, or observations:

_____________________________________________
_____________________________________________
_____________________________________________
_____________________________________________

