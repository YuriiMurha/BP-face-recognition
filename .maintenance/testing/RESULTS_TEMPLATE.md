# FaceNet Runtime Testing Results Template

Fill out this template after testing each model.

---

## Model: _________________

**Configuration:**
- Model Type: FaceNet (PU / TL / TLoss)
- Expected Accuracy: ___%
- Test Date: ___________
- Tester: ___________

---

## Executive Summary

**Overall Score:** ___/10

**Verdict:**
- [ ] Excellent - Primary choice
- [ ] Good - Viable alternative  
- [ ] Acceptable - Has limitations
- [ ] Poor - Not recommended

---

## Detailed Results

### Registration Phase

**Samples Captured:** ___/10

**Registration Quality Issues:**
- [ ] None - All samples clear
- [ ] Minor - 1-2 samples had issues
- [ ] Moderate - 3-5 samples had issues
- [ ] Major - More than 5 samples had issues

**Issues Noted:**
_____________________________________________

### Recognition Phase

#### Basic Recognition (10 tests)
**Score:** ___/10 (___%)

**Breakdown:**
| Test Type | Success Rate | Avg Confidence |
|-----------|--------------|----------------|
| Frontal, good light | ___% | ___% |
| Different distances | ___% | ___% |
| Different angles | ___% | ___% |
| Different lighting | ___% | ___% |

#### Robustness (10 tests)
**Score:** ___/10 (___%)

**Breakdown:**
| Test Type | Success Rate | Avg Confidence |
|-----------|--------------|----------------|
| Extreme angles | ___% | ___% |
| Poor lighting | ___% | ___% |
| Expressions | ___% | ___% |
| Movement | ___% | ___% |

#### False Positives (if tested)
**Score:** ___/___ (___% false positive rate)

| Person | Times Recognized As You | Rate |
|--------|------------------------|------|
| Person 2 | ___/___ | ___% |

### Performance Metrics

#### Speed
- **Registration:** ___ seconds for 10 samples
- **Recognition:** ___ ms per frame
- **Overall App Speed:** [ ] Fast [ ] Medium [ ] Slow

#### Confidence Scores
- **Average Confidence (Correct):** ___%
- **Average Confidence (Incorrect):** ___%
- **Lowest Successful Recognition:** ___%

---

## Strengths & Weaknesses

### Top 3 Strengths
1. 
2. 
3. 

### Top 3 Weaknesses
1. 
2. 
3. 

---

## Comparison to Other Models

**Better than:** _________________

**Worse than:** _________________

**Similar to:** _________________

---

## User Experience Notes

### What Worked Well
_____________________________________________
_____________________________________________

### What Was Frustrating
_____________________________________________
_____________________________________________

### Surprises
_____________________________________________
_____________________________________________

---

## Recommendation

**Should this be the default model?**
- [ ] Yes - Clear winner
- [ ] Maybe - Good option
- [ ] No - Not suitable

**Reasoning:**
_____________________________________________
_____________________________________________

---

## Additional Notes

_____________________________________________
_____________________________________________
_____________________________________________
_____________________________________________

---

*Template version: 1.0*
