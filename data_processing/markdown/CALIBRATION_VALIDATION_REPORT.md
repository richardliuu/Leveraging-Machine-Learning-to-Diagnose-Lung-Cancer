# Calibration R² Score Robustness Validation Report

## Executive Summary

After comprehensive testing with strict data isolation, bootstrap confidence intervals, and temporal validation, the calibration R² scores show **mixed robustness**:

- **Platt calibration**: More robust (R² = 0.896 ± 0.049)
- **Isotonic calibration**: Less robust (R² = 0.875 ± 0.051)
- **Critical finding**: Minimal actual improvement on held-out patients

## Key Findings

### 1. Data Isolation Results

Using strict nested cross-validation with complete data isolation:

| Method | Test R² | Val R² | CI Width | Overfitting Score |
|--------|---------|--------|----------|-------------------|
| Isotonic | 0.875 ± 0.051 | 0.924 ± 0.005 | ~0.074 | 0.050 |
| Platt | 0.896 ± 0.049 | 0.938 ± 0.012 | ~0.065 | 0.039 |

**Interpretation**: 
- Validation R² > Test R² by ~0.04-0.05, suggesting mild optimistic bias
- Confidence intervals are moderately wide (~0.07), indicating some uncertainty
- Overfitting scores < 0.1 are acceptable but present

### 2. Bootstrap Confidence Intervals

All folds show statistically significant R² scores (p < 0.0001), but:
- CI widths of 0.065-0.074 indicate moderate uncertainty
- No overlap with zero, confirming positive predictive ability
- But wide enough to question precise R² values

### 3. Temporal Validation

Progressive time-based validation shows:
- R² ranges from 0.861 to 0.942 across time periods
- Positive correlation (+0.400) suggests performance **improves** over time
- This could indicate data drift or learning from accumulated samples

**Red Flag**: Performance improving over time is unusual and may indicate:
- The model is learning temporal patterns rather than generalizable features
- Later patients may be easier to predict
- Possible data quality improvements over time

### 4. Patient-Level Holdout

Complete patient holdout validation reveals:
- Uncalibrated R²: 0.988
- Calibrated R²: 0.985
- **Calibration improvement: -0.003 (negligible)**

**Critical Finding**: Calibration provides no meaningful improvement on completely unseen patients!

### 5. Brier Score Decomposition

| Fold | Reliability | Resolution | Uncertainty |
|------|------------|------------|-------------|
| 1 | 0.001 | - | - |
| 2 | 0.064 | - | - |
| 3 | 0.005 | - | - |
| 4 | 0.009 | - | - |

Fold 2 shows much higher reliability score (worse calibration), explaining its historically poor performance.

## Robustness Assessment

### ✅ What IS Robust:
1. **Statistical significance**: All p-values < 0.0001
2. **Positive R² scores**: Consistently above 0.8
3. **Low overfitting scores**: All < 0.1
4. **Brier reliability**: Generally low (good)

### ⚠️ What is NOT Robust:
1. **No improvement on holdout patients**: -0.003 change
2. **Moderate CI widths**: ~0.07 uncertainty range
3. **Val-Test gap**: 0.04-0.05 difference suggests mild overfitting
4. **Temporal trend**: Performance changes over time
5. **Fold variance**: Std of 0.05 is non-negligible

## Root Causes of Limited Robustness

1. **Isotonic Regression Overfitting**: Non-parametric method memorizes training distribution
2. **Small Calibration Sets**: Only ~576 samples for calibration fitting
3. **Patient Heterogeneity**: Different patients may have fundamentally different patterns
4. **Limited Feature Space**: Surrogate model may already capture most signal

## Recommendations

### For Production Use:
1. **Use Platt calibration** over isotonic (more stable)
2. **Report confidence intervals**, not point estimates
3. **Monitor temporal drift** in deployment
4. **Validate on new patient cohorts** regularly

### For Improvement:
1. **Increase calibration set size** (currently only 15% of data)
2. **Try parametric calibration methods** (e.g., temperature scaling)
3. **Ensemble multiple calibrators** for stability
4. **Add regularization** to isotonic regression

## Final Verdict

**The calibration R² scores are NOT fully robust.**

While statistically significant, they show:
- Moderate uncertainty (CI width ~0.07)
- Mild overfitting (val-test gap ~0.05)
- No real-world improvement on holdout patients
- Temporal instability

**The scores are reliable enough for research** but require careful interpretation. The lack of improvement on patient holdout (-0.003) suggests calibration may be fitting noise rather than fixing genuine probability issues.

## Code Reproducibility

All validation code is available in:
- `models/robust_calibration_validation.py` - Main validation pipeline
- `models/temporal_validation.py` - Temporal and patient holdout tests

Run with:
```bash
python models/robust_calibration_validation.py
python models/temporal_validation.py
```

## Statistical Tests Performed

1. **Bootstrap CI** (n=500-1000): Uncertainty quantification
2. **Permutation Test** (n=500): Significance testing
3. **Spearman Correlation**: Temporal trend analysis
4. **GroupKFold CV**: Patient-aware splitting
5. **Brier Score Decomposition**: Calibration quality metrics
6. **Expected Calibration Error**: Reliability measurement

## Conclusion

The calibration provides statistically significant but practically minimal improvements. The R² scores of 0.875-0.896 are not artifacts of data leakage but show limited robustness due to:
- Moderate uncertainty
- Mild overfitting  
- No improvement on true holdout data

**Recommendation**: Use with caution and always report confidence intervals.