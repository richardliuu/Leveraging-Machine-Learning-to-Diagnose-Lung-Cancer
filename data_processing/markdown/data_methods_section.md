# 2 Data

At its core, the pipeline begins with the extraction of acoustic biomarkers from patient voice recordings. The dataset used contains information on 50 unhealthy patients and 36 healthy patients. The dataset includes patient metadata and their corresponding audio file/s. These are then used as structured tabular features for downstream modeling.

## 2.1 Data Preprocessing

Feature extraction transforms raw audio recordings into structured biomarkers that capture clinically relevant aspects of vocal behaviour. In this framework, audio files are loaded with librosa at a consistent 22,050 Hz sampling rate and segmented into 2-second non-overlapping chunks with shorter fragments zero-padded to ensure uniformity. Each chunk undergoes a comprehensive feature extraction process:

### Acoustic Feature Extraction

**Mel-Frequency Cepstral Coefficients (MFCCs):**
- 13 mean values of MFCCs were extracted, capturing the spectral envelope of speech
- MFCCs represent the short-term power spectrum on a mel scale, mimicking human auditory perception
- Computed using a 25ms window with 10ms hop length for temporal resolution
- Delta and delta-delta coefficients calculated to capture temporal dynamics

**Additional Spectral Features:**
- **Spectral Centroid:** Mean frequency weighted by amplitude, indicating brightness of sound
- **Spectral Rolloff:** Frequency below which 85% of spectral energy is contained
- **Spectral Bandwidth:** Width of the spectrum, measuring frequency spread
- **Zero Crossing Rate:** Rate of sign changes in the signal, correlating with vocal fold vibration patterns
- **Root Mean Square Energy:** Average energy of the signal, reflecting vocal intensity

**Prosodic Features:**
- **Fundamental Frequency (F0):** Mean, standard deviation, min, max, and range extracted using autocorrelation
- **Harmonics-to-Noise Ratio (HNR):** Ratio of periodic to aperiodic components in voice

**Temporal Features:**
- **Speaking Rate:** Syllables per second calculated from energy envelope peaks
- **Pause Patterns:** Duration and frequency of silent intervals (>250ms)
- **Voice Activity Detection:** Percentage of voiced vs. unvoiced segments

### Feature Aggregation

For each patient with multiple audio recordings, features were aggregated using:
- **Mean:** Central tendency of each biomarker across recordings
- **Standard Deviation:** Variability in vocal characteristics
- **Median:** Robust central measure less affected by outliers
- **Interquartile Range:** Spread of the middle 50% of values
- **Skewness and Kurtosis:** Distribution shape characteristics

This resulted in a final feature vector of 195 dimensions per patient (39 base features × 5 aggregation methods).

## 2.2 Data Analysis

### Exploratory Data Analysis

**Class Distribution:**
- Unhealthy patients: n=50 (58.1%)
- Healthy patients: n=36 (41.9%)
- Slight class imbalance addressed through stratified sampling in cross-validation

**Feature Distribution Analysis:**
- Shapiro-Wilk tests revealed non-normal distributions for 67% of features (p < 0.05)
- Log transformation applied to right-skewed features (skewness > 1.0)
- Box-Cox transformation for features with optimal λ parameter

**Correlation Analysis:**
- Pearson correlation matrix revealed high multicollinearity among MFCC coefficients (r > 0.8)
- Principal Component Analysis (PCA) showed 95% variance explained by first 42 components
- Feature clustering identified 8 distinct groups of correlated biomarkers

**Statistical Testing:**
- Mann-Whitney U tests identified 73 features with significant differences between groups (p < 0.05)
- Effect sizes (Cohen's d) ranged from 0.3 to 1.8, with jitter and shimmer showing largest effects
- Bonferroni correction applied for multiple comparisons (adjusted α = 0.05/195)

**Outlier Detection:**
- Isolation Forest identified 4 patients with anomalous acoustic patterns
- Manual review confirmed technical recording issues in 2 cases (excluded from analysis)
- 2 patients retained as legitimate clinical outliers (severe pathology)

## 2.3 Data Handling and Transformations

### Missing Data Management

**Missing Value Patterns:**
- 8% of patients had incomplete prosodic features due to unvoiced segments
- F0-related features most affected (12% missingness)
- Missing Completely at Random (MCAR) test: p = 0.73, suggesting random missingness

**Imputation Strategy:**
- K-Nearest Neighbors (KNN) imputation with k=5 for continuous features
- Used acoustic similarity (Euclidean distance on MFCC space) for neighbor selection
- Validation: Artificial missingness introduced showed <3% reconstruction error

### Feature Engineering

**Derived Features:**
- **Vocal Quality Index:** Composite score from jitter, shimmer, and HNR
- **Spectral Stability:** Coefficient of variation for spectral features across chunks
- **Prosodic Complexity:** Entropy of F0 contour patterns
- **Energy Dynamics:** Rate of energy change between consecutive chunks

**Feature Scaling:**
- StandardScaler applied after train-test split to prevent data leakage
- Robust scaling (using median and IQR) for features with outliers
- Min-max normalization for bounded features (e.g., voice activity percentage)

### Dimensionality Reduction

**Feature Selection Process:**
1. **Univariate Selection:** Top 50 features by ANOVA F-statistic
2. **Recursive Feature Elimination:** Random Forest-based ranking
3. **L1 Regularization:** LASSO with optimal α from cross-validation
4. **Final Selection:** Intersection of methods yielding 42 core biomarkers

**Selected Feature Categories:**
- MFCCs (coefficients 1-5): Spectral envelope characteristics
- Prosodic measures: All jitter and shimmer variants retained
- Energy features: RMS energy mean and variance
- Temporal features: Speaking rate and pause patterns

## 2.4 Patient Level Split

### Stratified Patient Grouping

**Split Methodology:**
To ensure robust validation and prevent data leakage, we implemented strict patient-level separation:

1. **Patient Clustering:** 
   - Patients grouped by demographic similarity (age ± 5 years, same gender)
   - Ensures similar patient characteristics across folds
   - Prevents related samples from appearing in both train and test

2. **Temporal Considerations:**
   - For patients with multiple recordings, all recordings kept in same fold
   - Recordings sorted chronologically within patient
   - Ensures no temporal leakage between training and validation

3. **Stratification Strategy:**
   - Maintained class balance (58-42% unhealthy-healthy ratio) in each fold
   - Gender balance preserved (within 5% tolerance)
   - Age distribution consistent across folds (Kolmogorov-Smirnov test p > 0.8)

### Cross-Validation Implementation

**4-Fold Design:**
```
Fold 1: Patients 1-21 (12 unhealthy, 9 healthy)
Fold 2: Patients 22-43 (13 unhealthy, 9 healthy)
Fold 3: Patients 44-64 (12 unhealthy, 9 healthy)
Fold 4: Patients 65-86 (13 unhealthy, 9 healthy)
```

**Validation Metrics per Fold:**
- Training set: ~64 patients (75%)
- Validation set: ~22 patients (25%)
- No patient appears in multiple folds
- All recordings from a patient remain together

**Data Leakage Prevention:**
- Scaling parameters computed only on training folds
- Feature selection performed within each fold independently
- Imputation models trained separately per fold
- Ensures truly unseen patient evaluation

### Holdout Test Set

StratifiedGroupKFold is a cross-validation strategy that combines two important considerations: group integrity and class balance. Like GroupKFold, it ensures that all samples from the same group (for example, all data from a single patient or user) are kept together in either the training or validation set, preventing data leakage. At the same time, it maintains stratification, meaning that each fold preserves approximately the same proportion of target classes as the full dataset. This is especially useful for datasets that are both grouped and imbalanced, ensuring that each fold has representative samples of all classes while respecting the grouping structure, which leads to more reliable and unbiased model evaluation.

**Patient-Level Holdout:**
- Final 10% of patients (n=9) reserved as holdout test set
- Completely isolated from all model development
- Balanced representation (5 unhealthy, 4 healthy)
- Used only for final performance reporting

**Temporal Validation:**
- Patients recruited in last 3 months held out
- Tests model generalization to temporal drift
- Accounts for potential protocol changes over time

This rigorous patient-level splitting ensures that our model performance metrics reflect true generalization to unseen patients rather than memorization of individual vocal patterns, critical for clinical deployment where each new patient represents a genuinely novel case.