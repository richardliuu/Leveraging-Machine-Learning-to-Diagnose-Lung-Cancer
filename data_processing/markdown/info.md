# Data Preprocessing Summary: jitter_shimmer.py

## Overview
The `jitter_shimmer.py` file processes audio data from unhealthy vocal samples to extract acoustic features for cancer stage analysis.

## Data Processing Pipeline (Processing Order)

### 1. Configuration Setup
- **Input Directory**: `data/wavfiles/unhealthy/38-`
- **Output File**: `data/jitter_shimmerlog.csv`
- **Sample Rate**: 22,050 Hz
- **Chunk Duration**: 2.0 seconds
- **Cancer Stage**: 1

### 2. File Processing Loop
- Iterates through all WAV files in the audio directory
- Extracts patient ID from filename (without extension)
- Processes each audio file individually

### 3. Audio Loading (using **librosa**)
- **Audio Loading**: `librosa.load(sr=22050)` - Loads and resamples WAV files to consistent sample rate

### 4. Audio Chunking
- Splits audio files into **2-second non-overlapping chunks** using `chunk_audio()` function
- Pads final chunk with zeros if shorter than target duration
- Creates multiple data points per audio file for temporal analysis

### 5. Feature Extraction (Per Chunk)
Each audio chunk undergoes comprehensive feature extraction using multiple libraries:

#### MFCCs First (using **librosa**)
- **MFCCs**: `librosa.feature.mfcc(n_mfcc=13)` - Extracts 13 coefficients, calculates mean values
- Captures vocal tract characteristics and phonetic content

#### Spectral Features (using **librosa**)
- **Zero Crossing Rate (ZCR)**: `librosa.feature.zero_crossing_rate()` - Voice activity and texture measure
- **Spectral Centroid**: `librosa.feature.spectral_centroid()` - Brightness/frequency center of spectrum
- **Spectral Rolloff**: `librosa.feature.spectral_rolloff()` - 85% energy frequency threshold
- **Spectral Bandwidth**: `librosa.feature.spectral_bandwidth()` - Frequency range concentration
- **RMS Energy**: `librosa.feature.rms()` - Overall amplitude/loudness

#### Statistical Features (using **scipy.stats** and **numpy**)
- **Crest Factor**: Custom calculation using `np.max()` and `np.sqrt(np.mean())` - Peak-to-RMS ratio
- **Kurtosis**: `scipy.stats.kurtosis()` - Distribution tail heaviness (vocal irregularities)
- **Skew**: `scipy.stats.skew()` - Distribution asymmetry

### 6. Metadata Addition
- **Patient ID**: Extracted from filename
- **Stage**: Cancer stage (1)
- **Chunk Number**: Sequential chunk index
- **Filename**: Original audio file name

### 7. Output Format
- **CSV file**: `data/jitter_shimmerlog.csv`
- **Metadata**: Patient ID, cancer stage, chunk number, filename
- **Append mode**: Supports incremental processing
- **Total features**: 21 acoustic features + 4 metadata columns

## Key Characteristics
- Processes unhealthy vocal samples specifically
- Designed for cancer stage classification (stage 1)
- Chunk-based analysis for temporal feature variation
- Comprehensive acoustic feature set covering spectral, cepstral, and statistical domains

# Data: Project Results and Performance Metrics

## Random Forest Model Performance

### Dataset Statistics
- **Total Samples**: 3,841
- **Unique Patients**: 117
- **Class Distribution**: 
  - Class 0 (Healthy): 1,842 samples
  - Class 1 (Cancer): 1,999 samples
  - Class Ratio: 1.09:1 (balanced)
- **Features Used**: 16 (after excluding rolloff, bandwidth, skew, zcr, rms)
- **Data Integrity**: No duplicate feature rows, all patients have consistent labels

### Cross-Validation Configuration
- **Validation Method**: GroupKFold Cross-Validation (4-fold)
- **Patient-aware splitting**: No data leakage between train/test
- **Model Configuration**:
  - Number of estimators: 200 trees
  - Max depth: 5
  - Min samples split: 12
  - Min samples leaf: 3
  - Criterion: Log loss
  - Class weight: Balanced
  - Random state: 42

### Overall Performance Metrics
- **Mean Accuracy**: 0.8964 ± 0.0413 (89.64% ± 4.13%)
- **Min Accuracy**: 0.8522 (85.22%)
- **Max Accuracy**: 0.9500 (95.00%)
- **Class 0 F1-score**: 0.8797 ± 0.0653 (87.97% ± 6.53%)
- **Class 1 F1-score**: 0.9022 ± 0.0350 (90.22% ± 3.50%)

### Per-Fold Results
| Fold | Accuracy | Patients | Samples | ROC AUC | Class 0 (precision/recall/f1) | Class 1 (precision/recall/f1) |
|------|----------|----------|---------|---------|--------------------------------|--------------------------------|
| 1 | 0.8522 | 29 | 961 | 0.8946 | 0.87/0.71/0.78 | 0.84/0.94/0.89 |
| 2 | 0.9229 | 29 | 960 | 0.9811 | 0.97/0.90/0.94 | 0.85/0.96/0.90 |
| 3 | 0.9500 | 29 | 960 | 0.9902 | 0.92/0.96/0.94 | 0.97/0.94/0.96 |
| 4 | 0.8604 | 30 | 960 | 0.9384 | 0.87/0.86/0.86 | 0.85/0.87/0.86 |

### Average Confusion Matrix
```
       Predicted 0  Predicted 1
Actual 0    398          63
Actual 1     37         463
```
- **True Negative Rate**: 86.3%
- **True Positive Rate**: 92.6%
- **False Positive Rate**: 13.7%
- **False Negative Rate**: 7.4%

## Most Representative Tree (MRT) Analysis

### Tree Selection Methodology
- **Selection Criterion**: Prediction similarity to ensemble
- **Metrics Used**:
  - Accuracy similarity to ensemble predictions
  - Probability similarity (via MSE)
  - Combined similarity score (average of both)

### MRT Statistics (Actual Results)
- **Selected Tree Index**: 112 (out of 200 trees)
- **Maximum Similarity Score**: 0.9965 (99.65%)
- **Mean Similarity Score**: 0.9644 ± 0.0249 (96.44% ± 2.49%)
- **Tree Depth**: 5
- **Number of Leaves**: 24
- **Total Node Count**: 47
- **Prediction Agreement with Ensemble**: 100.00%

### Visualization Outputs
- **Tree Similarity Distribution**: Generated histogram saved as `tree_similarity_distribution.png`
- **Representative Tree Structure**: Exported to `representative_tree.dot` for GraphViz visualization
- **High Similarity**: All trees show >90% similarity, indicating ensemble consistency

## Calibration Performance

### Calibration Methods Tested
1. **Isotonic Calibration**
   - Non-parametric monotonic regression
   - Out-of-bounds clipping applied
   
2. **Platt Calibration**
   - Sigmoid calibration via logistic regression
   - Logit transformation with epsilon clipping

### Robust Validation Results

#### Isotonic Calibration Performance
- **Test R² Score**: 0.875 ± 0.051
- **Validation R² Score**: 0.924 ± 0.005
- **Confidence Interval Width**: ~0.074
- **Overfitting Score**: 0.050
- **Statistical Significance**: p < 0.0001 (all folds)

#### Platt Calibration Performance
- **Test R² Score**: 0.896 ± 0.049
- **Validation R² Score**: 0.938 ± 0.012
- **Confidence Interval Width**: ~0.065
- **Overfitting Score**: 0.039
- **Statistical Significance**: p < 0.0001 (all folds)

### Calibration Quality Metrics

#### Expected Calibration Error (ECE)
- Measured across 10 probability bins
- Lower values indicate better calibration
- Calculated for both training and validation sets

#### Brier Score Decomposition
Per-fold Brier reliability scores:
- **Fold 1**: 0.001 (excellent calibration)
- **Fold 2**: 0.064 (moderate calibration)
- **Fold 3**: 0.005 (good calibration)
- **Fold 4**: 0.009 (good calibration)

### Temporal Validation
- **R² Range**: 0.861 to 0.942 across time periods
- **Temporal Correlation**: +0.400 (positive trend)
- **Interpretation**: Performance improves over time, suggesting temporal patterns

### Patient-Level Holdout Validation
- **Uncalibrated R²**: 0.988
- **Calibrated R²**: 0.985
- **Calibration Improvement**: -0.003 (negligible)
- **Finding**: No meaningful improvement on completely unseen patients

## Surrogate Model Performance

### Decision Tree Regressor Configuration
- **Max Depth**: 8
- **Min Samples Leaf**: 6
- **Min Samples Split**: 20
- **Max Leaf Nodes**: 25
- **Random State**: 42 (for reproducibility)
- **Criterion**: MSE (Mean Squared Error)
- **Splitter**: Best (exhaustive search for optimal splits)

### Training Strategy
- **Target Variable**: Calibrated RF probabilities (Platt calibration)
- **Training Set**: Same patient-aware splits as RF model
- **Validation**: 4-fold cross-validation with patient grouping
- **Clipping**: Applied to ensure probability bounds [0,1]
- **Purpose**: Interpretable approximation of complex RF ensemble

### Cross-Validation Performance (4-fold)
- **Mean Test R²**: 0.9402 (94.02%)
- **Mean Validation R²**: 0.9150 (91.50%)
- **Per-Fold Test R² Range**: 0.9261 - 0.9666
- **Per-Fold Val R² Range**: 0.8146 - 0.9613
- **Fold 2 Performance**: Highest test R² of 0.9666
- **Fold 3 Performance**: Highest validation R² of 0.9613

### Model Characteristics
- **Tree Complexity**: 47 total nodes, 24 leaf nodes
- **Interpretability**: Single tree structure allows direct rule extraction
- **Fidelity to RF**: 94% variance explained in RF predictions
- **Stability**: Perfect determinism across all random seeds
- **Clinical Utility**: Provides transparent decision rules for diagnosis

## Seed Stability Testing

### Random Forest Classification Model
- **Seeds Tested**: 51 random seeds (0-50)
- **Metrics Evaluated**: AUC, Accuracy, F1-Score, Precision, Recall
- **Cross-Validation**: 4-fold GroupKFold (patient-aware)

#### RF Stability Results
- **Mean AUC**: 0.9541 ± 0.0005 (CV: 0.048%)
- **Mean Accuracy**: 0.8902 ± 0.0010 (CV: 0.112%)
- **Mean F1-Score**: 0.8977 ± 0.0009 (CV: 0.106%)
- **95% CI Width**: < 0.0003 for all metrics
- **Convergence**: Achieved after ~30 seeds
- **Statistical Assessment**: EXCELLENT stability (CV < 0.2%)

### Surrogate Decision Tree Model
- **Seeds Tested**: 51 random seeds (0-50)
- **Model Type**: Decision Tree Regressor (interpretable surrogate)
- **Target**: Calibrated RF probabilities

#### Surrogate Stability Results
- **Test R²**: 0.9402 (perfectly consistent across all seeds)
- **Validation R²**: 0.9150 (perfectly consistent across all seeds)
- **Standard Deviation**: 0.0000 (complete determinism)
- **Coefficient of Variation**: 0.00%
- **Statistical Power**: 100% (detects any deviation)
- **Finding**: Model exhibits perfect determinism - no randomness detected

## Statistical Validation Tests

### Bootstrap Confidence Intervals
- **Bootstrap Samples**: 500-1000 iterations
- **Confidence Level**: 95%
- **Application**: R² score uncertainty quantification

### Permutation Testing
- **Permutations**: 500 iterations
- **Purpose**: Test if R² significantly better than random
- **All p-values**: < 0.0001 (highly significant)

### Cross-Validation Strategy
- **GroupKFold**: Patient-aware splitting
- **Nested CV**: Separate calibration, training, and test sets
- **Data Isolation**: Strict separation prevents leakage

## Key Performance Indicators

### Model Robustness Metrics
- **Val-Test Gap**: 0.04-0.05 (mild overfitting present)
- **Cross-fold Variance**: Std ~0.05 (moderate consistency)
- **Confidence Interval Width**: ~0.07 (moderate uncertainty)
- **Overfitting Scores**: All < 0.1 (acceptable range)

### Clinical Relevance Metrics
- **Binary Classification**: Healthy vs Cancer detection
- **Feature Importance**: Acoustic biomarkers ranked by importance
- **Prediction Confidence**: Calibrated probability scores
- **Patient-level Predictions**: Aggregated from chunk-level analysis

## Data Quality Checks

### Integrity Verification
- **Duplicate Detection**: Check for repeated feature rows
- **Label Consistency**: Verify patients have consistent labels
- **Class Balance**: Monitor class distribution and ratios
- **Patient Leakage**: Ensure no overlap between train/test patients

### Sample Statistics
- **Total Samples**: Full dataset size
- **Unique Patients**: Number of distinct patient IDs
- **Chunks per Patient**: Variable based on audio length
- **Features Used**: 21 acoustic features (after exclusions)

## Summary Statistics

### Best Performing Configuration
- **Model**: Random Forest with 200 trees
- **Accuracy**: 89.64% ± 4.13% (cross-validation mean)
- **Best Fold Performance**: 95.00% accuracy with ROC AUC of 0.9902
- **Calibration**: Platt calibration (more stable than isotonic)
- **Calibrated Test R²**: 0.896 ± 0.049
- **Calibrated Validation R²**: 0.938 ± 0.012
- **Statistical Significance**: All folds p < 0.0001

### Key Performance Highlights
- **Class Balance**: Near-perfect balance (1.09:1 ratio)
- **F1-Score Consistency**: Class 1 (90.22%) slightly better than Class 0 (87.97%)
- **ROC AUC Range**: 0.8946 to 0.9902 across folds
- **Most Representative Tree**: Tree #112 with 99.65% similarity to ensemble
- **Patient-Level Performance**: 117 patients with no data leakage
- **RF Seed Stability**: CV < 0.2% across 51 seeds (excellent)
- **Surrogate Fidelity**: 94% R² approximating RF predictions

### Model Stability Summary
| Model | Seeds Tested | Key Metric | Stability | Interpretation |
|-------|-------------|------------|-----------|----------------|
| Random Forest | 51 | AUC CV: 0.048% | Excellent | Publication-ready |
| Surrogate DT | 51 | R² Std: 0.000 | Perfect | Fully deterministic |
| Calibration | 4 folds | Val R²: 0.938 | High | Reliable probabilities |

### Statistical Testing Adequacy
- **Sample Size**: 51 seeds exceeds requirements (typically 20-30 needed)
- **Convergence**: Achieved at ~30 seeds for RF model
- **Power Analysis**: 100% power to detect meaningful differences
- **Conclusion**: Current testing is statistically robust and complete

### Recommendations for Deployment
1. Use Platt calibration over isotonic (better stability)
2. Deploy surrogate model for interpretable predictions (R² = 0.94)
3. Report confidence intervals with all predictions
4. Monitor temporal drift in production environment
5. Validate regularly on new patient cohorts
6. Use Tree #112 for single-tree explanations when needed
7. Document seed=42 for reproducibility in clinical settings

# Paper: Interpretable Random Forest Approach

## Title
"Interpretable Machine Learning for Lung Cancer Detection Using Acoustic Biomarkers: A Random Forest Approach with Clinical Validation"

## Abstract Components
- **Background**: Lung cancer early detection through non-invasive voice analysis
- **Methods**: Random forest with comprehensive interpretability framework
- **Results**: Model performance metrics, feature importance rankings, clinical validation
- **Conclusion**: Practical implications for screening and diagnostic support

## Methodology

### 1. Data Collection and Preprocessing
- **Dataset Description**: 
  - Number of patients (healthy vs. cancer stages)
  - Audio recording protocols and equipment
  - Demographic characteristics
- **Audio Processing**:
  - 22,050 Hz sampling rate standardization
  - 2-second chunk segmentation for temporal analysis
  - Quality control and noise filtering procedures

### 2. Feature Engineering
- **Acoustic Features** (21 total):
  - **Spectral Domain**: Zero-crossing rate, spectral centroid, rolloff, bandwidth
  - **Cepstral Domain**: 13 MFCCs for vocal tract modeling
  - **Statistical Domain**: Kurtosis, skewness, crest factor
  - **Energy Domain**: RMS energy for amplitude variations
- **Feature Selection**:
  - K-best selection methodology
  - Correlation analysis to remove redundancy
  - Clinical relevance criteria

### 3. Random Forest Architecture
- **Model Configuration**:
  - Number of trees (typically 100-500)
  - Maximum depth constraints for interpretability
  - Minimum samples per leaf for generalization
  - Bootstrap sampling and feature randomness
- **Hyperparameter Optimization**:
  - Grid search or Bayesian optimization approach
  - Cross-validation strategy (stratified k-fold)
  - Performance metrics for selection

### 4. Interpretability Framework

#### 4.1 Global Interpretability
- **Feature Importance Analysis**:
  - Mean Decrease in Impurity (MDI)
  - Permutation importance for robustness
  - SHAP values for unified importance metrics
- **Feature Interaction Detection**:
  - Two-way interaction strengths
  - Partial dependence plots for key features
  - Accumulated local effects (ALE) plots

#### 4.2 Local Interpretability
- **Individual Predictions**:
  - Decision path visualization
  - Local feature contributions
  - Counterfactual explanations
- **Representative Trees**:
  - Selection of most informative trees
  - Simplified decision rules extraction
  - Clinical decision support pathways

#### 4.3 Model Transparency
- **Tree Structure Analysis**:
  - Average tree depth and complexity
  - Common decision patterns across trees
  - Rule extraction and simplification
- **Uncertainty Quantification**:
  - Prediction confidence intervals
  - Out-of-bag (OOB) error estimation
  - Calibration curves and reliability diagrams

### 5. Validation Strategy

#### 5.1 Internal Validation
- **Cross-Validation**:
  - Stratified 5-fold or 10-fold CV
  - Leave-one-patient-out validation
  - Temporal validation on sequential data
- **Performance Metrics**:
  - Classification: AUC-ROC, precision, recall, F1
  - Calibration: Brier score, calibration plots
  - Clinical utility: Decision curve analysis

#### 5.2 External Validation
- **Independent Test Set**:
  - Separate patient cohort
  - Different recording conditions
  - Multi-center validation approach
- **Robustness Testing**:
  - Noise injection experiments
  - Recording quality variations
  - Device/microphone differences

### 6. Clinical Integration

#### 6.1 Decision Support System
- **Risk Stratification**:
  - Low/medium/high risk categories
  - Probability thresholds for referral
  - Integration with existing screening protocols
- **Interpretable Reports**:
  - Key acoustic markers identified
  - Comparison to healthy baselines
  - Actionable clinical insights

#### 6.2 Implementation Considerations
- **Computational Requirements**:
  - Real-time processing capabilities
  - Mobile/edge deployment feasibility
  - Storage and privacy requirements
- **User Interface**:
  - Clinician dashboard design
  - Patient-facing explanations
  - Quality assurance indicators

## Results Documentation

### 1. Model Performance
- **Primary Metrics**:
  - Overall accuracy and balanced accuracy
  - Per-class performance (healthy vs. stages)
  - Confusion matrices with confidence intervals
- **Comparative Analysis**:
  - Benchmark against other ML methods
  - Comparison with clinical standards
  - Incremental value assessment

### 2. Feature Importance Results
- **Top Predictive Features**:
  - Ranked list with importance scores
  - Clinical interpretation of each feature
  - Correlation with disease pathophysiology
- **Feature Stability**:
  - Consistency across CV folds
  - Bootstrap confidence intervals
  - Temporal stability analysis

### 3. Interpretability Insights
- **Decision Patterns**:
  - Common classification rules
  - Stage-specific acoustic signatures
  - Progression markers identification
- **Clinical Correlations**:
  - Alignment with known symptoms
  - Novel biomarker discoveries
  - Mechanistic hypotheses

## Discussion Points

### 1. Clinical Significance
- **Early Detection Impact**:
  - Potential for screening programs
  - Cost-effectiveness analysis
  - Patient accessibility benefits
- **Limitations**:
  - Confounding factors (smoking, age, comorbidities)
  - Generalizability constraints
  - False positive/negative implications

### 2. Technical Contributions
- **Interpretability Advances**:
  - Novel visualization techniques
  - Rule extraction methodologies
  - Uncertainty communication
- **Methodological Innovations**:
  - Feature engineering insights
  - Validation framework design
  - Integration strategies

### 3. Future Directions
- **Research Extensions**:
  - Longitudinal monitoring capabilities
  - Multi-modal integration (imaging, biomarkers)
  - Treatment response prediction
- **Clinical Trials**:
  - Prospective validation studies
  - Implementation pilot programs
  - Outcome assessment protocols

## Supplementary Materials

### 1. Technical Details
- **Code Availability**:
  - GitHub repository structure
  - Reproducibility instructions
  - Docker/environment specifications
- **Data Availability**:
  - De-identified dataset access
  - Data dictionary and metadata
  - Ethics approval documentation

### 2. Additional Analyses
- **Sensitivity Analyses**:
  - Hyperparameter variations
  - Feature subset experiments
  - Missing data handling
- **Subgroup Analyses**:
  - Age and gender stratification
  - Smoking status effects
  - Comorbidity adjustments

### 3. Implementation Resources
- **Clinical Guidelines**:
  - Standard operating procedures
  - Training materials for clinicians
  - Quality control checklists
- **Technical Documentation**:
  - API specifications
  - Integration guides
  - Troubleshooting manual