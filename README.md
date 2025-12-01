# Interpretable Lung Cancer Diagnosis from Voice Biomarkers

## Project Overview

This project presents a novel machine learning approach for lung cancer diagnosis using voice biomarkers, with a unique focus on **interpretable AI through surrogate modeling**. The system combines the high accuracy of Random Forest classification with the transparency of decision tree regression, providing clinicians with both reliable predictions and understandable explanations.

### Key Innovation: Random Forest with Decision Tree Surrogate Framework

- **Primary Model**: Random Forest Classifier for binary classification of lung cancer using ensemble learning
- **Surrogate Model**: Decision Tree Regressor trained to mimic Random Forest probability predictions, providing interpretable decision rules and transparent reasoning

This two-stage approach enables deployment in clinical settings where model explainability is crucial for medical decision-making.

### Why Voice-Based Screening?
- **Non-invasive**: No physical procedures or imaging required
- **Accessible**: Can be performed remotely with basic recording equipment  
- **Cost-Effective**: Minimal infrastructure requirements for widespread deployment
- **Early Detection**: Potential for identifying lung cancer indicators through voice biomarker patterns

## Our Approach

### The Challenge
Ensemble models like Random Forest provide excellent predictive performance but lack transparency in their decision-making process. Medical professionals need to understand *why* a model makes specific predictions to trust and effectively use AI-assisted diagnosis.

### Solution: Two-Stage Pipeline
We implemented an **interpretable framework** with distinct training stages:

#### Stage 1: Random Forest Training (`_model/`)
- Trains Random Forest Classifier on voice biomarker features
- Uses patient-grouped cross-validation to prevent data leakage
- Generates probability predictions for each sample
- Outputs predictions with feature data for surrogate training

#### Stage 2: Surrogate Model Training (`_surrogate_model/`)
- Trains Decision Tree Regressor to predict Random Forest probabilities
- Uses regression approach to match continuous probability outputs
- Provides interpretable decision rules and thresholds
- Validates surrogate fidelity with R² metrics

## Technical Architecture

### Stage 1: Random Forest Classifier (`_model/`)

**Purpose**: High-accuracy primary classification model for lung cancer diagnosis

**Architecture**:
- Ensemble of 200 decision trees with balanced class weights
- Log loss criterion for probabilistic predictions
- Bootstrap aggregating for improved generalization
- Out-of-bag (OOB) scoring for internal validation

**Key Parameters**:
```python
{
    'n_estimators': 200,
    'max_depth': 5,
    'min_samples_split': 25,
    'min_samples_leaf': 10,
    'max_features': 0.3,
    'class_weight': 'balanced',
    'criterion': 'log_loss'
}
```

**Validation Strategy**:
- 5-fold patient-grouped cross-validation
- Stratified sampling to maintain class distribution
- Prevents data leakage by ensuring no patient appears in both train and test sets

**Interpretability Tools**:
- SHAP (SHapley Additive exPlanations) analysis for feature importance
- Feature importance rankings across the ensemble
- Tree-based decision path analysis

**Module Structure**:
```
_model/modules/
├── config.py           # Hyperparameters and paths
├── cross_validation.py # CV implementation with patient grouping
├── data_methods.py     # Data loading and preprocessing
├── model.py            # Random Forest training and evaluation
├── evaluation.py       # Metrics calculation and visualization
└── shap_analysis.py    # SHAP-based interpretability
```

### Stage 2: Decision Tree Surrogate (`_surrogate_model/`)

**Purpose**: Interpretable proxy model that mimics Random Forest behavior

**Training Approach**:
- **Target Variable**: Random Forest probability predictions (not ground truth labels)
- **Model Type**: Decision Tree Regressor (continuous output matching probabilities)
- **Advantage**: Learns the Random Forest's decision boundaries while maintaining full transparency

**Architecture**:
```python
{
    'max_depth': 4,
    'min_samples_leaf': 20,
    'random_state': 42
}
```

**Key Features**:
- Transparent decision rules with explicit feature thresholds
- Visual tree representation for clinical review
- R² score validation to ensure fidelity to Random Forest
- UMAP visualization for probability space analysis

**Validation Metrics**:
- **Fidelity (R²)**: Measures how well surrogate replicates Random Forest predictions
- **MSE/MAE**: Quantifies prediction error between models
- **Decision Path Analysis**: Verifies logical consistency of decision rules

**Module Structure**:
```
_surrogate_model/modules/
├── config.py           # Surrogate-specific parameters
├── cross_validation.py # Surrogate model validation
├── data_methods.py     # RF prediction data loading
├── model.py            # Decision Tree Regressor
├── evaluation.py       # Fidelity metrics
└── umap_projection.py  # Dimensionality reduction visualization
```

### Architectural Decision: Why Random Forest + Surrogate?

#### Why Random Forest for Primary Model?

**Performance Advantages**:
- Robust performance with built-in feature selection and ensemble averaging
- Excellent handling of class imbalance through balanced class weights
- Higher resistance to overfitting through bootstrap aggregating
- Fast training time compared to deep learning approaches
- Reliable probability estimates for medical decision support

**Technical Rationale**:
- **Feature Nature**: Voice biomarkers are tabular numerical features ideal for tree-based methods
- **Stability**: Ensemble approach reduces variance and improves generalization
- **Efficiency**: Fast training and inference suitable for clinical deployment
- **No Feature Scaling Required**: Trees are invariant to monotonic transformations

#### Why Decision Tree Surrogate?

**Interpretability Benefits**:
- Provides explicit, human-readable decision rules
- Shows exact feature thresholds used for classification
- Visual tree structure for clinical review and validation
- Enables trust and verification by medical professionals

**Technical Advantages**:
- Learns Random Forest's decision boundaries directly from probability outputs
- Simpler structure while maintaining high fidelity to ensemble predictions
- Regression-based approach captures nuanced probability distributions
- Supports model debugging and failure mode analysis

**Clinical Value**:
The surrogate framework enables clinicians to understand *why* the AI makes specific predictions through transparent rules, while still benefiting from the Random Forest's superior accuracy and robustness.

## Data Processing Pipeline

### Pipeline Overview

INCLUDE AN IMAGE

```
Raw Audio (WAV files) 
    ↓
[Feature Extraction] → MFCCs, spectral features, temporal features
    ↓
train_data.csv (Voice biomarkers + labels)
    ↓
[Random Forest Training] → Cross-validation, SHAP analysis
    ↓
rf_predictions.csv (Features + RF probabilities)
    ↓
[Surrogate Training] → Decision Tree Regressor
    ↓
Interpretable Model + Visualizations
```

### Data Flow

1. **Audio Processing**: Extract voice biomarkers (MFCCs, spectral features) from WAV files
2. **RF Training**: Train Random Forest on features with patient-grouped CV
3. **Prediction Generation**: Create RF probability predictions for all samples
4. **Surrogate Training**: Train Decision Tree to mimic RF probability outputs
5. **Validation**: Assess surrogate fidelity and generate visualizations

## Project Structure

```
project/
├── _model/                         # Stage 1: Random Forest Training
│   ├── main.py                     # RF training pipeline orchestrator
│   └── modules/
│       ├── config.py               # Hyperparameters and paths
│       ├── cross_validation.py     # Patient-grouped CV implementation
│       ├── data_methods.py         # Data loading and preprocessing
│       ├── model.py                # RandomForest training and evaluation
│       ├── evaluation.py           # Metrics calculation and visualization
│       └── shap_analysis.py        # SHAP-based interpretability
│
├── _surrogate_model/               # Stage 2: Surrogate Training
│   ├── main.py                     # Surrogate training pipeline
│   └── modules/
│       ├── config.py               # Surrogate-specific parameters
│       ├── cross_validation.py     # Surrogate validation
│       ├── data_methods.py         # RF prediction data loading
│       ├── model.py                # Decision Tree Regressor
│       ├── evaluation.py           # Fidelity metrics and plots
│       └── umap_projection.py      # Dimensionality reduction viz
│
├── class_based_models/             # Reference implementations
│   ├── random_forest_classifier.py # Original monolithic RF implementation
│   └── surrogate.py                # Original surrogate implementation
│
├── data/                           # Datasets
│   ├── wavfiles/                   # Original audio recordings
│   │   ├── healthy/                # Healthy control recordings
│   │   └── unhealthy/              # Lung cancer patient recordings
│   ├── train_data.csv              # Processed voice features for RF
│   ├── rf_predictions.csv          # RF outputs for surrogate training
│   └── rf_surrogate_data.csv       # Alternative RF prediction format
│
├── data_processing/                # Data preparation scripts
│   ├── dataprocessing.py           # Main preprocessing pipeline
│   ├── feature_extraction.py       # Voice biomarker extraction
│   └── confusion_matrix.py         # Evaluation utilities
│
├── results/                        # Training outputs
│   ├── models/                     # Saved model files (.pkl)
│   ├── metrics/                    # Performance metrics (.csv)
│   └── plots/                      # Visualizations (.png)
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning repository)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/richardliuu/INSERT LINK
cd project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Python Packages
```
numpy
pandas
scikit-learn
matplotlib
seaborn
shap
umap-learn
joblib
```

### Usage

#### Stage 1: Train Random Forest Model

Navigate to the Random Forest training directory and run the pipeline:

```bash
cd _model
python main.py
```

**Outputs**:
- `results/models/random_forest_model.pkl` - Trained Random Forest model
- `results/metrics/cv_summary.csv` - Cross-validation performance metrics
- `results/metrics/feature_importance.csv` - Feature importance rankings
- `results/plots/feature_importance.png` - Feature importance visualization
- `results/plots/cv_results.png` - Cross-validation performance plots

**Configuration**:
Edit `_model/modules/config.py` to modify:
- Hyperparameters (n_estimators, max_depth, etc.)
- Cross-validation settings (n_folds, stratified vs grouped)
- File paths and output directories

#### Stage 2: Train Surrogate Model

After training the Random Forest and generating predictions:

```bash
cd _surrogate_model
python main.py
```

**Inputs**:
- `data/rf_predictions.csv` - Random Forest probability predictions

**Outputs**:
- `results/models/surrogate_model.pkl` - Trained surrogate decision tree
- `results/metrics/fidelity_metrics.csv` - R², MSE, MAE between RF and surrogate
- `results/plots/tree_visualization.png` - Decision tree structure
- `results/plots/umap_projection.png` - UMAP visualization of probability space

**Configuration**:
Edit `_surrogate_model/modules/config.py` to modify:
- Tree parameters (max_depth, min_samples_leaf)
- Probability thresholds for clustering
- UMAP projection parameters

### Data Format

**Input Data (train_data.csv)**:
```csv
patient_id,chunk,cancer_stage,mfcc_1,mfcc_2,...,mfcc_20,spectral_centroid,...
P001,1,1,0.234,-0.456,...,0.123,1234.56,...
```

**RF Predictions (rf_predictions.csv)**:
```csv
true_label,predicted_label,prob_class_0,prob_class_1,mfcc_1,mfcc_2,...
0,0,0.85,0.15,0.234,-0.456,...
1,1,0.12,0.88,-0.123,0.789,...
```

## Key Features

### Rigorous Validation Framework
- **Patient-Grouped Cross-Validation**: Prevents data leakage by ensuring no patient's samples appear in both training and testing sets within the same fold
- **Stratified Sampling**: Maintains class distribution across all CV splits for balanced evaluation
- **5-Fold Validation**: Robust performance estimation across multiple train-test partitions
- **Out-of-Bag (OOB) Scoring**: Internal validation during Random Forest training

### Interpretability Tools
- **SHAP Analysis**: 
  - Global feature importance across the Random Forest ensemble
  - Individual prediction explanations showing feature contributions
  - Summary plots for understanding model behavior patterns
  
- **Decision Tree Visualization**: 
  - Transparent decision rules with explicit feature thresholds
  - Visual tree structure for clinical review
  - Path tracing for individual predictions
  
- **Surrogate Fidelity Metrics**:
  - R² score quantifying how well surrogate mimics Random Forest
  - MSE/MAE for prediction error analysis
  - UMAP projections for visualizing probability space

### Modular Architecture
- **Separation of Concerns**: Each module handles specific functionality (data, model, evaluation)
- **Reusable Components**: Modules can be imported and used independently
- **Easy Configuration**: Centralized config files for all hyperparameters
- **Extensible Design**: Simple to add new models or evaluation metrics

## Model Performance

### Random Forest Evaluation Metrics

**Classification Metrics**:
- **Accuracy**: Overall correct prediction rate across both classes
- **Precision**: Proportion of positive predictions that are actually positive (reduces false alarms)
- **Recall/Sensitivity**: Proportion of actual positives correctly identified (critical for medical diagnosis)
- **F1-Score**: Harmonic mean of precision and recall, balanced metric for imbalanced datasets
- **ROC-AUC**: Area under ROC curve, measures model's discrimination ability across all thresholds

**Analysis Tools**:
- **Confusion Matrix**: Detailed breakdown of true/false positives/negatives
- **ROC Curve**: Trade-off between sensitivity and specificity
- **Precision-Recall Curve**: Performance at different probability thresholds
- **Feature Importance Rankings**: Top contributing features to predictions

### Surrogate Model Evaluation

**Fidelity Metrics** (How well surrogate mimics Random Forest):
- **R² Score**: Coefficient of determination (target: >0.80 for good fidelity)
- **Mean Squared Error (MSE)**: Average squared difference in probability predictions
- **Mean Absolute Error (MAE)**: Average absolute difference in probabilities

**Visualization**:
- **Tree Structure Diagram**: Complete decision path visualization
- **UMAP Projection**: 2D visualization of feature space and probability clusters
- **Probability Distribution Comparison**: RF vs Surrogate output distributions

### Cross-Validation Strategy
- **5-Fold Patient-Grouped CV**: Ensures no patient leakage between train and test sets
- **Stratified Splits**: Maintains class balance in each fold
- **Per-Fold Reporting**: Detailed metrics for each CV iteration
- **Aggregated Statistics**: Mean and standard deviation across folds

## Clinical Interpretability

### Understanding Model Decisions

Our two-stage approach provides multiple layers of interpretability:

#### 1. SHAP Analysis (Random Forest Level)
**Purpose**: Understand which voice biomarkers drive ensemble predictions

**Capabilities**:
- **Global Feature Importance**: Ranking of features by average impact on predictions
- **Individual Explanations**: For each prediction, see which features contributed positively or negatively
- **Feature Interactions**: Identify how combinations of features influence outcomes
- **Summary Plots**: Visualize feature impact distributions across the dataset

**Clinical Value**: Helps validate if the model focuses on medically relevant voice characteristics

#### 2. Decision Tree Visualization (Surrogate Level)
**Purpose**: Provide explicit, human-readable decision rules

**Capabilities**:
- **Transparent Thresholds**: Exact feature values used for splitting decisions (e.g., "If MFCC_3 > 0.45, then...")
- **Decision Paths**: Complete reasoning chain from root to leaf for any prediction
- **Visual Tree Structure**: Graphical representation of the entire decision logic
- **Rule Extraction**: Convert tree structure into IF-THEN statements

**Clinical Value**: Enables medical professionals to:
- Verify decision logic aligns with clinical knowledge
- Identify potential biases or spurious correlations
- Trust predictions through understanding the reasoning
- Explain decisions to patients in simple terms

#### 3. Surrogate Fidelity Validation
**Purpose**: Ensure the interpretable model accurately represents the Random Forest

**Metrics**:
- **R² Score**: Measures how well surrogate predictions match RF probabilities (target: >0.80)
- **Prediction Agreement**: Percentage of cases where surrogate and RF agree on classification
- **Error Analysis**: Identify cases where surrogate deviates significantly from RF

**Clinical Value**: Provides confidence that interpretations from the surrogate genuinely reflect the Random Forest's reasoning

#### 4. UMAP Probability Space Visualization
**Purpose**: Visualize how samples cluster based on RF probability predictions

**Capabilities**:
- 2D projection of high-dimensional feature space
- Color-coded by RF probability predictions
- Identifies confident vs uncertain prediction regions
- Reveals data structure and potential outliers

**Clinical Value**: Helps understand model confidence levels and identify edge cases requiring additional scrutiny

## Limitations and Disclaimers

### Research Purpose Only
This work is intended for **research and prototyping purposes only**. It should NOT be:
- **used for clinical use**
- **used as a substitute for professional medical diagnosis**
- **used for real-world deployment**

### Technical Limitations

#### Data Limitations
1. **Limited Dataset Size**: Small sample size may not capture full population variability
2. **Single Institution Data**: Recordings from one source may introduce site-specific biases
3. **Binary Classification Only**: Model only distinguishes healthy vs unhealthy, not cancer stages or types
4. **Demographic Bias**: May not generalize to populations with different:
   - Age distributions
   - Geographic/ethnic backgrounds
   - Language/accent variations

#### Model Limitations
1. **Surrogate Fidelity Trade-off**: 
   - Simpler surrogate tree provides interpretability but may not capture all ensemble complexity
   - High fidelity (R² > 0.90) is ideal but not always achievable
   - Some nuanced predictions may be oversimplified

2. **Feature Engineering Dependency**:
   - Voice biomarker extraction assumes current feature set is optimal
   - May miss relevant acoustic patterns not captured by MFCCs and spectral features
   - Feature selection process may introduce bias

3. **Recording Quality Sensitivity**:
   - Model performance may degrade with poor audio quality
   - Background noise, microphone variability can affect predictions
   - Requires standardized recording protocols for consistency

4. **Temporal Stability Unknown**:
   - Voice characteristics may change over time (disease progression, treatment effects)
   - Model may need retraining or recalibration for longitudinal use

### Clinical Considerations

#### Medical Context
- **Non-specific Indicator**: Voice changes can result from many conditions:
  - Other respiratory diseases (COPD, pneumonia)
  - Vocal cord disorders
  - Neurological conditions
  - Age-related changes
  - Smoking effects (independent of cancer)

- **Screening Tool Only**: Should be used as part of comprehensive diagnostic workup, not standalone diagnosis

- **False Positives/Negatives**: 
  - False positives cause unnecessary anxiety and follow-up procedures
  - False negatives delay potentially life-saving treatment

#### Deployment Considerations
- **Multi-class Reality**: Real-world deployment needs to handle:
  - Various cancer stages and types
  - Multiple confounding conditions
  - Healthy individuals with voice disorders
  - Uncertain/borderline cases

- **Clinical Workflow Integration**: Requires:
  - Clear guidelines on when to use the tool
  - Protocols for handling uncertain predictions
  - Integration with existing diagnostic pathways
  - Training for clinical staff

### Future Work

#### Immediate Priorities
1. **Dataset Expansion**:
   - Increase sample size (target: >1000 patients)
   - Multi-site data collection for generalizability
   - Diverse demographic representation
   - Include multiple cancer stages and types

2. **Model Improvements**:
   - Experiment with deep learning architectures (CNNs on spectrograms)
   - Ensemble methods combining multiple model types
   - Multi-task learning for stage classification
   - Uncertainty quantification for prediction confidence

3. **Enhanced Interpretability**:
   - Attention mechanisms for deep learning models
   - Counterfactual explanations ("What would need to change for different prediction?")
   - Interactive visualization tools for clinicians
   - Natural language explanations of predictions

#### Long-term Goals
1. **Clinical Validation**:
   - Prospective studies in real clinical settings
   - Comparison with existing screening methods
   - Health economic analysis (cost-effectiveness)
   - Patient outcomes and satisfaction studies

2. **Deployment Infrastructure**:
   - Web/mobile application for easy access
   - Real-time prediction capabilities
   - Integration with electronic health records (EHR)
   - Continuous monitoring and model updating

3. **Regulatory Pathway**:
   - FDA submission for clinical decision support tool
   - Clinical trial design and execution
   - Post-market surveillance plan

## Data Sources and Ethics

### Data Acquisition
Permission for the use of audio data and patient information was granted by **Dr. Haydar Ankishan**, Associate Professor at Stem Cell Institute of Ankara University, Turkey. The dataset contains:

- Voice recordings from individuals with various stages of lung cancer
- Healthy control recordings for comparison
- Associated medical history and staging information
- Proper anonymization and privacy protection measures

## Contributing

This research project welcomes contributions and collaboration in the following areas:

### Areas for Contribution

1. **Model Improvements**:
   - Implement alternative ensemble methods (XGBoost, LightGBM, CatBoost)
   - Experiment with deep learning architectures
   - Develop hybrid models combining multiple approaches
   - Optimize hyperparameters using advanced tuning methods

2. **Interpretability Methods**:
   - Implement additional explainability techniques (LIME, Integrated Gradients)
   - Develop interactive visualization tools
   - Create natural language explanation generation
   - Build counterfactual explanation systems

3. **Feature Engineering**:
   - Extract additional voice biomarkers (prosody, formants, jitter, shimmer)
   - Implement wavelet-based features
   - Develop deep feature learning from raw audio
   - Create domain-specific feature selection methods

4. **Validation Studies**:
   - Extend evaluation to external datasets
   - Conduct cross-institution validation
   - Perform sensitivity analysis on hyperparameters
   - Analyze subgroup performance (age, gender, ethnicity)

5. **Clinical Integration**:
   - Build user-friendly interfaces for clinicians
   - Develop EHR integration modules
   - Create clinical decision support workflows
   - Design patient-facing explanation tools

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** with clear, documented code
4. **Add tests** for new functionality
5. **Update documentation** (README, docstrings, comments)
6. **Commit changes** (`git commit -m 'Add amazing feature'`)
7. **Push to branch** (`git push origin feature/amazing-feature`)
8. **Open a Pull Request** with detailed description

### Contribution Guidelines

- Follow PEP 8 style guide for Python code
- Include docstrings for all functions and classes
- Add unit tests for new features
- Update requirements.txt if adding dependencies
- Provide clear commit messages
- Reference any related issues in PR description

## Acknowledgments

### Data Source
Special thanks to **Dr. Haydar Ankishan**, Associate Professor at the Stem Cell Institute of Ankara University, Turkey, for providing access to the lung cancer voice biomarker dataset and granting permission for its use in this research.

### Technical References
This project builds upon research in:
- Voice biomarker analysis for disease detection
- Interpretable machine learning in healthcare
- Surrogate modeling for ensemble interpretability
- SHAP-based explainable AI

## Contact

For questions about this research or potential collaborations:

- **Author**: Richard Liu
- **Institution**: Milliken Mills High School
- **Email**: richardliu200127@gmail.com
- **GitHub**: [@richardliuu](https://github.com/richardliuu)

## Disclaimer

**RESEARCH USE ONLY**: This software is for research and educational purposes only and is not intended for clinical diagnosis or treatment decisions. 

**NO WARRANTY**: This software is provided "as is" without warranty of any kind, express or implied. The authors assume no responsibility for errors, omissions, or consequences from use.

**MEDICAL ADVICE**: Always consult qualified healthcare professionals for medical advice. This tool is not a substitute for professional medical diagnosis.

**ETHICAL USE**: This tool should be used ethically and responsibly, with appropriate informed consent and data protection measures in place.

---

*Last Updated: November 2025*