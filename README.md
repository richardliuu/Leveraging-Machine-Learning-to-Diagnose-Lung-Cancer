# Interpretable Lung Cancer Diagnosis from Voice Biomarkers

## Project Overview

This project presents a novel machine learning approach for lung cancer diagnosis using voice biomarkers, with a unique focus on **interpretable AI through surrogate modeling**. The system combines the high accuracy of Random Forest classification with the transparency of decision trees, providing clinicians with both reliable predictions and understandable explanations.

### Key Innovation: Random Forest with Decision Tree Surrogate Framework

- **Primary Model**: Random Forest Classifier for high-accuracy lung cancer binary classification
- **Surrogate Model**: Decision Tree trained to mimic Random Forest predictions while providing interpretable decision rules

## Clinical Significance

Lung cancer can cause subtle changes in voice patterns due to effects on the respiratory system. This project investigates whether machine learning models can detect such changes using features extracted from speech recordings, offering a **non-invasive, accessible screening approach**.

### Why Voice-Based Screening?
- **Non-invasive**: No physical procedures required
- **Accessible**: Can be performed remotely with basic recording equipment  
- **Early Detection**: Potential to identify subtle vocal changes before clinical symptoms
- **Cost-Effective**: Minimal infrastructure requirements for widespread deployment

## Interpretability Approach

### The Challenge
Traditional deep learning models for medical diagnosis are "black boxes" - highly accurate but difficult for clinicians to understand and trust.

### Solution
We implemented an **interpretable framework** where:

1. **Random Forest Model** (`models/randfor.py`): Trained on voice biomarkers for optimal accuracy using ensemble learning
2. **Decision Tree Surrogate** (`models/decisiontree.py`): Trained to replicate Random Forest predictions using interpretable rules

## Technical Architecture

### Models Implemented

#### 1. Random Forest Classifier
- **Purpose**: Primary classification model for lung cancer binary classification
- **Architecture**: Ensemble of 200 decision trees with balanced class weights
- **Features**: Rigorous cross-validation, patient-grouped splits, hyperparameter optimization
- **Parameters**: Max depth 5, min samples split 12, log loss criterion
- **Interpretability**: SHAP analysis for feature importance across ensemble

#### 2. Decision Tree Surrogate  
- **Purpose**: Interpretable proxy model mimicking Random Forest behavior
- **Training**: Learns to predict Random Forest probability outputs rather than ground truth labels
- **Benefits**: Provides transparent decision rules and feature thresholds
- **Validation**: Fidelity assessment ensures surrogate accuracy with R² scores
- **Architecture**: Max depth 10, max leaf nodes 15, regression-based approach

### Architectural Decision: Why Random Forest?

After comprehensive evaluation of multiple architectures, the **Random Forest Classifier was selected as the primary model** for this lung cancer classification task. This decision was based on several key factors:

#### Performance Advantages
- **Random Forest Benefits**:
  - Robust performance with built-in feature selection and ensemble averaging
  - Excellent handling of class imbalance through balanced class weights
  - Higher resistance to overfitting through bootstrap aggregating
  - Fast model training time 

#### Technical Rationale
- **Feature Nature**: Voice biomarkers are tabular numerical features ideal for tree-based methods
- **Interpretability**: Tree-based models provide natural feature importance and decision paths
- **Stability**: Ensemble approach reduces variance and improves generalization
- **Efficiency**: Fast training and inference suitable for clinical deployment

The surrogate decision tree framework enables clinicians to understand Random Forest decisions through simple, interpretable rules while maintaining the ensemble's superior accuracy.

### Data Processing Pipeline

```
Voice Recordings → Feature Extraction → Model Training → Surrogate Analysis → Clinical Interpretation
```

## Project Structure

```
repository/
├── models/                     # Primary and surrogate models
│   ├── randfor.py              # Main Random Forest classifier with SHAP analysis
│   ├── surrogate.py            # Decision tree surrogate model for Random Forest
│   ├── rf_model.pkl            # Saved Random Forest model
│   └── mlp.py                  # Alternative MLP implementation for comparison
├── class_based_models/         # Legacy neural network implementations
│   ├── lung_cancer_mlp.py      # MLP baseline for comparison
│   └── lung_cancer_cnn.py      # CNN baseline for comparison
├── data/                       # Training datasets   
|   ├── wavfiles/               # Original audio files with lung cancer patients and healthy controls 
│   ├── train_data.csv          # Processed voice features for Random Forest
│   ├── rf_surrogate_data.csv   # Random Forest predictions for surrogate training
├── data_processing/            # Analysis and visualization tools
├── results/                    # Model outputs and visualizations
└── requirements.txt            # Project dependencies
```

## Getting Started

### Prerequisites
- Python 3.8+

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <folder name>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the models**
   
   **Primary Random Forest Model:**
   ```bash
   python models/randfor.py
   ```
   
   **Surrogate Decision Tree:**
   ```bash
   python models/decisiontree.py
   ```
   
   **MLP Baseline (for comparison):**
   ```bash
   python class_based_models/lung_cancer_mlp.py
   ```

## Key Features

### Rigorous Validation
- **Patient-Grouped Cross-Validation**: Prevents data leakage by ensuring no patient appears in both training and testing sets
- **Stratified Sampling**: Maintains class distribution across all splits

### Class Imbalance Handling
- **Stratified Cross-Validation**: Maintains class distribution across all splits
- **Balanced Metrics**: Focus on macro-averaged metrics for fair evaluation

### Interpretability Tools
- **SHAP Analysis**: Feature importance for both Random Forest and surrogate models
- **Decision Tree Visualization**: Transparent decision rules with feature thresholds from surrogate
- **Fidelity Assessment**: Validates surrogate accuracy against Random Forest using R² scores
- **Comparative Analysis**: Validates surrogate fidelity to original Random Forest predictions

## Model Performance

### Evaluation Metrics
- **Accuracy**: Overall classification performance
- **Precision/Recall**: Class-specific performance assessment  
- **F1-Score**: Balanced metric for imbalanced datasets
- **ROC-AUC**: Model discrimination ability
- **Confusion Matrix**: Detailed prediction analysis

### Cross-Validation Strategy
- **4-Fold GroupKFold**: Ensures robust performance estimation
- **Stratified Validation**: Maintains class distribution integrity

## Clinical Interpretability

### Understanding Model Decisions

1. **SHAP Feature Importance**: Identifies which voice biomarkers drive Random Forest predictions
2. **Decision Tree Rules**: Provides explicit thresholds and decision pathways from surrogate model
3. **Surrogate Validation**: Ensures interpretable model accurately represents Random Forest behavior through fidelity metrics
4. **Clinical Correlation**: Maps AI insights to medical knowledge and expectations

## Limitations and Disclaimers

### Research Purpose Only
This work is intended for **research and prototyping purposes only**. It is not approved for clinical use and should not be used for actual medical diagnosis without proper validation and regulatory approval.

### Technical Limitations
1. **Dataset Size**: Limited to available voice recordings from single institution
2. **Surrogate Fidelity**: Interpretable model may not capture all Random Forest ensemble complexity  
3. **Generalizability**: Performance may vary across different populations and recording conditions
4. **Feature Engineering**: Voice biomarker extraction may miss relevant patterns
5. **Bias**: The dataset provides bias to the model, as only 2 classes need to be identified. In deployment, that may not be the case. 

### Clinical Considerations
- Voice changes can result from many conditions beyond lung cancer
- Model predictions should supplement, not replace, clinical judgment
- Requires validation on larger, more diverse patient populations
- Need for longitudinal studies to assess temporal stability

### Future Work
- Expanding dataset size to provide model with a larger demographic 
- Improve model architecture to improve model accuracy
- Implementing more explainability tools to enhance human interpretability of the model

## Data Sources and Ethics

### Data Acquisition
Permission for the use of audio data and patient information was granted by **Dr. Haydar Ankishan**, Associate Professor at Stem Cell Institute of Ankara University, Turkey. The dataset contains:

- Voice recordings from individuals with various stages of lung cancer
- Healthy control recordings for comparison
- Associated medical history and staging information
- Proper anonymization and privacy protection measures

## Contributing

This research project welcomes contributions and collaboration in the following areas:

1. **Model Improvements**: Enhanced architectures or training strategies
2. **Interpretability Methods**: Novel approaches to model explanation
3. **Feature Engineering**: Advanced voice biomarker extraction techniques
4. **Validation Studies**: Extended evaluation on diverse datasets
5. **Clinical Integration**: Tools for healthcare professional adoption

## Contact

For questions about this research or potential collaborations:

- **Author**: Richard Liu
- **Institution**: Milliken Mills High School
- **Email**: richardliu200127@gmail.com

---

**Disclaimer**: This software is for research purposes only and is not intended for clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice. For using this repository, please ensure compliance with applicable medical data regulations and institutional policies when using or adapting this code.