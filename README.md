# Interpretable Lung Cancer Diagnosis from Voice Biomarkers

## 🎯 Project Overview

This project presents a novel machine learning approach for lung cancer diagnosis using voice biomarkers, with a unique focus on **interpretable AI through surrogate modeling**. The system combines the high accuracy of deep neural networks with the transparency of decision trees, providing clinicians with both reliable predictions and understandable explanations.

### Key Innovation: Dual-Model Interpretability Framework

- **Primary Model**: Multi-Layer Perceptron (MLP) for high-accuracy lung cancer stage classification
- **Surrogate Model**: Decision Tree trained to mimic MLP predictions while providing interpretable decision rules
- **SHAP Analysis**: Feature importance comparison between both models for validation and clinical insight

## 🏥 Clinical Significance

Lung cancer can cause subtle changes in voice patterns due to effects on the respiratory system. This project investigates whether machine learning models can detect such changes using features extracted from speech recordings, offering a **non-invasive, accessible screening approach**.

### Why Voice-Based Screening?
- **Non-invasive**: No physical procedures required
- **Accessible**: Can be performed remotely with basic recording equipment  
- **Early Detection**: Potential to identify subtle vocal changes before clinical symptoms
- **Cost-Effective**: Minimal infrastructure requirements for widespread deployment

## 🧠 Novel Interpretability Approach

### The Challenge
Traditional deep learning models for medical diagnosis are "black boxes" - highly accurate but difficult for clinicians to understand and trust.

### Our Solution
We implement a **surrogate model framework** where:

1. **MLP Model** (`class_based_models/lung_cancer_mlp.py`): Trained on voice biomarkers for optimal accuracy
2. **Decision Tree Surrogate** (`models/decisiontree.py`): Trained to replicate MLP predictions using interpretable rules
3. **Comparative SHAP Analysis**: Feature importance validation across both models
4. **Clinical Translation**: Provides explainable decision pathways for medical professionals

## 📊 Technical Architecture

### Models Implemented

#### 1. Multi-Layer Perceptron (MLP)
- **Purpose**: Primary classification model for lung cancer staging
- **Architecture**: Deep neural network with batch normalization and dropout
- **Features**: Rigorous cross-validation, SMOTEENN resampling, early stopping
- **Interpretability**: SHAP analysis for feature importance

#### 2. Decision Tree Surrogate  
- **Purpose**: Interpretable proxy model mimicking MLP behavior
- **Training**: Learns to predict MLP outputs rather than ground truth labels
- **Benefits**: Provides transparent decision rules and feature thresholds
- **Validation**: Fidelity assessment ensures surrogate accuracy

#### 3. Convolutional Neural Network (CNN)
- **Purpose**: Alternative approach using MFCC spectrograms
- **Input**: Mel-frequency cepstral coefficients from voice recordings
- **Architecture**: Convolutional layers for spatial pattern recognition

### Data Processing Pipeline

```
Voice Recordings → Feature Extraction → Model Training → Surrogate Analysis → Clinical Interpretation
```

## 🗂️ Project Structure

```
science2/
├── class_based_models/          # Primary models
│   ├── lung_cancer_mlp.py       # Main MLP with SHAP analysis
│   └── lung_cancer_cnn.py       # CNN implementation
├── models/                      # Surrogate and experimental models
│   ├── decisiontree.py          # Decision tree surrogate model
│   ├── mlp.py                   # Alternative MLP implementation
│   └── cnn.py                   # Alternative CNN implementation
├── data/                        # Training datasets   
|   ├── wavfiles/                # Original audio files with lung cancer patients and healthy controls 
│   ├── binary_features_log.csv  # Processed voice features
│   ├── surrogate_data.csv       # MLP predictions for surrogate training
│   └── binary_mfccs.npy         # MFCC features for CNN
├── data_processing/             # Analysis and visualization tools
├── results/                     # Model outputs and visualizations
└── requirements.txt             # Project dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for neural network training and running SHAP)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd science2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the models**
   
   **Primary MLP Model:**
   ```bash
   python class_based_models/lung_cancer_mlp.py
   ```
   
   **Surrogate Decision Tree:**
   ```bash
   python models/decisiontree.py
   ```
   
   **CNN Model:**
   ```bash
   python class_based_models/lung_cancer_cnn.py
   ```

## 📈 Key Features

### Rigorous Validation
- **Patient-Grouped Cross-Validation**: Prevents data leakage by ensuring no patient appears in both training and testing sets
- **Data Integrity Checks**: Comprehensive validation for duplicates and label consistency
- **Stratified Sampling**: Maintains class distribution across all splits

### Class Imbalance Handling
- **SMOTEENN Resampling**: Combines oversampling and undersampling for balanced training
- **Weighted Loss Functions**: Accounts for class imbalance in neural network training
- **Balanced Metrics**: Focus on macro-averaged metrics for fair evaluation

### Interpretability Tools
- **SHAP Analysis**: Feature importance for both MLP and surrogate models
- **Decision Tree Visualization**: Transparent decision rules with feature thresholds
- **Comparative Analysis**: Validates surrogate fidelity to original MLP

## 📊 Model Performance

### Evaluation Metrics
- **Accuracy**: Overall classification performance
- **Precision/Recall**: Class-specific performance assessment  
- **F1-Score**: Balanced metric for imbalanced datasets
- **ROC-AUC**: Model discrimination ability
- **Confusion Matrix**: Detailed prediction analysis

### Cross-Validation Strategy
- **4-Fold GroupKFold**: Ensures robust performance estimation
- **Patient-Level Splitting**: Prevents optimistic bias from patient overlap
- **Stratified Validation**: Maintains class distribution integrity

## 🔬 Clinical Interpretability

### Understanding Model Decisions

1. **SHAP Feature Importance**: Identifies which voice biomarkers drive predictions
2. **Decision Tree Rules**: Provides explicit thresholds and decision pathways  
3. **Surrogate Validation**: Ensures interpretable model accurately represents MLP behavior
4. **Clinical Correlation**: Maps AI insights to medical knowledge and expectations

### Benefits for Healthcare Professionals

- **Transparency**: Clear understanding of why predictions were made
- **Validation**: Ability to verify AI decisions against medical expertise
- **Trust**: Increased confidence in AI-assisted diagnosis
- **Education**: Insights into voice-disease relationships for research

## ⚠️ Limitations and Disclaimers

### Research Purpose Only
This work is intended for **research and prototyping purposes only**. It is not approved for clinical use and should not be used for actual medical diagnosis without proper validation and regulatory approval.

### Technical Limitations
1. **Dataset Size**: Limited to available voice recordings from single institution
2. **Surrogate Fidelity**: Interpretable model may not capture all MLP complexity  
3. **Generalizability**: Performance may vary across different populations and recording conditions
4. **Feature Engineering**: Voice biomarker extraction may miss relevant patterns

### Clinical Considerations
- Voice changes can result from many conditions beyond lung cancer
- Model predictions should supplement, not replace, clinical judgment
- Requires validation on larger, more diverse patient populations
- Need for longitudinal studies to assess temporal stability

## 📊 Data Sources and Ethics

### Data Acquisition
Permission for the use of audio data and patient information was granted by **Dr. Haydar Ankishan**, Associate Professor at Stem Cell Institute of Ankara University, Turkey. The dataset contains:

- Voice recordings from individuals with various stages of lung cancer
- Healthy control recordings for comparison
- Associated medical history and staging information
- Proper anonymization and privacy protection measures

## 🔧 Technical Dependencies

### Core Libraries
- **TensorFlow/Keras**: Neural network implementation
- **scikit-learn**: Machine learning utilities and evaluation
- **pandas/numpy**: Data manipulation and numerical computing
- **librosa**: Audio processing and feature extraction
- **SHAP**: Model interpretability and feature analysis
- **imbalanced-learn**: Class imbalance handling
- **matplotlib/seaborn**: Visualization and plotting

### Audio Processing
- **librosa**: MFCC extraction and audio analysis
- **soundfile**: Audio file I/O operations
- **noisereduce**: Audio preprocessing and noise reduction

## 🤝 Contributing

This research project welcomes contributions and collaboration in the following areas:

1. **Model Improvements**: Enhanced architectures or training strategies
2. **Interpretability Methods**: Novel approaches to model explanation
3. **Feature Engineering**: Advanced voice biomarker extraction techniques
4. **Validation Studies**: Extended evaluation on diverse datasets
5. **Clinical Integration**: Tools for healthcare professional adoption

## 📞 Contact

For questions about this research or potential collaborations:

- **Author**: Richard Liu
- **Institution**: Milliken Mills High School
- **Email**: richardliu200127@gmail.com

---

**Disclaimer**: This software is for research purposes only and is not intended for clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice. For using this repository, please ensure compliance with applicable medical data regulations and institutional policies when using or adapting this code.