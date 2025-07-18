# Summary 



# Introduction 
This repository contains the code for a proof-of-concept machine learning model that detects potential signs of lung cancer using audio recordings of human speech. It explores the relationship between vocal biomakers and health conditions like lung cancer, offering a non-invasive approach to voice-based screening. 

# Project Overview 
Lung cancer can cause subtle changes in voice due to effects on the respiratory system. This project investigates whether machine learning models can detect such changes using features extracted from speech. 

This work is intended for research and prototyping purposes only. 

# Models Used 
MLP (Multilayer Perceptron)
- Tabular features for classification

CNN (Convolutional Neural Network) 
- For MFCC-based (Mel-frequency cepstral coefficients) spectrogram inputs

# Methodology 


# Explainability
1. A surrogate model attempts to explain the predictions of the MLP


# Getting Started 
1. Clone the repository
2. Download dependencies
- pip isntall -r requirements.txt

3. Run the models' files 

# Data Acquisition
Permission for the use of audio data and patient information used in this project was granted by Dr. Haydar Ankishan, Associate Professor at Stem Cell Institute of Ankara University, Turkey. The data contains voice recordings of individuals diagnosed with various stages of lung cancer as well as healthy controls and their medical history.