import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shap
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text, export_graphviz
from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
#from class_based_models.lung_cancer_mlp import LungCancerMLP
from collections import Counter

"""
Potential Features:
    - Compare SHAP values for the MLP and surrogate model
"""

"""
Setting the seed to lock the training environment for reproducable results 
"""

import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

"""
The DataHandling Class handles and transforms the MLP performance data into training data for this surrogate model. 

The instantiated variables for training and validation are transformed through the class functions using LabelEncoder and StandardScaler.
Data Splits are handled in validation_split() 
"""

class DataHandling:
    def __init__(self):
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()

        self.reports = []
        self.conf_matrices = []
        self.details = []
        self.history = []
        self.predictions = []

        self.feature_cols = None
        self.duplicates = None
        self.patient_labels = None
        self.groups = None
        self.inconsistent_patients = None

        self.data = None
        self.X = None
        self.y = None
        self.X_val = None
        self.y_val = None
        self.num_classes = None

    """
    Training samples are handled such that the same patient does not appear in the training or validation set. 
    This is to prevent data leakage, causing the model's performance to be skewed.
    """

    def load_data(self):
        self.data = pd.read_csv("data/surrogate_data.csv")
        self.X = self.data.drop(columns=["segment", "true_label", "patient_id", "predicted_label"])
        self.y = self.data['predicted_label']
        self.feature_cols = self.X.columns.tolist()
        
        # Data Leakage Prevention
        self.groups = self.data['patient_id']
        self.duplicates = self.data.duplicated(subset=self.feature_cols)
        self.patient_labels = self.data.groupby('patient_id')['true_label'].nunique()
        self.inconsistent_patients = self.patient_labels[self.patient_labels > 1]

    def split(self, X, y, data, train_idx, test_idx):
        self.X_train = X.iloc[train_idx]
        self.y_train = y.iloc[train_idx]
        self.X_test = X.iloc[test_idx]
        self.y_test = y.iloc[test_idx]

        self.train_patients = set(data.iloc[train_idx]['patient_id'])
        self.test_patients = set(data.iloc[test_idx]['patient_id'])

    def transform(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        self.y_train = self.encoder.fit_transform(self.y_train)
        self.y_test = self.encoder.transform(self.y_test)
        self.num_classes = len(self.encoder.classes_)
    
    def validation_split(self):

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, 
            self.y_train, 
            test_size=0.15, 
            stratify=self.y_train, 
            random_state=SEED
        )

        self.X_train, self.y_train = SMOTEENN(sampling_strategy='auto', random_state=SEED).fit_resample(self.X_train, self.y_train)

""" 
The class is to check the fidelity of the surrogate model and the MLP 

This is based on the accuracy of the surrogate model, as its predictions are to replicate the predicted labels of the MLP

NOTE R2 Value can be considered to also check for fidelity of the surrogate and MLP. Looking for around 70% fidelity 

"""
class FidelityCheck():
    def __init__(self):
        self.fidelity = None

    def comparison(self):
        mlp_accuracy = LungCancerMLP().predict(handler.X.val)
        mlp_accuracy = np.argmax(mlp_accuracy, axis=1)

        surrogate_preds = self.model.predict(handler.X_val)

        self.fidelity = accuracy_score(mlp_accuracy, surrogate_preds)
        print(f"Fidelity to MLP: {self.fidelity:.2%}")

"""
Pipeline function includes the paramaters of the model and performance logging
"""

def pipeline(self):
    gkf = GroupKFold(n_splits=4)

    # For Grid Search tuning 
    params = {
        'max_depth': [6, 10, 15],
        'criterion': ['entropy'],
        'splitter': ['best', 'random'],
        'min_samples_leaf': [2, 6, 10],
        'min_samples_split': [5, 10, 15],
        'max_leaf_nodes': [5, 15, 20],
    }

    """
    grid = GridSearchCV(DecisionTreeClassifier(random_state=SEED), params, cv=5, scoring='accuracy')
    grid.fit(handler.X_train, handler.y_train)

    print("Best parameters:", grid.best_params_)
    print("Best score:", grid.best_score_)
    """

    for fold, (train_idx, test_idx) in enumerate(gkf.split(handler.X, handler.y, handler.groups)):
        handler.split(handler.X, handler.y, handler.data, train_idx, test_idx)
        handler.transform()
        handler.validation_split()

        """
        Insert the parameters for the DecisionTreeClassifier

        The goal of the model is to be a surrogate for the 'black box' MLP model
        The parameters impact fidelity with the 'black box' model

        Parameters: 
            criterion:     
            max_depth:    
            min_samples_leaf:  
            min_samples_split:  
            max_leaf_nodes:
            splitter='best' or 'random'
            random_state=SEED: This variable will be set to the seed to ensure training produces the same results everytime
        """

        model = DecisionTreeClassifier(
            criterion="entropy", 
            max_depth=6, 
            min_samples_leaf=6,
            min_samples_split=15,
            max_leaf_nodes=20,
            splitter='best',
            class_weight='balanced',
            random_state=SEED
            )
        
        model.fit(handler.X_train, handler.y_train, sample_weight=None)

        """
        Logged metrics to view model performance 

        Includes: 
            Accuracy: model.score()
            Confusion Matrix: confusion_matrix()
            Prediction Function: model.predict()
            F1 score: f1_score()
            Tree Size: node_count and max_depth

        Explainability tools: 
            plot_tree()
                - 

            export_graphviz() and export_text()
                - prints out the 
        """

        #prob = model.predict_proba(handler.X_test)
        #print(prob)

        accuracy = model.score(handler.X_test, handler.y_test, sample_weight=None)
        y_pred = model.predict(handler.X_test)
        print(np.unique(y_pred, return_counts=True))

        c_matrix = confusion_matrix(handler.y_test, y_pred)

        print(classification_report(handler.y_test, y_pred, target_names=[str(cls) for cls in handler.encoder.classes_]))
        print(accuracy)
        print(c_matrix)
        print("Node Size", model.tree_.node_count)
        print("Max Depth", model.tree_.max_depth)
        plot_tree(model, feature_names=handler.feature_cols, class_names=[str(cls) for cls in handler.encoder.classes_], filled=True)
        print(export_graphviz(model, feature_names=handler.feature_cols))


        X_val_df = pd.DataFrame(handler.X_val, columns=handler.feature_cols)

        print(X_val_df.shape)  # Should be (56, 37)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_df)

        print(np.array(shap_values).shape)  # (56, 37, 2)

        # Class 1 SHAP values for binary classification
        shap.summary_plot(shap_values[..., 1], X_val_df)

handler = DataHandling()
handler.load_data()
pipeline(handler)
