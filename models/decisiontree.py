import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.combine import SMOTEENN
from ..class_based_models import lung_cancer_mlp

SEED = 42

# Class for dealing with preprocessing the data 

"""
I think I will want to clean up all the X variables as I think they could be
passed through functions to reduce clutter in the classes

Also makes the pipeline cleaner, with less variables are more ability to 
edit the code 
"""

class DataHandling:
    def __init__(self,data=r"C:\Users\richa\OneDrive\Desktop\science2\data\surrogate_data.csv"):
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler() 
        self.smote = SMOTEENN()
        self.data = data


        # The long list of variables that I want to get rid of 
        # Also exists in the other model files as well
        self.reports = []
        self.conf_matrices = []
        self.details = []
        self.history = []
        self.roc_aucs = []
        self.predictions = []

        self.feature_cols = None
        self.groups = None
        self.duplicates = None
        self.patient_labels = None
        self.inconsistent_patients = None

        self.X = None
        self.y = None
        self.X_train_fold = None
        self.X_test_fold = None
        self.y_train_fold = None
        self.y_test_fold = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train_encoded = None
        self.y_test_encoded = None
        self.X_train_resampled = None
        self.y_train_resampled
        self.num_classes = None
        self.y_test_cat = None
        self.y_train_cat = None
        self.X_val = None
        self.y_val = None

    @staticmethod
    def to_categorical(labels, num_classes=None):
        labels = np.array(labels, dtype=int)
        if num_classes is None:
            num_classes = np.max(labels) + 1
        return np.eye(num_classes)[labels]

    def load_data(self):
        # Insert file directory 
        data = pd.read_csv("data/surrogate_data.csv")

        # This may change because data from the predictions of the MLP are to be used instead 
        # Predicting for y_pred of the MLP
        self.X = data.drop(columns=["segment", "cancer_stage", "patient_id"])

        # Change this to the y_pred of MLP
        self.y = data['cancer_stage']
        self.groups = data['patient_id']
        self.duplicates = data.duplicated(subset=self.feature_cols)
        self.patient_labels = data.groupby('patient_id')['cancer_stage'].nunique()
        self.inconsistent_patients = self.patient_labels[self.patient_labels > 1]

    def split(self, X, y, data, train_idx, test_idx):
        self.X_train_fold = X.iloc[train_idx]
        self.y_train_fold = y.iloc[train_idx]
        self.X_test_fold = X.iloc[test_idx]
        self.y_test_fold = y.iloc[test_idx]

        self.train_patients = set(data.iloc[train_idx]['patient_id'])
        self.test_patients = set(data.iloc[test_idx]['patient_id'])

    def transform(self):
        self.X_train_scaled = self.scaler.fit_transform(self.X_train_fold)
        self.X_test_scaled = self.scaler.transform(self.X_test_fold)
        self.y_train_encoded = self.encoder.fit_transform(self.y_train_fold)
        self.y_test_encoded = self.encoder.transform(self.y_test_fold)
        self.num_classes = len(self.encoder.classes_)

        # Could decide whether smote is used or not 
        self.X_train_resampled, self.y_train_resampled = self.smote.fit_resample(self.X_train_scaled, self.y_train_encoded)

    def put_to_categorical(self):
        self.y_train_cat = self.to_categorical(self.y_train_resampled, num_classes=self.num_classes)
        self.y_test_cat = self.to_categorical(self.y_test_encoded, num_classes=self.num_classes)
    
    def validation_split(self):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train_resampled, 
            self.y_train_cat, 
            test_size=0.1, 
            stratify=self.y_train_resampled, 
            random_state=SEED
        )

    def get_data(self):
        return self.X, self.y, self.X_train_fold, self.y_train_fold, self.X_test_fold, self.y_test_fold

class DecisionTreeSurrogate:
    def __init__(self, num_classes):
        self.model = self._buildmodel()
        self.num_classes = num_classes

    def _buildmodel(self):
        self.model = DecisionTreeClassifier(max_depth=6, random_state=SEED)
        
        """
        Insert all the parameteers and the history will be used again
        for graphing 
        """ 
        history = self.model.fit(

        )

        return self.model

    def evaluate(self, X_test, y_test, encoder):
        # Go back and check if feature_names is the variable to be called 
        plot_tree(self.model, feature_names=self.feature_names, filled=True)

        preds = np.argmax(self.model.predict(X_test), axis=1)

        # Find out whether predicting on a decision tree is the same as a neural network
        report = classification_report(
                    y_test, preds, 
                    target_names=[str(cls) for cls in encoder.classes_],
                    output_dict=True 
                )
        
        return report
    
    def graph():    
        plt.plot(history[])


        plt.show()

class FidelityCheck():
    def __init__(self):
        self.fidelity = None

    # Need to import the MLP for fidelity check 
    def comparison(self):
        self.fidelity = accuracy_score(lung_cancer_mlp.LungCancerMLP.predict(self.X_val), self.model.predict(self.X_val))
        print(self.fidelity)

# Running the pipeline

"""
Make modifications to the pipeline startup
to ensure that it works with a
decision tree

Some features may nnot be needed like the metrics 
"""
def pipeline():
    gkf = GroupKFold(n_splits=4)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(handler.X, handler.y, handler.groups)):
        handler.split(handler.X, handler.y, pd.read_csv(handler.data), train_idx, test_idx)
        handler.transform()
        handler.put_to_categorical()
        handler.validation_split()

        # Fill in model specifications
        model = ()

        # Tune this to fit decision tree architecture 
        history = model.train(
            handler.X_train, handler.y_train, handler.X_val, handler.y_val
        )

        report, c_matrix, auc = model.evaluate(
            handler.X_test_scaled, handler.y_test_encoded, handler.encoder
        )

        y_pred = model.predict(handler.X_test_scaled)
        handler.predictions.append(y_pred)

        c_matrix = confusion_matrix(handler.y_test_encoded, y_pred)
        print(c_matrix)

        handler.reports.append(report)
        handler.conf_matrices.append(c_matrix)
        handler.roc_aucs.append(auc)
        handler.history.append(history.history)
        handler.details.append({
            "fold": fold + 1,
            "train_samples": len(handler.X_train_fold),
            "test_samples": len(handler.X_test_fold),
            "accuracy": report['accuracy'],
            "epochs_trained": len(history.history['loss']),
        }) 

handler = DataHandling()
handler.load_data()
pipeline(handler)
