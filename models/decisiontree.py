import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.combine import SMOTEENN
from ..class_based_models import lung_cancer_mlp

SEED = 42

# Data preprocessing 
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
        self.groups = None
        self.duplicates = None
        self.patient_labels = None
        self.inconsistent_patients = None

        self.X = None
        self.y = None
        self.X_val = None
        self.y_val = None
        self.num_classes = None

    @staticmethod
    def to_categorical(labels, num_classes=None):
        labels = np.array(labels, dtype=int)
        if num_classes is None:
            num_classes = np.max(labels) + 1
        return np.eye(num_classes)[labels]

    def load_data(self):
        data = pd.read_csv("data/surrogate_data.csv")

        self.X = data.drop(columns=["segment", 
                                    "true_label", 
                                    "patient_id"
                                    ])
        
        self.y = data['predicted_label']
        self.groups = data['patient_id']
        self.duplicates = data.duplicated(subset=self.feature_cols)
        self.patient_labels = data.groupby('patient_id')['true_label'].nunique()
        self.inconsistent_patients = self.patient_labels[self.patient_labels > 1]

        # Storing this information may be useful for analysing the model
        self.meta = data[['segment', 'true_label', 'predicted_label', 'patient_id']]

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

        # Could decide whether smote is used or not 
        self.X_train, self.y_train = SMOTEENN().fit_resample(self.X_train, self.y_train)
    
    def validation_split(self):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, 
            self.y_train, 
            test_size=0.1, 
            stratify=self.y_train, 
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

        self.model.fit(self.X_train, self.y_train, sample_weight=None)

        self.preds = self.model.predict(self.X_test_scaled)

        return self.model, self.history, self.preds

    def evaluate(self, X_test, y_test, encoder):
        plot_tree(self.model, feature_names=handler.feature_cols, filled=True)
        report = classification_report(
                    y_test, self.preds, 
                    target_names=[str(cls) for cls in encoder.classes_],
                    output_dict=True 
                )
        
        y_pred = self.predict(X_test)
        
        return report, y_pred
      
    def graph(self):    
        plt.plot(self.history['history'])
        plt.xlabel()
        plt.ylabel()
        plt.title()
        plt.show()

class FidelityCheck():
    def __init__(self):
        self.fidelity = None

    # Need to import the MLP for fidelity check 
    def comparison(self):
        mlp_accuracy = lung_cancer_mlp.LungCancerMLP().predict(handler.X.val)
        mlp_accuracy = np.argmax(mlp_accuracy, axis=1)

        surrogate_preds = self.model.predict(handler.X_val)

        self.fidelity = accuracy_score(mlp_accuracy, surrogate_preds)
        print(f"Fidelity to MLP: {self.fidelity:.2%}")

        f1 = f1_score(mlp_accuracy, surrogate_preds)

"""
Make modifications to the pipeline startup
to ensure that it works with a
decision tree

Some features may nnot be needed like the metrics 
"""
def pipeline(self):
    gkf = GroupKFold(n_splits=4)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(handler.X, handler.y, handler.groups)):
        handler.split(handler.X, handler.y, train_idx, test_idx)
        handler.transform()
        handler.validation_split()

        model = DecisionTreeClassifier(num_classes=handler.num_classes)
        model._buildmodel()

        report= model.evaluate(
            handler.X_test, 
            handler.y_test, 
            handler.encoder
        )

        y_pred = model.predict(handler.X_test)
        handler.predictions.append(y_pred)

        c_matrix = confusion_matrix(handler.y_test, y_pred)
        print(c_matrix)

        handler.reports.append(report)
        handler.conf_matrices.append(c_matrix)
        handler.details.append({
            "fold": fold + 1,
            "train_samples": len(handler.X_train),
            "test_samples": len(handler.X_test),
            "accuracy": report['accuracy'],
        }) 

handler = DataHandling()
handler.load_data()
pipeline(handler)
