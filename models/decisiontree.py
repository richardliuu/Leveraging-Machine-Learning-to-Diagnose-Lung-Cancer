import pandas as pd 
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder, StandardScaler

SEED = 42

# Manual categorical function
def to_categorical(labels, num_classes=None):
    labels = np.array(labels, dtype=int)
    if num_classes is None:
        num_classes = np.max(labels) + 1
    return np.eye(num_classes)[labels]

class DataHandling:
    def __init__(self):
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler() 

        self.reports = []
        self.conf_matrices = []
        self.details = []
        self.history = []
        self.roc_aucs = []
        self.predictions = []

        self.X = None
        self.y = None

    def load_data(self):
        # Insert file directory 
        data = pd.read_csv("data/")

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
        self.y_train_cat = to_categorical(self.y_train_resampled, num_classes=self.num_classes)
        self.y_test_cat = to_categorical(self.y_test_encoded, num_classes=self.num_classes)
    
    def validation_split(self):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
        self.X_train_resampled, self.y_train_cat, test_size=0.1, 
        stratify=self.y_train_resampled, random_state=SEED
        )

    def get_data(self):
        return self.X, self.y, self.X_train_fold, self.y_train_fold, self.X_test_fold, self.y_test_fold

class DecisionTreeSurrogate:
    def __init__(self):
        pass 

    def _buildmodel(self):
        model = DecisionTreeClassifier(max_depth=10, random_state=SEED)
        model.fit()

        return model
    
    def train():
        pass
        





