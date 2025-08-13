import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor

# --- Step 1: Data Handler Class ---
class DataHandler:
    def __init__(self, csv_paths):
        self.data = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)

    def preprocess(self):
        self.data.dropna(inplace=True)
        self.X = self.data.drop(columns=["segment", "true_label", "patient_id", "predicted_label", "mlp_prob"])
        self.y = self.data["mlp_prob"]

    def split_data(self, test_size=0.2, random_state=42):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

# --- Step 2: Feature Selection ---
def select_k_best_features(X_train, y_train, X_test, k=10):
    selector = SelectKBest(score_func=f_regression, k=k)
    X_train_new = selector.fit_transform(X_train, y_train)
    X_test_new = selector.transform(X_test)
    selected_features = X_train.columns[selector.get_support()]
    return X_train_new, X_test_new, selected_features

# --- Step 3: Train Surrogate Model ---
def train_surrogate(X_train, y_train, X_test, y_test):
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

# --- Step 4: Evaluate Fidelity ---
def print_predictions_comparison(y_true, y_pred, num_samples=100):
    print(f"\nComparing first {num_samples} predictions (MLP vs Surrogate):\n")
    print(f"{'Index':>5} | {'MLP Prediction':>15} | {'Surrogate Prediction':>20} | {'Difference':>10}")
    print("-" * 60)
    for i in range(min(num_samples, len(y_true))):
        diff = abs(y_true[i] - y_pred[i])
        print(f"{i:5d} | {y_true[i]:15.4f} | {y_pred[i]:20.4f} | {diff:10.4f}")

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\nFidelity Metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²:  {r2:.4f}\n")

# --- Step 5: Pipeline ---
def pipeline(csv_paths, k_features=10):
    handler = DataHandler(csv_paths)
    handler.preprocess()
    X_train, X_test, y_train, y_test = handler.split_data()

    X_train_k, X_test_k, selected = select_k_best_features(X_train, y_train, X_test, k=k_features)
    print("Selected Features:", list(selected))

    model, preds = train_surrogate(X_train_k, y_train, X_test_k, y_test)
    print_predictions_comparison(y_test.to_numpy(), preds)

# --- Run Example ---
if __name__ == "__main__":
    csv_files = [
        "results_fold_1.csv",
        "results_fold_2.csv",
        "results_fold_3.csv",
        "results_fold_4.csv"
    ]
    pipeline(csv_files, k_features=10)
