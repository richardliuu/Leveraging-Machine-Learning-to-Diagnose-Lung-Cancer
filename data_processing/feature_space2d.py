import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load your dataset
df = pd.read_csv("data/train_data_normalized.csv")

# Select features (exclude labels and IDs)
feature_cols = df.drop(columns=['cancer_stage','patient_id','chunk','filename','rolloff','bandwidth','skew','zcr','rms']).columns
X = df[feature_cols].values
y = df['cancer_stage'].values

# Reduce dimensions to 2D
reducer = umap.UMAP(n_components=2, random_state=42)
X_2d = reducer.fit_transform(X)

# Plot 2D feature space
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=X_2d[:, 0], 
    y=X_2d[:, 1], 
    hue=y,           # Color by cancer stage
    palette='coolwarm',
    alpha=0.7
)
plt.title("2D UMAP Projection of Feature Space")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(title='Cancer Stage')
plt.grid(True, alpha=0.3)
plt.show()
