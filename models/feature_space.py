import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Load dataset
df = pd.read_csv("data/train_data_normalized.csv")

# Select features
feature_cols = df.drop(columns=['cancer_stage','patient_id','chunk','filename','rolloff','bandwidth','skew','zcr','rms']).columns
X = df[feature_cols].values
y = df['cancer_stage'].values

# 3D UMAP projection
reducer = umap.UMAP(n_components=3, random_state=42)
X_3d = reducer.fit_transform(X)

# 3D plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    X_3d[:,0], X_3d[:,1], X_3d[:,2],
    c=y, cmap='coolwarm', alpha=0.7
)

ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')
ax.set_title('3D UMAP Feature Space')
fig.colorbar(scatter, ax=ax, label='Cancer Stage')
plt.show()
