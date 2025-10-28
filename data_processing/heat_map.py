import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your CSV with features
csv_path = r"data/jitter_shimmerlog.csv"
df = pd.read_csv(csv_path)

# Keep only numeric columns (drop patient_id, segment, etc.)
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Compute Pearson correlation
corr = numeric_df.corr(method='pearson')

# Plot heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
plt.title("Feature Correlation Heatmap", fontsize=16)
plt.show()
