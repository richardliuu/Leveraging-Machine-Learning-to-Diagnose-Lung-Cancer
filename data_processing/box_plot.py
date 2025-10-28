import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

# Example: Make a simple dataset
data = pd.read_csv("data/jitter_shimmerlog.csv")
df = pd.DataFrame(data)

print(df.columns)

comparison = "centroid"
class_column = "cancer_stage"

# Get unique class names
classes = df[class_column].unique()

# Split feature values by each class
data_to_plot = [df[df[class_column] == cls][comparison] for cls in classes]

# Create boxplot
plt.boxplot(data_to_plot, labels=classes)
plt.title(f"Boxplot of MFCC3 between Class 0 and 1")
plt.ylabel("MFCC3 Mean Value")
plt.xlabel("Class")
plt.show()