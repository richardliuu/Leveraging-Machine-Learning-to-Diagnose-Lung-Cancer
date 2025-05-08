import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd

# Insert values 
data = {
    'Stage': ("Healthy", "Stage 1", "Stage 2", "Stage 3", "Stage 4"), 
    'Value': ("")

}

df = pd.DataFrame(data)

sns.set_theme()

plt.figure(figsize=(6, 6))
plt.pie(df["Percentage"], labels=df[], autopct='%1.1f%%', startangle=140)
plt.title("Class Distribution of Lung Cancer Stages")

plt.show()

