import matplotlib.pyplot as plt 
import pandas as pd

# Insert values 
data = {
    'Stage': [""], 
    'Number of Diagnosis': [],
}

df = pd.DataFrame(data)

plt.figure(figsize=(6, 6))
plt.pie(df["Number of Diagnosis"], labels=df["Stage"], autopct='%1.1f%%', startangle=140)
plt.title("Class Distribution of Lung Cancer Stages")
plt.axis('equal')   
plt.show()

