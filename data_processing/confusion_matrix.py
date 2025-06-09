import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np 

# Hardcode confusion matrix in the array
conf_matrix = np.array([[]])
classes = ['Class 0', 'Class 1'] 

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title('(Model Name) Confusion Matrix')
plt.show()