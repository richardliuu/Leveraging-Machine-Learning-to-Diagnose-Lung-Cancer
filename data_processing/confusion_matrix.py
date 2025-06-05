import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np 

# Hardcoded confusion matrix
conf_matrix = np.array([[240, 49], [18, 307]])
classes = ['Class 0', 'Class 1']  # Replace with your class names if needed

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title('MLP (Blue) Confusion Matrix')
plt.show()


