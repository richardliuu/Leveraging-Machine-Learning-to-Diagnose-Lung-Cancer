import matplotlib.pyplot as plt

# CNN.py Results

"""

             precision    recall  f1-score   support

           0       0.88      0.89      0.89       236
           1       0.93      0.92      0.93       373

    accuracy                           0.91       609
   macro avg       0.91      0.91      0.91       609
weighted avg       0.91      0.91      0.91       609

Confusion Matrix:
 [[210  26]
 [ 28 345]]
ROC AUC Score (macro): 0.9695

Step 3: Results Summary

============================================================
CROSS-VALIDATION SUMMARY
============================================================
Per-fold results:
Fold 1: 0.8399 accuracy (21 patients, 612 samples, 7 epochs)
Fold 2: 0.8145 accuracy (21 patients, 620 samples, 24 epochs)
Fold 3: 0.9192 accuracy (21 patients, 619 samples, 22 epochs)
Fold 4: 0.9113 accuracy (21 patients, 609 samples, 6 epochs)

Overall Performance:
Mean Accuracy: 0.8712 ± 0.0450
Min Accuracy:  0.8145
Max Accuracy:  0.9192

Class-wise F1-scores:
Class 0: 0.8488 ± 0.0670
Class 1: 0.8808 ± 0.0303

Average Confusion Matrix:
[[244  45]
 [ 34 292]]
"""

# I find it strange that the ROC AUC has quite a big difference with the F1 scores and fold accuracies. 
# Not a single fold accuracy crosses over the ROC for some folds

# Formula 
"""
CM

A B 
C D

tpr = d / (d + c)
fpr = b / (b + a)
"""

# Simulated TPR/FPR point
mlp_tpr = 303 / (303 + 22)
mlp_fpr = 46 / (46 + 244)

cnn_tpr = 292 / (292 + 34)
cnn_fpr = 45 / (45 + 244)

plt.figure(figsize=(8, 6))
plt.plot([0, mlp_fpr, 1], [0, mlp_tpr, 1], color='blue', label=f"MLP (AUC = 0.9468)")
plt.plot([0, cnn_fpr, 1], [0, cnn_tpr, 1], color='red', label=f"CNN (AUC = 0.9442)")
plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Random chance (AUC = 0.500)')

# Styling
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Macro ROC AUC Curves of Models")
plt.legend(loc='lower right')
plt.grid(True)
plt.show()