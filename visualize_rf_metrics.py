import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

fold_metrics = {
    'Fold 1': {
        'accuracy': 0.8943,
        'precision_0': 0.92,
        'recall_0': 0.86,
        'f1_0': 0.89,
        'precision_1': 0.88,
        'recall_1': 0.93,
        'f1_1': 0.90,
        'roc_auc': 0.9465,
        'train_accuracy': 0.9397,
        'support_0': 475,
        'support_1': 499
    },
    'Fold 2': {
        'accuracy': 0.9316,
        'precision_0': 0.96,
        'recall_0': 0.89,
        'f1_0': 0.93,
        'precision_1': 0.91,
        'recall_1': 0.97,
        'f1_1': 0.94,
        'roc_auc': 0.9835,
        'train_accuracy': 0.9270,
        'support_0': 449,
        'support_1': 501
    },
    'Fold 3': {
        'accuracy': 0.8586,
        'precision_0': 0.93,
        'recall_0': 0.77,
        'f1_0': 0.84,
        'precision_1': 0.81,
        'recall_1': 0.94,
        'f1_1': 0.87,
        'roc_auc': 0.9343,
        'train_accuracy': 0.9530,
        'support_0': 469,
        'support_1': 500
    },
    'Fold 4': {
        'accuracy': 0.8618,
        'precision_0': 0.84,
        'recall_0': 0.88,
        'f1_0': 0.86,
        'precision_1': 0.89,
        'recall_1': 0.85,
        'f1_1': 0.87,
        'roc_auc': 0.9399,
        'train_accuracy': 0.9426,
        'support_0': 449,
        'support_1': 499
    }
}

confusion_matrices = {
    'Fold 1': np.array([[409, 66], [37, 462]]),
    'Fold 2': np.array([[401, 48], [17, 484]]),
    'Fold 3': np.array([[360, 109], [28, 472]]),
    'Fold 4': np.array([[395, 54], [77, 422]])
}

fig = plt.figure(figsize=(20, 12))

fold_names = list(fold_metrics.keys())
accuracies = [fold_metrics[f]['accuracy'] for f in fold_names]
f1_class0 = [fold_metrics[f]['f1_0'] for f in fold_names]
f1_class1 = [fold_metrics[f]['f1_1'] for f in fold_names]
roc_aucs = [fold_metrics[f]['roc_auc'] for f in fold_names]
precision_class0 = [fold_metrics[f]['precision_0'] for f in fold_names]
precision_class1 = [fold_metrics[f]['precision_1'] for f in fold_names]
recall_class0 = [fold_metrics[f]['recall_0'] for f in fold_names]
recall_class1 = [fold_metrics[f]['recall_1'] for f in fold_names]
train_accuracies = [fold_metrics[f]['train_accuracy'] for f in fold_names]

ax1 = plt.subplot(3, 3, 1)
x = np.arange(len(fold_names))
width = 0.35
ax1.bar(x - width/2, accuracies, width, label='Test Accuracy', color='steelblue', alpha=0.8)
ax1.bar(x + width/2, train_accuracies, width, label='Train Accuracy', color='lightcoral', alpha=0.8)
ax1.set_xlabel('Fold')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy per Fold')
ax1.set_xticks(x)
ax1.set_xticklabels([f.split()[-1] for f in fold_names])
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.82, 0.96)

for i, (test_acc, train_acc) in enumerate(zip(accuracies, train_accuracies)):
    ax1.text(i - width/2, test_acc + 0.002, f'{test_acc:.3f}', ha='center', va='bottom', fontsize=9)
    ax1.text(i + width/2, train_acc + 0.002, f'{train_acc:.3f}', ha='center', va='bottom', fontsize=9)

ax2 = plt.subplot(3, 3, 2)
ax2.plot(range(1, 5), f1_class0, 'o-', label='Class 0 (No Cancer)', linewidth=2, markersize=8, color='green')
ax2.plot(range(1, 5), f1_class1, 's-', label='Class 1 (Cancer)', linewidth=2, markersize=8, color='red')
ax2.set_xlabel('Fold')
ax2.set_ylabel('F1-Score')
ax2.set_title('F1-Score per Fold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(1, 5))
ax2.set_ylim(0.82, 0.95)

for i, (f1_0, f1_1) in enumerate(zip(f1_class0, f1_class1), 1):
    ax2.text(i, f1_0 + 0.002, f'{f1_0:.3f}', ha='center', va='bottom', fontsize=9)
    ax2.text(i, f1_1 - 0.002, f'{f1_1:.3f}', ha='center', va='top', fontsize=9)

ax3 = plt.subplot(3, 3, 3)
ax3.plot(range(1, 5), roc_aucs, 'D-', linewidth=2, markersize=8, color='purple')
ax3.set_xlabel('Fold')
ax3.set_ylabel('ROC AUC Score')
ax3.set_title('ROC AUC Score per Fold')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(1, 5))
ax3.set_ylim(0.92, 1.0)

for i, auc in enumerate(roc_aucs, 1):
    ax3.text(i, auc + 0.002, f'{auc:.4f}', ha='center', va='bottom', fontsize=9)

ax4 = plt.subplot(3, 3, 4)
ax4.plot(range(1, 5), precision_class0, 'o-', label='Class 0 (No Cancer)', linewidth=2, markersize=8, color='green')
ax4.plot(range(1, 5), precision_class1, 's-', label='Class 1 (Cancer)', linewidth=2, markersize=8, color='red')
ax4.set_xlabel('Fold')
ax4.set_ylabel('Precision')
ax4.set_title('Precision per Fold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xticks(range(1, 5))
ax4.set_ylim(0.80, 0.97)

ax5 = plt.subplot(3, 3, 5)
ax5.plot(range(1, 5), recall_class0, 'o-', label='Class 0 (No Cancer)', linewidth=2, markersize=8, color='green')
ax5.plot(range(1, 5), recall_class1, 's-', label='Class 1 (Cancer)', linewidth=2, markersize=8, color='red')
ax5.set_xlabel('Fold')
ax5.set_ylabel('Recall')
ax5.set_title('Recall per Fold')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.set_xticks(range(1, 5))
ax5.set_ylim(0.75, 0.98)

ax6 = plt.subplot(3, 3, 6)
overfitting_gaps = [train_accuracies[i] - accuracies[i] for i in range(len(fold_names))]
bars = ax6.bar(range(1, 5), overfitting_gaps, color='orange', alpha=0.7)
ax6.set_xlabel('Fold')
ax6.set_ylabel('Overfitting Gap')
ax6.set_title('Overfitting Gap (Train - Test Accuracy)')
ax6.grid(True, alpha=0.3, axis='y')
ax6.set_xticks(range(1, 5))

for i, (bar, gap) in enumerate(zip(bars, overfitting_gaps)):
    ax6.text(bar.get_x() + bar.get_width()/2, gap + 0.001, f'{gap:.4f}', ha='center', va='bottom', fontsize=9)

for idx, (fold_name, cm) in enumerate(confusion_matrices.items(), 1):
    ax = plt.subplot(3, 4, 8 + idx)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    ax.set_title(f'{fold_name} Confusion Matrix')

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'F1 (Class 0)', 'F1 (Class 1)', 'Precision (Class 0)', 
               'Precision (Class 1)', 'Recall (Class 0)', 'Recall (Class 1)', 'ROC AUC'],
    'Mean': [
        np.mean(accuracies),
        np.mean(f1_class0),
        np.mean(f1_class1),
        np.mean(precision_class0),
        np.mean(precision_class1),
        np.mean(recall_class0),
        np.mean(recall_class1),
        np.mean(roc_aucs)
    ],
    'Std Dev': [
        np.std(accuracies),
        np.std(f1_class0),
        np.std(f1_class1),
        np.std(precision_class0),
        np.std(precision_class1),
        np.std(recall_class0),
        np.std(recall_class1),
        np.std(roc_aucs)
    ]
})

ax_table = plt.subplot(3, 1, 3)
ax_table.axis('tight')
ax_table.axis('off')

table_data = []
table_data.append(['Metric', 'Mean ± Std Dev'])
for _, row in metrics_df.iterrows():
    table_data.append([row['Metric'], f"{row['Mean']:.4f} ± {row['Std Dev']:.4f}"])

table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                       colWidths=[0.3, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

header_row = table[(0, 0)], table[(0, 1)]
for cell in header_row:
    cell.set_facecolor('#4CAF50')
    cell.set_text_props(weight='bold', color='white')

for i in range(1, len(table_data)):
    if i % 2 == 0:
        table[(i, 0)].set_facecolor('#f0f0f0')
        table[(i, 1)].set_facecolor('#f0f0f0')

plt.suptitle('Random Forest Classifier - Cross-Validation Performance Metrics', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])

plt.savefig('rf_performance_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("RANDOM FOREST CROSS-VALIDATION METRICS SUMMARY")
print("="*60)
print("\nPer-Fold Results:")
for fold_name in fold_names:
    metrics = fold_metrics[fold_name]
    print(f"\n{fold_name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score (Class 0/1): {metrics['f1_0']:.4f} / {metrics['f1_1']:.4f}")
    print(f"  Precision (Class 0/1): {metrics['precision_0']:.4f} / {metrics['precision_1']:.4f}")
    print(f"  Recall (Class 0/1): {metrics['recall_0']:.4f} / {metrics['recall_1']:.4f}")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Overfitting Gap: {metrics['train_accuracy'] - metrics['accuracy']:.4f}")

print("\n" + "="*60)
print("OVERALL STATISTICS")
print("="*60)
for _, row in metrics_df.iterrows():
    print(f"{row['Metric']:.<25} {row['Mean']:.4f} ± {row['Std Dev']:.4f}")

print("\n" + "="*60)
print("Visualization saved as 'rf_performance_metrics.png'")
print("="*60)