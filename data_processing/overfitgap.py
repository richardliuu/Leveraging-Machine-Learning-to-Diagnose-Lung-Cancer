import matplotlib.pyplot as plt

# Data
folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4']
overfitting_gaps = [0.1007, 0.0178, 0.0589, 0.0527]

# Create bar chart
plt.figure(figsize=(8,5))
bars = plt.bar(folds, overfitting_gaps, color='skyblue', edgecolor='black')

# Add values on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.002, f'{height:.3f}', ha='center', va='bottom')

# Labels and title
plt.ylabel('Overfitting Gap (Train Accuracy - Test Accuracy)')
plt.title('Overfitting Gap per Fold')
plt.ylim(0, max(overfitting_gaps)*1.2)  # add some space on top

plt.show()
