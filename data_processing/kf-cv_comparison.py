
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

y_sets = [[82.7, 81.0, 87.2, 95.1], [83.2, 82.3, 92.1, 92.0], [0, 0, 0, 0], [0, 0, 0, 0]]
x_sets = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]

adjusted_x_sets = [[x + 1 for x in x_set] for x_set in x_sets]

for i in range(len(adjusted_x_sets)):
    plt.plot(adjusted_x_sets[i], y_sets[i], marker='o', linestyle='-', label=f'Set {i+1}')

# Sets only integer values for the graph 
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title("Model Performance across 4 Folds with Cross Validation")
plt.grid(True)
plt.show()