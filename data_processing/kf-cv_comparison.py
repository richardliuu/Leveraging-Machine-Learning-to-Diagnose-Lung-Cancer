import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

y_sets = [[83.33, 85.00, 88.53, 95.73]]
x_sets = [[0, 1, 2, 3]]

adjusted_x_sets = [[x + 1 for x in x_set] for x_set in x_sets]

for i in range(len(adjusted_x_sets)):
    plt.plot(adjusted_x_sets[i], y_sets[i], marker='o', linestyle='-', label=f'Set {i+1}')

# Sets only integer values for the graph 
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

# Set x-axis limits from 0 to 100

plt.ylim(0, 100)

plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title("MLP Performance Across 4 Folds with Cross Validation")
plt.grid(True)
plt.show()
