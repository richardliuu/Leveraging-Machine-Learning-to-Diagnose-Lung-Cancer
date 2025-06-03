# This file will be used to compare the models' performance on Group K fold-cross validation
# Very import that all the models conduct the same cross validation or else the results are not right 

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Include legend 
# Models: 1, 2, 3, 4 for each corresponding bracket 
# Model 1 Done: 

y_sets = [[82.7, 81.0, 87.2, 95.1], [84, 92, 71, 90], [53, 60, 32, 22], [60, 32, 32, 54]]
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