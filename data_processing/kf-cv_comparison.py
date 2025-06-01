# This file will be used to compare the models' performance on Group K fold-cross validation
# Very import that all the models conduct the same cross validation or else the results are not right 

import matplotlib.pyplot as plt 

# Hard coding the performance values of the models 

model1 = []

model2 = []

model3 = []

model4 = []  

# Graphing the results

plt.plot('model1', label='m', color='')
plt.plot()
plt.plot()
plt.plot()
plt.title("Model Performance across 4 Folds with Cross Validation")
plt.ylabel("Accuracy")
plt.xlabel("Fold")

plt.show()