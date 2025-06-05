import matplotlib.pyplot as plt 

# Graphing 

num_apples = [2, 4, 5]

plt.title("Number of Apples for each Type")
plt.xlabel("Types of Apples")
plt.ylabel("Number of Apples")
plt.plot(num_apples, color='green')
plt.show()