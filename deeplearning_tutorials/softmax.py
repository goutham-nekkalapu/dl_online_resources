

"""softmax."""

import numpy as np

def softmax(x,k):
    """Compute softmax values for each sets of scores in x."""
    x = np.dot(x,k)
    return np.exp(x)/np.sum(np.exp(x), axis = 0)


scores = [3.0, 1.0, 0.2]
result = softmax(scores,1)
print (result)
sum = np.sum(result)
print("sum is : ",sum)

scores = [3.0, 1.0, 0.2]
result = softmax(scores,10)
print (result)
sum = np.sum(result)
print("sum is : ",sum)


scores = [3.0, 1.0, 0.2]
result = softmax(scores,0.1)
print (result)
sum = np.sum(result)
print("sum is : ",sum)

"""
# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
"""
