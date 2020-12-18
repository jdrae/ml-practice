"""
activation functions should be nonlinear.
because linear function makes hidden layers meaningless.
"""

import numpy as np

# step function
def _step_function(x):
    if x>0:
        return 1
    else:
        return 0

def step_function(x):
    # y = x>0
    # return y.astype(np.int)
    return np.array(x>0, dtype=np.int)

import matplotlib.pylab as plt
x = np.arange(-5.0,5.0,0.1)
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
# plt.show()

# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
# plt.show()


# ReLU: Rectified Linear Unit
def relu(x):
    return np.maximum(0,x)

y = relu(x)
plt.plot(x,y)
plt.ylim(-1, 5)
# plt.show()

if __name__ == '__main__':
    x = np.array([-1.0,1.0,2.0])
    y = x>0
    print(y)
    print(y.astype(np.int))
    print("================")

