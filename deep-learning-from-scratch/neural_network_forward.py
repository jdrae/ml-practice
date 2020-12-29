import numpy as np


'''
y1 = x1*w11 + x2*w12 -> 1*1 + 2*2
y2 = x1*w21 + x2*w22 -> 1*3 + 2*4
y3 = x1*w31 + x2*w32 -> 1*5 + 2*6
'''
X = np.array([1,2])
W = np.array([[1,3,5],
              [2,4,6]])
Y = np.dot(X,W)
print(Y)

from activation_function import sigmoid
from output_layer import identity_function

def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5], [0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    print(z1.shape)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    print(z2.shape)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network, x)
print(y)