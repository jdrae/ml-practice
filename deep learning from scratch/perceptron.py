# linear -> single-layer perceptron
def _AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

print(_AND(0,0))
print(_AND(0,1))
print(_AND(1,0))
print(_AND(1,1))
print("================")

import numpy as np
x = np.array([0,1]) # 입력
w = np.array([0.5,0.5]) # 가중치
b = -0.7 # 편향
print("w*x:\t",w*x)
print("np.sum(w*x):\t",np.sum(w*x))
print("np.sum(w*x)+b:\t",np.sum(w*x)+b) # 부동소수점 수에 의한 연산 오차
print("================")

def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))
print("================")

def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# nonlinear XOR -> multi-layer perceptron

def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y

print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))
print("================")

