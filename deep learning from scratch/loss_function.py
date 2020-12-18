import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta)) # np.log(0) is negative infinity

def cross_entropy_error_batch(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)

    batch_size = y.shape[0]
    return -np.sum(t*np.log(y)) / batch_size
    # if given with labels like [2,7,0,9,4] (not ont-hot-encoding)
    # return -np.sum(np.log(y[np.arange(batch_size),t])) / batch_size 

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)
    
    print(x_train.shape)
    print(t_train.shape)