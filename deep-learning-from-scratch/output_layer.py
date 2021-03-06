import numpy as np

# regression - identity function
def identity_function(x):
    return x


# classification - softmax function
def softmax(x):
    # c = np.max(a)
    # exp_a = np.exp(a-c) # to avoid overflow
    # sum_exp_a = np.sum(exp_a)
    # y = exp_a / sum_exp_a
    # return y
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))

if __name__ == '__main__':
    a = np.array([0.3,2.9,4.0]) # max element becomes max value => softmax can be skipped
    y = softmax(a)
    print(y) # an output is between 0 and 1.0
    print(np.sum(y)) # always 1  => interpreted as 'probability' of belonging class