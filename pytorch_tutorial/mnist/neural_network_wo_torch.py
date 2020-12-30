from load_data import x_train, y_train
import math
import torch

# print(x_train, y_train)
# print(x_train.shape)
# print(y_train.min(), y_train.max())
n, c = x_train.shape

weights = torch.rand(784,10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias) # @ stands for the dot product operation

# negative log-likelihood
def nil(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nil

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean() # if largest value matches the target value


"""
one forward pass
"""
bs = 64 # batch size

xb = x_train[0:bs] # predictions
yb = y_train[0:bs]

preds = model(xb)

print(loss_func(preds,yb))
print(accuracy(preds,yb))


"""
forwardpass & backprop
"""
lr = 0.5
epochs = 2

for epoch in range(epochs):
    for i in range((n-1)//bs+1):
        start_i = i * bs
        end_i = start_i + bs
        
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]

        pred = model(xb)
        loss = loss_func(pred,yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

print(loss_func(model(xb),yb), accuracy(model(xb),yb))

