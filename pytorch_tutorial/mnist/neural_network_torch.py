from load_data import x_train, y_train, x_valid, y_valid
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

n, c = x_train.shape
bs = 64 # batch size
lr = 0.5
epochs = 2

train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)

xb = x_train[0:bs] # predictions
yb = y_train[0:bs]

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784,10)
        # self.weights = nn.Parameter(torch.randn(784,10) / math.sqrt(784))
        # self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return self.lin(xb)
        # return xb @ self.weights + self.bias

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        # twice as large as that for the training set
        # validation set does not need backpropagation and thus takes less memory
        DataLoader(valid_ds, batch_size=bs*2),
    )

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

loss_func = F.cross_entropy # combine: negative log likelihood loss & log softmax activation

def loss_batch(model, loss_func, xb, yb, opt=None): # for the validation set, we don’t pass an optimizer
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean() # if largest value matches the target value


"""
forwardpass & backprop
"""

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        # for i in range((n-1)//bs+1):
        #     xb, yb = train_ds[i*bs : i*bs+bs]
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)
            # pred = model(xb) # (xb, yb) are loaded automatically from the data loader
            # loss = loss_func(pred,yb)
            # loss.backward()
            # opt.step()
            # opt.zero_grad()
        
        model.eval()
        with torch.no_grad():
            #     for p in model.parameters():
            #         p -= p.grad * lr
            #     model.zero_grad()
            losses, nums = zip(
                *[loss_batch(model,loss_func, xb,yb) for xb,yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses,nums)) / np.sum(nums)
        print(epoch, val_loss)

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)

if __name__ == "__main__":
    model, opt = get_model()
    fit(epochs, model, loss_func, opt, train_dl, valid_dl)

