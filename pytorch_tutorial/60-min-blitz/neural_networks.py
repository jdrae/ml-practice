import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,3) 
        self.conv2 = nn.Conv2d(6,16,3)
        # Affine: y = Wx + b
        self.fc1 = nn.Linear(16*6*6,120) # imgsize: 6*6
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2)) # window: (2,2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # if it's a squared number

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
"""
Net(
  (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=576, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
"""

# backward method is set by 'autograd'
params = list(net.parameters())
print(len(params))
print(params[0].size()) # weight of conv1

# put data
input = torch.randn(1,1,32,32)
out = net(input)
print(out)

net.zero_grad() # gradient buffer 0
out.backward(torch.randn(1,10))


"""
Loss Function
"""

output = net(input)
target = torch.randn(10) # sample target
target = target.view(1,-1) # reshape like output
criterion = nn.MSELoss()

loss = criterion(output,target)
print(loss)

print(loss.grad_fn) # MSELoss
print(loss.grad_fn.next_functions[0][0]) # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) # ReLU


"""
Backprop
"""

net.zero_grad()
print("conv1.bias.grad before backward")
print(net.conv1.bias.grad)

loss.backward(retain_graph = True) # to use .backward() in line 106

print("conv2.bias.grad after backward")
print(net.conv2.bias.grad)


"""
SGD; Stochastic Gradient Descent
"""

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad() # to clear gradients
ouput = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # update weights
