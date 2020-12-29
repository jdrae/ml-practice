import torch
import math

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# tensor (x, x^2, x^3)
p = torch.tensor([1,2,3])
xx = x.unsqueeze(-1).pow(p) # shape (2000, 3)

model = torch.nn.Sequential(
    torch.nn.Linear(3,1), # omputes output from input and holds internal Tensors for its weight and bias.
    torch.nn.Flatten(0,1) # flatens the output of the linear layer to a 1D tensor, to match the shape of 'y'
)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

for t in range(2000):
    y_pred = model(xx)

    loss = loss_fn(y_pred,y) # returns a Tensor containing the loss
    if t % 100 == 99:
        print(t, loss.item())
    
    # Zero the gradients before running the backward pass.
    optimizer.zero_grad()
    # model.zero_grad()

    loss.backward()

    # update parameters
    optimizer.step()
    # with torch.no_grad():
    #    for param in model.parameters(): # each parameter is a tensor
    #        param -= learning_rate * param.grad

linear_layer = model[0]


print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:,0].item()} x + {linear_layer.weight[:,1].item()} x^2 + {linear_layer.weight[:,2].item()} x^3')


