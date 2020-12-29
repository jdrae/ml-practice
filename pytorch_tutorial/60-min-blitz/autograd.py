import torch

x = torch.ones(2,2, requires_grad = True) # requires_grad tracks every operation in tensor

y = x+2
print(y.grad_fn) # AddBackward0

z = y*y*3
out = z.mean()
print(z, out) # MulBackward0 MeanBackward0


a = torch.randn(2,2)
a = ((a*3)/(a-1))
print(a.requires_grad) #false
a.requires_grad_(True)

b = (a*a).sum()
print(b.grad_fn) # SumBackward0


"""
Gradient
"""

out.backward()
print(x.grad) # d(out)/dx

# vector-Jacobian
x = torch.randn(3, requires_grad=True)
y = x*2
while y.data.norm() <1000:
    y = y*2

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

print(x.requires_grad)
print((x**2).requires_grad)

with torch.no_grad(): # stop tracking
    print((x**2).requires_grad)

print(x.requires_grad) # True
y = x.detach()
print(y.requires_grad) # False
print(x.eq(y).all()) # tensor(True)


