from __future__ import print_function
import torch

"""
Matrix
"""

a = torch.empty(5,3)
b = torch.rand(5,3)
c = torch.zeros(5,3,dtype=torch.long)
d = torch.tensor([5.5,3])
e = b.new_ones(5,3,dtype=torch.double)
f = torch.randn_like(e, dtype=torch.float)

print(f.size())


"""
Operation
"""

g = e + f
# g = torch.add(e,f)

result = torch.empty(5,3)
torch.add(e,f,out=result)

b.add_(e) # in-place

# Indexing
print(b[:,1])


# Resize & Reshape
x = torch.rand(4,4)
y = x.view(16)
z = x.view(-1,8) # set dimension automatically via -1
print(x.size(), y.size(), z.size())

one_value = torch.rand(1)
print(one_value.item()) # if tensor has one value


"""
Numpy
"""

a = torch.ones(5)
b = a.numpy()
a.add_(1) # synchronize with b

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a,1,out=a) # synchronize with b


"""
CUDA
"""

if torch.cuda.is_available():
    device = torch.device("cuda")

    y = torch.ones_like(one_value, device=device)
    x = x.to(device)

    z = x+y
    print(z)
    print(z.to("cpu", torch.double))


