# autograd package provides automatic differentiation for 
# all operations on tensors. torch.autograd is an engine for comp.
# vector jacobian product (it also applies chain rule)

import torch
import numpy as np

# all the opartions are tracked
x = torch.randn(3, requires_grad=True)
y = x + 2
z = y * y * 3
z.backward()
print(x.grad) #dz/dx

#there're some situations when you dont want to use grad then you 
# can use x.requires_grad(False), x.detach()
b = x.detach()
print(x.requires_grad)
print(b.requires_grad)
x.requires_grad_(False)
print(x.requires_grad)

