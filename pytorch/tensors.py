import torch
import numpy as np

#generating tensors
x = torch.empty(1)
print("empty(1):", x)
x = torch.empty(2, 3)
print("empty(2, 3)", x)
x = torch.rand(5, 3)
print("rand(5, 3): ", x)
x = torch.zeros(5, 3)
print("zeros(5, 3): ", x)

#check size
print(x.shape, x.size())

#datatypes
print(x.dtype)
x = torch.zeros(5, 3, dtype=torch.float16)
print(x)
x = torch.tensor([5.5, 3], requires_grad=True)

x = torch.ones(2, 2)
y = torch.rand(2, 2)
print(x + y)
print(torch.add(x, y))
print(x[:, 0])
print(x[1, 1].item())

#reshape with torch view
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)

#torch to numpy
a = torch.randn(2, 2)
b = a.numpy()
a = np.ones(5)
b = torch.tensor(a)

#gpu support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.rand(2, 2).to(device)
x = torch.rand(2, 2, device=device) #better