import torch
import numpy as np

x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8, 10, 12, 14, 16], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return w * x

def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

x_test = 5.0
print(f"prediction before training: {x_test} = {forward(x_test).item() : .3f}")

learning_rate = 0.01
epochs = 100

for epoch in range(epochs):
    y_pred = forward(x)
    l = loss(y, y_pred)
    l.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
    w.grad.zero_()  #emptying the gradients
    if not ((epoch + 1) % 10):
        print(f"epoch {epoch + 1}: w = {w.item() : .3f} loss = {l.item() : .3f}")

