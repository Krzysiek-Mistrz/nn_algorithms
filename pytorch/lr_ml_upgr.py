import torch
import torch.nn as nn

x = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8], [10], [12], [14], [16]], dtype=torch.float32)
x_test = torch.tensor([[5]], dtype=torch.float32)  # Poprawione na 2D
n_samples, n_features = x.shape

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim) 

    def forward(self, x):
        return self.lin(x)

input_size, output_size = n_features, n_features
model = LinearRegression(input_size, output_size)

print(f"Prediction before training: {x_test.item()} = {model(x_test).item():.3f}")

learning_rate = 0.01
epochs = 100

loss = nn.MSELoss()
#parametry tj.wagi i bias
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    y_pred = model(x)
    l = loss(y, y_pred)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch + 1) % 10 == 0:
        w, b = model.lin.weight, model.lin.bias
        print(f"Epoch {epoch + 1}: w = {w.item():.3f}, loss = {l.item():.3f}")

print(f"Prediction after training: {x_test.item()} = {model(x_test).item():.3f}")
