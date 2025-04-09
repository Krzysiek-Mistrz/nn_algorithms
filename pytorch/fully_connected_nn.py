import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784    #bo img jest 28x28
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

#zbiory danych
train_dataset = torchvision.datasets.MINST(root='./data',
                                           train=True,
                                           transforms=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MINST(root='./data',
                                           train=False,
                                           transforms=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

examples = iter(test_loader)
example_data, example_targets = examples.next()

#fully connected nn with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, *args, **kwargs):
        super(NeuralNet).__init__(*args, **kwargs)
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

#if model is pushed to device the tensors needs to be pushed as well
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

#loss & optimizer
#also crossentropyloss needs raw values at the end thats why we 
# dont use softmax at the end
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        #calc
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if not (i+1) % 100:
            w, _ = model.lin.weight, model.lin.bias
            print(f"epoch {epoch + 1}/{num_epochs}, step {i+1}/{n_total_steps}, loss {loss.item()}, w {w.item()}")

with torch.no_grad():
    n_correct = 0
    n_samples = len(test_loader.dataset)
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()
    acc = n_correct / n_samples
    print(f"accuracy of network {n_samples} test images {100*acc}")