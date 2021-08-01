import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import numpy as np
from random import sample

# Create fake data
nsamples = 10000
train_labels = random.choices([0,1,2,3,4,5,6,7,8,9], k=1)
train_dataset = np.random.rand(100, 34)

train_dataset = []
valid_dataset = []
test_dataset = []

for i in range(nsamples):
    train_dataset.append((np.random.rand(1, 34), random.choices([0,1,2,3,4], k=1)[0]))
    valid_dataset.append((np.random.rand(1, 34), random.choices([0,1,2,3,4], k=1)[0]))
    test_dataset.append((np.random.rand(1, 34), random.choices([0,1,2,3,4], k=1)[0]))

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(34, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

model = MLP()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
mean_train_losses = []
mean_valid_losses = []
valid_acc_list = []
epochs = 15

# Train model
for epoch in range(epochs):
    model.train()
    train_losses = []
    valid_losses = []
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images.float())
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(valid_loader):
            outputs = model(images.float())
            loss = loss_fn(outputs, labels)
            valid_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    mean_train_losses.append(np.mean(train_losses))
    mean_valid_losses.append(np.mean(valid_losses))
    accuracy = 100*correct/total
    valid_acc_list.append(accuracy)
    print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}, valid acc : {:.2f}%'\
         .format(epoch+1, np.mean(train_losses), np.mean(valid_losses), accuracy))

# set the model to inference mode and export
model.eval()
# Input to the model
x = torch.randn(1, 1, 34, requires_grad=True)
out = model(x)

# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "simple_model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})
