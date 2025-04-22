import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report

from customdataset import CarsAndTrucks

#hyperparameters
num_workers = 0
learning_rate = 1e-3
batch_size = 32
num_epochs = 20 #I will say 20 epochs is enough

#transform
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#loading data
dataset = CarsAndTrucks(csv_file="data.csv", root_dir="data", transform=transforms)

print(int(len(dataset)))

train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
print(f"Dataset length: {len(dataset)}")
print(f"Train set length: {len(train_set)}")
print(f"Test set length: {len(test_set)}")

sample_image, sample_label = dataset[7888]
print(f"Sample image shape: {sample_image.shape}")
print(f"Sample label: {sample_label}")
image, label = train_set[0]
print(image.size())

class_names = ['car', 'truck']

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5) # 12, 124, 124
        self.pool = nn.MaxPool2d(2, 2) # 12, 62, 62
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5) #(24, 58, 58)
        self.fcl = nn.Linear(24*29*29, 120)
        self.fcl2 = nn.Linear(120, 84)
        self.fcl3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fcl(x))
        x = F.relu(self.fcl2(x))
        x = self.fcl3(x)

        return x

net = NeuralNet()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

for epoch in range(num_epochs):
    print(f"training epoch {epoch}...")
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        images, labels = data

        optimizer.zero_grad()

        outputs = net(images)

        loss = loss_func(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Loss: {running_loss / len(train_loader):.4f}")

torch.save(net.state_dict(), "trained_model_20_epochs.pth")
net = NeuralNet()
net.load_state_dict(torch.load("trained_model_20_epochs.pth"))

correct = 0
total = 0

net.eval()

all_predicted = []
all_labels = []

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct +=(predicted == labels).sum().item()

        all_predicted.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total

print(f"accuracy: {accuracy}%")

print("\nDetailed Classification Report:")
print(classification_report(
    all_labels, all_predicted,
    target_names=['Truck', 'Car'],
    zero_division=0
))