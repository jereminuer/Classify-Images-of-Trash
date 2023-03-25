
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import random
import numpy as np
import matplotlib.pyplot as plt


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, padding="same")
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 5, padding="same")
        
        self.fc1 = nn.Linear(in_features = 12*8*8, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 60)
        self.out = nn.Linear(in_features = 60, out_features = 5)
        
    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)
        
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)
        
        t = torch.flatten(t, start_dim = 1)
        t = F.relu(self.fc1(t))
        
        t = F.relu(self.fc2(t))
        output = self.out(t)
        
        return output


with open("/Users/jereminuer/downloads/recycled_32/recycled_32_train.npz", 'rb') as fo:
    data = np.load("/Users/jereminuer/downloads/recycled_32/recycled_32_train.npz")
    x, y = data['x'], data['y']


with open("/Users/jereminuer/downloads/recycled_32/recycled_32_test.npz", 'rb') as fo:
    data = np.load("/Users/jereminuer/downloads/recycled_32/recycled_32_test.npz")
    x1, y1 = data['x'], data['y']


x = x.astype('float32')
x1 = x1.astype('float32')
y = y.astype('int64')
y1 = y1.astype('int64')

x = x / 255
x1 = x1 / 255

train_set = torch.from_numpy(x)
test_set = torch.from_numpy(x1)
label_train = torch.from_numpy(y)
label_test = torch.from_numpy(y1)

train_set.shape


train_set_loader = torch.utils.data.DataLoader([[train_set[i], label_train[i]] for i in range(len(label_train))], batch_size = 64, shuffle=True)
test_set_loader = torch.utils.data.DataLoader([[test_set[i], label_test[i]] for i in range(len(label_test))], batch_size = 64, shuffle=True)


test = next(iter(train_set_loader))
image, label = test

network = Network()
optimizer = optim.Adam(network.parameters(), lr=0.001)


for epoch in range(7):
    total_correct = 0
    denom = 0
    for batch in train_set_loader:
        batch = next(iter(train_set_loader))
        images, labels = batch
        
        predictions = network(images)
        loss = F.cross_entropy(predictions, labels)
        print(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_correct += predictions.argmax(dim=1).eq(labels).sum().item()
        denom += 64
        accuracy = total_correct/denom *100
        print(accuracy)
    print(accuracy)


print('Computer Vision: Types of Trash')
print('Accuracy: ', accuracy)
print('Epochs: 7')





