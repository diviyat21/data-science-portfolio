import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

exec(open("./sprites.py").read())
x_classes = list(range(0, 13))
y_classes = list(range(0, 13))
z_classes = list(range(0, 5))

# Loading the data
# Training data
train_inputs = np.reshape(trainingpix, (676, 1, 15, 15)).astype(float)
train_inputs = torch.from_numpy(train_inputs).float()

x_train_labels = torch.from_numpy(traininglabels[:, 0]).long()
x_train_labels = x_train_labels - 1

y_train_labels = torch.from_numpy(traininglabels[:, 1]).long()
y_train_labels = y_train_labels - 1

z_train_labels = torch.from_numpy(traininglabels[:, 2]).long()

# Testing data
test_inputs = np.reshape(testingpix, (169, 1, 15, 15))
test_inputs = torch.from_numpy(test_inputs).float()

x_test_labels = torch.from_numpy(testinglabels[:, 0]).long()
test_labels = x_test_labels - 1

y_test_labels = torch.from_numpy(testinglabels[:, 1]).long()
y_test_labels = y_test_labels - 1

z_test_labels = torch.from_numpy(testinglabels[:, 2]).long()


class dataHandler(Dataset):

    def __init__(self, inputs, labels, transform=None):
        self.inputs = inputs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.inputs)


x_trainData = dataHandler(inputs=train_inputs, labels=x_train_labels)
x_testData = dataHandler(test_inputs, x_test_labels)

y_trainData = dataHandler(inputs=train_inputs, labels=y_train_labels)
y_testData = dataHandler(test_inputs, y_test_labels)

z_trainData = dataHandler(inputs=train_inputs, labels=z_train_labels)
z_testData = dataHandler(test_inputs, z_test_labels)

x_trainloader = DataLoader(dataset=x_trainData, batch_size=4, shuffle=True)
x_testloader = DataLoader(dataset=x_testData, batch_size=4, shuffle=True)

y_trainloader = DataLoader(dataset=y_trainData, batch_size=4, shuffle=True)
y_testloader = DataLoader(dataset=y_testData, batch_size=4, shuffle=True)

z_trainloader = DataLoader(dataset=z_trainData, batch_size=4, shuffle=True)
z_testloader = DataLoader(dataset=z_testData, batch_size=4, shuffle=True)


# Convolutional Neural Network

class NetA(nn.Module):
    def __init__(self):
        super(NetA, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(16 * 2 * 2, 40)
        self.fc2 = nn.Linear(40, 24)
        # final layer outputs 13 classes
        self.fc3 = nn.Linear(24, 13)

    def forward(self, x):
        # x begins as 15x15, 1 channel
        x = self.pool(F.relu(self.conv1(x)))
        # after convolution with 3x3 kernel, image is 13x13, then pooled to 6x6
        x = self.pool(F.relu(self.conv2(x)))
        # after convolution with 3x3 kernel, image is 4x4, then pooled to 2x2
        # 16 outputs of size 2x2
        x = x.view(-1, 16 * 2 * 2)
        # flatten 16x2x2 tensor to 16x2x2=64 length vector

        # 64 dimensions to 40
        x = F.relu(self.fc1(x))
        # 40 dimensions to 24
        x = F.relu(self.fc2(x))
        # 24 dimensions to 13
        x = self.fc3(x)
        return x


class NetB(nn.Module):
    def __init__(self):
        super(NetB, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(16 * 2 * 2, 40)
        self.fc2 = nn.Linear(40, 24)
        # final layer outputs 13 classes
        self.fc3 = nn.Linear(24, 13)

    def forward(self, x):
        # x begins as 15x15, 1 channel
        x = self.pool(F.relu(self.conv1(x)))
        # after convolution with 3x3 kernel, image is 13x13, then pooled to 6x6
        x = self.pool(F.relu(self.conv2(x)))
        # after convolution with 3x3 kernel, image is 4x4, then pooled to 2x2
        # 16 outputs of size 2x2
        x = x.view(-1, 16 * 2 * 2)
        # flatten 16x2x2 tensor to 16x2x2=64 length vector

        # 64 dimensions to 40
        x = F.relu(self.fc1(x))
        # 40 dimensions to 24
        x = F.relu(self.fc2(x))
        # 24 dimensions to 13
        x = self.fc3(x)
        return x


class NetC(nn.Module):
    def __init__(self):
        super(NetC, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(16 * 2 * 2, 40)
        self.fc2 = nn.Linear(40, 24)
        # final layer outputs 13 classes
        self.fc3 = nn.Linear(24, 5)

    def forward(self, x):
        # x begins as 15x15, 1 channel
        x = self.pool(F.relu(self.conv1(x)))
        # after convolution with 3x3 kernel, image is 13x13, then pooled to 6x6
        x = self.pool(F.relu(self.conv2(x)))
        # after convolution with 3x3 kernel, image is 4x4, then pooled to 2x2
        # 16 outputs of size 2x2
        x = x.view(-1, 16 * 2 * 2)
        # flatten 16x2x2 tensor to 16x2x2=64 length vector

        # 64 dimensions to 40
        x = F.relu(self.fc1(x))
        # 40 dimensions to 24
        x = F.relu(self.fc2(x))
        # 24 dimensions to 13
        x = self.fc3(x)
        return x


# Create the networks
netA = NetA()
netA.train()
netB = NetB()
netB.train()
netC = NetC()
netC.train()

# Define loss function and optimiser
criterion = nn.CrossEntropyLoss()
x_optimizer = optim.SGD(netA.parameters(), lr=0.001, momentum=0.9)
y_optimizer = optim.SGD(netB.parameters(), lr=0.001, momentum=0.9)
z_optimizer = optim.SGD(netC.parameters(), lr=0.001, momentum=0.9)

# Train the network, time the training
np.random.seed(52)
start = timeit.default_timer()

for epoch in range(25):  # loop over dataset 25 times
    x_running_loss = 0.0
    x_correct = 0
    x_total = 0
    y_running_loss = 0.0
    y_correct = 0
    y_total = 0
    z_running_loss = 0.0
    z_correct = 0
    z_total = 0

    # iterate through batches
    for i, (x_data, y_data,z_data) in enumerate(zip(x_trainloader, y_trainloader,z_trainloader)):
        # get the inputs
        x_inputs, x_labels = x_data
        y_inputs, y_labels = y_data
        z_inputs, z_labels = z_data

        # zero the parameter gradients
        x_optimizer.zero_grad()
        y_optimizer.zero_grad()
        z_optimizer.zero_grad()

        # forward + backward + optimize
        x_outputs = netA(x_inputs)  # forward pass
        x_loss = criterion(x_outputs, x_labels)  # compute loss

        y_outputs = netB(y_inputs)
        y_loss = criterion(y_outputs, y_labels)

        z_outputs = netC(z_inputs)
        z_loss = criterion(z_outputs, z_labels)

        x_loss.backward()  # backwards pass
        y_loss.backward()
        z_loss.backward()
        x_optimizer.step()  # compute gradient and update weights
        y_optimizer.step()
        z_optimizer.step()

        x_running_loss += x_loss.item()
        y_running_loss += y_loss.item()
        z_running_loss += z_loss.item()

        # training accuracy
        _, x_predicted = torch.max(x_outputs, 1)  # choose the class with max probability
        _, y_predicted = torch.max(y_outputs, 1)
        _, z_predicted = torch.max(z_outputs, 1)
        x_total += x_labels.size(0)  # add to total count
        y_total += y_labels.size(0)
        z_total += z_labels.size(0)

        x_correct += (x_predicted == x_labels).sum().item()  # add to correct count
        y_correct += (y_predicted == y_labels).sum().item()
        z_correct += (z_predicted == z_labels).sum().item()

        # print statistics
        print('[%5d] x loss: %.3f' % (i + 1, x_loss),'y loss: %.3f' % (y_loss),'z loss: %.3f' % (z_loss),
              'Total Training Accuracy: %d %%' % (
                    100 * (x_correct + y_correct + z_correct) / (x_total + y_total + z_total)), end='\r',
              flush=True)

stop = timeit.default_timer()
timeTaken = stop - start

print('\nFinished Training')
print('Training time: ', timeTaken)


# Test the network

netA.eval()  # set the networks to evaluation mode
netB.eval()
netC.eval()

np.random.seed(54)
test_correct = 0
test_total = 0

x_test_correct = 0
x_test_total = 0
y_test_correct = 0
y_test_total = 0
z_test_correct = 0
z_test_total = 0

with torch.no_grad():
    # for each batch
    for x_data,y_data,z_data in zip(x_testloader, y_testloader,z_testloader):
        x_test_inputs, x_test_labels = x_data
        y_test_inputs, y_test_labels = y_data
        z_test_inputs, z_test_labels = z_data

        x_test_outputs = netA(x_test_inputs)  # predict the probability of each class
        y_test_outputs = netB(y_test_inputs)
        z_test_outputs = netC(z_test_inputs)
        _, x_test_predicted = torch.max(x_test_outputs, 1)  # choose the class with max probability
        _, y_test_predicted = torch.max(y_test_outputs, 1)
        _, z_test_predicted = torch.max(z_test_outputs, 1)
        x_test_total += x_test_labels.size(0)  # add to total count
        y_test_total += y_test_labels.size(0)
        z_test_total += z_test_labels.size(0)
        x_test_correct += (x_test_predicted == x_test_labels).sum().item()  # add to correct count
        y_test_correct += (y_test_predicted == y_test_labels).sum().item()
        z_test_correct += (z_test_predicted == z_test_labels).sum().item()

print('Accuracy of the network on the 169 test images: %d %%' % (
        100 * (x_test_correct+y_test_correct+z_test_correct) / (x_test_total+y_test_total+z_test_total)))

