import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


exec(open("./sprites.py").read())
classes = list(range(0, 13))

# transform to normalise the images to have mean 0 and std dev 1
'''
transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.5, ), (0.5, ))

])
'''
# Loading the data
# Training data
train_inputs = np.reshape(trainingpix, (676, 1, 15, 15)).astype(float)
train_inputs = torch.from_numpy(train_inputs).float()
train_labels = torch.from_numpy(traininglabels[:, 0]).long()
train_labels = train_labels - 1  # targets should be between 0 and 12 (13 classes)
# class indices start at 0 so target should contain indices in the range [0, nb_classes-1].

# Testing data
test_inputs = np.reshape(testingpix, (169, 1, 15, 15))
test_inputs = torch.from_numpy(test_inputs).float()
test_labels = torch.from_numpy(testinglabels[:, 0]).long()
test_labels = test_labels - 1

class dataHandler(Dataset):

    def __init__(self,inputs, labels, transform=None):

        self.inputs = inputs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.inputs)


trainData = dataHandler(inputs=train_inputs, labels=train_labels)
print("Train dataloader for X: ", trainData.inputs.shape)
print("Train dataloader for Y: ", trainData.labels.shape)
testData = dataHandler(test_inputs, test_labels)
print(testData.inputs.shape)
print(testData.labels.shape)

trainloader = DataLoader(dataset=trainData, batch_size=4, shuffle=True)
testloader = DataLoader(dataset=testData, batch_size=4, shuffle=True)


# Convolutional Neural Network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

# Create the network
net = Net()
net.train()

# Define loss function and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the network, time the training
np.random.seed(52)
start = timeit.default_timer()

for epoch in range (25): #loop over dataset 25 times
    running_loss = 0.0
    correct = 0
    total = 0
    # iterate through batches
    for i, data in enumerate(trainloader):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs) # forward pass
        loss = criterion(outputs, labels) # compute loss
        loss.backward() # backwards pass
        optimizer.step() # compute gradient and update weights

        running_loss += loss.item()

        # training accuracy
        _, predicted = torch.max(outputs, 1)  # choose the class with max probability
        total += labels.size(0)  # add to total count
        correct += (predicted == labels).sum().item()  # add to correct count

        # print statistics
        print('[%5d] loss: %.3f' % (i + 1, loss), 'Training Accuracy: %d %%' % (100 * correct / total), end='\r',
              flush=True)

stop = timeit.default_timer()
timeTaken = stop - start

print('\nFinished Training')
print('Training time: ', timeTaken)


# Test the network

net.eval()  # set the network to evaluation mode

np.random.seed(54)
test_correct = 0
test_total = 0
with torch.no_grad():
    # for each batch
    for data in testloader:
        test_inputs, test_labels = data
        test_outputs = net(test_inputs)  # predict the probability of each class
        _, test_predicted = torch.max(test_outputs, 1)  # choose the class with max probability
        test_total += test_labels.size(0)  # add to total count
        test_correct += (test_predicted == test_labels).sum().item()  # add to correct count

print('Accuracy of the network on the 169 test images: %d %%' % (
        100 * test_correct / test_total))
