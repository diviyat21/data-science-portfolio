import timeit
import numpy as np
import torch
import torch.nn as nn
from torch import tanh as torch_tanh
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image


exec(open("./sprites.py").read())
plt.close()
classes = list(range(0, 13))

# Loading the data
# Training data
train_inputs = np.reshape(trainingpix, (676, 1, 15, 15))
train_inputs = torch.from_numpy(train_inputs).float()
train_labels = torch.from_numpy(traininglabels).long()
train_labels = train_labels - 1  # targets should be between 0 and 12 (13 classes)
# class indices start at 0 so target should contain indices in the range [0, nb_classes-1].



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


trainloader = DataLoader(dataset=trainData, batch_size=4, shuffle=True)


# Autoencoder
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        # images are 15x15, single channel.

        # encoding
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)  # halve image size
        self.conv2 = nn.Conv2d(16, 8, 3)

        # decoding
        self.iconv1 = nn.ConvTranspose2d(8, 16, 6)
        self.iconv2 = nn.ConvTranspose2d(16, 8, 6)
        self.iconv3 = nn.ConvTranspose2d(8, 1, 4)

    def encoder(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return (x)

    def decoder(self, x):
        x = F.relu(self.iconv1(x))
        x = F.relu(self.iconv2(x))
        x = torch_tanh(self.iconv3(x))
        return (x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Training
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 15, 15)
    return x
    ## output comes from the decoder



num_epochs = 30
learning_rate = 1e-3
model = autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

np.random.seed(52)
start = timeit.default_timer()
for epoch in range(num_epochs):
    # Iterate num_epoch number of times.
    for data in trainloader:
        # data contains the number of images specified in batch_size
        # this will loop until all 60000 MNIST images are provided to the AE
        inputs, _ = data
        # pass the images through the AE
        output = model(inputs)
        # compare the images to the output
        loss = criterion(output, inputs)
        # update the gradients to minimise the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch [{}/{}], loss:{}'.format(epoch + 1, num_epochs, loss.item()))
    if epoch % 10 == 0:
        imageGrid = to_img(output.data)
        save_image(imageGrid, './imageGrid_{}.png'.format(epoch))

stop = timeit.default_timer()
timeTaken = stop - start

print('\nFinished Training')
print('Training time: ', timeTaken)

# original input example
imageGrid = to_img(train_inputs[0])
save_image(imageGrid, './original.png')

