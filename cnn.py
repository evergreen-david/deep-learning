import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms, datasets

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

EPOCHS = 40
BATCH_SIZE = 64
EPOCHS = 40
BATCH_SIZE = 64

train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./.data',
        train=True,
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize( (0.1307,), (0.3081,) )
        ])
    ),
    batch_size = BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./.data',
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize( (0.1307, ), (0.3081,) )
        ])
    ),
    batch_size = BATCH_SIZE, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1=nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)

    def forward(self, x):
        x = F.relu( F.max_pool2d(self.conv1(x), 2))
        x = F.relu( F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu( self.fc1(x) )
        x = F.dropout( x, training = self.training)
        x = self.fc2()
        return F.log_softmax( x, dim=1)

model = CNN().to(DEVICE)
optimizer = optim.SGD( model.parameters(),lr = 0.01, momentum = 0.5)












