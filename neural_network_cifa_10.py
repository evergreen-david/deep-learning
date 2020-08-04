import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import wandb
import torchvision
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--hidden_dim', default=100, type=int)
parser.add_argument('--batch_sze', default=256, type=int)

dataset_train = torchvision.datasets.CIFAR10(root = './cifar10',  download = True, train=True, transform=torchvision.transforms.ToTensor() )
dataset_valid = torchvision.datasets.CIFAR10(root = './cifar10',  download = True, train=True, transform=torchvision.transforms.ToTensor() )

dataloader_train = DataLoader(dataset_train, batch_size = 128, shuffle = True)
dataloader_valid = DataLoader(dataset_valid, batch_size = 128, shuffle = True)

wandb.config.update({
    'learning_rate': 0.01,
	'batch_size': 512,
	'hidden_dim': 100
})

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.component_1 = nn.Linear(3*32*32, wandb.config.hidden_dim ) 
        self.component_2 = nn.ReLU()
        self.component_3 = nn.Linear(50,10)

    def forward(self, inputs):
        # dim = N, 3, 32, 32
        inputs = inputs.view(-1, 3*32*32)
        predictions = self.component_1(x)
        predictions = self.component_2(predictions)
        predictions = self.component_3(predictions)

        return predictions

net = MLP()
optimizer = optim.SGD(net.parameters(), lr = 0.01) 
loss_fn = nn.CrossEntropyLoss()

i = 0
while True:
    training_loss = 0
    # training loop
    for x,y in dataloader_train:

        predictions = net( x )
        loss = loss_fn( predictions, y )
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        training_loss += loss.item()
    
    training_loss /= len(dataloader_train)


    validation_loss = 0
    # for validation
    for x,y in dataloader_valid:
        with torch.no_grad()
            predictions = net( x )
            loss = loss_fn( predictions, y )
        
        validation_loss += loss.item()

    validation_loss += len(dataloader_valid)

    wandb.log({
        "Training loss": loss.item(),
        "Validation loss": val_loss.item()})

    # if current validation loss is lower than lowest validation loss:
    # update the lowest validation loss, and best parameters
    if validation_loss < lowest_valid_loss:
        torch.save(net.state_dict(), "./net.pth")
        lowest_valid_loss = val_loss
        counter = 0
    else:
        counter += 1

    if counter == early_stopping_tolerance:
            break

    i += 1    
