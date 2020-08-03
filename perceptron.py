import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Load dataset
data = pd.read_csv("synthetic_regression_train.csv")

# get x and y with torch.tensor
x = torch.tensor(data[['x1','x2']].values, dtype=torch.float32)
y = torch.tensor(data.y.values, dtype=torch.float32).view(-1,1)

class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()

        self.component_1 = nn.Linear(2,3)
        self.component_2 = nn.ReLU()
        self.component_3 = nn.Linear(3,1)

    def forward(self, inputs):
        predictions = self.component_1(x)
        predictions = self.component_2(predictions)
        predictions = self.component_3(predictions)

        return predictions

net = Perceptron()
optimizer = optim.SGD(net.parameters(), lr = 0.01) 
loss_fn = nn.MSELoss()

# update w for 100 steps
for i in range(100):
    predictions = net(x)
    loss = loss_fn(predictions, y)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(loss)
