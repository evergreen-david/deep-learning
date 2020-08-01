import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Load dataset
data = pd.read_csv("synthetic_regression_train.csv")

# get x and y with torch.tensor
x = torch.tensor(data[['x1','x2']].values, dtype=torch.float32)
y = torch.tensor(data.y.values, dtype=torch.float32).view(-1,1)

component_1 = nn.Linear(2,3)
component_2 = nn.ReLU()
component_3 = nn.Linear(3,1)

optimizer_1 = optim.SGD(component_1.parameters(), lr = 0.01) 
optimizer_3 = optim.SGD(component_3.parameters(), lr = 0.01) 

loss_fn = nn.MSELoss()

# update w for 100 steps
for i in range(100):

    predictions = component_1(x)
    predictions = component_2(predictions)
    predictions = component_3(predictions)

    loss = loss_fn(predictions, y)
    
    loss.backward()
    
    optimizer_1.step()
    optimizer_3.step()

    # gradient to zero
    optimizer_1.zero_grad()
    optimizer_3.zero_grad()

    print(loss)
