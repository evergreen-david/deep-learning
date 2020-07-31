import torch
import pandas as pd

# Load dataset
data = pd.read_csv("synthetic_regression_train.csv")

# get x and y with torch.tensor
x = torch.tensor(data.x1.values, dtype=torch.float32)
y = torch.tensor(data.y.values, dtype=torch.float32)

# program, or function, or the model
def program(inputs, w, b):
    predictions = w * inputs + b
    return predictions

# loss function, measure how current program is good
def loss_fn(predictions, outputs):
    error = predictions - outputs
    squared_error = error ** 2
    return squared_error.mean()

# define the initial w, b with torch.tensor
w = torch.tensor(1., requires_grad=True)
b = torch.tensor(1., requires_grad=True)

# update w for 100 steps
for i in range(100):
    # calculate loss
    loss = loss_fn(program(x, w, b), y)

    # get the gradient, save the gradient value at 'w.grad'
    # if dw > 0 : need to decrease w
    # if dw < 0 : need to increase w
    loss.backward()

    # update w for one step
    with torch.no_grad():
        w -= 0.01 * w.grad
        b -= 0.01 * b.grad

        # let w.grad to be zero,
        # or torch accumulates all gradients
        w.grad.zero_()
        b.grad.zero_()

    print(loss)

