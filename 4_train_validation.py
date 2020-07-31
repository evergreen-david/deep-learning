import torch
import pandas as pd


# Load dataset
train = pd.read_csv("synthetic_regression_train.csv")
valid = pd.read_csv("synthetic_regression_valid.csv")

# get train data
train_x = torch.tensor(train.x1.values, dtype=torch.float32)
train_y = torch.tensor(train.y.values, dtype=torch.float32)

# get validation data
valid_x = torch.tensor(valid.x1.values, dtype=torch.float32)
valid_y = torch.tensor(valid.y.values, dtype=torch.float32)

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
b = torch.tensor(-2., requires_grad=True)

# let's find the lowest valid loss
lowest_valid_loss = float('inf')

# see if lowest validation error is updated or not
# if it is not updated for enough iterations, let's stop training
early_stopping_tolerance = 100
counter = 0

# update until early_stopped
i = 0
while True:
    # calculate loss
    loss = loss_fn(program(train_x, w, b), train_y)

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

    # let's see validation loss
    val_loss = loss_fn(program(valid_x, w, b), valid_y)

    # if current validation loss is lower than lowest validation loss:
    # update the lowest validation loss, and best parameters
    if val_loss < lowest_valid_loss:
        i_best = i
        w_best = w
        b_best = b
        lowest_valid_loss = val_loss
        counter = 0
    else:
        counter += 1

    if counter == early_stopping_tolerance:
        break

    i += 1

    print(f"iteration: {i}  |  train loss: {loss}  |  validation loss: {val_loss}")

