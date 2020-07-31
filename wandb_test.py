# (ë¦¬ëˆ…ìŠ¤ í„°ë¯¸ë„ ëª…ë ¹ì–´ ì‚¬ìš©)
# wandb ì„¤ì¹˜
# !pip install wandb -q

# login
# !wandb login

# initialization
# !wandb init


import torch
import wandb
import pandas as pd


# initialize wandb

# set configuration parameter
wandb.config.update({'learning_rate': 0.01})


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
b = torch.tensor(1., requires_grad=True)

# let's find the lowest valid loss
lowest_valid_loss = float('inf')

# see if lowest validation error is updated or not
# if it is not updated for enough iterations, let's stop training
early_stopping_tolerance = 100
counter = 0

# update w for 1000 steps
i = 0
while True:
		# calculate loss
		loss = loss_fn(program(train_x, w, b), train_y)

		# get the gradient, save the gradient value at 'w.grad'
		# if dw > 0 : need to decrease w
		# if dw < 0 : need to increase w
		loss.backward()

		# update w for one step
		# use wandb parameter : learning_rate
		with torch.no_grad():
				w -= wandb.config.learning_rate * w.grad
				b -= wandb.config.learning_rate * b.grad

				# let w.grad to be zero,
				# or torch accumulates all gradients
				w.grad.zero_()
				b.grad.zero_()

		# let's see validation loss
		val_loss = loss_fn(program(valid_x, w, b), valid_y)

		# log loss to wandb
		wandb.log({"Training loss": loss.item(),
							 "Validation loss": val_loss.item()})

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


