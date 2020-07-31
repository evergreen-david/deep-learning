import pandas as pd

# Load dataset
data = pd.read_csv("synthetic_regression_train.csv")

# get x and y
x = data.x1.values
y = data.y.values

# program, or function, or the model
def program(inputs, w):
    predictions = w * inputs
    return predictions

# loss function, measure how current program is good
def loss_fn(predictions, outputs):
    error = predictions - outputs
    squared_error = error ** 2
    return squared_error.mean()

# define the initial w
w = 1

# update w for 100 steps
for i in range(100):
    # w_new : slightly bigger than w
    w_new = w + 0.00001

    # calculate loss
    loss = loss_fn(program(x, w), y)
    loss_new = loss_fn(program(x, w_new), y)

    # get the gradient
    # if dw > 0 : need to decrease w
    # if dw < 0 : need to increase w
    dw = (loss_new - loss) / (w_new - w)

    # update w for one step
    w = w - 0.01 * dw

    print(loss)

