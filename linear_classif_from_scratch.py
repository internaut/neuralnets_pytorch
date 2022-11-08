"""
Linear classifier from scratch with PyTorch. Implements a simple linear model used for binary classification in 2D
space on synthetic data.

Roughly follows the TensorFlow example implementation in *Deep Learning with Python* p. 79 by François Chollet, but
adapts it to PyTorch.

Nov. 2022

author: Markus Konrad <post@mkonrad.net>
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

#%% constants

N_SAMPLES_PER_CLASS = 1000

# hyperparameters

N_TRAIN_STEPS = 40
LEARNING_RATE = 0.1

#%% generate synthetic data

# generate features for two classes
# they have different means but same covariance
feat_0 = np.random.multivariate_normal(
    mean=[0, 3],
    cov=[[1, 0.5],[0.5, 1]],
    size=N_SAMPLES_PER_CLASS
)

feat_1 = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0.5],[0.5, 1]],
    size=N_SAMPLES_PER_CLASS
)

# concatenate
inputs = np.vstack((feat_0, feat_1)).astype(np.float32)

# set targets, i.e. class membership
targets = np.vstack((
    np.zeros((N_SAMPLES_PER_CLASS, 1), dtype="float32"),
    np.ones((N_SAMPLES_PER_CLASS, 1), dtype="float32"))
)

del feat_0, feat_1

#%%

def linear_model(x, W, b):
    """Defines a linear model `y = xW + b`."""
    return torch.matmul(x, W) + b


def mse(y_true, y_pred):
    """Mean squared error loss function."""
    return torch.mean(torch.square(y_pred - y_true))


def train_step(x, y, W, b):
    """
    Training step: For features `x` and labels `y` do the following:
    (1) perform the forward calculations, i.e. predict the outcomes `y_pred` given `x` and model parameters `W`, `b`
    (2) calculate the loss between `y_pred` and known outcomes `y`
    (3) perform back propagation to obtain the gradients of the loss w.r.t. to `W` and `b`
    (4) update the parameters by moving them in the opposite direction of the gradients in order to minimize the loss
    """
    assert len(x) == len(y)

    # reset parameter gradients to 0
    for param in (W, b):
        param.requires_grad_(True)   # (re)attach to computation graph
        if param.grad is not None:
             param.grad.data = torch.zeros_like(param.grad.data)

    # (1)
    y_pred = linear_model(x, W, b)
    # (2)
    loss = mse(y, y_pred)
    # (3)
    loss.backward()

    # (4)
    for param in (W, b):
        # update parameter p with learning rate r and gradient g
        # p := p - r * g
        param.data.add_(-LEARNING_RATE * param.grad.data)
        param.detach_()   # detach from computation graph

    # return the loss as scalar
    return loss.item()


def plot_data(x, y, title, W=None, b=None):
    """
    Plot the 2D points given in `x` along with their class labels `y`. If `W` and `b` are given, also plot the
    "separation line" produced by the linear model.
    """
    if W is not None and b is not None:
        # plot the separation line
        x_min = np.min(x[:, 0]) - 0.5
        x_max = np.max(x[:, 0]) + 0.5
        line_x = np.linspace(x_min, x_max)
        W_np = W.numpy()

        # parameters are given for the linear model: y = w1x1 + w2x2 + b
        # we use a threshold of y = 0.5 for the binary classification; hence we can rearrange the eq. like follows:
        #   0.5 = w1x1 + w2x2 + b
        #   x2 = -w1/w2 x1 + (0.5 - b) / w2
        # this gives a standard line equation y=mx+n that we can plot
        line_y = - W_np[0] / W_np[1] * line_x + (0.5 - b.item()) / W_np[1]
        plt.plot(line_x, line_y, "-r")

        # always draw in the same limits
        plt.xlim(x_min, x_max)
        plt.ylim(np.min(x[:, 1]) - 0.5, np.max(x[:, 1]) + 0.5)

    scat = plt.scatter(x[:, 0], x[:, 1], c=y[:, 0])
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.title(title)
    plt.legend(*scat.legend_elements(), title="class")
    plt.show()

#%% plot input data

plot_data(inputs, targets, "input data")

#%% train the model and plot how it learns the separation line

# initialize model parameters with random values for W and b = 0
W = torch.rand((2, 1), requires_grad=True)
b = torch.zeros((1,), requires_grad=True)

# convert inputs and targets to tensors
x = torch.tensor(inputs)
y = torch.tensor(targets)

# train the model
# we don't use batchwise training here but train on the full data
for step in range(1, N_TRAIN_STEPS + 1):
    loss = train_step(x, y, W, b)
    if step == 1 or step % 10 == 0:
        plot_data(inputs, targets, f"model at step {step} (loss = {loss:.4f})", W, b)
    print(f"Loss at step {step}: {loss:.4f}")
