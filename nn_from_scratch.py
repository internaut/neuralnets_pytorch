"""
Neural network from scratch with PyTorch. Implements a simple two-layer sequential neural network that is trained and
evaluated on the MNIST dataset.

Roughly follows the TensorFlow example implementation in *Deep Learning with Python* p. 63 by François Chollet, but
adapts it to PyTorch.

Nov. 2022

author: Markus Konrad <post@mkonrad.net>
"""

#%%

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#%%

# constants
IMG_W = 28
IMG_H = 28
N_LABELS = 10

# hyperparameters
BATCH_SIZE = 64
DENSE_UNITS = 512
EPOCHS = 20
LEARNING_RATE = 0.001

#%% load data

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

#%% torch module implementations


class Flatten(nn.Module):
    """A "flatten" layer: flattens all input except for the first axis."""
    def forward(self, x):
        return x.flatten(start_dim=1)   # don't flatten the first axis: this is the axis for the individual samples

    def parameters(self, recurse=True): return []


class Dense(nn.Module):
    """A dense layer with parameters W and b performing a linear transformation `xW + b`."""
    def __init__(self, input_size, units):
        super().__init__()

        w_shape = (input_size, units)
        # initialize weights with random values
        # scale by inverse of units -- otherwise output will get too large later on and Softmax will produce inf values
        self.W = torch.rand(w_shape, dtype=torch.float, requires_grad=True) * (1.0 / units)
        # somehow necessary, see https://discuss.pytorch.org/t/grad-is-none-even-when-requires-grad-true/29826/2
        self.W.retain_grad()
        # bias
        self.b = torch.zeros(w_shape[1:], requires_grad=True)

    def forward(self, x):
        """
        Perform linear transformation `xW + b`.

        If `x` is of shape `(M, N)` (M samples in the batch, each with N features) and the layer is set up to have U
        units (i.e. W is of shape `(N, U)` and b is of length `U`), the output will be of shape `(M, U)`.
        """
        return torch.matmul(x, self.W) + self.b

    def parameters(self, recurse=True): return [self.W, self.b]


class ReLU(nn.Module):
    """A rectified linear unit activation performing `max(x, 0)` on input `x`."""
    def forward(self, x):
        return torch.max(x, torch.tensor(0.0))

    def parameters(self, recurse=True): return []


class Softmax(nn.Module):
    """A softmax activation transforming the input `x` so that the output is in [0, 1] range and sums up to 1."""
    def forward(self, x):
        e = torch.exp(x)
        return e / torch.sum(e)

    def parameters(self, recurse=True): return []


class Sequential(nn.Module):
    """Module that implements a sequential feed-forward network."""
    def __init__(self, modules):
        super().__init__()
        self.modules = modules

    def forward(self, x):
        """Sequentially calculate the output of each layer and pass it on to the next layer."""
        for mod in self.modules:
            x = mod(x)
        return x

    def parameters(self, recurse=True):
        """Collect parameters from all layers of this network."""
        params = []
        for mod in self.modules:
            params.extend(mod.parameters())
        return params

#%%


def crossentropy_singlelabel(y_true, y_pred, clip=1e-15):
    """
    Cross-entropy loss for single label classification (i.e. exactly one label per sample).

    `y_true` is a vector of length N where an integer in range [0, C] denotes the correct class for each of the
    N samples. `y_pred` is a matrix of shape (N, C) where each row represents a classification probability across the C
    classes.

    For each of the N samples, we calculate the loss as `-log(y_c)` where `y_c` is the estimated probability for the
    known correct class `c`. If it's high, the negative log will be low, i.e. the loss will be low. If it's low, the
    negative log and hence the loss will be high.
    """
    assert y_true.ndim == 1
    assert y_pred.ndim == 2
    assert len(y_true) == len(y_pred)
    torch.clip_(y_pred, clip, 1 - clip)
    # take the predicted scores from the indices of the known true outcomes
    p = torch.take_along_dim(y_pred, y_true[:, None], dim=1)
    return -torch.log(p)


def accuracy_singlelabel(y_true, y_pred):
    """
    Accuracy for single label classification. Simply gives the fraction of correctly classified samples in `y_pred`
    (as number of matches to `y_true`).
    """
    matches = (torch.argmax(y_pred, dim=1) == y_true).clone().detach().to(torch.float)
    return torch.mean(matches)


def train_step(model, x, y):
    """
    Training step: For a batch of features `x` and labels `y` do the following:
    (1) perform the forward calculations, i.e. predict the outcomes `pred` given `x`
    (2) calculate the loss between `pred` and known outcomes `y`
    (3) perform back propagation to obtain the gradients of the loss w.r.t. to all parameters
    (4) update the parameters by moving them in the opposite direction of the gradients in order to minimize the loss
    """
    assert len(x) == len(y)

    # reset parameter gradients to 0
    for param in model.parameters():
        param.requires_grad_(True)   # (re)attach to computation graph
        if param.grad is not None:
             param.grad.data = torch.zeros_like(param.grad.data)

    # (1)
    pred = model(x)
    # (2)
    loss = torch.mean(crossentropy_singlelabel(y, pred))

    # (3)
    loss.backward()

    # (4)
    for param in model.parameters():
        # update parameter p with learning rate r and gradient g
        # p := p - r * g
        param.data.add_(-LEARNING_RATE * param.grad.data)
        param.detach_()   # detach from computation graph

    # return loss as scalar
    return loss.item()


def test_step(model, x, y):
    """
    Testing step: measure accuracy for a batch of predictions `model(x)` and ground truth `y`.
    """
    assert len(x) == len(y)

    with torch.no_grad():   # no need to retain computation graph for forward-only calculations
        pred = model(x)
        batch_acc = accuracy_singlelabel(y, pred)

    # return batch accuracy as scalar
    return batch_acc.item()



#%% define the model

model = Sequential((
    Flatten(),
    Dense(IMG_H * IMG_W, DENSE_UNITS),
    ReLU(),
    Dense(DENSE_UNITS, N_LABELS),
    Softmax()
))

#%% train and test the model

for i in range(1, EPOCHS+1):
    print(f"epoch {i}")

    # perform training step on each batch
    for b, (x, y) in enumerate(train_dataloader, 1):
        loss = train_step(model, x, y)

        if b == 1 or b % 100 == 0:
            print(f"> batch {b}: loss = {loss:.4f}")

    # perform testing step on each batch
    acc_sum = 0.0
    n_batches = 0
    for x, y in test_dataloader:
        acc_sum += test_step(model, x, y)
        n_batches += 1
    mean_acc = acc_sum / n_batches

    print(f"\nmean acc. = {mean_acc:.4f}\n\n")

print("done.")
