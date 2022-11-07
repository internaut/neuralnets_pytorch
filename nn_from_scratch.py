import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#%%

IMG_W = 28
IMG_H = 28

BATCH_SIZE = 64
DENSE_UNITS = 512
EPOCHS = 20
LEARNING_RATE = 0.001

#%%

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

#%%


class Flatten(nn.Module):
    def forward(self, x):
        return x.flatten(1)

    def parameters(self, recurse=True): return []


class Dense(nn.Module):
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
        y = torch.matmul(x, self.W) + self.b
        return y

    def parameters(self, recurse=True): return [self.W, self.b]


class ReLU(nn.Module):
    def forward(self, x):
        return torch.max(x, torch.tensor(0.0))

    def parameters(self, recurse=True): return []


class Softmax(nn.Module):
    def forward(self, x):
        e = torch.exp(x)
        return e / torch.sum(e)

    def parameters(self, recurse=True): return []


class Sequential(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules = modules

    def forward(self, x):
        for mod in self.modules:
            x = mod(x)
        return x

    def parameters(self, recurse=True):
        params = []
        for mod in self.modules:
            params.extend(mod.parameters())
        return params

#%%


def crossentropy(y_true, y_pred):
    assert y_true.ndim == 1
    assert y_pred.ndim == 2
    assert len(y_true) == len(y_pred)
    torch.clip_(y_pred, 1e-15, 1 - 1e-15)
    p = torch.take_along_dim(y_pred, y_true[:, None], dim=1)
    return -torch.log(p)


def accuracy(y_true, y_pred):
    matches = (torch.argmax(y_pred, dim=1) == y_true).clone().detach().to(torch.float)
    return torch.mean(matches)


def train_step(model, x, y):
    assert len(x) == len(y)

    for param in model.parameters():
        param.requires_grad_(True)
        if param.grad is not None:
             param.grad.data = torch.zeros_like(param.grad.data)

    pred = model(x)
    loss = torch.mean(crossentropy(y, pred))

    loss.backward()
    for param in model.parameters():
        param.data.add_(-LEARNING_RATE * param.grad.data)
        param.detach_()

    return loss.item()


def test_step(model, x, y):
    assert len(x) == len(y)

    with torch.no_grad():
        pred = model(x)
        batch_acc = accuracy(y, pred)

    return batch_acc.item()



#%%

model = Sequential((
    Flatten(),
    Dense(IMG_H * IMG_W, DENSE_UNITS),
    ReLU(),
    Dense(DENSE_UNITS, 10),
    Softmax()
))

for i in range(1, EPOCHS+1):
    print(f"epoch {i}")

    for b, (x, y) in enumerate(train_dataloader):
        loss = train_step(model, x, y)

        if b % 100 == 0:
            print(f"> batch {b+1}: loss = {loss:.4f}")

    acc_sum = 0.0
    n_batches = 0
    for x, y in test_dataloader:
        acc_sum += test_step(model, x, y)
        n_batches += 1
    mean_acc = acc_sum / n_batches

    print(f"\nmean acc. = {mean_acc:.4f}\n\n")