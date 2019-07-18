import random

random.seed(42)



from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import TensorDataset

iris = load_iris()


X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=1)

# Convert to pytorch tensors
X_train, y_train, X_test, y_test = map(
    torch.tensor, (X_train, y_train, X_test, y_test)
)

X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).long()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).long()


train_ds = TensorDataset(X_train, y_train)

train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


class IrisNet(nn.Module):
    def __init__(self, inputshape, outputshape, hiddenunits=10):
        super(IrisNet, self).__init__()
        self.input_layer = nn.Linear(inputshape, hiddenunits)
        self.hidden_layer = nn.Linear(hiddenunits, hiddenunits)
        self.output_layer = nn.Linear(hiddenunits, outputshape)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        X = F.relu(self.input_layer(X))
        X = self.hidden_layer(X)
        X = self.output_layer(X)
        X = self.softmax(X)

        return X


net = IrisNet(4, 3)


criterion = nn.CrossEntropyLoss()  # cross entropy loss
optimizer = SGD(net.parameters(), lr=0.01)
loss_func = F.cross_entropy

epochs = 10
before_training_pred = net(X_train)
print('Train Loss BEFORE training', accuracy(before_training_pred, y_train))


for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = net(xb)
        loss = loss_func(pred, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print(loss)

    print(loss_func(net(xb), yb), accuracy(pred, yb))


after_training_pred = net(X_train)
print('Train Loss After Training', accuracy(after_training_pred, y_train))


y_test_pred = net(X_test)
print('Test Loss', accuracy(y_test_pred, y_test))
