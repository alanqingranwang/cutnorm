import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import cutnorm
from cutnorm import tools, compute_cutnorm


def build_linear_lr_model(input_dim, output_dim):
    # We don't need the softmax layer here since CrossEntropyLoss already
    # uses it internally.
    model = torch.nn.Sequential()
    model.add_module("linear", torch.nn.Linear(input_dim, output_dim))
    return model


def build_deep_lr_model(input_dim,
                        output_dim,
                        num_hidden,
                        hidden_dim=None):
    if hidden_dim == None:
        hidden_dim = input_dim
    model = torch.nn.Sequential()
    if num_hidden > 0:
        model.add_module("input", torch.nn.Linear(input_dim, hidden_dim))
        model.add_module("input tanh", torch.nn.Tanh())
        for i in range(num_hidden):
            model.add_module("linear" + str(i),
                             torch.nn.Linear(hidden_dim, hidden_dim))
            model.add_module("tanh" + str(i), torch.nn.Tanh())
        model.add_module("output", torch.nn.Linear(hidden_dim, output_dim))
    else:
        model.add_module("input", torch.nn.Linear(hidden_dim, output_dim))
    return model


def train(model, loss, optimizer, x_val, y_val):
    x = Variable(x_val, requires_grad=False)
    y = Variable(y_val, requires_grad=False)

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x)
    fx = fx.squeeze()
    loss_output = loss.forward(fx, y)

    # Backward
    loss_output.backward()

    # Update parameters
    optimizer.step()

    return loss_output.data[0]


def predict(model, x_val):
    x = Variable(x_val, requires_grad=False)
    output = model.forward(x).squeeze()
    pred = F.sigmoid(output)
    return pred.data.numpy() > 0.5


def main():
    # Generate train and test data
    X, Y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=0,
        n_clusters_per_class=2)
    trX, teX, trY, teY = train_test_split(X, Y)
    trX = torch.from_numpy(trX).float()
    teX = torch.from_numpy(teX).float()
    trY = torch.from_numpy(trY).float()

    n_examples, n_features = trX.size()
    n_classes = 1
    num_hidden = 2
    model = build_deep_lr_model(n_features, n_classes, num_hidden)
    loss = torch.nn.BCEWithLogitsLoss(size_average=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    batch_size = 50

    for i in range(100):
        cost = 0.
        num_batches = n_examples // batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += train(model, loss, optimizer, trX[start:end],
                          trY[start:end])
            predY = predict(model, teX)

            acc = np.equal(predY, teY).sum() / len(predY)
            print("Epoch %d, cost = %f, acc = %.2f%%" %
                  (i + 1, cost / num_batches, acc))


if __name__ == "__main__":
    main()
