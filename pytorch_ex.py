import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
from data_util import load_mnist
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
sys.path.append('../cutnorm')

import cutnorm
from cutnorm import tools, compute_cutnorm


def build_model(input_dim, output_dim):
    # We don't need the softmax layer here since CrossEntropyLoss already
    # uses it internally.
    model = torch.nn.Sequential()
    model.add_module("linear", torch.nn.Linear(input_dim, output_dim))
    return model


def train(model, loss, optimizer, x_val, y_val):
    x = Variable(x_val, requires_grad=False)
    y = Variable(y_val, requires_grad=False)

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x)
    output = loss.forward(fx, y)

    # Backward
    output.backward()

    # Update parameters
    optimizer.step()

    return output.data[0]


def predict(model, x_val):
    x = Variable(x_val, requires_grad=False)
    output = model.forward(x)
    print(output.data.numpy().argmax(axis=1))
    return output.data.numpy().argmax(axis=1)


def main():
    # Generate tain and test data
    X, Y = make_classification(n_samples=1000, n_features=10, n_informative=8, 
                                   n_redundant=0, n_clusters_per_class=2)
    Y = np.array([Y, -(Y-1)]).T  # The model currently needs one column for each class
    trX, teX, trY, teY = train_test_split(X, Y)
    trX = torch.from_numpy(trX).float()
    teX = torch.from_numpy(teX).float()
    trY = torch.from_numpy(trY).float()
    # torch.manual_seed(42)
    # trX, teX, trY, teY = load_mnist(onehot=False)
    # trX = torch.from_numpy(trX).float()
    # teX = torch.from_numpy(teX).float()
    # trY = torch.from_numpy(trY).long()

    n_examples, n_features = trX.size()
    # n_classes = 10
    n_classes = 2 
    model = build_model(n_features, n_classes)
    loss = torch.nn.BCEWithLogitsLoss(size_average=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    batch_size = 50 

    for i in range(100):
        cost = 0.
        num_batches = n_examples // batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += train(model, loss, optimizer,
                          trX[start:end], trY[start:end])
        predY = predict(model, teX)
        predY = predY.reshape((-1, 2))

        count = 0
        for i in range(len(predY)):
            if np.array_equal(predY[i], teY[i]):
                count += 1
        print("Epoch %d, cost = %f, acc = %.2f%%"
              % (i + 1, cost / num_batches, count/len(predY)))


if __name__ == "__main__":
    main()
