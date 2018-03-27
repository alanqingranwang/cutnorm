import numpy as np
import torch
import copy
import pickle
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

def build_deep_lr_model(input_dim, output_dim, n_hidden, hidden_dim=-1):
    if hidden_dim < 1:
        hidden_dim = input_dim
    model = torch.nn.Sequential()
    if n_hidden > 0:
        model.add_module("input", torch.nn.Linear(input_dim, hidden_dim))
        model.add_module("input tanh", torch.nn.Tanh())
        for i in range(n_hidden):
            model.add_module("linear" + str(i),
                             torch.nn.Linear(hidden_dim, hidden_dim))
            model.add_module("tanh" + str(i), torch.nn.Tanh())
        model.add_module("output", torch.nn.Linear(hidden_dim, output_dim))
    else:
        model.add_module("input", torch.nn.Linear(hidden_dim, output_dim))
    return model

def train_batch(model, loss, optimizer, x_val, y_val):
    x = Variable(x_val, requires_grad=False)
    y = Variable(y_val, requires_grad=False)

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x)
    loss_output = loss.forward(fx, y)

    # Backward
    loss_output.backward()

    # Update parameters
    optimizer.step()

    return loss_output.data[0]

def predict(model, x_val):
    x = Variable(x_val, requires_grad=False)
    output = model.forward(x)
    pred = F.sigmoid(output)
    n_samples, n_classes = pred.size()
    res = np.zeros((n_samples, n_classes))
    res[np.arange(n_samples), pred.data.numpy().argmax(axis=1)] = 1
    return res

def acc(pred_y, test_y):
    return np.equal(pred_y, test_y).all(axis=1).sum() / len(pred_y)

def train_model(data_container, n_hidden=0, n_epochs_train=100, n_batches=10):
    # Extract data
    n_features = data_container.n_features
    n_samples = data_container.n_samples
    n_classes = data_container.n_classes
    batch_size = n_samples // n_batches

    # Construct model
    model = build_deep_lr_model(n_features, n_classes, n_hidden=n_hidden)
    loss_function = torch.nn.BCEWithLogitsLoss(size_average=True)
    optimizer = optim.Adam(model.parameters())

    # Prep data for training
    train_x = torch.from_numpy(data_container.train_x).float()
    train_y = torch.from_numpy(data_container.train_y).float()

    model_state_list = []

    for i in range(n_epochs_train):
        loss = 0.
        for k in range(n_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            loss += train_batch(model, loss_function, optimizer,
                                train_x[start:end], train_y[start:end])
        if i % (n_epochs_train // 10) == 0:
            print("Epoch %d, average batch loss = %f" % (i + 1,
                                                         loss / n_batches))
        model_state_list.append(copy.deepcopy(model.state_dict()))

    # Save Model
    return model_state_list


class GaussDataContainer:
    def __init__(self):
        # Generate tain and test data
        self.X = np.load("gauss_quant_X.npy")
        self.Y = np.load("gauss_quant_Y_onehot.npy")
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.X, self.Y)
        self.n_samples, self.n_features = self.train_x.shape
        _, self.n_classes = self.Y.shape


def main():
    gauss_data = GaussDataContainer()

    # n_hidden_list = np.arange(5)
    n_hidden_list = [4]
    models_params_list = []
    for n_hidden in n_hidden_list:
        print("Training model with", n_hidden, "layers")
        models_params = train_model(gauss_data, n_hidden=n_hidden)
        models_params_list.append({
            "n_hidden": n_hidden,
            "models_params": models_params
        })
    # Dump to file
    pickle.dump(models_params_list, open("nn_models_params.p", "wb"))


if __name__ == "__main__":
    main()

