import numpy as np
from sklearn.datasets import make_gaussian_quantiles

n_samples = 5000
n_features = 10
n_classes = 3
X, Y = make_gaussian_quantiles(
    n_samples=n_samples, n_features=n_features, n_classes=n_classes)
Y_onehot = np.zeros((n_samples, n_classes))
Y_onehot[np.arange(n_samples), Y] = 1

np.save("gauss_quant_X", X)
np.save("gauss_quant_Y", Y)
np.save("gauss_quant_Y_onehot", Y_onehot)
