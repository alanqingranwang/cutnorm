import numpy as np
import pickle
import matplotlib.pyplot as plt

res = pickle.load(open("nn_models_analysis.p", "rb"))
# Plot
for computed_res in res:
    label = "% Acc n_hidden = " + str(computed_res['n_hidden'])
    plt.plot(
        np.arange(len(computed_res['acc'])), computed_res['acc'], label=label)
plt.legend()
plt.show()
for computed_res in res:
    label = "cutnorm n_hidden = " + str(computed_res['n_hidden'])
    plt.plot(
        np.arange(len(computed_res['succ_cutnorm'])),
        computed_res['succ_cutnorm'],
        label=label)
plt.legend()
plt.show()
