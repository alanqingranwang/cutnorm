import numpy as np
import pickle
import torch
from torch.autograd import Variable
import train_gauss_nn
import matplotlib.pyplot as plt
from cutnorm import compute_cutnorm


def compute_data_corr_penult(model, gauss_data):
    # Get layer
    *_, penult_layer, output_layer = model.modules()

    # Create var for input
    limit = 1000
    x = Variable(
        torch.from_numpy(gauss_data.X[:limit]).float(), requires_grad=False)
    n_samples, n_features = x.size()

    # Create vector for feature extraction
    penult_embedding = torch.zeros((n_samples, n_features))
    print('penult_embedding: ', penult_embedding.size())

    # def function to copy output
    def copy_data(m, i, o):
        print('o', o.size())
        penult_embedding.copy_(o.data)

    # Attach function as hook to layer
    h = penult_layer.register_forward_hook(copy_data)

    output = model.forward(x)

    # Remove hook
    h.remove()

    return np.corrcoef(penult_embedding.numpy().transpose())


def process_models(models_params_list, gauss_data):
    processed_data = []
    limit = 1000
    orig_data_corrcoef = np.corrcoef(gauss_data.X[:limit].transpose())
    for config in models_params_list:
        print("Processing config n_hidden = ", config['n_hidden'])
        acc = []
        corr_mat = []
        cutnorm_sets = []
        # Original data
        corr_mat.append(orig_data_corrcoef)
        for model_state in config['models_params']:
            # Load model
            model = train_gauss_nn.build_deep_lr_model(gauss_data.n_features,
                                                       gauss_data.n_classes,
                                                       config['n_hidden'])
            model.load_state_dict(model_state)

            # Compute accuracy on holdout set
            model_pred_y = train_gauss_nn.predict(
                model,
                torch.from_numpy(gauss_data.test_x).float())
            model_acc = train_gauss_nn.acc(model_pred_y, gauss_data.test_y)
            acc.append(model_acc)

            # Compute corrcoef on a subset of data
            corr_mat.append(compute_data_corr_penult(model, gauss_data))

        succ_cutnorm = []
        for i in range(len(corr_mat)):
            print("Computing cutnorm for n_hidden=", config['n_hidden'],
                  "between epoch", 0, i)
            cutn_round, cutn_sdp, info = compute_cutnorm(
                corr_mat[0], corr_mat[i])
            succ_cutnorm.append(cutn_round)

            S,T = info['cutnorm_sets']
            cutnorm_sets.append(np.linalg.norm(S))

        processed_data.append({
            "n_hidden": config['n_hidden'],
            "acc": acc,
            "succ_cutnorm": succ_cutnorm,
            "cutnorm_set": cutnorm_sets
        })
    return processed_data


def main():
    gauss_data = train_gauss_nn.GaussDataContainer()

    models_params_list = pickle.load(open("nn_models_params.p", "rb"))

    res = process_models(models_params_list, gauss_data)
    # Dump to file
    pickle.dump(res, open("nn_models_analysis.p", "wb"))


if __name__ == "__main__":
    main()
