import numpy as np
import pickle
import torch
from torch.autograd import Variable
import vary_num_nodes
import matplotlib.pyplot as plt
from cutnorm import compute_cutnorm


def compute_data_corr_penult(model, gauss_data, hidden_dim):
    # Get layer
    *_, penult_layer, output_layer = model.modules()

    # Create var for input
    limit = 100
    x = Variable(
        torch.from_numpy(gauss_data.X[:limit]).float(), requires_grad=False)
    n_samples, n_features = x.size()

    # Create vector for feature extraction
    penult_embedding = torch.zeros((n_samples, hidden_dim))

    # def function to copy output
    def copy_data(m, i, o):
        penult_embedding.copy_(o.data)

    # Attach function as hook to layer
    h = penult_layer.register_forward_hook(copy_data)

    output = model.forward(x)

    # Remove hook
    h.remove()

    return np.corrcoef(penult_embedding.numpy())


def process_models(models_params_list, gauss_data):
    processed_data = []
    for config in models_params_list:
        print("Processing config hidden_dim=", config['hidden_dim'])
        acc = []
        corr_mat = []
        for model_state in config['models_params']:
            # Load model
            model = vary_num_nodes.build_deep_lr_model(gauss_data.n_features,
                                                       gauss_data.n_classes,
                                                       1,
                                                       config['hidden_dim'])
            model.load_state_dict(model_state)

            # Compute accuracy on holdout set
            model_pred_y = vary_num_nodes.predict(
                model,
                torch.from_numpy(gauss_data.test_x).float())
            model_acc = vary_num_nodes.acc(model_pred_y, gauss_data.test_y)
            acc.append(model_acc)

            # Compute corrcoef on a subset of data
            corr_mat.append(compute_data_corr_penult(model, gauss_data, config['hidden_dim']))

        succ_cutnorm = []
        for i in range(len(corr_mat)):
            print("Computing cutnorm for hidden_dim=", config['hidden_dim'],
                  "between epoch", 0, i)
            cutn_round, cutn_sdp, info = compute_cutnorm(
                corr_mat[0], corr_mat[i])
            succ_cutnorm.append(cutn_round)

        processed_data.append({
            "hidden_dim": config['hidden_dim'],
            "acc": acc,
            "succ_cutnorm": succ_cutnorm
        })
    return processed_data


def main():
    gauss_data = vary_num_nodes.GaussDataContainer()

    models_params_list = pickle.load(open("nn_models_params.p", "rb"))

    res = process_models(models_params_list, gauss_data)
    # Dump to file
    pickle.dump(res, open("nn_models_analysis.p", "wb"))


if __name__ == "__main__":
    main()
