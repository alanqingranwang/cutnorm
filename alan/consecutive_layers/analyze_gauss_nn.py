import numpy as np
import pickle
import torch
from torch.autograd import Variable
import consecutive_layers
import matplotlib.pyplot as plt
from cutnorm import compute_cutnorm


def compute_data_corr_penult(model, gauss_data):
    # Get layer
    #*_, penult_layer, output_layer = model.modules()
    layers = []
    for m in model.modules():
        layers.append(m)

    # Create var for input
    limit = 100
    x = Variable(
        torch.from_numpy(gauss_data.X[:limit]).float(), requires_grad=False)
    n_samples, n_features = x.size()

    # Create vector for feature extraction
    penult_embedding = torch.zeros((n_samples, n_features))

    # def function to copy output
    # def copy_data(m, i, o):
    #     print('o', o.size())
    #     penult_embedding.copy_(o.data)

    layers_to_inspect = []
    if len(layers) == 2:
        layers_to_inspect.append(layers[1]) 
    else:
        for i in range(2, len(layers), 2):
            layers_to_inspect.append(layers[i])
    # Attach function as hook to layer
    corrcoef_list = []
    for layer in layers_to_inspect:
        layer_embedding = torch.zeros((n_samples, n_features))
        def copy_data(m, i, o):
            layer_embedding.copy_(o.data)

        h = layer.register_forward_hook(copy_data)
        output = model.forward(x)
        h.remove()
        corrcoef_list.append(np.corrcoef(layer_embedding.numpy()))
    #h = penult_layer.register_forward_hook(copy_data)

    #output = model.forward(x)

    # Remove hook
    #h.remove()
    print(np.array(corrcoef_list[0]).shape)
    return corrcoef_list


def process_models(models_params_list, gauss_data):
    processed_data = []
    for config in models_params_list:
        print("Processing config n_hidden = ", config['n_hidden'])
        acc = []
        corr_mat = []
        for model_state in config['models_params']:
            # Load model
            model = consecutive_layers.build_deep_lr_model(gauss_data.n_features,
                                                       gauss_data.n_classes,
                                                       config['n_hidden'])
            model.load_state_dict(model_state)

            # Compute accuracy on holdout set
            model_pred_y = consecutive_layers.predict(
                model,
                torch.from_numpy(gauss_data.test_x).float())
            model_acc = consecutive_layers.acc(model_pred_y, gauss_data.test_y)
            acc.append(model_acc)

            # Compute corrcoef on a subset of data
            corr_mat.append(compute_data_corr_penult(model, gauss_data))

        succ_cutnorm = []
        for i in range(len(corr_mat)):
            for idx_layer in range(len(corr_mat[i])-1):
                print("Computing cutnorm for n_hidden=", config['n_hidden'],
                      "between layer", idx_layer, idx_layer+1)

                cutn_round, cutn_sdp, info = compute_cutnorm(
                    corr_mat[i][idx_layer], corr_mat[i][idx_layer+1])
                succ_cutnorm.append(cutn_round)

        processed_data.append({
            "n_hidden": config['n_hidden'],
            "acc": acc,
            "succ_cutnorm": succ_cutnorm
        })
    return processed_data


def main():
    gauss_data = consecutive_layers.GaussDataContainer()

    models_params_list = pickle.load(open("nn_models_params.p", "rb"))

    res = process_models(models_params_list, gauss_data)
    # Dump to file
    pickle.dump(res, open("nn_models_analysis.p", "wb"))


if __name__ == "__main__":
    main()
