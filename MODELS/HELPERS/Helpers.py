import copy
import itertools
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import json
import datetime
import os

from FL_HELPERS.FL_constants import CLIENT_INFO_TRAINING


def generate_data(OD_tensor, timestamps, lookback, ahead, train_ratio):

    length, height, width = OD_tensor.shape
    data_size = length - lookback - ahead

    X = np.zeros((data_size, lookback, height, width, 1))
    Y = np.zeros((data_size, ahead, height, width, 1))

    for i in range(data_size):
        X[i, :, :, :, 0] = OD_tensor[i:i+lookback, :, :]
        Y[i, :, :, :, 0] = OD_tensor[i+lookback:i+lookback+ahead, :, :]


    # Split the data into train and test sets
    train_size = int(data_size*train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]


    # Normalize the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, height*width)).reshape(-1, lookback, height, width, 1)
    Y_train = scaler.transform(Y_train.reshape(-1, height*width)).reshape(-1, ahead, height, width, 1)

    X_test = scaler.transform(X_test.reshape(-1, height*width)).reshape(-1, lookback, height, width, 1)
    Y_test = scaler.transform(Y_test.reshape(-1, height*width)).reshape(-1, ahead, height, width, 1)

    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    return X_train, Y_train, X_test, Y_test, scaler




def compute_metrics(Y_true, Y_pred, metric='MSE'):

    if metric == 'MAE':
        MAE = np.mean(np.abs(Y_pred - Y_true))
        return MAE
    elif metric == 'MSE':
        MSE = np.mean((Y_pred - Y_true)**2)
        return MSE
    elif metric == 'RMSE':
        RMSE = np.sqrt(np.mean((Y_pred - Y_true)**2))
        return RMSE
    elif metric == 'NRMSE':
        NRMSE = np.sqrt(np.mean((Y_pred - Y_true)**2)) / (np.max(Y_true) - np.min(Y_true))
        return NRMSE
    elif metric == 'MAPE':
        MAPE = np.mean(np.abs((Y_true - Y_pred) / Y_true)) * 100
        return MAPE
    elif metric == 'sMAPE':
        sMAPE = np.mean(2 * np.abs(Y_true - Y_pred) / (np.abs(Y_true) + np.abs(Y_pred))) * 100
        return sMAPE
    elif metric == 'R2':
        SS_res = np.sum((Y_pred - Y_true)**2)
        SS_tot = np.sum((Y_true - np.mean(Y_true))**2)
        R2 = 1 - (SS_res / SS_tot)
        return R2
    




def save_training_results(client, results_path, test_metrics=None, client_name='Name', is_FL=False):
    """Saves training results to a JSON file, appending to a list.

    Args:
        client: The client object containing training results.
        results_path: The path to the JSON file.
        client_name: The name of the current run.
    """

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Extract relevant data
    if is_FL:
        train_loss_list = client.FL_train_loss_list
        val_loss_list = client.FL_val_loss_list
        val_metrics_dict = client.FL_val_metrics_dict
    else:
        train_loss_list = client.trainer.train_loss_list
        val_loss_list = client.trainer.val_loss_list
        val_metrics_dict = client.trainer.val_metrics_dict

    # Create a JSON object for the current run
    run_data = {
        "client_name": client_name,
        "train_loss_list": train_loss_list,
        "val_loss_list": val_loss_list,
        "replaced_modules": client.replaced_modules if len(client.replaced_modules) > 0 else None,
        "hyperparameters": {
            "FL_rounds": None if getattr(client, 'is_FL') == False else getattr(client.trainer.args, 'FL_rounds', None),
            "FL_scheme": "LocalLearning" if getattr(client, 'is_FL') == False else getattr(client.trainer.args, 'FL_scheme', None),
            "to_weight_clients": getattr(client.trainer.args, 'to_weight_clients', None),
            "use_CSV": getattr(client.trainer.args, 'use_CSV', None),
            "batch_size": getattr(client.trainer.args, 'batch_size', None),
            "epochs": getattr(client.trainer.args, 'epochs', None),
            "embed_dim": getattr(client.trainer.args, 'embed_dim', None),
            "rnn_units": getattr(client.trainer.args, 'rnn_units', None),
            "num_layers": getattr(client.trainer.args, 'num_layers', None),
            "hidden_dim_node": getattr(client.trainer.args, 'hidden_dim_node', None),
            "num_layers_node": getattr(client.trainer.args, 'num_layers_node', None),
            "num_heads": getattr(client.trainer.args, 'num_heads', None),
            "input_dim": getattr(client.trainer.args, 'input_dim', None),
            "cheb_k": getattr(client.trainer.args, 'cheb_k', None),
            "hyperGNN_dim1": getattr(client.trainer.args, 'hyperGNN_dim1', None),
            "hyperGNN_dim2": getattr(client.trainer.args, 'hyperGNN_dim2', None),
            "lr_init": getattr(client.trainer.args, 'lr_init', None),
            "lr_decay": getattr(client.trainer.args, 'lr_decay', None),
            "lr_decay_rate": getattr(client.trainer.args, 'lr_decay_rate', None),
            "lr_decay_steps": getattr(client.trainer.args, 'lr_decay_steps', None),
            "early_stop_patience": getattr(client.trainer.args, 'early_stop_patience', None),
            "device": getattr(client.trainer.args, 'device', None),
            "lookback": getattr(client.trainer.args, 'lookback', None),
            "lookahead": getattr(client.trainer.args, 'lookahead', None),
            "lstm_layer": getattr(client.trainer.args, 'lstm_layer', None),
            "gru_layer": getattr(client.trainer.args, 'gru_layer', None),
            "log_dir": getattr(client.trainer.args, 'log_dir', None),
            "model_name": getattr(client.trainer.model, 'model_name', None),
        },
        "recorded_at": datetime.datetime.now().isoformat(),
        "load_best_model": getattr(client.trainer.args, 'load_best_model', None),
        "test_metrics": test_metrics
    }

    # Add validation metrics to the JSON object
    for metric, values in val_metrics_dict.items():
        run_data[metric] = values


    # Add hyperparameters to the JSON object
    # run_data["hyperparameters"] = client.trainer.hyperparameters
     
    try:
        with open(results_path, "r") as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []

    existing_data.append(run_data)

    with open(results_path, "w") as f:
        json.dump(existing_data, f, sort_keys=True, indent=4, separators=(',', ': '))

    # print(f"{CLIENT_INFO_TRAINING} {client_name} Training process is saved to {results_path}")


def modulewise_update(model, trainer, aggregated_weights, params):
    
    model = copy.deepcopy(model)
    trainer = copy.deepcopy(trainer)
    
    val_losses = []

    module_index = [[0,4*params.num_layers_node+2], 
                    [4*params.num_layers_node+2,4*params.num_layers_node+2+4],
                    [4*params.num_layers_node+2+4,len(aggregated_weights)]]

    best_loss = float('inf')
    for k in range(3):
        # Replace the weights for GRU module
        new_weights = model.get_weights()[0]
        new_weights[module_index[k][0]:module_index[k][1]] = aggregated_weights[module_index[k][0]:module_index[k][1]]

        # Set the new weights to the model and calculate the validation loss
        model.set_weights(new_weights)
        trainer.model = model
        val_loss, avg_metrics = trainer.val_epoch()
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = new_weights
        # print(f"Sum model after replace module {k} module_scores { DEBUG_sum_weights(model)}")
    
    module_scores = np.array(val_losses)/sum(np.array(val_losses))
    layer_scores = np.zeros(len(aggregated_weights))
    for k in range(3):
        layer_scores[module_index[k][0]:module_index[k][1]] = module_scores[k]

    return best_weights, layer_scores




def all_subsets_update(model, trainer, aggregated_weights, params):

    model = copy.deepcopy(model)
    trainer = copy.deepcopy(trainer)
    
    module_index = [[0, 4*params.num_layers_node+2], 
                    [4*params.num_layers_node+2, 4*params.num_layers_node+2+4],
                    [4*params.num_layers_node+2+4, len(aggregated_weights)]]

    best_loss = float('inf')
    val_losses = []
    is_replaced = [None, None, None]
    modules = {0,1,2}
    subsets = powerset(modules)

    for S in subsets:

        new_weights = model.get_weights()[0]
        for k in S:
            # Replace the weights for the subset S of modules
            new_weights[module_index[k][0]:module_index[k][1]] = aggregated_weights[module_index[k][0]:module_index[k][1]]

        # Set the new weights to the model and calculate the validation loss
        model.set_weights(new_weights)
        trainer.model = model
        val_loss, avg_metrics = trainer.val_epoch()
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = new_weights
            # print(f'[Debug] Replaced: {S}')
            for m in modules:
                if m in S:
                    is_replaced[m] = 1
                else:
                    is_replaced[m] = 0
        # print(f"Sum model after replace module {k} module_scores { DEBUG_sum_weights(model)}")


    shapley_scores = calculate_shapley_values(modules, subsets, val_losses)


    # This part is to calculate the layer scores, i.e. for each layer for convenience later
    shapley_scores = np.array(shapley_scores)
    layer_scores = np.zeros(len(aggregated_weights))

    for k in range(3):
        layer_scores[module_index[k][0]:module_index[k][1]] = shapley_scores[k]

    return best_weights, layer_scores, is_replaced




def only_one_module_update(model, aggregated_weights, params, module):

    model = copy.deepcopy(model)
    
    module_index = [[0,4*params.num_layers_node+2], 
                    [4*params.num_layers_node+2,4*params.num_layers_node+2+4],
                    [4*params.num_layers_node+2+4,len(aggregated_weights)]]
    
    weights = model.get_weights()[0]

    if module == 'GRU':
        weights[module_index[0][0]:module_index[0][1]] = aggregated_weights[module_index[0][0]:module_index[0][1]]
    if module == 'Attention':
        weights[module_index[1][0]:module_index[1][1]] = aggregated_weights[module_index[1][0]:module_index[1][1]]
    if module == 'GCN':
        weights[module_index[2][0]:module_index[2][1]] = aggregated_weights[module_index[2][0]:module_index[2][1]]
    
    return weights




def client_validate_weights(model, trainer, aggregated_weights, params):

    layer_scores = None
    is_replaced = None

    # Update the weights based on the FL scheme
    if params.FL_scheme == 'FedAvg':
        pass

    elif params.FL_scheme == 'Attentive':
        pass

    elif params.FL_scheme == 'AttentiveCSV':
        aggregated_weights, layer_scores, is_replaced = all_subsets_update(model, trainer, aggregated_weights, params)

    elif params.FL_scheme == 'Module-wise':
        aggregated_weights, layer_scores = modulewise_update(model, trainer, aggregated_weights, params)

    elif params.FL_scheme == 'Only-LSTM-module':
        aggregated_weights = only_one_module_update(model, aggregated_weights, params, 'GRU')

    elif params.FL_scheme == 'Only-Attention-module':
        aggregated_weights = only_one_module_update(model, aggregated_weights, params, 'Attention')

    elif params.FL_scheme == 'Only-AGCRN-module':
        aggregated_weights = only_one_module_update(model, aggregated_weights, params, 'Attention')

    elif params.FL_scheme == 'ClientSideValidation':
        aggregated_weights, layer_scores, is_replaced = all_subsets_update(model, trainer, aggregated_weights, params)

    return aggregated_weights, layer_scores, is_replaced



def DEBUG_sum_weights(model):
    return sum([np.sum(w) for w in model.get_weights()[0]])

def powerset(s):
  """
  This function generates all subsets of a set, but returns a list of lists instead of sets.
  Args:
    s: The input set.
  Returns:
    A list of all subsets of the input set as lists.
  """
  x = list(s)
  return [list(subset) for i in range(len(x) + 1) for subset in itertools.combinations(x, i)]


def calculate_shapley_values(modules, subsets, validation_losses):
    n = len(modules)
    shapley_values = [0] * n  # Initialize a list to store Shapley values for each module
    module_list = list(modules)
    factorial = {i: math.factorial(i) for i in range(len(modules) + 1)}
    
    for module in modules:
        module_index = module_list.index(module)  # Find index of module in the list
        
        # Iterate over all subsets of modules except the current module
        for subset in subsets:
            if module not in subset:
                subset_with_module = subset + [module]
                
                # Convert subsets to indices
                subset_index = subsets.index(subset)
                subset_with_module_index = subsets.index(sorted(subset_with_module))
                
                # Calculate the marginal contribution
                marginal_contribution = validation_losses[subset_with_module_index] - validation_losses[subset_index]
                
                # Calculate the weight
                weight = factorial[len(subset)] * factorial[n - len(subset) - 1] / factorial[n]
                
                # Add the weighted marginal contribution to the Shapley value
                shapley_values[module_index] += weight * marginal_contribution
    
    return shapley_values