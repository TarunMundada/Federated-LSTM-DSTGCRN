"""
This file contains the functions to run the server and client for Federated Learning.
The functions are feed into the subprocess.Popen() function to run the server and client script.
Each client has its own particular data loader.

---
Thien Pham, Oct 2024
"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from FL_HELPERS.FL_constants import *
from FL_HELPERS.FL_components import FL_Server, FL_Client
from MODELS.HELPERS.dataloader import get_dataloader, get_dataloader_OD
from MODELS.HELPERS.Helpers import generate_data, save_training_results
from MODELS.LocalModels import LocalModel
import platform, os



def run_server(params, num_clients, is_FL=True):
    """
    Runs the server for Federated Learning.
    Args:
        params (Hyperparameters): The hyperparameters.
        num_clients (int): The number of clients. Defaults to None.
        is_FL (bool, optional): Whether to run Federated Learning or not. Defaults to True.
    """

    # INITIATE GLOBAL MODEL
    global_model = LocalModel(params)
    
    # INITIATE SERVER OF FL
    server = FL_Server(global_model=global_model.model,
                        num_clients=num_clients,
                        params=params,
                        is_FL=is_FL)

    # TRAIN
    server.train()




def run_client(params, client_config, is_FL=True, device=None):
    """
    This function is used to feed into the subprocess.Popen() function to run the client script.
    Each model has its own particular data loader.
    Args:
        params (Hyperparameters): The hyperparameters.
        client_config (dict): Configuration for the client.
        is_FL (bool, optional): Whether to run Federated Learning or not. Defaults to True.
    Returns:
        FL_Client: The client object.
    """
    import numpy as np
    
    model_name = client_config.get('model_name')
    framework = client_config.get('framework')
    client_name = client_config.get('client_name')
    data_path = client_config.get('data_path')
    results_path = client_config.get('results_path')

    print(f"{CLIENT_INFO_TRAINING} {client_name}: Running on {platform.system()} OS")
    if DATA_INFO_VERBOSE:
        print(f"{CLIENT_INFO_TRAINING} {client_name} Current working directory: {os.getcwd()}")

    data_path = os.path.join(os.getcwd(), data_path)


    # Prepare the data, adapted for each model
    if model_name == 'FedConvLSTM':
        OD_tensor_client = np.load(data_path)

        # Generate the data
        X_train, Y_train, X_test, Y_test, scaler = generate_data(OD_tensor_client, 
            None, 
            params.lookback, 
            params.ahead, 
            params.train_ratio
        )
        loss='mse'
        data_loader=(X_train, Y_train)
    
    elif model_name == 'Fed-LSTM-DSTGCRN' or model_name in ['FedGRU', 'FedLSTM', 'FedARIMA', 'FedLR', 'FedAGCRN']:
        import torch
        print(f"{'-'*90}\n{GENERAL_INFO} There are {torch.cuda.device_count()} GPUs available\n{'-'*90}")
        if torch.cuda.is_available():
            if "cuda" in device:
                torch.cuda.set_device(int(device.split(":")[1]))

        else:
            params.device = "cpu"
            
        if "OD" not in data_path:
            (train_loader, val_loader, test_loader, scaler, num_nodes, num_features, num_obsevations) = get_dataloader(
                data_path,
                params,
                normalizer=params.normalizer,
                single=True,
                name=client_name,
                device=device
            )
        else:
            (train_loader, val_loader, test_loader, scaler, num_nodes, num_features, num_obsevations) = get_dataloader_OD(
                data_path,
                params,
                normalizer=params.normalizer,
                single=True,
                name=client_name,
                device=device
            )
        
        # Parameters for FL_Client constructor
        # Here, None are used to make the calling of constructor consistent with the other models
        X_test, Y_test = None, None
        loss = torch.nn.MSELoss().to(params.device)
        data_loader = (train_loader, val_loader, test_loader, scaler)

    
    print(f"{CLIENT_INFO_TRAINING} {client_name} data loaded")

    params.num_nodes = num_nodes
    params.num_features = num_features
    print(f"{CLIENT_INFO_TRAINING} {client_name} has {num_nodes} nodes, each has {num_features} feature(s) and {num_obsevations} timestamps")
    
    # Initiate the model
    initial_model = LocalModel(params)

    # Initiate the client object
    client = FL_Client(initial_model, name=client_name, data_loader=data_loader, params=params, results_path=results_path, framework=framework, is_FL=is_FL)

    # Start Learning Process
    if is_FL:
        # Federated Learning
        client.FL_train(loss=loss,  
                epochs=params.epochs, 
                batch_size=params.batch_size, 
                val_ratio=params.val_ratio, 
                verbose=params.local_model_training_verbose)
        train_log_path = os.path.join(os.getcwd(), results_path, f"{client_name}_FL_training_logs.json")
        
    else:
        # Centralized Learning
        client.CL_train(loss=loss,  
                epochs=params.epochs, 
                batch_size=params.batch_size, 
                val_ratio=params.val_ratio, 
                verbose=params.local_model_training_verbose)
        train_log_path = os.path.join(os.getcwd(), results_path, f"{client_name}_CL_training_logs.json")

    # Evaluate the model
    # print(f"{'-'*90}\n{CLIENT_INFO_TESTING} {client_name} Metrics ({'Federated' if is_FL else 'Centralized'} Learning)\n{'-'*90}")

    metric_path = os.path.join(os.getcwd(), results_path, "Metrics-Logs.md")
    test_metrics = client.test(X_test, Y_test, scaler, loss, results_path=metric_path)
    
    save_training_results(client, train_log_path, test_metrics, client_name, is_FL=is_FL)

    return client






