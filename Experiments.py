"""
This script starts a server and multiple client processes for federated learning.

The server process runs a Federated Learning server that orchestrates the federated learning process.
The client processes simulate multiple clients participating in the federated learning process.

---
Github: https://github.com/nhat-thien/Federated-LSTM-DSTGCRN
"""

from datetime import datetime
import subprocess
import time
import json
import os
from FL_HELPERS.FL_constants import GENERAL_INFO
from Hyperparameters import get_hyperparameters
from TestCase import get_clients_configs, TEST_CASES

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Huge faster than INTEL MKL
os.environ['MKL_THREADING_LAYER'] = 'GNU' 

#--------------------------------------------------------------------
# Because we known beforehand the number of GPUs in the machine. 
# You can set it to [0] if you have only one GPU, then make change 
# accordingly where GPU_IDs is used.
#--------------------------------------------------------------------
GPU_IDs = [0, 1, 2, 3]


def main(is_FL, TEST_CASE_ID=1, PARAM_ID=0):
    
    #----------------------------------------------------------------
    # START THE SERVER
    #----------------------------------------------------------------
    server_start_time = time.time()
    num_clients = len(TEST_CASES[TEST_CASE_ID].clients_names)
    python_command = (
            f"from FL_HELPERS.FL_subprocess import run_server;"
            f"from TestCase import get_clients_configs, TEST_CASES;"
            f"from Hyperparameters import get_hyperparameters;"
            f"clients_configs = get_clients_configs(TEST_CASES[{TEST_CASE_ID}]);"
            f"params = get_hyperparameters(clients_configs[0].get('model_name'), {is_FL});"
            f"run_server(params[{PARAM_ID}], {num_clients}, is_FL={is_FL})"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    server_process = subprocess.Popen(["python", "-c", python_command])
    time.sleep(5)
    print(f"{'-'*90}\n{GENERAL_INFO} SERVER STARTED\n{'-'*90}")
    


    #----------------------------------------------------------------
    # RUN SUBPROCESSES FOR CLIENTS
    #----------------------------------------------------------------
    client_processes = []
    clients_configs = get_clients_configs(TEST_CASES[TEST_CASE_ID])

    for i, client_config in enumerate(clients_configs):

        python_command = (
            f"from FL_HELPERS.FL_subprocess import run_client;"
            f"from TestCase import get_clients_configs, TEST_CASES;"
            f"from Hyperparameters import get_hyperparameters;"
            f"clients_configs = get_clients_configs(TEST_CASES[{TEST_CASE_ID}]);"
            f"params = get_hyperparameters(clients_configs[{i}].get('model_name'), {is_FL});"
            f"params[{PARAM_ID}].device = 'cuda:{GPU_IDs[i]}';"
            f"run_client(params[{PARAM_ID}], clients_configs[{i}], is_FL={is_FL}, device='cuda:{GPU_IDs[i]}')"
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        client_process = subprocess.Popen(["python", "-c", python_command])
        print(f"{GENERAL_INFO} {client_config['client_name']}: Client machine started")
        client_processes.append(client_process)
        time.sleep(1)
    

    # Wait for the server process to finish
    server_process.wait()


    # Measure the time when the server stops
    server_stop_time_seconds = time.time() - server_start_time
    print(f"{'='*90}\nServer stopped at: {server_stop_time_seconds:0.2f} seconds\n{'='*90}\n\n") 
    
    
    # Load the JSON file
    server_log_path = os.path.join(os.getcwd(), f"{TEST_CASES[TEST_CASE_ID].results_path}/SERVER_training_logs.json")
    try:
        with open(server_log_path, "r") as file:
            server_logs = json.load(file)
    except FileNotFoundError:
        server_logs = []


    # Add a record for the server start and stop times
    record = {
        "started_at": datetime.fromtimestamp(server_start_time).isoformat(),
        "stopped_at": datetime.now().isoformat(),
        "duration": server_stop_time_seconds,
        "clients": TEST_CASES[TEST_CASE_ID].clients_names,
        "model_name": TEST_CASES[TEST_CASE_ID].model_name,
        "NOTE": f'Testcase {TEST_CASE_ID}: FL for all clients FedAvg, Attentive, AttentiveCSV, CSV'
    }
    server_logs.append(record)


    # Save the updated data to the JSON file
    os.makedirs(os.path.dirname(server_log_path), exist_ok=True)
    with open(server_log_path, "w") as file:
        json.dump(server_logs, file)





if __name__ == "__main__":


    for is_FL in [False]:
        
        #--------------------------------------------------------------------
        # Test case ID, see TestCase.py
        #--------------------------------------------------------------------
        for TEST_CASE_ID in [0]:


            #----------------------------------------------------------------
            # Get hyperparameters for the corresponding test case
            #----------------------------------------------------------------
            clients_configs = get_clients_configs(TEST_CASES[TEST_CASE_ID])
            model_name = clients_configs[0].get('model_name')
            parmas = get_hyperparameters(model_name, is_FL)
            print(f"{GENERAL_INFO} There are {len(parmas)} combinations of hyperparameters")


            #----------------------------------------------------------------
            # MAIN LOOP over all hyperparameters configurations
            #----------------------------------------------------------------
            for i in range(len(parmas)):
                print(f"{GENERAL_INFO} TEST CASE {TEST_CASE_ID}: {model_name} -- {'CENTRALIZED LEARNING' if is_FL == False else f'FL SCHEME: {parmas[i].FL_scheme}'} -- CSV: {parmas[i].use_CSV}")
                main(is_FL, TEST_CASE_ID=TEST_CASE_ID, PARAM_ID=i)
