class TestCase:
    """
    Represents a test case for a machine learning model.
    Args:
        model_name (str): The name of the machine learning model.
        framework (str): The framework used for the model.
        results_path (str): The path to store the results of the test case.
        clients_names (list): A list of client names.
        data_paths (list): A list of data paths for each client.
    """

    def __init__(self, model_name, framework, results_path, clients_names, data_paths):
        self.model_name = model_name
        self.framework = framework
        self.results_path = results_path
        self.clients_names = clients_names
        self.data_paths = data_paths


TRANSPORT_MODES = ['NYC-bike', 'NYC-taxi', 'CHI-taxi']
TRANSPORT_OPERATORS = ['LyonPT', 'Orange']

TEST_CASES = [
    TestCase(#0
        'Fed-LSTM-DSTGCRN',
        'PyTorch',
        'RESULTS/TransportModes',
        TRANSPORT_MODES,
        [f'DATA/TransportModes/{mode}/tripdata_full.csv' for mode in TRANSPORT_MODES],
    ),
    TestCase(#1
        'FedLSTM',
        'PyTorch',
        'RESULTS/TransportModes',
        TRANSPORT_MODES,
        [f'DATA/TransportModes/{mode}/tripdata_full.csv' for mode in TRANSPORT_MODES],
    ),
    TestCase(#1
        'FedGRU',
        'PyTorch',
        'RESULTS/TransportModes',
        TRANSPORT_MODES,
        [f'DATA/TransportModes/{mode}/tripdata_full.csv' for mode in TRANSPORT_MODES],
    ),
    TestCase(#1
        'FedAGCRN',
        'PyTorch',
        'RESULTS/TransportModes',
        TRANSPORT_MODES,
        [f'DATA/TransportModes/{mode}/tripdata_full.csv' for mode in TRANSPORT_MODES],
    ),
    TestCase(#3
        'Fed-LSTM-DSTGCRN',
        'PyTorch',
        'RESULTS/OD_Data',
        TRANSPORT_OPERATORS,
        [f'DATA/OD_Data/{client}_2022.csv' for client in TRANSPORT_OPERATORS],
    ),
    TestCase(#2
        'FedLSTM',
        'PyTorch',
        'RESULTS/OD_Data',
        TRANSPORT_OPERATORS,
        [f'DATA/OD_Data/{client}_2021.csv' for client in TRANSPORT_OPERATORS],
    ),
    TestCase(#2
        'FedGRU',
        'PyTorch',
        'RESULTS/OD_Data',
        TRANSPORT_OPERATORS,
        [f'DATA/OD_Data/{client}_2021.csv' for client in TRANSPORT_OPERATORS],
    ),
    TestCase(#2
        'FedAGCRN',
        'PyTorch',
        'RESULTS/OD_Data',
        TRANSPORT_OPERATORS,
        [f'DATA/OD_Data/{client}_2021.csv' for client in TRANSPORT_OPERATORS],
    )
]


def get_clients_configs(test_case):
    """
    Generates a list of client configurations based on the given test case.
    Args:
        test_case (TestCase): The test case object.
    Returns:
        list: A list of client configurations.
    """

    clients_conf = []
    for client_name, data_path in zip(test_case.clients_names, test_case.data_paths):
        conf = {
            'model_name': test_case.model_name,
            'framework': test_case.framework,
            'results_path': test_case.results_path,
            'client_name': client_name,
            'data_path': data_path
        }
        clients_conf.append(conf)
    return clients_conf
