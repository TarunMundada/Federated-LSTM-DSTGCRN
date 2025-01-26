from FL_HELPERS.FL_constants import *
import pickle
import torch
from joblib import load, dump
from MODELS.LocalModels import LocalModel


def parse_data(data):

    if isinstance(data, bytes):
        return data
    
    buffer = BytesIO()

    if isinstance(data, np.ndarray):
        np.save(buffer, data, allow_pickle=True)

    elif 'keras' in str(type(data)):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        with h5py.File(buffer, 'w') as f:
            save_model(data, f, include_optimizer=False)

    elif 'sklearn' in str(type(data)):
        dump(data, buffer)
    
    elif isinstance(data, torch.nn.Module):  # Handle torch models
        torch.save(data.state_dict(), buffer)
    
    elif isinstance(data, list):  # Handle lists
        pickle.dump(data, buffer)

    buffer.seek(0)
    return buffer.read()



def load_data(file: BytesIO, params):

    data = file.read()[:-3]
    file.seek(0)
    file.truncate(len(data))

    if b'torch' in data: #Handle torch models
        """
        Resetting num_nodes to 1 ensures that the checkpoint dimensions match the model dimensions.
        This does not affect the weights.
        """
        params.num_nodes = 1
        model = LocalModel(verbose=False, params=params).model
        model.load_state_dict(torch.load(file, weights_only=True))
        return model
    
    elif b'sklearn' in data:
        return load(file)
    
    elif b'NUMPY' in data:  
        return np.load(file)
    
    elif (b'ndarray' in data) or isinstance(data, list):
        return pickle.load(file)
    
    elif b'HDF' in data:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        with h5py.File(file, 'r') as f:
            model = load_model(f, compile=False)
        return model
    else:
        return data
        


def socket_send(connection, data):

    bufdata = parse_data(data)
    result = connection.send(bufdata)

    connection.send(b'End')
    connection.recv(BUFFER_SIZE)

    return result



def socket_receive(connection, bufsize, params):

    buffer = BytesIO()

    while True:
        data = connection.recv(bufsize)
        if not data:
            break
        buffer.write(data)
        buffer.seek(-4, 2)
        if b'End' in buffer.read():
            connection.send(b'Complete')
            break
        
    buffer.seek(0)

    return load_data(buffer, params)



def find_free_port():
    """Finds a free port on the system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind to any available port
        return s.getsockname()[1]  # Get the assigned port
    



