"""
The FL_Server class is responsible for handling the server-side operations, such as:
- Accepting connections from clients
- Distributing the global model
- Aggregating the weights received from the clients

The FL_Client class is responsible for handling the client-side operations, such as:
- Connecting to the server
- Training the local model
- Sending the weights to the server
- Receiving the aggregated weights from the server

The FL_Server and FL_Client classes communicate with each other using sockets.
---
Thien Pham, Oct 2024
"""


from MODELS.HELPERS.Helpers import DEBUG_sum_weights, client_validate_weights, compute_metrics
from FL_HELPERS.FL_constants import *
from FL_HELPERS.FL_socket import *
import torch
from Hyperparameters import Hyperparameters
from MODELS.LSTM_DSTGCRN.Trainers import Trainer
import pandas as pd



"""----------------------------------------------------------------------
SERVER
"""
class FL_Server():

    def __init__(self, global_model, num_clients, params, is_FL=True, port=PORT, multi_system=False):
        self.s = socket.socket()
        self.executor = concurrent.futures.ThreadPoolExecutor(num_clients)
        self.global_model = global_model
        self.num_clients = num_clients
        self.rounds = params.FL_rounds
        self.FL_scheme = params.FL_scheme
        self.params = params
        self.port = port 
        self.is_FL = is_FL 
        self.multi_system = multi_system
        self.results = []
        self.connections = []


    def __initiate_socket(self):
        self.s.bind(('', self.port))
        print(f"{SERVER_INFO_CONNECTION} socket binded to {self.port}")

        self.s.listen(self.num_clients)
        print(f"{SERVER_INFO_CONNECTION} socket is listening...")


    # This function runs in the first round to accept all connections and distribute the global model
    def __accept_connection(self):
        connection, addr = self.s.accept()
        self.connections.append(connection)
        print(f'{SERVER_INFO_CONNECTION} Got connection from {addr[0]}:{addr[1]}')
        
        # Broadcast the global model
        res = self.executor.submit(self.__handle_client, connection, True, True)
        self.results.append(res)


    def __close_connections(self):
        for connection in self.connections:
            connection.close()
        print(f"{'-'*90}\n{SERVER_INFO_CONNECTION} SERVER CLOSED CONNECTIONS\n{'-'*90}")


    def __handle_client(self, connection, is_send_rounds=True, is_send_model=True):
        
        if is_send_rounds:
            connection.send(str(self.rounds).encode(FORMAT))

        if is_send_model:
            # The server only send the model object once, after the initial connections
            # to distribute the model to all clients
            socket_send(connection, self.global_model)
        else:
            # Later, the server send only the weights, to precerve the model object at clients
            socket_send(connection, self.global_model.get_weights()[0])

        new_from_client = socket_receive(connection, BUFFER_SIZE, self.params)
        return new_from_client




    #--------------------------------------
    # Main federated scheme 🟥
    #--------------------------------------
    def __FL_aggregate(self, new_weights_list, layer_scores_list=None):


        if self.FL_scheme in ['FedAvg', 'Only-LSTM-module', 'Only-Attention-module', 'Only-AGCRN-module']:
            
            aggregated_weights = []
            
            for layer in range(len(new_weights_list[0])):
                sum_layer = np.zeros_like(new_weights_list[0][layer])
                for client_weights in new_weights_list:
                    sum_layer += client_weights[layer]
                aggregated_weights.append(sum_layer / self.num_clients)


        if self.FL_scheme=='Attentive' or self.FL_scheme=='AttentiveCSV':
            
            aggregated_weights = []

            for layer in range(len(new_weights_list[0])):
                new_layer = np.zeros_like(new_weights_list[0][layer])

                for client_idx, _ in enumerate(new_weights_list):
                    EuclideanDistances = []

                    for other_client_idx, _ in enumerate(new_weights_list):
                        
                        # Calculate the distance between the current layer and the corresponding layer of other clients
                        dis = np.linalg.norm(new_weights_list[client_idx][layer] - new_weights_list[other_client_idx][layer])
                        EuclideanDistances.append(dis)
                    
                    EuclideanDistances = 1/(1+np.array(EuclideanDistances))
    
                    scores = np.array(EuclideanDistances)/sum(EuclideanDistances)

                    for i, local_weight in enumerate(new_weights_list):
                        new_layer += local_weight[layer] * scores[i]

                
                # Append to the aggregated weights list
                aggregated_weights.append(new_layer/self.num_clients)
                # Why divide by the number of clients ? Because the scores are already sum to one, dividing makes the weights downscaled ?
                # aggregated_weights.append(new_layer)


        if self.FL_scheme=='Module-wise' or self.FL_scheme=='ClientSideValidation':
            
            aggregated_weights = []

            if self.params.to_weight_clients:
                layer_scores_2d = np.array(layer_scores_list).reshape(len(layer_scores_list), -1)
                sum_layer_scores = np.sum(layer_scores_2d, axis=0)
                scores_2d = (layer_scores_2d)/sum_layer_scores
                
                # Weight the weights by the layer scores
                for i, _ in enumerate(new_weights_list):
                    current_layer = np.zeros_like(new_weights_list[0][i])
                    for new_weight, score in zip(new_weights_list, scores_2d[:,i]):
                        current_layer += score*new_weight[i]
                    aggregated_weights.append(current_layer)

            else:
                # Just take the average of the weights as FedAvg
                for i, _ in enumerate(new_weights_list):
                    current_layer = np.zeros_like(new_weights_list[0][i])
                    for new_weight in new_weights_list:
                        current_layer += new_weight[i]
                    aggregated_weights.append(current_layer/self.num_clients)

        self.global_model.set_weights(aggregated_weights)



    #--------------------------------------
    # Main federated loop
    #--------------------------------------
    def __FL_loop(self):

        for i in range(self.rounds):
            start_time = time.time()
            new_weights_list = []
            layer_scores_list = []
            for f in concurrent.futures.as_completed(self.results):
                received_data = f.result() #List
                new_weights_list.append(received_data[0])
                layer_scores_list.append(received_data[1])

            self.__FL_aggregate(new_weights_list, layer_scores_list)

            self.results = []

            for connection in self.connections:
                # Because the model is already distributed, I don't need to send the model again (is_send_model=False)
                res = self.executor.submit(self.__handle_client, connection, False, False)
                self.results.append(res)

            print(f'{SERVER_INFO_TRAINING} ✅ FL ROUND {i+1}/{self.rounds} COMPLETED | {round(time.time()-start_time, 2)} seconds')



    #🟪
    def train(self):

        self.__initiate_socket()

        # The first round to accept all connections and distribute the global model
        for _ in range(self.num_clients):
            self.__accept_connection()
        print(f"{SERVER_INFO_CONNECTION} Broadcasted the global model to all clients.")

        if self.is_FL:
            if len(self.connections) == self.num_clients:
                print(f"{'-'*90}\n{SERVER_INFO_TRAINING} Received connections from {self.num_clients} clients. Begin the learning process\n{'-'*90}")
                # The main federated learning loop
                self.__FL_loop()
                print(f"{SERVER_INFO_TRAINING} ✅ FEDERATED LEARNING PROCESS COMPLETED")

        else:
            if len(self.connections) == self.num_clients:
            # Without this part, the server will close the connections after distributing the model
                for f in concurrent.futures.as_completed(self.results):
                    model = f.result()

        # This stupid line took me 2 days to figure out the error :(
        # self.__close_connections()

        if not self.multi_system:
            pass




"""----------------------------------------------------------------------
CLIENT
"""
class FL_Client():

    def __init__(self, initial_model, name, data_loader, params, results_path, framework='TensorFlow', is_FL=None, server_ip=LOCAL_IP, server_port=PORT):

        self.initial_model = initial_model
        self.name = name
        self.server_ip = server_ip
        self.server_port = server_port
        self.s = socket.socket()
        self.params = params
        self.framework = framework
        self.results_path = results_path
        self.FL_train_loss_list = []
        self.FL_val_loss_list = []
        self.FL_val_metrics_dict = {}
        self.replaced_modules = []
        self.is_FL = is_FL
        
        if framework == 'TensorFlow':
            self.X = data_loader[0] 
            self.Y = data_loader[1]
            self.samplesize, self.sample, self.height, self.width, self.channels = self.X.shape[:]
            
        elif framework == 'PyTorch':
            self.train_loader = data_loader[0]
            self.val_loader = data_loader[1]
            self.test_loader = data_loader[2]
            self.scaler = data_loader[3]

        elif framework == 'StatsModels':

            time_series_data = data_loader

            # Split the data into training, validation, and test sets
            train_size = int(len(time_series_data) * 0.7)
            val_size = int(len(time_series_data) * 0.2)
            test_size = len(time_series_data) - train_size - val_size

            self.train_data = time_series_data[:train_size]
            self.val_data = time_series_data[train_size:train_size + val_size]
            self.test_data = time_series_data[train_size + val_size:]

        print(f"{'-'*90}\n{GENERAL_INFO} {self.name}: We use {self.params.device}\n{'-'*90}")

            

    # FEDERATED LEARNING TRAIN
    def FL_train(self, loss=None, optimizer=None, epochs=None, batch_size=None, val_ratio=0.1, verbose=0):

        self.s.connect((self.server_ip, self.server_port))
        print(f"{CLIENT_INFO_CONNECTION} {self.name} CONNECTED SUCCESSFULLY to {self.server_ip}:{self.server_port}")
        
        # Receive the number of rounds and the model from the server
        rounds = int(self.s.recv(BUFFER_SIZE).decode(FORMAT))
        
        # Receive the model from the server, otherwise, the client keeps waiting
        self.model = socket_receive(self.s, BUFFER_SIZE, self.params)

        # I use the initial model because the num_nodes parameter of the model received from the server is not adapted to the client's data
        self.model = self.initial_model.model

        # Initialize the layer scores
        self.model.layer_scores = np.ones(len(self.model.get_weights()[0]))/3

        if DEBUG_VERBOSE:
            print(f'[Debug] {self.name} SUM OF WEIGHTS (initial): {sum([np.sum(w) for w in self.model.get_weights()[0]]):.2f}')

        # Load the best model if specified
        if self.params.load_best_model != '':
            self.load_best_model(os.path.join(self.params.load_best_model, f"best_model_{self.name}.pth"))

        print(f'{CLIENT_INFO_TRAINING} {self.name} FEDERATED TRAINING STARTED.')
        
        for i in range(rounds):
   
            self.local_train(loss=loss, optimizer=optimizer, epochs=epochs, batch_size=batch_size, val_ratio=val_ratio, verbose=verbose)
            
            # Append the loss and metrics to the global lists
            self.FL_val_loss_list = self.FL_val_loss_list + self.trainer.val_loss_list
            self.FL_train_loss_list = self.FL_train_loss_list + self.trainer.train_loss_list
            for metric, value in self.trainer.val_metrics_dict.items():
                if metric in self.FL_val_metrics_dict:
                    self.FL_val_metrics_dict[metric] = self.FL_val_metrics_dict[metric] + value
                else:
                    self.FL_val_metrics_dict[metric] = value


            if DEBUG_VERBOSE:
                print(f'[Debug] {self.name} SUM OF WEIGHTS (before aggregation): {sum([np.sum(w) for w in self.model.get_weights()[0]]):.2f}')

            #🟥
            to_be_sent = [self.model.get_weights()[0], self.model.layer_scores]
            socket_send(self.s, to_be_sent)

            print(f'{CLIENT_INFO_TRAINING} {self.name} Round {i+1}/{rounds} completed')

            # Receive aggregated weights from the server
            aggregated_weights = socket_receive(self.s, BUFFER_SIZE, self.params)


            # Validate the weights before updating the model
            aggregated_weights, layer_scores, is_replaced = client_validate_weights(self.model, self.trainer, aggregated_weights, self.params)
            self.model.layer_scores = layer_scores

            # Store the modules that have been replaced
            if is_replaced:
                self.replaced_modules.append(is_replaced)
            
            # Update the model weights
            if i < rounds-1:
                self.model.set_weights(aggregated_weights)
            
            if DEBUG_VERBOSE:
                print(f'[Debug] {self.name} SUM OF WEIGHTS (after aggregation): {sum([np.sum(w) for w in self.model.get_weights()[0]]):.2f}')

        # FL_train_log_path = os.path.join(os.getcwd(), self.results_path, f"{self.name}_FL_training_logs.json")
        # save_training_results(self, FL_train_log_path, client_name=self.name, is_FL=True)
        
        print(f'{CLIENT_INFO_TRAINING} {self.name} Federated Training completed !')
        


    # CENTRALIZED LEARNING TRAIN
    def CL_train(self, loss=None, optimizer=None, epochs=None, batch_size=None, val_ratio=0.1, callbacks=[], verbose=0):
        
        self.s.connect((self.server_ip, self.server_port))
        print(f"{CLIENT_INFO_CONNECTION} {self.name} [CONNECTED] to {self.server_ip}:{self.server_port}")
        
        rounds = int(self.s.recv(BUFFER_SIZE).decode(FORMAT))

        # Receive the model from the server, otherwise, the client keeps waiting
        self.model = socket_receive(self.s, BUFFER_SIZE, self.params)

        # I use the initial model because the num_nodes parameter of the model received from the server is not adapted to the client's data
        self.model = self.initial_model.model

        # Initialize the layer scores
        self.model.layer_scores = np.ones(len(self.model.get_weights()[0]))/3

        if DEBUG_VERBOSE:
            print(f'[Debug] {self.name} SUM OF WEIGHTS (initial): {sum([np.sum(w) for w in self.model.get_weights()[0]]):.2f}')
        
        # Load the best model if specified
        if self.params.load_best_model != '':
            self.load_best_model(os.path.join(self.params.load_best_model, f"best_model_{self.name}.pth"))

        print(f'{CLIENT_INFO_TRAINING} {self.name} CENTRALIZED TRAINING STARTED.')
        self.local_train(loss=loss, optimizer=optimizer, epochs=epochs, batch_size=batch_size, val_ratio=val_ratio, callbacks=callbacks, verbose=verbose)
        if DEBUG_VERBOSE:
            print(f'[Debug] {self.name} SUM OF WEIGHTS (after centralized training): {sum([np.sum(w) for w in self.model.get_weights()[0]]):.2f}')
        
        socket_send(self.s, self.model)
        print(f'{CLIENT_INFO_TRAINING} {self.name} CENTRALIZED TRAINING COMPLETED !')

        self.s.close()
        


    def local_train(self, loss=None, optimizer=None, epochs=None, batch_size=None, val_ratio=0.1, callbacks=[], verbose=0):

        #🟩
        if self.framework == 'TensorFlow':
            
            # Remove old logs
            if self.server_ip != LOCAL_IP:
                if os.name == "nt":
                    os.system("if exist LOGS rmdir /s /q LOGS")
                else:
                    os.system(f'rm -rf ./LOGS/')

            params = Hyperparameters(model_name='FedConvLSTM')
            
            log_dir = f"{LOG_PATH}/{self.name}/centralized_run/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard( log_dir=log_dir, update_freq="epoch")
            optimizer = Adam(learning_rate=params.learning_rate)
            # Use GPU if available
            with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                if tf.config.list_physical_devices('GPU'):
                    print(f"{CLIENT_INFO_TRAINING} {self.name} Using GPU")
                else:
                    print(f"{CLIENT_INFO_TRAINING} {self.name} Using CPU")
                self.model.compile(loss=loss, optimizer=optimizer)
                L = int(self.samplesize*(1-val_ratio)) 

                self.model.fit(self.X[L:],
                            self.Y[L:],
                            validation_data=(self.X[L:], self.Y[L:]),
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=[tensorboard_callback]+callbacks,
                            verbose=verbose)

                if self.server_ip != LOCAL_IP:
                    os.system(f'tensorboard --logdir={LOG_PATH}')



        #🟧
        if self.framework == 'PyTorch':
            # Choose device and move the model to the device
            self.model = self.model.to(self.params.device)
            # Print model info
            if self.params.client_model_parameter_verbose:
                print(f"{CLIENT_INFO_MODEL} {self.name} 🟧 Model parameters:")
                print_model_info(CLIENT_INFO_TRAINING, CLIENT_INFO_MODEL, self.name, self.params)
                for name, param in self.model.named_parameters():
                    print(f"{CLIENT_INFO_MODEL} {self.name} - {name}: {param.shape}")
                print(f"{CLIENT_INFO_MODEL} {self.name} Total number of parameters: {sum(p.numel() for p in self.model.parameters())}")

            # Initialize optimizer
            optimizer = torch.optim.Adam(
                                        params=self.model.parameters(),
                                        lr=self.params.lr_init,
                                        eps=1.0e-8,
                                        weight_decay=0,
                                        amsgrad=False,
                                        )
            lr_scheduler = None
            if self.params.lr_decay:
                print(f"{CLIENT_INFO_TRAINING} {self.name} Applying learning rate decay.")
                lr_decay_steps = [int(i) for i in self.params.lr_decay_steps]
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer, milestones=lr_decay_steps, gamma=self.params.lr_decay_rate
                )
            
            trainer = Trainer(self.model,
                            loss,
                            optimizer,
                            self.train_loader,
                            self.val_loader,
                            self.test_loader,
                            self.scaler,
                            self.params,
                            lr_scheduler=lr_scheduler,
                            verbose=self.params.local_model_training_verbose,
                            name=self.name
                            )
            
            trainer.train()

            self.trainer=trainer

        if self.framework == 'statsmodels':
            return self.trainer.avg_metrics



    # Evaluate function for client 
    def test(self, X=None, Y=None, scaler=None, loss=None, results_path=None):

        print(f"{CLIENT_INFO_TESTING} {self.name} 🔍 Testing...")
        
        if self.framework == 'TensorFlow':
            self.model.compile(loss=loss)
            self.model.evaluate(X, Y)

            Y_predicted = self.model.predict(X)
            Y_predicted = scaler.inverse_transform(Y_predicted.reshape(-1, self.height*self.width))
            Y_predicted = Y_predicted.reshape(Y.shape)

            Y_original = scaler.inverse_transform(Y.reshape(-1, self.height*self.width))
            Y_original = Y_original.reshape(Y.shape)
            
            MAE = compute_metrics(Y_predicted, Y_original, metric='MAE')
            MSE = compute_metrics(Y_predicted, Y_original, metric='MSE')
            NRMSE = compute_metrics(Y_predicted, Y_original, metric='NRMSE')
            RMSE  = compute_metrics(Y_predicted, Y_original, metric='RMSE')
            MAPE  = compute_metrics(Y_predicted, Y_original, metric='MAPE')
            R2    = compute_metrics(Y_predicted, Y_original, metric='R2')
            

            print(f"{CLIENT_INFO_TESTING} {self.name} Metrics:")
            print(f"{CLIENT_INFO_TESTING} {self.name} 📊 MAE: {MAE:.2f} (passengers)")
            print(f"{CLIENT_INFO_TESTING} {self.name} 📊 MAPE: {MAPE:.2f} (%)")
            print(f"{CLIENT_INFO_TESTING} {self.name} 📊 MSE: {MSE:.2f} (passengers)^2")
            print(f"{CLIENT_INFO_TESTING} {self.name} 📊 RMSE: {RMSE:.2f} (passengers)")
            print(f"{CLIENT_INFO_TESTING} {self.name} 📊 NRMSE: {NRMSE:.2f}")
            print(f"{CLIENT_INFO_TESTING} {self.name} 📊 R2: {R2:.2f}")

        if self.framework == 'PyTorch':
            avg_metrics, true_pred_y = self.trainer.test(results_path=results_path)
            if self.params.save_predictions:
                df = pd.DataFrame(true_pred_y)
                if self.params.load_best_model != '':
                    df.to_csv(os.path.join(self.params.load_best_model, f"{self.name}_predictions.csv"), index=False)
                else:
                    df.to_csv(os.path.join(self.results_path, f"{self.name}_{self.params.FL_scheme}_{self.params.model_name}_predictions.csv"), index=False)
            return avg_metrics


    def load_best_model(self, load_path):
        """Loads the best model from the specified path."""
        if os.path.exists(load_path):
            print(f"{CLIENT_INFO_TRAINING} {self.name} Loading best model from {load_path}")
            self.model.load_state_dict(torch.load(load_path, weights_only=True))
        else:
            print(f"{CLIENT_INFO_TRAINING} {self.name} Best model not found at {load_path}. Learn from scratch.")



# Helper functions
def print_model_info(CLIENT_INFO_TRAINING, CLIENT_INFO_MODEL, name, params):
    print(f'{CLIENT_INFO_TRAINING} {name} Running on {params.device}')
    print(f'{CLIENT_INFO_TRAINING} {name} Running on {params.device}')
    print(f"{CLIENT_INFO_MODEL} {name} Embed Dimension: {params.embed_dim}")
    print(f"{CLIENT_INFO_MODEL} {name} Number of Layers: {params.num_layers}")
    print(f"{CLIENT_INFO_MODEL} {name} RNN Units: {params.rnn_units}")
    print(f"{CLIENT_INFO_MODEL} {name} Chebyshev Polynomial Order: {params.cheb_k}")
    print(f"{CLIENT_INFO_MODEL} {name} Initial Learning Rate: {params.lr_init}")
    print(f"{CLIENT_INFO_MODEL} {name} Number of Heads: {params.num_heads}")
    print(f"{CLIENT_INFO_MODEL} {name} Hidden Dimension of Node: {params.hidden_dim_node}")
    print(f"{CLIENT_INFO_MODEL} {name} Number of Layers of Node: {params.num_layers_node}")
    