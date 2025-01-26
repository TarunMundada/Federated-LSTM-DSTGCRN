from datetime import datetime
import itertools

FL_SCHEMES = ['FedAvg', 'ClientSideValidation', 'Attentive', 'Module-wise', 'Only-LSTM-module', 'Only-Attention-module', 'Only-AGCRN-module']


def get_hyperparameters(model_name, is_FL):

    if is_FL:
        hyperparameters = {
            'is_FL': [True],
            'FL_rounds': [2],
            'FL_scheme': FL_SCHEMES[1:2],
            'to_weight_clients': [False],
            'use_CSV': [False],
            'batch_size': [32],
            'epochs': [500],
            'num_layers_node': [2],
            'hidden_dim_node': [16],
            'embed_dim': [16],
            'num_heads': [4],
            'hyperGNN_dim1': [8],
            'hyperGNN_dim2': [16],
            'num_layers': [2],
            'rnn_units': [32],
            'lr_init': [0.001],
            'attention_layer': [True],
            'gru_layer': [False],
            'lstm_layer': [True]
        }
    else:
        hyperparameters = {
            'is_FL': [False],
            'FL_rounds': [1],
            'FL_scheme': ['LocalLearning'],
            'to_weight_clients': [False],
            'use_CSV': [None],
            'batch_size': [32],
            'epochs': [5],
            'num_layers_node': [2],
            'hidden_dim_node': [16],
            'embed_dim': [16],
            'num_heads': [4],
            'hyperGNN_dim1': [8],
            'hyperGNN_dim2': [16],
            'num_layers': [2],
            'rnn_units': [32],
            'lr_init': [0.001],
            'gru_layer': [False],
            'lstm_layer': [False]
        }

    hyperparameter_combinations = list(itertools.product(*hyperparameters.values()))

    CONFIGURATIONS = []
    for combination in hyperparameter_combinations:
        config_dict = dict(zip(hyperparameters.keys(), combination))
        CONFIGURATIONS.append(Hyperparameters(
            model_name=model_name,
            **config_dict
        ))
    return CONFIGURATIONS



class Hyperparameters:
    """
    Class representing hyperparameters for a federated learning model.
    """
    
    def __init__(self, 
                 is_FL=None,
                 model_name=None,
                 FL_rounds=None, 
                 FL_scheme=None, 
                 use_CSV=None, 
                 batch_size=None, 
                 epochs=None, 
                 num_layers_node=None, 
                 hidden_dim_node=None, 
                 embed_dim=None, 
                 num_heads=None, 
                 hyperGNN_dim1=None, 
                 hyperGNN_dim2=None, 
                 num_layers=None, 
                 rnn_units=None,  
                 FL_verbose=True, 
                 to_weight_clients=None, 
                 val_ratio=0.2, 
                 test_ratio=0.1, 
                 load_best_model='', 
                 seed=42, 
                 lookback=10, 
                 lookahead=1, 
                 input_dim=3, 
                 output_dim=1,
                 cheb_k=3,
                 lr_init=None, 
                 lr_decay=False, 
                 lr_decay_rate=0.5, 
                 lr_decay_steps=[50, 100, 200, 300], 
                 early_stop=True, 
                 early_stop_patience=100, 
                 dynamic_embed=True, 
                 attention_layer=True,
                 gru_layer=False, 
                 lstm_layer=True
                 ):

        #===================================================================================================
        # Common hyperparameters
        self.is_FL = is_FL
        self.FL_rounds = FL_rounds
        self.FL_scheme = FL_scheme
        self.use_CSV = use_CSV
        self.to_weight_clients = to_weight_clients
        self.FL_verbose = FL_verbose
        self.model_name = model_name
        self.client_model_parameter_verbose = False
        self.local_model_training_verbose = True
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.load_best_model = load_best_model
        self.seed = seed


        #===================================================================================================
        if self.model_name=='Fed-LSTM-DSTGCRN':
                        
            # Data hyperparameters-----------------------------------
            self.batch_size = batch_size
            self.epochs = epochs

            # Periods------------------------------------------------
            self.lookback = lookback
            self.lookahead = lookahead

            # LSTM layer --------------------------------------------
            self.num_layers_node = num_layers_node

            # Hidden dim for the LSTM (d_g)--------------------------
            self.hidden_dim_node = hidden_dim_node

            # Embedding dim (d_e) after linear layer-----------------
            self.embed_dim = embed_dim

            # Attention layer, should be divisible by self.embed_dim
            self.num_heads = num_heads

            # GNN-  -------------------------------------------------
            self.hyperGNN_dim1 = hyperGNN_dim1
            self.hyperGNN_dim2 = hyperGNN_dim2

            # Number of layers in GCRN-------------------------------
            self.num_layers = num_layers

            # Hidden dim for RNN i.e. the last Conv2d layer (d_h)
            self.rnn_units = rnn_units

            self.input_dim = input_dim  # Number of features
            self.output_dim = output_dim
            
            # Chebyshev polynomials----------------------------------
            self.cheb_k = cheb_k

            # Normalization------------------------------------------
            self.normalizer = "std" #max01 # None
            self.normalized_col = None # Means all columns
            self.column_wise = False

            # Train--------------------------------------------------
            self.lr_init = lr_init
            self.lr_decay = lr_decay
            self.lr_decay_rate = lr_decay_rate
            self.lr_decay_steps = lr_decay_steps

            self.early_stop = early_stop
            self.early_stop_patience = early_stop_patience

            # Log-----------------------------------------------------
            self.log_dir = f"LOGS/{self.model_name}-" + datetime.now().strftime("%Y%m%d-%H%M")

            self.save_arrays_EDA = False
            self.device = "cuda:0"
            self.log_step = 10

            # Ablation study------------------------------------------
            self.TNE = False # Time Specific (naive solution)
            self.dynamic_embed = dynamic_embed # Static or not
            self.attention_layer = attention_layer # Attention layer in the dynamic module -> w/o Attention
            self.gru_layer = gru_layer # GRU layer in the dynamic module -> w/o GRU
            self.lstm_layer = lstm_layer # LSTM layer in the dynamic module -> w/o LSTM
            self.saved_model_path = None
            self.save_predictions = True





        #===================================================================================================
        if self.model_name in ['FedLSTM','FedGRU', 'FedARIMA', 'FedLR', 'FedAGCRN']:
                        
            # Data hyperparameters-----------------------------------
            self.batch_size = batch_size
            self.epochs = epochs

            # Periods------------------------------------------------
            self.lookback = lookback
            self.lookahead = lookahead

            # LSTM layer --------------------------------------------
            self.num_layers_node = num_layers_node

            # Hidden dim for the LSTM (d_g)--------------------------
            self.hidden_dim_node = hidden_dim_node

            # Embedding dim (d_e) after linear layer-----------------
            self.embed_dim = embed_dim

            # Attention layer, should be divisible by self.embed_dim
            self.num_heads = num_heads

            # GNN----------------------------------------------------
            self.hyperGNN_dim1 = hyperGNN_dim1
            self.hyperGNN_dim2 = hyperGNN_dim2

            # Number of layers in GCRN-------------------------------
            self.num_layers = num_layers

            # Hidden dim for RNN i.e. the last Conv2d layer (d_h)
            self.rnn_units = rnn_units 

            self.input_dim = input_dim  # Number of features
            self.output_dim = output_dim
            
            # Chebyshev polynomials----------------------------------
            self.cheb_k = cheb_k

            # Normalization------------------------------------------
            self.normalizer = "std" #max01 # None
            self.normalized_col = None # Means all columns
            self.column_wise = False

            # Train:--------------------------------------------------
            self.lr_init = lr_init
            self.lr_decay = lr_decay
            self.lr_decay_rate = lr_decay_rate
            self.lr_decay_steps = lr_decay_steps

            self.early_stop = early_stop
            self.early_stop_patience = early_stop_patience

            # Log-----------------------------------------------------
            self.log_dir = f"LOGS/{self.model_name}-" + datetime.now().strftime("%Y%m%d-%H%M")

            self.save_arrays_EDA = False

            self.device = "cuda:0"
            self.log_step = 10

            # Ablation study------------------------------------------
            self.TNE = False # Time Specific (naive solution)
            self.dynamic_embed = dynamic_embed # Static or not
            self.attention_layer = attention_layer # Attention layer in the dynamic module -> w/o Attention
            self.gru_layer = gru_layer # GRU layer in the dynamic module -> w/o GRU
            self.lstm_layer = lstm_layer # GRU layer in the dynamic module -> w/o GRU
            self.saved_model_path = None
            self.save_predictions = True
