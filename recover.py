import torch
import numpy as np
from MODELS.LSTM_DSTGCRN.LSTM_DSTGCRN import LSTM_DSTGCRN
from MODELS.HELPERS.normalization import StandardScaler
from MODELS.HELPERS.Utils import evaluate_metrics
from MODELS.HELPERS.load_dataset import load_and_transform_data
from torch.utils.data import DataLoader, TensorDataset
from Hyperparameters import get_hyperparameters

args = get_hyperparameters(model_name="LSTM_DSTGCRN", is_FL=True)[0]

# # Override critical inference-time values to match training
# args.input_dim = 1
# args.output_dim = 1
# args.lookback = 12
# args.lookahead = 3
# args.batch_size = 64
# args.num_nodes = 35
# args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Configurable paths ===
CHECKPOINT_PATH = 'checkpoints/round_1_epoch_19_FIXED.pt'
DATA_FILE = 'DATA/TransportModes/NYC-taxi/tripdata_full.csv'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# === Step 2: Load and Crop Data to Match Checkpoint ===
data = load_and_transform_data(DATA_FILE)

# Crop to match checkpoint shape: 35 nodes, 3 input features
data = data[:, :35, :3]  # [T, 35, 3]
T, N, D = data.shape
scaler = StandardScaler(mean=np.mean(data), std=np.std(data))

# === Step 3: Set Args to Match Training Setup ===
args.input_dim = 3
args.num_nodes = 35
args.output_dim = 1
args.lookback = 12
args.lookahead = 3
args.batch_size = 64
args.device = DEVICE
args.embed_dim = 16
args.rnn_units = 32
args.hidden_dim_node = 16
args.hyperGNN_dim1 = 8
args.hyperGNN_dim2 = 16
args.num_layers_node = 2
args.num_heads = 4
args.num_layers = 2
args.dynamic_embed = True
args.attention_layer = True
args.lstm_layer = True

# === Step 4: Prepare Dataloader ===
def create_dataloader(data, lookback, lookahead, batch_size):
    X, Y = [], []
    for i in range(data.shape[0] - lookback - lookahead):
        X.append(data[i:i+lookback])
        Y.append(data[i+lookback:i+lookback+lookahead])
    X = np.array(X)
    Y = np.array(Y)
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

test_loader = create_dataloader(data, args.lookback, args.lookahead, args.batch_size)

# === Step 5: Load Model and Weights ===
model = LSTM_DSTGCRN(args).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

print("✅ Checkpoint keys:", checkpoint.keys())
print("✅ Type of weights:", type(checkpoint['weights']))

if isinstance(checkpoint['weights'], dict):
    model.load_state_dict(checkpoint['weights'])
    print("✅ Model weights loaded successfully.")
else:
    raise ValueError("❌ Checkpoint 'weights' is not a state_dict.")

model.eval()

# === Step 6: Run Inference ===
y_true, y_pred = [], []
with torch.no_grad():
    for source, label in test_loader:
        source = source[..., :args.input_dim].to(DEVICE)
        label = label[..., :args.output_dim].to(DEVICE)

        output, _ = model(source)

        # Inverse transform
        label = label * scaler.std + scaler.mean
        output = output * scaler.std + scaler.mean

        y_true.append(label.cpu().numpy())
        y_pred.append(output.cpu().numpy())

y_true = np.concatenate(y_true, axis=0)
y_pred = np.concatenate(y_pred, axis=0)

# === Step 7: Save Results ===
np.save('y_true_round1_epoch19.npy', y_true)
np.save('y_pred_round1_epoch19.npy', y_pred)

print("✅ Saved y_true and y_pred to .npy files.")