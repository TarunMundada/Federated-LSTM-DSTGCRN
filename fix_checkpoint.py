import torch
import numpy as np
from MODELS.LSTM_DSTGCRN.LSTM_DSTGCRN import LSTM_DSTGCRN
from Hyperparameters import get_hyperparameters

args = get_hyperparameters(model_name="LSTM_DSTGCRN", is_FL=True)[0]

# Override critical inference-time values to match training
args.input_dim = 3
args.output_dim = 1
args.lookback = 12
args.lookahead = 3
args.batch_size = 64
args.num_nodes = 35
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHECKPOINT_PATH = 'checkpoints/round_1_epoch_19.pt'
FIXED_PATH = 'checkpoints/round_1_epoch_19_FIXED.pt'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model ===
model = LSTM_DSTGCRN(args).to(DEVICE)

# === Load broken checkpoint (NumPy weights) ===
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
numpy_weights = checkpoint['weights']
print(f"✅ Loaded NumPy weights: {len(numpy_weights)} tensors")

# === Inject weights manually ===
with torch.no_grad():
    for i, param in enumerate(model.parameters()):
        param.copy_(torch.tensor(numpy_weights[i]))

# === Save fixed checkpoint ===
torch.save({
    'epoch': checkpoint['epoch'],
    'round': checkpoint['round'],
    'weights': model.state_dict()
}, FIXED_PATH)

print(f"✅ Saved fixed checkpoint to {FIXED_PATH}")

# print("\n==== MODEL PARAMETER SHAPES ====")
# for i, param in enumerate(model.parameters()):
#     print(f"Param {i}: {tuple(param.shape)}")

# print("\n==== NUMPY WEIGHT SHAPES ====")
# for i, weight in enumerate(numpy_weights):
#     print(f"Weight {i}: {weight.shape}")
