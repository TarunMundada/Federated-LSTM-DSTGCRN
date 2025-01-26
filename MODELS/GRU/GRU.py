import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, params):
        super(GRU, self).__init__()
        params.num_nodes = 1 if not hasattr(params, 'num_nodes') else params.num_nodes
        self.params = params
        self.model_name = "GRU"
        self.num_nodes = params.num_nodes
        self.input_dim = params.input_dim
        self.hidden_dim = params.hidden_dim_node
        self.num_layers = params.num_layers_node
        self.output_dim = params.output_dim
        self.lookahead = params.lookahead

        if torch.cuda.is_available():
            self.device = params.device
        else:
            self.device = "cpu"

        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)  # Use GRU here
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        # x shape: (batch_size, lookback, num_nodes, input_dim)
        batch_size = x.size(0)
        lookback = x.size(1)  # lookback 
        num_nodes = x.size(2)  # number of nodes 

        # Ensure the number of nodes matches the expected value
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, but got {num_nodes}")

        x = x.permute(0, 2, 1, 3).contiguous()  # (batch_size, num_nodes, lookback, input_dim)

        # Flatten for GRU, keeping the sequence length
        x = x.view(batch_size * self.num_nodes, lookback, self.input_dim) 

        h0 = torch.zeros(self.num_layers, batch_size * self.num_nodes, self.hidden_dim).to(self.device)
        
        # GRU only needs the hidden state, not the cell state
        out, _ = self.gru(x, h0)  # out: (batch_size * num_nodes, lookback, hidden_dim)

        # Take the output of the last time step for each node
        out = out[:, -self.lookahead:, :]  # (batch_size * num_nodes, lookahead, hidden_dim)

        # Pass through the fully connected layer for each time step
        out = out.view(batch_size * self.num_nodes, self.lookahead, self.hidden_dim)
        out = out.reshape(-1, self.hidden_dim)  # Prepare for FC layer
        out = self.fc(out)  # (batch_size * num_nodes * lookahead, output_dim)

        # Reshape back to (batch_size, lookahead, num_nodes, output_dim)
        out = out.view(batch_size, self.lookahead, self.num_nodes, self.output_dim)

        return out, _
    
    def get_weights(self):
        names = []
        weights = []
        for name, param in self.named_parameters():
            if param.requires_grad:  # Only include trainable weights
                weights.append(param.data.cpu().numpy())  # Move to CPU and convert to NumPy array
                names.append(name)
        return [weights, names]
    
    def set_weights(self, weights):
        for param, weight in zip(self.parameters(), weights):
            param.data = torch.from_numpy(weight).to(param.device)
