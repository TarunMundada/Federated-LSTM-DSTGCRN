"""
Description:
    Architecture of the DSTGCRN (Dynamic Spatial-Temporal Graph Convolutional Recurrent Network) model.
    Note: Code segments are adapted from 'https://github.com/LeiBAI/AGCRN' and 'https://github.com/wengwenchao123/DDGCRN'
Authors: LeiBAI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from collections import OrderedDict 



class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, hyperGNN_dim1, hyperGNN_dim2):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out)
        )
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.hyperGNN_dim1 = hyperGNN_dim1
        self.hyperGNN_dim2 = hyperGNN_dim2
        self.embed_dim = embed_dim
        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(dim_in, self.hyperGNN_dim1)),
                    ("sigmoid1", nn.Sigmoid()),
                    ("fc2", nn.Linear(self.hyperGNN_dim1, self.hyperGNN_dim2)),
                    ("sigmoid2", nn.Sigmoid()),
                    ("fc3", nn.Linear(self.hyperGNN_dim2, self.embed_dim)),
                ]
            )
        )

    def forward(self, x, node_embeddings):
        # Get the batch size and number of nodes
        batch_size = x.shape[0]

        # Ensure node_embeddings is in shape [N, D] (remove batch dim if it's 1)
        if node_embeddings.dim() == 3 and node_embeddings.shape[0] == 1:
            node_embeddings2 = node_embeddings.squeeze(0)  # Shape: [N, D]

        # Check the number of nodes (N) and dimension (D)
        node_num, embed_dim = node_embeddings2.shape

        # Initialize support matrix with an identity matrix and expand it to have batch dimension
        supports1 = torch.eye(node_num).to(node_embeddings2.device)
        supports1 = supports1.unsqueeze(0).expand(batch_size, -1, -1).to(node_embeddings2.device)  # Shape: [B, N, N]

        # Compute the filter from input x and node embeddings
        filter = self.fc(x)  # Assuming the output has shape [B, N, dim_in]
        

        # Expand node_embeddings to have batch dimension and apply element-wise multiplication
        node_embeddings_expanded = node_embeddings2.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [B, N, D]
        nodevec = torch.tanh(torch.mul(node_embeddings_expanded, filter))  # Shape: [B, N, dim_in]

        # Compute the supports (laplacian) by ensuring dimensions match
        supports2 = AVWGCN.get_laplacian(F.relu(torch.matmul(nodevec, nodevec.transpose(2, 1))), supports1)  # Shape: [B, N, N]
        supports3 = AVWGCN.get_laplacian(F.relu(torch.matmul(nodevec, nodevec.transpose(2, 1))), supports2)  # Shape: [B, N, N]

        # Apply graph convolution for each support
        x_g1 = torch.einsum("bnm,bmc->bnc", supports1, x)  # Shape: [B, N, dim_in]
        x_g2 = torch.einsum("bnm,bmc->bnc", supports2, x)  # Shape: [B, N, dim_in]
        x_g3 = torch.einsum("bnm,bmc->bnc", supports3, x)  # Shape: [B, N, dim_in]

        # Stack the graph convolutions
        x_g = torch.stack([x_g1, x_g2, x_g3], dim=1)  # Shape: [B, 3, N, dim_in]

        # Compute the weights and bias using node embeddings
        weights = torch.einsum("nd,dkio->nkio", node_embeddings2, self.weights_pool)  # Shape: [N, cheb_k, dim_in, dim_out]
        bias = torch.matmul(node_embeddings2, self.bias_pool)  # Shape: [N, dim_out]

        # Permute x_g to align dimensions: [B, N, cheb_k, dim_in]
        x_g = x_g.permute(0, 2, 1, 3)

        # Compute the final graph convolution
        x_gconv = torch.einsum("bnki,nkio->bno", x_g, weights) + bias  # Shape: [B, N, dim_out]

        return x_gconv, supports2  # Return x_gconv and the second support matrix (supports2)






    @staticmethod
    def get_laplacian(graph, I, normalize=True):
        if normalize:
            graph = graph + I
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.matmul(torch.matmul(D, graph), D)

        return L


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, hyperGNN_dim1, hyperGNN_dim2):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim, hyperGNN_dim1, hyperGNN_dim2)
        self.update = AVWGCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim, hyperGNN_dim1, hyperGNN_dim2)

    def forward(self, x, state, node_embeddings):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r, adjmatrix = self.gate(input_and_state, node_embeddings)
        z_r = torch.sigmoid(z_r)
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc, _ = self.update(candidate, node_embeddings)
        hc = torch.tanh(hc)
        h = r * state + (1 - r) * hc  # B, num_nodes, hidden_dim
        return h, adjmatrix

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, hyperGNN_dim1, hyperGNN_dim2, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, "At least one DCRNN layer in the Encoder."
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim, hyperGNN_dim1, hyperGNN_dim2))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(
                AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim, hyperGNN_dim1, hyperGNN_dim2)
            )
        # # Initialize adjacency matrix, considering the shape and data type you need
        # self.adjmatrix = torch.zeros(32, node_num, node_num)

    def forward(self, x, init_state, node_embeddings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        # shape of node_embeddings: (T, N, D)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            adjmatrices = []
            for t in range(seq_length):
                state, adjmatrix = self.dcrnn_cells[i](
                    current_inputs[:, t, :, :], state, node_embeddings
                )
                inner_states.append(state)
                adjmatrices.append(adjmatrix)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        # adjmatrices: adj for the last layer
        return current_inputs, output_hidden, adjmatrices

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


class AGCRN(nn.Module):
    def __init__(self, args):
        super(AGCRN, self).__init__()

        # Here, I set the number of nodes to 1 for dummy calls, i.e.,
        # for the calls where num_nodes does not matter.
        self.model_name = "AGCRN"
        args.num_nodes = 1 if not hasattr(args, 'num_nodes') else args.num_nodes
        self.num_nodes = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.hidden_dim_node = args.hidden_dim_node
        self.output_dim = args.output_dim
        self.lookahead = args.lookahead
        self.num_layers = args.num_layers
        self.embed_dim = args.embed_dim
        self.batch = args.batch_size

        # For FL training
        self.layer_scores = []

        self.node_embeddings = nn.Parameter(
            torch.randn(1, self.num_nodes, args.embed_dim), requires_grad=True
        )

        self.encoder = AVWDCRNN(
            args.num_nodes,
            args.input_dim,
            args.rnn_units,
            args.cheb_k,
            args.embed_dim,
            args.hyperGNN_dim1, 
            args.hyperGNN_dim2,
            args.num_layers,
        )

        # Predictor
        self.end_conv = nn.Conv2d(
            1,
            args.lookahead * self.output_dim,
            kernel_size=(1, self.hidden_dim),
            bias=True,
        )

    def forward(self, source):
        # Source: B, T_1, N, D
        # Target: B, T_2, N, D

        # node_embeddings = nn.Parameter(
        #     torch.randn(1, self.num_nodes, self.embed_dim).to(source.device), requires_grad=True
        # )

        init_state = self.encoder.init_hidden(
            source.shape[0]
        )  # num_layers, B, N, hidden
        output, _, adjmatrices = self.encoder(
            source, init_state, self.node_embeddings
        )  # B, T, N, hidden
        output = output[:, -1:, :, :]  # B, 1, N, hidden

        # CNN based predictor
        output = self.end_conv((output))  # B, T*C, N, 1
        output = output.squeeze(-1).reshape(
            -1, self.lookahead, self.output_dim, self.num_nodes
        )
        output = output.permute(0, 1, 3, 2)  # B, T, N, C
        adjmatrices = torch.stack(adjmatrices, dim=1)

        return output, adjmatrices
    
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
