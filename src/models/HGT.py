import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv

MAIN_TYPES = ('customer', 'product')


class HGT(nn.Module):
    def __init__(self, metadata, hidden_dim=64, out_dim=32, num_layers=2, heads=4):
        super().__init__()
        node_types = metadata[0]
        self.lin_in = nn.ModuleDict({nt: nn.LazyLinear(hidden_dim) for nt in node_types})
        self.convs = nn.ModuleList([HGTConv(hidden_dim, hidden_dim, metadata, heads) for _ in range(num_layers)])
        self.lin_out = nn.ModuleDict({nt: nn.Linear(hidden_dim, out_dim) for nt in MAIN_TYPES})
        self.predictor = nn.Sequential(nn.Linear(2 * out_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def encode(self, x_dict, edge_index_dict):
        h = {nt: self.lin_in[nt](x).relu() for nt, x in x_dict.items()}
        for conv in self.convs:
            h = conv(h, edge_index_dict)
        return {nt: self.lin_out[nt](h[nt]) for nt in MAIN_TYPES}

    def decode(self, h_customer, h_product):
        return self.predictor(torch.cat([h_customer, h_product], dim=-1)).squeeze(-1)
