import torch
import torch.nn as nn
from torch_geometric.utils import scatter

MAIN_TYPES = ('customer', 'product')


def attribute_features(x_dict, edge_index_dict):
    out = {}
    for nt in MAIN_TYPES:
        columns = [x_dict[nt]]
        for (src_type, _, dst_type), (src, dst) in sorted(edge_index_dict.items()):
            if dst_type == nt and src_type not in MAIN_TYPES:
                columns.append(scatter(x_dict[src_type][src], dst, dim=0,
                                       dim_size=x_dict[nt].size(0), reduce='sum'))
        out[nt] = torch.cat(columns, dim=1)
    return out


class MLP(nn.Module):
    def __init__(self, metadata, hidden_dim=64, out_dim=32):
        super().__init__()
        self.encoders = nn.ModuleDict({
            nt: nn.Sequential(nn.LazyLinear(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
            for nt in MAIN_TYPES
        })
        self.predictor = nn.Sequential(nn.Linear(2 * out_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def encode(self, x_dict, edge_index_dict):
        features = attribute_features(x_dict, edge_index_dict)
        return {nt: self.encoders[nt](features[nt]) for nt in MAIN_TYPES}

    def decode(self, h_customer, h_product):
        return self.predictor(torch.cat([h_customer, h_product], dim=-1)).squeeze(-1)
