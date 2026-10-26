import torch
from torch_geometric.nn import GATConv, to_hetero


class GATEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, hop_depth=2, heads=4):
        super().__init__()
        if hop_depth < 1:
            raise ValueError("hop_depth muss mindestens 1 sein.")
        self.convs = torch.nn.ModuleList()
        for i in range(hop_depth):
            last = i == hop_depth - 1
            self.convs.append(GATConv(
                (-1, -1),
                out_channels if last else hidden_channels // heads,
                heads=heads,
                concat=not last,
                add_self_loops=False,
                residual=True,
            ))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu()
        return x


class GAT(torch.nn.Module):
    def __init__(self, metadata, hidden_dim=64, out_dim=32, hop_depth=2, heads=4):
        super().__init__()
        self.encoder = to_hetero(GATEncoder(hidden_dim, out_dim, hop_depth, heads), metadata)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(out_dim * 2, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, 1)
        )

    def encode(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)

    def decode(self, h_customer, h_product):
        return self.predictor(torch.cat([h_customer, h_product], dim=-1)).squeeze(-1)
