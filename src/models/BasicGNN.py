import torch
from torch_geometric.nn import to_hetero, SAGEConv


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, hop_depth=2):
        super().__init__()

        if hop_depth < 1:
            raise ValueError("hop_depth muss mindestens 1 sein.")

        self.hop_depth = hop_depth

        self.convs = torch.nn.ModuleList()

        if hop_depth == 1:
            self.convs.append(
                SAGEConv((-1, -1), out_channels)
            )
        else:
            self.convs.append(
                SAGEConv((-1, -1), hidden_channels)
            )

            for _ in range(hop_depth - 2):
                self.convs.append(
                    SAGEConv((hidden_channels, hidden_channels), hidden_channels)
                )

            self.convs.append(
                SAGEConv((hidden_channels, hidden_channels), out_channels)
            )

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if i < len(self.convs) - 1:
                x = x.relu()

        return x


class Model(torch.nn.Module):
    def __init__(
        self,
        hetero_metadata,
        hidden_dim=64,
        out_dim=32,
        hop_depth=2
    ):
        super().__init__()

        self.hop_depth = hop_depth

        base_gnn = GNNEncoder(
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            hop_depth=hop_depth
        )

        self.encoder = to_hetero(
            base_gnn,
            hetero_metadata
        )

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(out_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def encode(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)

    def decode(self, h_customer, h_product):
        cat_features = torch.cat(
            [h_customer, h_product],
            dim=-1
        )

        return self.predictor(cat_features).squeeze(-1)
