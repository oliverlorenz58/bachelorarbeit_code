import torch
import torch.nn as nn
from torch_geometric.utils import scatter

BASKET = ('product', 'in_basket_of', 'customer')
MAIN_TYPES = ('customer', 'product')


class COPHGNN(nn.Module):
    def __init__(self, metadata, in_channels, emb_dim=12):
        super().__init__()
        node_types, edge_types = metadata
        self.embed = nn.ModuleDict({nt: nn.Linear(in_channels[nt], emb_dim) for nt in node_types})   # Gl. 1
        self.attribute_edges = [et for et in edge_types if et[0] not in MAIN_TYPES and et[2] in MAIN_TYPES]
        self.order_customer = nn.Linear(emb_dim, emb_dim, bias=False)   # Gl. 4: W_Order1
        self.order_products = nn.Linear(emb_dim, emb_dim, bias=False)   # Gl. 4: W_Order2
        self.out1 = nn.Linear(2 * emb_dim, emb_dim)                     # Gl. 5
        self.out2 = nn.Linear(emb_dim, 1)

    def encode(self, x_dict, edge_index_dict):
        e = {nt: self.embed[nt](x) for nt, x in x_dict.items()}

        # Gl. 2
        h = {nt: e[nt] for nt in MAIN_TYPES}
        for src_type, rel, dst_type in self.attribute_edges:
            src, dst = edge_index_dict[(src_type, rel, dst_type)]
            h[dst_type] = h[dst_type] + scatter(e[src_type][src], dst, dim=0,
                                                dim_size=h[dst_type].size(0), reduce='sum')

        # Gl. 4
        order = self.order_customer(h['customer'])
        if BASKET in edge_index_dict:
            product, customer = edge_index_dict[BASKET]
            order = order + self.order_products(scatter(h['product'][product], customer, dim=0,
                                                        dim_size=order.size(0), reduce='mean'))
        return {'customer': order, 'product': h['product']}

    def decode(self, h_order, h_product):
        return self.out2(torch.relu(self.out1(torch.cat([h_order, h_product], dim=-1)))).squeeze(-1)
