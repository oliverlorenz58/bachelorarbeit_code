import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Node2Vec, TransformerConv

BUY = ('customer', 'buys', 'product')
FUSED_TYPES = ('customer', 'product')


class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, grid_range=(-3.0, 3.0)):
        super().__init__()
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h + grid_range[0]
        self.register_buffer("grid", grid.expand(in_features, -1).contiguous())
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, grid_size + spline_order) * 0.1)

    def b_splines(self, x):
        grid, x = self.grid, x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
            right = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases

    def forward(self, x):
        base = F.linear(F.silu(x), self.base_weight)
        spline = torch.einsum("bic,oic->bo", self.b_splines(x), self.spline_weight)
        return base + spline


class AttentionFusion(nn.Module):
    def __init__(self, orig_dim, topo_dim, hidden_dim):
        super().__init__()
        self.orig_embed = nn.Linear(orig_dim, hidden_dim)
        self.topo_embed = nn.Linear(topo_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)        # Gl. 1
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q_o = nn.Parameter(torch.randn(hidden_dim) * 0.1)            # Gl. 2
        self.q_s = nn.Parameter(torch.randn(hidden_dim) * 0.1)

    def forward(self, x_orig, x_topo):
        x_o = self.W_o(self.orig_embed(x_orig))
        x_s = self.W_s(self.topo_embed(x_topo))
        alpha = torch.stack([torch.tanh(x_o @ self.q_o), torch.tanh(x_s @ self.q_s)], dim=-1)
        alpha = F.softmax(alpha, dim=-1)                                   # Gl. 3
        return alpha[:, 1:2] * x_s + alpha[:, 0:1] * x_o                   # Gl. 4


class GraphTransformerLayer(nn.Module):
    def __init__(self, metadata, dim, heads, dropout):
        super().__init__()
        node_types, edge_types = metadata
        self.dropout = dropout
        self.attn = HeteroConv({
            et: TransformerConv((dim, dim), dim // heads, heads=heads, dropout=dropout, root_weight=False)
            for et in edge_types
        }, aggr='sum')
        self.out_proj = nn.ModuleDict({nt: nn.Linear(dim, dim) for nt in node_types})   # O_h, Gl. 7
        self.norm1 = nn.ModuleDict({nt: nn.LayerNorm(dim) for nt in node_types})
        self.norm2 = nn.ModuleDict({nt: nn.LayerNorm(dim) for nt in node_types})
        self.ffn = nn.ModuleDict({
            nt: nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)) for nt in node_types
        })

    def forward(self, h_dict, edge_index_dict):
        messages = self.attn(h_dict, edge_index_dict)
        out = {}
        for nt, h in h_dict.items():
            attn = self.out_proj[nt](messages[nt]) if nt in messages else torch.zeros_like(h)
            h_tilde = self.norm1[nt](h) + F.dropout(attn, self.dropout, self.training)       # Gl. 8
            h_bar = F.dropout(self.ffn[nt](h_tilde), self.dropout, self.training)            # Gl. 9
            out[nt] = self.norm2[nt](h_bar) + h_tilde                                        # Gl. 10
        return out


class GraphExternalAttention(nn.Module):
    def __init__(self, dim, num_memory_units=20, num_heads=1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim muss durch num_heads (H_ext) teilbar sein")
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.U_k = nn.Parameter(torch.randn(num_heads, num_memory_units, self.head_dim) * 0.02)
        self.U_v = nn.Parameter(torch.randn(num_heads, num_memory_units, self.head_dim) * 0.02)

    def forward(self, x):
        n = x.shape[0]
        x = x.view(n, self.num_heads, self.head_dim)
        s = torch.einsum("nhd,hmd->nhm", x, self.U_k)                       # Gl. 11/12
        a = F.softmax(s - torch.logsumexp(s, dim=0, keepdim=True), dim=-1)  # Gl. 13+14
        return torch.einsum("nhm,hmd->nhd", a, self.U_v).reshape(n, -1)     # Gl. 15


class Returnformer(nn.Module):
    def __init__(self, metadata, in_channels, topo_dim, hidden_dim=128, num_layers=2, num_heads=4,
                 gea_memory_units=20, gea_heads=1, dropout=0.45):
        super().__init__()
        node_types = metadata[0]
        self.topo_dim = topo_dim
        self.fusion = nn.ModuleDict({
            nt: AttentionFusion(in_channels[nt] - topo_dim, topo_dim, hidden_dim) for nt in FUSED_TYPES
        })
        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(in_channels[nt], hidden_dim) for nt in node_types if nt not in FUSED_TYPES
        })
        self.layers = nn.ModuleList([
            GraphTransformerLayer(metadata, hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.gea = GraphExternalAttention(hidden_dim, gea_memory_units, gea_heads)
        self.kan = nn.Sequential(KANLinear(2 * hidden_dim, hidden_dim), KANLinear(hidden_dim, 1))

    def encode(self, x_dict, edge_index_dict):
        h = {}
        for nt, x in x_dict.items():
            if nt in FUSED_TYPES:
                h[nt] = self.fusion[nt](x[:, :-self.topo_dim], x[:, -self.topo_dim:])
            else:
                h[nt] = self.input_proj[nt](x)
        for layer in self.layers:
            h = layer(h, edge_index_dict)
        for nt in FUSED_TYPES:
            h[nt] = self.gea(h[nt])
        return h

    def decode(self, h_customer, h_product):
        return self.kan(torch.cat([h_customer, h_product], dim=-1)).squeeze(-1)


def node2vec_embedding(graph, dim=64, walk_length=20, context_size=10,
                       walks_per_node=10, epochs=1, batch_size=256, lr=0.01, seed=0):
    torch.manual_seed(seed)
    n_customer, n_product = graph['customer'].num_nodes, graph['product'].num_nodes
    src, dst = graph[BUY].edge_index
    nodes = torch.cat([src.unique(), dst.unique() + n_customer])
    local = torch.full((n_customer + n_product,), -1, dtype=torch.long)
    local[nodes] = torch.arange(len(nodes))
    edge_index = torch.stack([local[src], local[dst + n_customer]])
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    node2vec = Node2Vec(edge_index, embedding_dim=dim, walk_length=walk_length, context_size=context_size,
                        walks_per_node=walks_per_node, p=1.0, q=1.0, num_nodes=len(nodes), sparse=True)
    optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=lr)
    node2vec.train()
    for _ in range(epochs):
        for pos_rw, neg_rw in node2vec.loader(batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            node2vec.loss(pos_rw, neg_rw).backward()
            optimizer.step()
    return nodes, node2vec.embedding.weight.detach().clone()


def add_structural_embeddings(graphs, cache_file=None, **params):
    def fingerprint(g):
        return g[BUY].num_edges, int(g[BUY].edge_id.sum())

    cache = None
    if cache_file is not None and Path(cache_file).exists():
        cache = torch.load(cache_file, weights_only=False)
        if cache['params'] != params or any(cache['fingerprint'].get(g.fold) != fingerprint(g) for g in graphs):
            print(f"{cache_file} passt nicht zu diesen Einstellungen/Fold-Graphen, Node2Vec wird neu berechnet")
            cache = None
        else:
            print(f"Node2Vec geladen: {cache_file}")

    if cache is None:
        cache = {'params': params, 'fingerprint': {}, 'embedding': {}}
        for g in graphs:
            t0 = time.time()
            cache['embedding'][g.fold] = node2vec_embedding(g, **params)
            cache['fingerprint'][g.fold] = fingerprint(g)
            print(f"Fold {g.fold:2d}: Node2Vec fertig nach {time.time() - t0:.0f} s", flush=True)
        if cache_file is not None:
            Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
            torch.save(cache, cache_file)
            print(f"Node2Vec gespeichert: {cache_file}")

    for g in graphs:
        nodes, embedding = cache['embedding'][g.fold]
        n_customer, n_product = g['customer'].num_nodes, g['product'].num_nodes
        topo = torch.zeros(n_customer + n_product, embedding.size(1))
        topo[nodes] = embedding
        g['customer'].x = torch.cat([g['customer'].x, topo[:n_customer]], dim=1)
        g['product'].x = torch.cat([g['product'].x, topo[n_customer:]], dim=1)
    return graphs
