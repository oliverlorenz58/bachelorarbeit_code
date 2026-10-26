import torch
from torch_geometric.data import HeteroData

from KomplexDataLoader import BUY, REV_BUY, ExplicitRelationalDataLoader

VAL = -1
COUNT_FEATURES = ('sales', 'returns')


def assign_folds(num_edges, k, val_frac, seed):
    perm = torch.randperm(num_edges, generator=torch.Generator().manual_seed(seed))
    n_val = round(val_frac * num_edges)
    fold = torch.empty(num_edges, dtype=torch.long)
    fold[perm[:n_val]] = VAL
    fold[perm[n_val:]] = torch.arange(num_edges - n_val) % k
    return fold


def label_features(nodes, y, num_nodes, global_rate):
    sales = torch.bincount(nodes, minlength=num_nodes).float()
    returns = torch.bincount(nodes, weights=y, minlength=num_nodes).float()
    has_history = sales > 0
    rate = torch.where(has_history, returns / sales.clamp(min=1), global_rate)
    return {'sales': sales, 'returns': returns, 'rate': rate, 'hasHistory': has_history.float()}


def visible_nodes(data, edge_index):
    nodes = {'customer': edge_index[0], 'product': edge_index[1]}
    for group_type, membership in data.group_nodes.items():
        member_type = membership[0]
        member, group = data[tuple(membership)].edge_index
        group_of = torch.full((data[member_type].num_nodes,), -1, dtype=torch.long)
        group_of[member] = group
        nodes[group_type] = group_of[nodes[member_type]]
    return nodes


def node_features(static_x, label_feats, names, log_counts=True):
    cols = [static_x]
    for name in names:
        col = label_feats[name]
        is_count = name.split('_')[-1] in COUNT_FEATURES
        cols.append((torch.log1p(col) if log_counts and is_count else col).unsqueeze(1))
    x, _, _ = ExplicitRelationalDataLoader._scaleTensor(torch.cat(cols, dim=1))
    return x


def make_graph(data, visible, target, fold, log_counts=True):
    edge_index, y, edge_id = data[BUY].edge_index, data[BUY].y, data[BUY].edge_id
    y_visible = y[visible]
    global_rate = y_visible.mean()

    g = HeteroData()
    nodes = visible_nodes(data, edge_index[:, visible])
    names = {}
    for node_type, idx in nodes.items():
        keep = idx >= 0
        feats = label_features(idx[keep], y_visible[keep], data[node_type].num_nodes, global_rate)
        names[node_type] = list(data.label_features)
        if node_type in data.group_features:
            group_name, group_of, num_groups = data.group_features[node_type]
            group_feats = label_features(group_of[idx[keep]], y_visible[keep], num_groups, global_rate)
            for name in data.label_features:
                feats[f"{group_name}_{name}"] = group_feats[name][group_of]
                names[node_type].append(f"{group_name}_{name}")
        g[node_type].x = node_features(data[node_type].x, feats, names[node_type], log_counts)
    for node_type in data.node_types:
        if node_type not in nodes:
            g[node_type].x = data[node_type].x
    for edge_type in data.edge_types:
        if edge_type not in (BUY, REV_BUY):
            g[edge_type].edge_index = data[edge_type].edge_index

    g[BUY].edge_index = edge_index[:, visible]
    g[BUY].edge_id = edge_id[visible]
    g[REV_BUY].edge_index = g[BUY].edge_index.flip(0)

    g[BUY].edge_label_index = edge_index[:, target]
    g[BUY].edge_label = y[target]
    g[BUY].edge_label_id = edge_id[target]

    g.fold = fold
    g.global_rate = global_rate.item()
    g.customer_features = data.customer_features + names['customer']
    g.product_features = data.product_features + names['product']
    return g


def build_folds(data, k=5, val_frac=0.1, seed=42, log_counts=True):
    fold = assign_folds(data[BUY].num_edges, k, val_frac, seed)
    in_folds = fold != VAL
    folds = [make_graph(data, in_folds & (fold != i), fold == i, i, log_counts) for i in range(k)]
    val = make_graph(data, in_folds, fold == VAL, VAL, log_counts) if val_frac > 0 else None
    return folds, val
