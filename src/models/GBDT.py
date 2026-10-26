import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from models.MLP import attribute_features

BUY = ('customer', 'buys', 'product')


class GBDT:
    def __init__(self, metadata=None, max_iter=500, learning_rate=0.1, max_leaf_nodes=31,
                 early_stopping_rounds=20, seed=0):
        self.classifier = HistGradientBoostingClassifier(
            max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=early_stopping_rounds,
            scoring='roc_auc', random_state=seed,
        )

    def eval(self):
        return self

    def encode(self, x_dict, edge_index_dict):
        return attribute_features(x_dict, edge_index_dict)

    def decode(self, h_customer, h_product):
        features = torch.cat([h_customer, h_product], dim=-1).numpy()
        return torch.tensor(self.classifier.decision_function(features), dtype=torch.float)


def train_gbdt(model, folds, val=None, seed=0):
    features, labels = [], []
    for g in folds:
        h = model.encode(g.x_dict, g.edge_index_dict)
        customer, product = g[BUY].edge_label_index
        features.append(torch.cat([h['customer'][customer], h['product'][product]], dim=-1).numpy())
        labels.append(g[BUY].edge_label.numpy())
    model.classifier.set_params(random_state=seed)
    model.classifier.fit(np.concatenate(features), np.concatenate(labels))

    history = {'iterations': int(model.classifier.n_iter_)}
    if val is not None:
        h = model.encode(val.x_dict, val.edge_index_dict)
        customer, product = val[BUY].edge_label_index
        logits = model.decode(h['customer'][customer], h['product'][product])
        history['val_auc'] = roc_auc_score(val[BUY].edge_label.numpy(), logits.numpy())
    print(f"{history['iterations']} Bäume" +
          (f" | Val-AUC {history['val_auc']:.4f}" if 'val_auc' in history else ""))
    return model, [history]
