import copy
import ctypes
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import LinkNeighborLoader

from KomplexDataLoader import BUY

RESULTS_DATA = Path(__file__).parent / "cache" / "results.json"
RESULTS_DOC = Path(__file__).parent / "Ergebnisse.md"
MODELS = ["BasicGNN (1 Hop)", "BasicGNN (2 Hops)", "BasicGNN (3 Hops)", "Returnformer", "COP-HGNN",
          "GAT", "HGT", "MLP", "GBDT"]


def model_name(name, graph):
    return name if graph == "ausgelagert" else f"{name} – einfacher Graph"


RESULTS_ORDER = MODELS + [model_name(m, "einfach") for m in MODELS]


def _fixed_malloc_thresholds():
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.mallopt(-3, 128 * 1024)  # M_MMAP_THRESHOLD
            libc.mallopt(-1, 128 * 1024)  # M_TRIM_THRESHOLD
        except OSError:
            pass


_fixed_malloc_thresholds()


def make_loader(graph, num_neighbors, batch_size, shuffle):
    return LinkNeighborLoader(
        graph, num_neighbors=list(num_neighbors), batch_size=batch_size, shuffle=shuffle,
        edge_label_index=(BUY, graph[BUY].edge_label_index), edge_label=graph[BUY].edge_label,
    )


@torch.no_grad()
def predict(model, graph, num_neighbors=None, batch_size=1024, embeddings=True):
    model.eval()
    if num_neighbors is None:
        batches = [graph]
    else:
        batches = make_loader(graph, num_neighbors, batch_size, shuffle=False)

    parts = []
    for batch in batches:
        h = model.encode(batch.x_dict, batch.edge_index_dict)
        customer, product = batch[BUY].edge_label_index
        h_customer, h_product = h['customer'][customer], h['product'][product]
        logits = model.decode(h_customer, h_product)
        parts.append((logits, h_customer, h_product) if embeddings else (logits,))
    out = tuple(torch.cat(p) for p in zip(*parts))
    return out if embeddings else out[0]


def auc(model, graphs, num_neighbors=None, batch_size=1024):
    logits = torch.cat([predict(model, g, num_neighbors, batch_size, embeddings=False) for g in graphs])
    labels = torch.cat([g[BUY].edge_label for g in graphs])
    return roc_auc_score(labels.numpy(), logits.numpy())


def train(model, folds, val=None, epochs=30, lr=1e-3, batch_size=1024,
          num_neighbors=(15, 10), patience=5, seed=0, eval_neighbors=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    optimizer = None
    criterion = torch.nn.BCEWithLogitsLoss()
    loaders = [make_loader(g, num_neighbors, batch_size, shuffle=True) for g in folds]

    history, best = [], {'auc': -1.0, 'epoch': 0, 'state': None}
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, count = 0.0, 0
        for i in rng.permutation(len(loaders)):
            for batch in loaders[i]:
                h = model.encode(batch.x_dict, batch.edge_index_dict)
                if optimizer is None:  # erst nach dem ersten Forward (Lazy-Parameter)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                optimizer.zero_grad()
                customer, product = batch[BUY].edge_label_index
                loss = criterion(model.decode(h['customer'][customer], h['product'][product]),
                                 batch[BUY].edge_label)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(customer)
                count += len(customer)

        val_auc = auc(model, [val], eval_neighbors, batch_size) if val is not None else float('nan')
        history.append({'epoch': epoch, 'loss': loss_sum / count, 'val_auc': val_auc})
        print(f"Epoche {epoch:02d} | Loss {loss_sum / count:.4f} | Val-AUC {val_auc:.4f}", flush=True)

        if val is None:
            continue
        if val_auc > best['auc']:
            best = {'auc': val_auc, 'epoch': epoch, 'state': copy.deepcopy(model.state_dict())}
        elif epoch - best['epoch'] >= patience:
            print(f"Early Stopping, beste Epoche {best['epoch']} (Val-AUC {best['auc']:.4f})")
            break

    if best['state'] is not None:
        model.load_state_dict(best['state'])
    return model, history


@torch.no_grad()
def generate_embeddings(model, graphs, reason=None, num_neighbors=None, batch_size=1024):
    n_customer, n_product = len(graphs[0].customer_features), len(graphs[0].product_features)
    edge_ids = torch.cat([g[BUY].edge_label_id for g in graphs])
    sorted_ids = edge_ids.sort().values
    out = {}
    for g in graphs:
        logits, h_customer, h_product = predict(model, g, num_neighbors, batch_size)
        customer, product = g[BUY].edge_label_index
        part = {
            'edge_id': g[BUY].edge_label_id,
            'state': torch.cat([h_customer, h_product], dim=1),
            'ground_truth': g[BUY].edge_label,
            'customer': g['customer'].x[customer, :n_customer],
            'product': g['product'].x[product, :n_product],
            'logit': logits,
            'fold': torch.full((len(customer),), g.fold),
        }
        if reason is not None:
            part['reason'] = reason[customer]
        rows = torch.searchsorted(sorted_ids, part['edge_id'])
        for key, value in part.items():
            if key not in out:
                out[key] = torch.empty((len(edge_ids),) + tuple(value.shape[1:]), dtype=value.dtype)
            out[key][rows] = value
        del logits, h_customer, h_product, part
    out['feature_names'] = {'customer': graphs[0].customer_features, 'product': graphs[0].product_features}
    return out


def embedding_auc(embeddings):
    y, logit, fold = embeddings['ground_truth'].numpy(), embeddings['logit'].numpy(), embeddings['fold'].numpy()
    in_folds = fold >= 0
    val = roc_auc_score(y[~in_folds], logit[~in_folds]) if (~in_folds).any() else None
    return roc_auc_score(y[in_folds], logit[in_folds]), val


def record_results(name, auc_values, durations, params, history=None, state_dim=None, note=None):
    results = json.loads(RESULTS_DATA.read_text()) if RESULTS_DATA.exists() else {}
    entry = {
        'stand': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'auc': {k: round(float(v), 4) for k, v in auc_values.items() if v is not None},
        'dauer_s': {k: round(v, 1) for k, v in durations.items()},
        'state_dim': state_dim,
        'rechner': f"{os.cpu_count()} CPU-Kerne, {torch.get_num_threads()} Torch-Threads, "
                   f"GPU: {'ja' if torch.cuda.is_available() else 'nein'}",
        'params': params,
        'hinweis': note,
    }
    if history and 'epoch' in history[0]:
        best = max(history, key=lambda h: h['val_auc'])
        entry['training'] = f"{len(history)} Epochen, beste Epoche {best['epoch']}"
    elif history and 'iterations' in history[0]:
        entry['training'] = f"{history[0]['iterations']} Bäume"
    results[name] = entry
    RESULTS_DATA.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DATA.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    RESULTS_DOC.write_text(_results_markdown(results))

    print(f"{name}: " + ", ".join(f"AUC {k} {v:.4f}" for k, v in entry['auc'].items())
          + f" | Dauer {_duration(sum(durations.values()))} -> {RESULTS_DOC.name}")


def _duration(seconds):
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    return f"{minutes} min {seconds:02d} s" if minutes else f"{seconds} s"


def _results_markdown(results):
    names = [n for n in RESULTS_ORDER if n in results] + sorted(set(results) - set(RESULTS_ORDER))
    auc_cols = ["Train (OOF)", "Validierung", "Test"]
    time_cols = []
    for n in names:
        time_cols += [k for k in results[n]['dauer_s'] if k not in time_cols]

    lines = [
        "# Ergebnisse der Embedding-Modelle", "",
        "Wird beim Ausführen der Notebooks automatisch geschrieben (`Training.record_results`, Daten in "
        "`cache/results.json`). Ein neuer Lauf eines Modells ersetzt dessen Eintrag.", "",
        "## AUC", "",
        "| Modell | " + " | ".join(auc_cols) + " | state-Dim | Training | Stand |",
        "|---|" + "---|" * (len(auc_cols) + 3),
    ]
    for n in names:
        r = results[n]
        aucs = [f"{r['auc'][c]:.4f}" if c in r['auc'] else "–" for c in auc_cols]
        lines.append(f"| {n} | " + " | ".join(aucs) + f" | {r.get('state_dim') or '–'} | "
                     f"{r.get('training') or '–'} | {r['stand']} |")
    lines += [
        "",
        "- **Train (OOF):** jede Trainingskante aus dem Fold-Graphen, in dem sie versteckt war.",
        "- **Validierung:** 10 % der Trainingskanten, nur für Early Stopping.",
        "- **Test:** Testdatensatz mit eigenen Folds, nie für Training oder Modellauswahl verwendet.",
        "", "## Rechendauer", "",
        "| Modell | " + " | ".join(time_cols) + " | Gesamt |",
        "|---|" + "---|" * (len(time_cols) + 1),
    ]
    for n in names:
        d = results[n]['dauer_s']
        lines.append(f"| {n} | " + " | ".join(_duration(d[c]) if c in d else "–" for c in time_cols)
                     + f" | {_duration(sum(d.values()))} |")
    rechner = {results[n]['rechner'] for n in names}
    lines += ["", f"Rechner: {', '.join(sorted(rechner))}.", "", "## Hyperparameter"]
    for n in names:
        r = results[n]
        lines += ["", f"### {n}", ""]
        lines += [f"- **{k}:** {_format_param(v)}" for k, v in r['params'].items()]
        if r.get('hinweis'):
            lines.append(f"- **Hinweis:** {r['hinweis']}")
    return "\n".join(lines) + "\n"


def _format_param(value):
    if isinstance(value, dict):
        return ", ".join(f"{k}={_format_param(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_param(v) for v in value) + "]"
    return f"`{value}`" if isinstance(value, str) else str(value)
