# Ergebnisse der Embedding-Modelle

Wird beim Ausführen der Notebooks automatisch geschrieben (`Training.record_results`, Daten in `cache/results.json`). Ein neuer Lauf eines Modells ersetzt dessen Eintrag.

## AUC

| Modell | Train (OOF) | Validierung | Test | state-Dim | Training | Stand |
|---|---|---|---|---|---|---|
| BasicGNN (1 Hop) | 0.7197 | 0.7210 | 0.7208 | 64 | 12 Epochen, beste Epoche 7 | 2026-10-02 16:12 |
| BasicGNN (2 Hops) | 0.7220 | 0.7218 | 0.7217 | 64 | 12 Epochen, beste Epoche 7 | 2026-10-02 16:19 |
| BasicGNN (3 Hops) | 0.7230 | 0.7223 | 0.7216 | 64 | 12 Epochen, beste Epoche 7 | 2026-10-02 16:41 |
| Returnformer | 0.7169 | 0.7188 | 0.7190 | 256 | 11 Epochen, beste Epoche 6 | 2026-10-02 22:14 |
| COP-HGNN | 0.7188 | 0.7208 | 0.7209 | 24 | 30 Epochen, beste Epoche 28 | 2026-10-02 16:10 |
| GAT | 0.7237 | 0.7223 | 0.7214 | 64 | 16 Epochen, beste Epoche 11 | 2026-10-02 17:09 |
| HGT | 0.7210 | 0.7220 | 0.7219 | 64 | 13 Epochen, beste Epoche 8 | 2026-10-02 17:39 |
| MLP | 0.7211 | 0.7209 | 0.7212 | 64 | 18 Epochen, beste Epoche 13 | 2026-10-02 16:04 |
| GBDT | 0.7299 | 0.7203 | 0.7208 | 48 | 313 Bäume | 2026-10-02 16:05 |
| BasicGNN (1 Hop) – einfacher Graph | 0.7223 | 0.7215 | 0.7211 | 64 | 16 Epochen, beste Epoche 11 | 2026-10-05 22:29 |
| BasicGNN (2 Hops) – einfacher Graph | 0.7246 | 0.7216 | 0.7205 | 64 | 16 Epochen, beste Epoche 11 | 2026-10-05 22:35 |
| BasicGNN (3 Hops) – einfacher Graph | 0.7202 | 0.7208 | 0.7191 | 64 | 8 Epochen, beste Epoche 3 | 2026-10-05 22:41 |
| Returnformer – einfacher Graph | 0.7166 | 0.7191 | 0.7176 | 256 | 7 Epochen, beste Epoche 2 | 2026-10-05 23:27 |
| COP-HGNN – einfacher Graph | 0.7184 | 0.7203 | 0.7205 | 24 | 21 Epochen, beste Epoche 16 | 2026-10-05 22:49 |
| GAT – einfacher Graph | 0.7233 | 0.7216 | 0.7199 | 64 | 11 Epochen, beste Epoche 6 | 2026-10-05 22:37 |
| HGT – einfacher Graph | 0.7215 | 0.7212 | 0.7214 | 64 | 13 Epochen, beste Epoche 8 | 2026-10-05 22:55 |
| MLP – einfacher Graph | 0.7218 | 0.7213 | 0.7207 | 64 | 21 Epochen, beste Epoche 16 | 2026-10-05 22:45 |
| GBDT – einfacher Graph | 0.7272 | 0.7202 | 0.7206 | 48 | 236 Bäume | 2026-10-05 22:46 |

- **Train (OOF):** jede Trainingskante aus dem Fold-Graphen, in dem sie versteckt war.
- **Validierung:** 10 % der Trainingskanten, nur für Early Stopping.
- **Test:** Testdatensatz mit eigenen Folds, nie für Training oder Modellauswahl verwendet.

## Rechendauer

| Modell | Graph + Folds (Train) | Training | Embeddings (Train) | Graph + Folds (Test) | Embeddings (Test) | Node2Vec (Train) | Node2Vec (Test) | Gesamt |
|---|---|---|---|---|---|---|---|---|
| BasicGNN (1 Hop) | 2 s | 1 min 34 s | 19 s | 2 s | 2 s | – | – | 1 min 58 s |
| BasicGNN (2 Hops) | 2 s | 6 min 48 s | 7 s | 2 s | 6 s | – | – | 7 min 04 s |
| BasicGNN (3 Hops) | 2 s | 20 min 43 s | 12 s | 2 s | 10 s | – | – | 21 min 09 s |
| Returnformer | 2 s | 1 h 23 min | 1 min 55 s | 2 s | 2 min 04 s | 0 s | 0 s | 1 h 27 min |
| COP-HGNN | 2 s | 5 min 03 s | 2 s | 2 s | 1 s | – | – | 5 min 10 s |
| GAT | 2 s | 27 min 31 s | 30 s | 2 s | 34 s | – | – | 28 min 39 s |
| HGT | 2 s | 28 min 11 s | 35 s | 2 s | 40 s | – | – | 29 min 29 s |
| MLP | 2 s | 1 min 59 s | 2 s | 2 s | 1 s | – | – | 2 min 05 s |
| GBDT | 2 s | 15 s | 3 s | 2 s | 3 s | – | – | 24 s |
| BasicGNN (1 Hop) – einfacher Graph | 2 s | 2 min 08 s | 2 s | 3 s | 2 s | – | – | 2 min 18 s |
| BasicGNN (2 Hops) – einfacher Graph | 2 s | 4 min 53 s | 5 s | 3 s | 6 s | – | – | 5 min 09 s |
| BasicGNN (3 Hops) – einfacher Graph | 3 s | 5 min 51 s | 10 s | 4 s | 9 s | – | – | 6 min 17 s |
| Returnformer – einfacher Graph | 2 s | 28 min 22 s | 1 min 15 s | 2 s | 1 min 41 s | 1 s | 1 s | 31 min 24 s |
| COP-HGNN – einfacher Graph | 4 s | 3 min 30 s | 1 s | 4 s | 1 s | – | – | 3 min 41 s |
| GAT – einfacher Graph | 3 s | 9 min 02 s | 21 s | 2 s | 21 s | – | – | 9 min 50 s |
| HGT – einfacher Graph | 3 s | 17 min 32 s | 21 s | 2 s | 24 s | – | – | 18 min 21 s |
| MLP – einfacher Graph | 5 s | 3 min 18 s | 1 s | 3 s | 2 s | – | – | 3 min 29 s |
| GBDT – einfacher Graph | 4 s | 22 s | 8 s | 3 s | 3 s | – | – | 40 s |

Rechner: 16 CPU-Kerne, 10 Torch-Threads, GPU: nein, 16 CPU-Kerne, 8 Torch-Threads, GPU: nein.

## Hyperparameter

### BasicGNN (1 Hop)

- **customer_features:** [`yearOfBirth`, `isMale`, `premier`]
- **product_features:** [`avgGbpPrice`, `avgDiscountValue`]
- **label_features:** [`sales`, `returns`, `rate`, `hasHistory`]
- **article_node:** True
- **k:** 5
- **val_frac:** 0.1
- **split_seed:** 42
- **log_counts:** True
- **model:** `models.BasicGNN`
- **model_params:** hidden_dim=64, out_dim=32, hop_depth=1
- **train_params:** epochs=30, lr=0.001, batch_size=1024, num_neighbors=[15], patience=5, seed=0

### BasicGNN (2 Hops)

- **customer_features:** [`yearOfBirth`, `isMale`, `premier`]
- **product_features:** [`avgGbpPrice`, `avgDiscountValue`]
- **label_features:** [`sales`, `returns`, `rate`, `hasHistory`]
- **article_node:** True
- **k:** 5
- **val_frac:** 0.1
- **split_seed:** 42
- **log_counts:** True
- **model:** `models.BasicGNN`
- **model_params:** hidden_dim=64, out_dim=32, hop_depth=2
- **train_params:** epochs=30, lr=0.001, batch_size=1024, num_neighbors=[15, 10], patience=5, seed=0

### BasicGNN (3 Hops)

- **customer_features:** [`yearOfBirth`, `isMale`, `premier`]
- **product_features:** [`avgGbpPrice`, `avgDiscountValue`]
- **label_features:** [`sales`, `returns`, `rate`, `hasHistory`]
- **article_node:** True
- **k:** 5
- **val_frac:** 0.1
- **split_seed:** 42
- **log_counts:** True
- **model:** `models.BasicGNN`
- **model_params:** hidden_dim=64, out_dim=32, hop_depth=3
- **train_params:** epochs=30, lr=0.001, batch_size=1024, num_neighbors=[15, 10, 5], patience=5, seed=0

### Returnformer

- **customer_features:** [`yearOfBirth`, `isMale`, `premier`]
- **product_features:** [`avgGbpPrice`, `avgDiscountValue`]
- **label_features:** [`sales`, `returns`, `rate`, `hasHistory`]
- **article_node:** True
- **k:** 5
- **val_frac:** 0.1
- **split_seed:** 42
- **log_counts:** True
- **node2vec:** dim=32, walk_length=20, context_size=10, walks_per_node=10, epochs=1, seed=0
- **model:** `models.Returnformer`
- **model_params:** hidden_dim=128, num_layers=2, num_heads=4, gea_memory_units=20, gea_heads=1, dropout=0.45
- **train_params:** epochs=30, lr=0.001, batch_size=1024, num_neighbors=[15, 10], patience=5, seed=0, eval_neighbors=[15, 10]
- **Hinweis:** Node2Vec (Train, Test) aus dem Cache geladen, deshalb 0 s. Ursprüngliche Berechnung: Train ca. 36 min (6 Fold-Graphen), Test ca. 33 min (5 Fold-Graphen).

### COP-HGNN

- **customer_features:** [`yearOfBirth`, `isMale`, `premier`]
- **product_features:** [`avgGbpPrice`, `avgDiscountValue`]
- **label_features:** [`sales`, `returns`, `rate`, `hasHistory`]
- **article_node:** True
- **basket:** True
- **k:** 5
- **val_frac:** 0.1
- **split_seed:** 42
- **log_counts:** True
- **model:** `models.COPHGNN`
- **model_params:** emb_dim=12
- **train_params:** epochs=30, lr=0.001, batch_size=1024, num_neighbors=[32, 10], patience=5, seed=0

### GAT

- **customer_features:** [`yearOfBirth`, `isMale`, `premier`]
- **product_features:** [`avgGbpPrice`, `avgDiscountValue`]
- **label_features:** [`sales`, `returns`, `rate`, `hasHistory`]
- **article_node:** True
- **k:** 5
- **val_frac:** 0.1
- **split_seed:** 42
- **log_counts:** True
- **model:** `models.GAT`
- **model_params:** hidden_dim=64, out_dim=32, hop_depth=2, heads=4
- **train_params:** epochs=30, lr=0.001, batch_size=1024, num_neighbors=[15, 10], patience=5, seed=0, eval_neighbors=[15, 10]

### HGT

- **customer_features:** [`yearOfBirth`, `isMale`, `premier`]
- **product_features:** [`avgGbpPrice`, `avgDiscountValue`]
- **label_features:** [`sales`, `returns`, `rate`, `hasHistory`]
- **article_node:** True
- **k:** 5
- **val_frac:** 0.1
- **split_seed:** 42
- **log_counts:** True
- **model:** `models.HGT`
- **model_params:** hidden_dim=64, out_dim=32, num_layers=2, heads=4
- **train_params:** epochs=30, lr=0.001, batch_size=1024, num_neighbors=[15, 10], patience=5, seed=0, eval_neighbors=[15, 10]

### MLP

- **customer_features:** [`yearOfBirth`, `isMale`, `premier`]
- **product_features:** [`avgGbpPrice`, `avgDiscountValue`]
- **label_features:** [`sales`, `returns`, `rate`, `hasHistory`]
- **article_node:** True
- **k:** 5
- **val_frac:** 0.1
- **split_seed:** 42
- **log_counts:** True
- **model:** `models.MLP`
- **model_params:** hidden_dim=64, out_dim=32
- **train_params:** epochs=30, lr=0.001, batch_size=1024, num_neighbors=[10], patience=5, seed=0

### GBDT

- **customer_features:** [`yearOfBirth`, `isMale`, `premier`]
- **product_features:** [`avgGbpPrice`, `avgDiscountValue`]
- **label_features:** [`sales`, `returns`, `rate`, `hasHistory`]
- **article_node:** True
- **k:** 5
- **val_frac:** 0.1
- **split_seed:** 42
- **log_counts:** True
- **model:** `models.GBDT`
- **model_params:** max_iter=500, learning_rate=0.1, max_leaf_nodes=31, early_stopping_rounds=20
- **train_params:** seed=0
- **Hinweis:** Logits der Trainings-Folds sind in-sample (die Bäume wurden auf genau diesen Kanten trainiert), AUC Train (OOF) ist deshalb zu optimistisch. Validierung und Test sind out-of-sample.

### BasicGNN (1 Hop) – einfacher Graph

- **customer_features:** [`yearOfBirth`, `isMale`, `premier`]
- **product_features:** [`avgGbpPrice`, `avgDiscountValue`]
- **label_features:** [`sales`, `returns`, `rate`, `hasHistory`]
- **graph:** `einfach`
- **article_node:** False
- **article_features:** True
- **k:** 5
- **val_frac:** 0.1
- **split_seed:** 42
- **log_counts:** True
- **model:** `models.BasicGNN`
- **model_params:** hidden_dim=64, out_dim=32, hop_depth=1
- **train_params:** epochs=30, lr=0.001, batch_size=1024, num_neighbors=[15], patience=5, seed=0

### BasicGNN (2 Hops) – einfacher Graph

- **customer_features:** [`yearOfBirth`, `isMale`, `premier`]
- **product_features:** [`avgGbpPrice`, `avgDiscountValue`]
- **label_features:** [`sales`, `returns`, `rate`, `hasHistory`]
- **graph:** `einfach`
- **article_node:** False
- **article_features:** True
- **k:** 5
- **val_frac:** 0.1
- **split_seed:** 42
- **log_counts:** True
- **model:** `models.BasicGNN`
- **model_params:** hidden_dim=64, out_dim=32, hop_depth=2
- **train_params:** epochs=30, lr=0.001, batch_size=1024, num_neighbors=[15, 10], patience=5, seed=0

### BasicGNN (3 Hops) – einfacher Graph

- **customer_features:** [`yearOfBirth`, `isMale`, `premier`]
- **product_features:** [`avgGbpPrice`, `avgDiscountValue`]
- **label_features:** [`sales`, `returns`, `rate`, `hasHistory`]
- **graph:** `einfach`
- **article_node:** False
- **article_features:** True
- **k:** 5
- **val_frac:** 0.1
- **split_seed:** 42
- **log_counts:** True
- **model:** `models.BasicGNN`
- **model_params:** hidden_dim=64, out_dim=32, hop_depth=3
- **train_params:** epochs=30, lr=0.001, batch_size=1024, num_neighbors=[15, 10, 5], patience=5, seed=0

### Returnformer – einfacher Graph

- **customer_features:** [`yearOfBirth`, `isMale`, `premier`]
- **product_features:** [`avgGbpPrice`, `avgDiscountValue`]
- **label_features:** [`sales`, `returns`, `rate`, `hasHistory`]
- **graph:** `einfach`
- **article_node:** False
- **article_features:** True
- **k:** 5
- **val_frac:** 0.1
- **split_seed:** 42
- **log_counts:** True
- **node2vec:** dim=32, walk_length=20, context_size=10, walks_per_node=10, epochs=1, seed=0
- **model:** `models.Returnformer`
- **model_params:** hidden_dim=128, num_layers=2, num_heads=4, gea_memory_units=20, gea_heads=1, dropout=0.45
- **train_params:** epochs=30, lr=0.001, batch_size=1024, num_neighbors=[15, 10], patience=5, seed=0, eval_neighbors=[15, 10]
- **Hinweis:** Node2Vec (Train, Test) aus dem Cache geladen, die ursprüngliche Rechenzeit ist nicht enthalten.

### COP-HGNN – einfacher Graph

- **customer_features:** [`yearOfBirth`, `isMale`, `premier`]
- **product_features:** [`avgGbpPrice`, `avgDiscountValue`]
- **label_features:** [`sales`, `returns`, `rate`, `hasHistory`]
- **graph:** `einfach`
- **article_node:** False
- **article_features:** True
- **basket:** True
- **k:** 5
- **val_frac:** 0.1
- **split_seed:** 42
- **log_counts:** True
- **model:** `models.COPHGNN`
- **model_params:** emb_dim=12
- **train_params:** epochs=30, lr=0.001, batch_size=1024, num_neighbors=[32, 10], patience=5, seed=0

### GAT – einfacher Graph

- **customer_features:** [`yearOfBirth`, `isMale`, `premier`]
- **product_features:** [`avgGbpPrice`, `avgDiscountValue`]
- **label_features:** [`sales`, `returns`, `rate`, `hasHistory`]
- **graph:** `einfach`
- **article_node:** False
- **article_features:** True
- **k:** 5
- **val_frac:** 0.1
- **split_seed:** 42
- **log_counts:** True
- **model:** `models.GAT`
- **model_params:** hidden_dim=64, out_dim=32, hop_depth=2, heads=4
- **train_params:** epochs=30, lr=0.001, batch_size=1024, num_neighbors=[15, 10], patience=5, seed=0, eval_neighbors=[15, 10]

### HGT – einfacher Graph

- **customer_features:** [`yearOfBirth`, `isMale`, `premier`]
- **product_features:** [`avgGbpPrice`, `avgDiscountValue`]
- **label_features:** [`sales`, `returns`, `rate`, `hasHistory`]
- **graph:** `einfach`
- **article_node:** False
- **article_features:** True
- **k:** 5
- **val_frac:** 0.1
- **split_seed:** 42
- **log_counts:** True
- **model:** `models.HGT`
- **model_params:** hidden_dim=64, out_dim=32, num_layers=2, heads=4
- **train_params:** epochs=30, lr=0.001, batch_size=1024, num_neighbors=[15, 10], patience=5, seed=0, eval_neighbors=[15, 10]

### MLP – einfacher Graph

- **customer_features:** [`yearOfBirth`, `isMale`, `premier`]
- **product_features:** [`avgGbpPrice`, `avgDiscountValue`]
- **label_features:** [`sales`, `returns`, `rate`, `hasHistory`]
- **graph:** `einfach`
- **article_node:** False
- **article_features:** True
- **k:** 5
- **val_frac:** 0.1
- **split_seed:** 42
- **log_counts:** True
- **model:** `models.MLP`
- **model_params:** hidden_dim=64, out_dim=32
- **train_params:** epochs=30, lr=0.001, batch_size=1024, num_neighbors=[10], patience=5, seed=0

### GBDT – einfacher Graph

- **customer_features:** [`yearOfBirth`, `isMale`, `premier`]
- **product_features:** [`avgGbpPrice`, `avgDiscountValue`]
- **label_features:** [`sales`, `returns`, `rate`, `hasHistory`]
- **graph:** `einfach`
- **article_node:** False
- **article_features:** True
- **k:** 5
- **val_frac:** 0.1
- **split_seed:** 42
- **log_counts:** True
- **model:** `models.GBDT`
- **model_params:** max_iter=500, learning_rate=0.1, max_leaf_nodes=31, early_stopping_rounds=20
- **train_params:** seed=0
- **Hinweis:** Logits der Trainings-Folds sind in-sample (die Bäume wurden auf genau diesen Kanten trainiert), AUC Train (OOF) ist deshalb zu optimistisch. Validierung und Test sind out-of-sample.
