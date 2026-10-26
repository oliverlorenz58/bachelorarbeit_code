# Retourenprävention mit einem RL-Agenten auf Graph-Embeddings

Code zur Bachelorarbeit. Ein RL-Agent (kontextueller Bandit) entscheidet für jeden Kauf, ob vor dem Bestellabschluss
eine Retourenwarnung angezeigt wird. Seinen Zustand bekommt er von einem vortrainierten Prediction-Modell
(Graph Neural Network oder Baseline) auf dem ASOS-GraphReturns-Datensatz (McGowan et al. 2023). Wie Kund:innen auf
eine Warnung reagieren, liefert eine Nutzersimulation: stochastisch oder LLM-basiert mit Personas aus MatrAIx
(Li et al. 2026).

## Aufbau

```
code/
├── pyproject.toml, uv.lock, .python-version   Python-Umgebung (uv, Python 3.12)
├── data/                       ASOS-Rohdaten (nicht enthalten, siehe Einrichtung)
├── src/
│   ├── embeddings/             Stufe 1: Prediction-Modelle trainieren, Embeddings erzeugen
│   │   ├── KomplexDataLoader.py    Rohdaten -> heterogener Graph (Feature-Auswahl, Graphstruktur)
│   │   ├── BuildFolds.py           Out-of-Fold-Aufteilung, label-abhängige Features pro Fold
│   │   ├── Training.py             Training, Embedding-Erzeugung, Ergebnisübersicht
│   │   ├── <Modell>_Embeddings.ipynb   ein Notebook je Modell: BasicGNN, GAT, HGT, COP-HGNN,
│   │   │                               Returnformer, MLP, GBDT
│   │   ├── Ergebnisse.md           AUC, Rechendauer, Hyperparameter aller Läufe (automatisch geschrieben)
│   │   └── cache/                  erzeugte Embedding-Dateien
│   ├── models/                 Modelle
│   │   ├── BasicGNN.py, GAT.py, HGT.py, COPHGNN.py, Returnformer.py, MLP.py, GBDT.py   Prediction-Modelle
│   │   ├── RLModel.py              Rewardmodelle des Agenten (LinearModel, DQN)
│   │   ├── Explorer.py             Explorationsstrategien LinUCB, LinTS, NeuralUCB, NeuralTS, Greedy
│   │   └── Evaluator.py            Evaluationsmetriken
│   ├── simulation/             Nutzersimulation
│   │   ├── Simulator.py            gemeinsame Schnittstelle evaluate(truth, action, context)
│   │   ├── Stochastisch.py         stochastische Simulation (binär und 14-teilig), Rewardgestaltungen
│   │   └── MatrAIx.py              LLM-basierte Simulation aus dem Antwort-Puffer
│   └── agent/                  Stufe 2: RL-Agent
│       ├── RLAgent.py              Agent: Training und Auswertung
│       ├── Experimente.py          alle Experimente der Arbeit (parallel, fortsetzbar)
│       ├── Agent.ipynb             ein einzelner Lauf interaktiv, mit Lernkurve
│       ├── Ergebnisse.csv          eine Zeile pro Lauf
│       └── Tuning.csv              Hyperparameter-Tuning der neuronalen Agenten
└── matrAIx/                    LLM-basierte Nutzersimulation (Puffer)
    ├── selectCustomers.py          Persona-Kohorte aus MatrAIx Persona 1M
    ├── Puffer.py                   Zuordnung Kunde -> Persona, Prompts, Anfragen über OpenRouter, Bericht
    └── daten/                      Personadaten, Puffer und abgeleitete Dateien
```

## Ablauf

```
data/*.p ─► KomplexDataLoader ─► BuildFolds ─► Training (Notebook je Modell) ─► src/embeddings/cache/<modell>/*.pt
                                                                                        │
matrAIx: selectCustomers ─► Puffer (zuordnung, kontext, fuellen) ─► daten/              │
                                              │                                         ▼
                                              └──► simulation/MatrAIx ◄── Experimente.py / Agent.ipynb ──► Ergebnisse.csv
                                                   simulation/Stochastisch ◄──┘   (RLAgent, models/Explorer, RLModel)
```

**Stufe 1 (Embeddings).** `KomplexDataLoader` baut aus den drei Tabellen einen heterogenen Graphen mit Kunden-,
Produkt- und Kaufkanten. Im ausgelagerten Graphen (Standard) sind Land, Produkttyp, Marke und Artikel eigene Knoten,
im einfachen Graphen stehen sie als Features in den Kunden- und Produktknoten (`GRAPH = "einfach"` im Notebook).
`BuildFolds` teilt die Käufe in 5 Folds und trennt 10 % als Validierungsmenge für Early Stopping ab. Für Fold *i*
bestehen der Graph und die label-abhängigen Features (Käufe, Retouren, Retourenquote, Historie) nur aus den Käufen
der anderen Folds, die Käufe von Fold *i* sind die Zielkanten. `Training.generate_embeddings` speichert für jeden
Kauf das Embedding aus dem Graphen, in dem er versteckt war, dazu Label, Logit, Fold und die Retourengrund-Anteile
des Kunden. Der Testdatensatz bekommt eigene Folds und wird mit demselben, nicht neu trainierten Modell eingebettet.

**Stufe 2 (Agent).** Der Agent wählt pro Kauf eine Aktion (binär: nicht warnen / warnen, 14-teilig: nicht warnen /
Warnung mit einem von 13 Retourengründen), der Simulator liefert Reward und Outcome. Lineare Agenten (LinUCB,
LinTS) schätzen den Reward mit einer Ridge-Regression je Aktion, neuronale (NeuralUCB, NeuralTS) mit einem Netz
(DQN); beide werden nach je 64 Käufen aktualisiert. Als State dient das Embedding (Architektur 4.1) oder die
Prediction des Prediction-Modells (4.2, Zusatz
`– Prediction`), ohne Agent wird ab einer Prediction von 50 % gewarnt (`Schwellenwert 50 %`). Trainiert wird auf
den Trainingsdaten, ausgewertet auf dem Testdatensatz.

**Nutzersimulation.** `Stochastisch.py`: Nach einer Warnung bricht ein Kunde mit Wahrscheinlichkeit 0,5 ab, wenn
der Kauf retourniert worden wäre, sonst mit 0,2. Die Rewards je Outcome und die daraus folgende Entscheidungsschwelle
p* stehen in `REWARDS` (Standard `basis`). `MatrAIx.py`: Jeder Kunde ist über Alter und Geschlecht einer festen
Persona zugeordnet, deren Reaktion auf die Warnung je Situation (Preisstufe, Sale, Eindruck) vorab von einem LLM
erfragt wurde (`matrAIx/daten/reaktionen.jsonl`). Beim Training wird kein Modell aufgerufen.

## Leakage-Schutz

- Die Spalten der Knotentabellen, die der Datensatz aus allen Retouren berechnet (`customerReturnRate`,
  `salesPerProduct`, Retourengrund-Anteile usw.), sind als Feature gesperrt (`KomplexDataLoader.LEAKY_COLUMNS`).
  Label-abhängige Features entstehen nur pro Fold aus den sichtbaren Käufen.
- Kein Kauf steht im Graphen seiner eigenen Vorhersage (Kontrollen in jedem Notebook).
- Die Retourengrund-Anteile (`reason`) bekommt nur der Simulator, nie das Modell oder der Agent.
- Trainings- und Testdatensatz werden nie gemischt. Standardisierung und PCA der States kommen nur aus den
  Trainingsdaten, das Tuning nutzt nur die Validierungskäufe der Trainingsdaten.

## Einrichtung

1. Umgebung: `uv sync` im Ordner `code` (Python 3.12, CPU genügt, keine GPU nötig). Die Notebooks laufen mit dem
   Kernel dieser Umgebung.
2. ASOS-Daten (McGowan et al. 2023): die sechs Pickles `customer_nodes_{training,testing}.p`,
   `product_nodes_{training,testing}.p` und `event_table_{training,testing}.p` nach `code/data/` legen.
3. Nur für die LLM-Simulation: Personadaten mit `python selectCustomers.py laden` im Ordner `matrAIx` (Hugging
   Face, feste Revision, Prüfsummen). Neue Anfragen an ein LLM brauchen einen OpenRouter-Key in der
   Umgebungsvariable `OPENROUTER_API_KEY`. Liegt der Puffer `matrAIx/daten/reaktionen.jsonl` vor, ist kein
   API-Zugang nötig.

## Ausführen

Reihenfolge wie in der Arbeit. Alle Schritte überspringen bereits fertige Läufe.

1. **Embeddings** (Ordner `src/embeddings`): jedes `<Modell>_Embeddings.ipynb` ausführen. Es schreibt
   `cache/<modell>/{train,test}_embeddings.pt` und trägt AUC, Rechendauer und Hyperparameter in `Ergebnisse.md`
   ein. BasicGNN je einmal mit `HOP_DEPTH = 1, 2, 3`. Für den einfachen Graphen `GRAPH = "einfach"` setzen
   (Ausgabe unter `cache/einfach/`).
2. **Puffer der LLM-Simulation** (Ordner `matrAIx`, nur bei Bedarf neu):
   ```
   python selectCustomers.py                 # Kohorte -> daten/kohorte.json
   python Puffer.py zuordnung                # Kunde -> Persona
   python Puffer.py kontext                  # Situation je Kauf
   python Puffer.py prompts [Aufgabe]        # Trockenlauf ohne Modellaufruf
   python Puffer.py modelle [Suche]          # passende Modelle bei OpenRouter mit Preisen
   python Puffer.py probe                    # 3 Testanfragen, Kostenschätzung
   python Puffer.py fuellen [haupt|treue|ohne_persona]
   python Puffer.py bericht
   ```
   `--modell <OpenRouter-Slug>` wählt ein anderes Modell, `--anbieter <Endpunkt>` legt den Anbieter fest (Name
   dann `<Slug>@<Anbieter>`). Im Puffer liegen Antworten von
   `qwen/qwen3.8-27b@deepinfra/bf16` und `anthropic/claude-haiku-5.5`.
3. **Agent** (Ordner `src/agent`):
   ```
   python Experimente.py raster basis        # alle Embeddings x alle Strategien, 5 Seeds
   python Experimente.py raster fehlalarm
   python Experimente.py architekturen       # Prediction als State und Schwellenwert 50 %
   python Experimente.py tuning 1            # ebenso 2 und 3; Ergebnis in NEURAL_PARAMS übernommen
   python Experimente.py tests               # Nebenexperimente: Rewards, Graph, Aktionsraum, Nutzersimulation
   MATRAIX_MODELL=anthropic/claude-haiku-5.5 python Experimente.py tests   # LLM-Simulation mit Haiku
   ```
   `tests` wählt die Varianten aus den Ergebnissen von `raster basis` und `architekturen` (`best_configs`) und
   braucht deshalb diese zuerst. `WORKERS=<n>` legt die Zahl paralleler Läufe fest (Standard 5, je etwa 1 GB RAM).
4. **Einzelner Lauf:** `Agent.ipynb` (Konfiguration in der ersten Code-Zelle nach den Imports) ergibt dieselben
   Werte wie die entsprechende Zeile in `Ergebnisse.csv` und zeigt die Lernkurve.

Laufzeiten auf 16 CPU-Kernen ohne GPU: Embeddings je Modell zwischen etwa 30 s (GBDT) und 1,5 h (Returnformer,
siehe `Ergebnisse.md`), Returnformer braucht dabei bis zu etwa 9 GB RAM. Ein Agentenlauf dauert linear etwa
1–2 min, neuronal etwa 6–7 min.

## Ergebnisdateien

- `src/embeddings/Ergebnisse.md`: AUC (Train-OOF, Validierung, Test), Rechendauer und Hyperparameter je
  Prediction-Modell.
- `src/agent/Ergebnisse.csv`: eine Zeile je Lauf mit Konfiguration (`Aktionsraum`, `Rewards`, `Nutzersimulation`,
  `Embedding`, `Strategie`, `Seed`), Metriken auf dem Testdatensatz, Laufzeiten und Parametern.
- `src/agent/Tuning.csv`: eine Zeile je Tuning-Lauf und Epoche, erwarteter Reward auf den Validierungskäufen.

Metriken (`models/Evaluator.py`):

| Metrik | Bedeutung |
|---|---|
| Ø Reward | mittlerer Reward je Kauf |
| Precision, Recall, F1 | Warnung als Vorhersage einer Retoure |
| FPR | Anteil der Käufe ohne Retoure, bei denen gewarnt wurde |
| RCR | Anteil der korrekten Warnungen, nach denen abgebrochen wurde |
| FCR | Anteil der Käufe ohne Retoure, die nach einer Warnung abgebrochen wurden |
| VR | Anteil der Retouren, die durch eine Warnung verhindert wurden |
| WR | Warnrate |
| WRRA | Warnrate geteilt durch die Retourenquote |

## Reproduzierbarkeit

Folds, Startgewichte, Training und Agent laufen mit festen Seeds, die Agentenläufe mit einem Torch-Thread je
Prozess. Mit den vorhandenen Embedding-Dateien und dem Puffer reproduziert ein erneuter Agentenlauf die Zeilen in
`Ergebnisse.csv` exakt.

Das Neighbor-Sampling von pyg-lib (`LinkNeighborLoader`) lässt sich nicht seeden. Ein erneutes Training der
Prediction-Modelle ergibt deshalb dieselben Folds und Features, aber nicht bit-gleiche Embeddings (im Test für
BasicGNN mit 1 Hop: AUC 0,7218 / 0,7217 / 0,7212 statt 0,7197 / 0,7210 / 0,7208).
