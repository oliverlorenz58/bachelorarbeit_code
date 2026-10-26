import contextlib
import csv
import multiprocessing as mp
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_ind

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from RLAgent import load_embeddings, make_agent, prepare_states, set_seed
from simulation.MatrAIx import MatrAIx, lade_kontext
from simulation.Stochastisch import Stochastisch_13, Stochastisch_Binär

CACHE = HERE.parent / "embeddings" / "cache"
ERGEBNISSE = HERE / "Ergebnisse.csv"

# Name -> (Unterordner in embeddings/cache, Endung des Dateinamens)
EMBEDDINGS = {
    "BasicGNN (1 Hop)": ("basicgnn", "_1hop"),
    "BasicGNN (2 Hops)": ("basicgnn", "_2hop"),
    "BasicGNN (3 Hops)": ("basicgnn", "_3hop"),
    "Returnformer": ("returnformer", ""),
    "COP-HGNN": ("cop", ""),
    "GAT": ("gat", ""),
    "HGT": ("hgt", ""),
    "MLP": ("mlp", ""),
    "GBDT": ("gbdt", ""),
}
EINFACH = " – einfacher Graph"
PREDICTION = " – Prediction"
THRESHOLD = "Schwellenwert 50 %"
PCA_DIMS = {"Returnformer": 64}

ACTION_SPACE = "binär"
REWARD_DESIGN = "fehlalarm"
P_TRUE_CANCEL = 0.5
P_FALSE_CANCEL = 0.2
NUTZERSIMULATION = "stochastisch"
MATRAIX_VARIANTE = "A"
MATRAIX_MODELL = os.environ.get("MATRAIX_MODELL", "qwen/qwen3.8-27b@deepinfra/bf16")
MATRAIX = f"matraix ({MATRAIX_MODELL})"

REWARDREIHE = ["retoure_3", "retoure_2", "retoure_1.5", "basis_x2", "fehlalarm_0.25", "retoure_0.58",
               "fehlalarm_1", "fehlalarm_2"]
# Test -> (Aktionsraum, Rewards, Graph, Nutzersimulation), je genau eine Änderung gegenüber
# binär, basis, ausgelagert, stochastisch
TESTS = {
    "Rewards": ("binär", "fehlalarm", "", "stochastisch"),
    "Graph": ("binär", "basis", EINFACH, "stochastisch"),
    "Aktionsraum": ("14", "basis", "", "stochastisch"),
    "Nutzersimulation": ("binär", "basis", "", MATRAIX),
    **{f"Rewards {r}": ("binär", r, "", "stochastisch") for r in REWARDREIHE},
}
STRATEGIES = ["LinUCB", "LinTS", "NeuralUCB", "NeuralTS"]

NET_PARAMS = dict(hidden_dim=128, num_layers=2, lr=1e-3)
NEURAL_PARAMS = dict(hidden_dim=64, num_layers=2, lr=1e-4)   # aus dem Tuning, nur NeuralUCB und NeuralTS
NEURAL_EPOCHS = 6
RIDGE_LAMBDA = 1.0
EXPLORER_PARAMS = {
    "LinUCB": dict(alpha=1.0),
    "LinTS": dict(v=1.0),
    "Greedy": dict(epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9998),
    "NeuralUCB": dict(m=NET_PARAMS["hidden_dim"], lamda=1.0, nu=0.1),
    "NeuralTS": dict(m=NET_PARAMS["hidden_dim"], lamda=1.0, nu=0.1),
}
BATCH_SIZE = 64
EPOCHS = 3
SEEDS = [0, 1, 2, 3, 4]

# etwa 1 GB je Lauf, Returnformer beim Laden bis etwa 3 GB
WORKERS = int(os.environ.get("WORKERS", 5))
THREADS = 1

TUNING = HERE / "Tuning.csv"
TUNING_EMBEDDINGS = ["BasicGNN (2 Hops)", "HGT", "COP-HGNN"]
# Stufe -> (Änderungen gegenüber NET_PARAMS, EXPLORER_PARAMS und BATCH_SIZE, Seeds)
TUNING_STAGES = {
    "1": ([dict(lr=lr, nu=nu) for lr in (1e-4, 3e-4, 1e-3) for nu in (0.01, 0.1, 1.0)], [0]),
    "2": ([dict(lr=1e-4, epochs=6), dict(lr=3e-5, epochs=6), dict(lr=1e-4, batch_size=256),
           dict(lr=1e-4, hidden_dim=64), dict(lr=1e-4, hidden_dim=256), dict(lr=1e-4, num_layers=1),
           dict(lr=1e-4, num_layers=3), dict(lr=1e-4, lamda=0.1), dict(lr=1e-4, lamda=10.0)], [0]),
    "3": ([dict(lr=1e-3), dict(lr=1e-4), dict(lr=1e-4, epochs=6), dict(lr=3e-5, epochs=6),
           dict(lr=1e-4, hidden_dim=64), dict(lr=1e-4, hidden_dim=64, epochs=6)], [0, 1, 2]),
}
TUNING_PARAMS = ["lr", "nu", "lamda", "hidden_dim", "num_layers", "batch_size", "epochs"]
TUNING_COLUMNS = (["Stufe", "Embedding", "Strategie", "Seed"] + TUNING_PARAMS
                  + ["Epoche", "E[Reward]", "Precision", "Recall", "F1", "FPR", "WR", "Dauer Training (s)", "Stand"])

BASELINES = {"random": "Zufall", "always_warn": "Immer warnen", "always_nothing": "Niemals warnen"}
METRICS = ["Ø Reward", "Precision", "Recall", "F1", "FPR", "RCR", "FCR", "VR", "WR", "WRRA"]
COLUMNS = (["Aktionsraum", "Rewards", "Nutzersimulation", "Embedding", "Strategie", "Seed"] + METRICS
           + ["state-Dim", "PCA", "Dauer Daten (s)", "Dauer Training (s)", "Dauer Auswertung (s)",
              "Epochen", "Parameter", "Rechner", "Stand"])


def simulator(action_space, rewards, nutzersimulation=NUTZERSIMULATION):
    if nutzersimulation.startswith("matraix"):
        if action_space != "binär":
            raise ValueError("Die MatrAIx-Simulation gibt es nur im binären Aktionsraum")
        return MatrAIx(rewards=rewards, modell=MATRAIX_MODELL, p_grund_retoure=P_TRUE_CANCEL,
                       p_grund_behalten=P_FALSE_CANCEL), 2
    if action_space == "binär":
        return Stochastisch_Binär(p_true_cancel=P_TRUE_CANCEL, p_false_cancel=P_FALSE_CANCEL, rewards=rewards), 2
    return Stochastisch_13(p_true_cancel=0.75, p_wrong_reason_cancel=0.1, p_false_cancel=P_FALSE_CANCEL,
                           rewards=rewards), 14


def embedding_file(embedding, split):
    model = embedding.removesuffix(EINFACH)
    folder, suffix = EMBEDDINGS[model]
    return CACHE / ("einfach" if embedding != model else "") / folder / f"{split}_embeddings{suffix}.pt"


LOAD_LOCK = None


def init_worker(lock):
    global LOAD_LOCK
    LOAD_LOCK = lock
    torch.set_num_threads(THREADS)


def load_data(embedding):
    model = embedding.removesuffix(PREDICTION)
    state = "embedding" if model == embedding else "prediction"
    pca_dim = PCA_DIMS.get(model.removesuffix(EINFACH)) if state == "embedding" else None
    with LOAD_LOCK or contextlib.nullcontext():
        train = load_embeddings(embedding_file(model, "train"), state)
        test = load_embeddings(embedding_file(model, "test"), state)
        original_dim = train['state'].shape[1]
        explained = prepare_states(train, test, pca_dim)
    pca = f"{original_dim} -> {train['state'].shape[1]}, {explained:.2%} Varianz" if pca_dim else ""
    return train, test, pca


def kontext_anhaengen(data, split, nutzersimulation):
    if nutzersimulation.startswith("matraix"):
        data['context'] = lade_kontext(split, MATRAIX_VARIANTE)[data['edge_id']]


def agent_params(strategy):
    if strategy.startswith("Neural"):
        return {**EXPLORER_PARAMS[strategy], "m": NEURAL_PARAMS["hidden_dim"]}, NEURAL_PARAMS, NEURAL_EPOCHS
    return EXPLORER_PARAMS[strategy], NET_PARAMS, EPOCHS


def run_agent(action_space, rewards, embedding, strategy, seed, nutzersimulation=NUTZERSIMULATION):
    t0 = time.perf_counter()
    sim, num_actions = simulator(action_space, rewards, nutzersimulation)
    train, test, pca = load_data(embedding)
    kontext_anhaengen(train, "training", nutzersimulation)
    kontext_anhaengen(test, "testing", nutzersimulation)
    t1 = time.perf_counter()

    set_seed(seed)
    explorer, net, epochs = agent_params(strategy)
    agent = make_agent(strategy, explorer, train['state'].shape[1], num_actions, sim,
                       ridge_lambda=RIDGE_LAMBDA, batch_size=BATCH_SIZE, **net)
    agent.train(train, epochs, label=f"[{action_space} | {rewards} | {embedding} | {strategy}] ")
    t2 = time.perf_counter()

    set_seed(seed)
    result = agent.evaluate(test, 'best')
    t3 = time.perf_counter()

    params = dict(explorer)
    params.update(dict(ridge_lambda=RIDGE_LAMBDA) if strategy.startswith("Lin") else net)
    params['batch_size'] = BATCH_SIZE
    return [row(action_space, rewards, embedding, strategy, seed, result, train['state'].shape[1], pca,
                (t1 - t0, t2 - t1, t3 - t2), params, epochs, nutzersimulation)]


def run_baselines(action_space, rewards, seed, nutzersimulation=NUTZERSIMULATION):
    t0 = time.perf_counter()
    folder, suffix = next(iter(EMBEDDINGS.values()))
    test = load_embeddings(CACHE / folder / f"test_embeddings{suffix}.pt")
    test['state'] = torch.zeros(len(test['ground_truth']), 1)
    kontext_anhaengen(test, "testing", nutzersimulation)
    sim, num_actions = simulator(action_space, rewards, nutzersimulation)
    agent = make_agent("Greedy", {}, 1, num_actions, sim)
    t1 = time.perf_counter()

    rows = []
    for mode, name in BASELINES.items():
        t2 = time.perf_counter()
        set_seed(seed)
        result = agent.evaluate(test, mode)
        rows.append(row(action_space, rewards, "–", name, seed, result, "", "",
                        (t1 - t0, 0.0, time.perf_counter() - t2), {}, "", nutzersimulation))
    return rows


def run_threshold(action_space, rewards, embedding, seed, nutzersimulation=NUTZERSIMULATION):
    t0 = time.perf_counter()
    test = load_embeddings(embedding_file(embedding.removesuffix(PREDICTION), "test"), "prediction")
    kontext_anhaengen(test, "testing", nutzersimulation)
    sim, num_actions = simulator(action_space, rewards, nutzersimulation)
    agent = make_agent("Greedy", {}, 1, num_actions, sim)
    t1 = time.perf_counter()
    set_seed(seed)
    result = agent.evaluate(test, 'threshold')
    return [row(action_space, rewards, embedding, THRESHOLD, seed, result, 1, "",
                (t1 - t0, 0.0, time.perf_counter() - t1), {}, "", nutzersimulation)]


def split_validation(data):
    def pick(mask):
        return {k: v[torch.from_numpy(mask)] if torch.is_tensor(v) else v[mask] for k, v in data.items()}
    val = data['fold'] < 0
    return pick(~val), pick(val)


def expected_metrics(agent, data, sim):
    with torch.no_grad():
        warn = agent.model.model(data['state']).argmax(1).numpy() == 1
    returned = data['ground_truth'] == 1
    r = sim.rewards
    warn_returned = sim.p_true_cancel * r['Warn_Returned_Canceled'] + (1 - sim.p_true_cancel) * r['Warn_Returned_NotCanceled']
    warn_kept = sim.p_false_cancel * r['Warn_NotReturned_Canceled'] + (1 - sim.p_false_cancel) * r['Warn_NotReturned_NotCanceled']
    reward = np.where(warn, np.where(returned, warn_returned, warn_kept),
                      np.where(returned, r['NoWarn_Returned'], r['NoWarn_NotReturned']))
    tp, fp, fn = (warn & returned).sum(), (warn & ~returned).sum(), (~warn & returned).sum()
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"E[Reward]": reward.mean(), "Precision": precision, "Recall": recall,
            "F1": 2 * precision * recall / (precision + recall) if tp else 0.0,
            "FPR": fp / max((~returned).sum(), 1), "WR": warn.mean()}


def run_tuning(job):
    stage, embedding, strategy, seed, changes = job
    net = {k: changes.get(k, v) for k, v in NET_PARAMS.items()}
    explorer = {**EXPLORER_PARAMS[strategy], "m": net["hidden_dim"],
                **{k: changes[k] for k in ("lamda", "nu") if k in changes}}
    params = {**net, **explorer, "batch_size": changes.get("batch_size", BATCH_SIZE), "epochs": changes.get("epochs", EPOCHS)}
    try:
        fit, val = split_validation(load_embeddings(embedding_file(embedding, "train")))
        prepare_states(fit, val, PCA_DIMS.get(embedding))
        set_seed(seed)
        sim, num_actions = simulator("binär", "basis")
        agent = make_agent(strategy, explorer, fit['state'].shape[1], num_actions, sim,
                           batch_size=params["batch_size"], **net)
        rows = []
        for epoch in range(1, params["epochs"] + 1):
            t0 = time.perf_counter()
            agent.train(fit, 1, label=f"[Tuning {stage} | {embedding} | {strategy} | {changes} | Seed {seed} | Epoche {epoch}] ")
            rows.append({"Stufe": stage, "Embedding": embedding, "Strategie": strategy, "Seed": seed,
                         **{k: params[k] for k in TUNING_PARAMS}, "Epoche": epoch,
                         **{k: round(float(v), 5) for k, v in expected_metrics(agent, val, sim).items()},
                         "Dauer Training (s)": round(time.perf_counter() - t0, 1),
                         "Stand": datetime.now().strftime("%Y-%m-%d %H:%M")})
        return rows
    except Exception:
        print(f"FEHLER in {job}:\n{traceback.format_exc()}", flush=True)
        return []


def row(action_space, rewards, embedding, strategy, seed, result, state_dim, pca, durations, params, epochs="",
        nutzersimulation=NUTZERSIMULATION):
    return {
        "Aktionsraum": action_space, "Rewards": rewards, "Nutzersimulation": nutzersimulation,
        "Embedding": embedding, "Strategie": strategy,
        "Seed": seed,
        **{m: round(result[m], 4) for m in METRICS},
        "state-Dim": state_dim, "PCA": pca,
        "Dauer Daten (s)": round(durations[0], 1),
        "Dauer Training (s)": round(durations[1], 1),
        "Dauer Auswertung (s)": round(durations[2], 1),
        "Epochen": epochs,
        "Parameter": ", ".join(f"{k}={v}" for k, v in params.items()),
        "Rechner": f"{os.cpu_count()} CPU-Kerne, {WORKERS} Läufe parallel mit je {THREADS} Threads, GPU: nein",
        "Stand": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def run_job(job):
    action_space, rewards, embedding, strategy, seed, *rest = job
    nutzersimulation = rest[0] if rest else NUTZERSIMULATION
    try:
        if strategy == "Baselines":
            return run_baselines(action_space, rewards, seed, nutzersimulation)
        if strategy == THRESHOLD:
            return run_threshold(action_space, rewards, embedding, seed, nutzersimulation)
        return run_agent(action_space, rewards, embedding, strategy, seed, nutzersimulation)
    except Exception:
        print(f"FEHLER in {job}:\n{traceback.format_exc()}", flush=True)
        return []


def finished():
    if not ERGEBNISSE.exists():
        return set()
    with open(ERGEBNISSE, newline="", encoding="utf-8") as f:
        return {(r["Aktionsraum"], r["Rewards"], r["Embedding"], r["Strategie"], int(r["Seed"]), r["Nutzersimulation"])
                for r in csv.DictReader(f)}


def job_key(job):
    action_space, rewards, embedding, strategy, seed, *rest = job
    return (action_space, rewards, embedding, "Zufall" if strategy == "Baselines" else strategy, seed,
            rest[0] if rest else NUTZERSIMULATION)


def append(rows, path=ERGEBNISSE, columns=COLUMNS):
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def best_configs():
    df = pd.read_csv(ERGEBNISSE)
    df = df[(df["Aktionsraum"] == "binär") & (df["Rewards"] == "basis") & (df["Nutzersimulation"] == "stochastisch")
            & df["Embedding"].str.removesuffix(PREDICTION).isin(EMBEDDINGS) & df["Strategie"].isin(EXPLORER_PARAMS)]
    df = df.assign(Modell=df["Embedding"].str.removesuffix(PREDICTION))

    def beste(gruppe):
        laeufe = gruppe.groupby(["Embedding", "Strategie"])
        mittel = laeufe[["Ø Reward", "FPR"]].mean()
        top = mittel["Ø Reward"].idxmax()
        gleichwertig = [k for k, g in laeufe if k == top or
                        ttest_ind(laeufe.get_group(top)["Ø Reward"], g["Ø Reward"], equal_var=False).pvalue >= 0.05]
        return mittel.loc[gleichwertig, "FPR"].idxmin()

    return ([beste(g[g["Strategie"].isin(STRATEGIES)]) for _, g in df.groupby("Modell")],
            [beste(g) for _, g in df.groupby("Strategie")])


def variante(embedding, graph):
    model = embedding.removesuffix(PREDICTION)
    return model + graph + embedding.removeprefix(model)


def grid_jobs(rewards=REWARD_DESIGN):
    jobs = [(ACTION_SPACE, rewards, "–", "Baselines", seed) for seed in SEEDS]
    return jobs + [(ACTION_SPACE, rewards, embedding, strategy, seed)
                   for strategy in EXPLORER_PARAMS for seed in SEEDS for embedding in EMBEDDINGS]


def test_jobs():
    per_model, per_strategy = best_configs()
    configs = sorted(set(per_model) | set(per_strategy))
    jobs = [(action_space, rewards, variante(embedding, graph), strategy, seed, nutzersimulation)
            for action_space, rewards, graph, nutzersimulation in TESTS.values() for seed in SEEDS
            for embedding, strategy in configs]
    for action_space, rewards, graph, nutzersimulation in TESTS.values():
        if nutzersimulation.startswith("matraix") or action_space != ACTION_SPACE or rewards != "basis":
            jobs += [(action_space, rewards, "–", "Baselines", seed, nutzersimulation) for seed in SEEDS]
        if graph:
            jobs += [(action_space, rewards, variante(model + PREDICTION, graph), THRESHOLD, seed, nutzersimulation)
                     for seed in SEEDS for model in EMBEDDINGS]
    # lange Läufe zuerst
    return sorted(jobs, key=lambda job: 2 * job[3].startswith("Neural") + (job[0] == "14"), reverse=True)


def architektur_jobs():
    jobs = [("binär", "basis", embedding + PREDICTION, strategy, seed)
            for strategy in [*EXPLORER_PARAMS, THRESHOLD] for seed in SEEDS for embedding in EMBEDDINGS]
    return sorted(jobs, key=lambda job: job[3].startswith("Neural"), reverse=True)


def tuning_jobs(stage):
    changes, seeds = TUNING_STAGES[stage]
    return [(stage, embedding, strategy, seed, c) for c in changes for seed in seeds
            for strategy in ("NeuralUCB", "NeuralTS") for embedding in TUNING_EMBEDDINGS]


def tuning(stage):
    def key(r):
        return (r["Embedding"], r["Strategie"], str(r["Seed"]), *(str(r[k]) for k in TUNING_PARAMS))
    done = set()
    if TUNING.exists():
        with open(TUNING, newline="", encoding="utf-8") as f:
            done = {key(r) for r in csv.DictReader(f) if int(r["Epoche"]) == int(r["epochs"])}
    jobs = []
    for job in tuning_jobs(stage):
        stage_, embedding, strategy, seed, changes = job
        net = {k: changes.get(k, v) for k, v in NET_PARAMS.items()}
        params = {**net, **EXPLORER_PARAMS[strategy], **changes, "batch_size": changes.get("batch_size", BATCH_SIZE),
                  "epochs": changes.get("epochs", EPOCHS)}
        if key({"Embedding": embedding, "Strategie": strategy, "Seed": seed, **params}) not in done:
            jobs.append(job)
    print(f"{len(jobs)} Tuning-Läufe offen (Stufe {stage}), {WORKERS} parallel -> {TUNING}", flush=True)
    with mp.get_context("fork").Pool(WORKERS, initializer=torch.set_num_threads, initargs=(THREADS,),
                                     maxtasksperchild=1) as pool:
        for rows in pool.imap_unordered(run_tuning, jobs):
            append(rows, TUNING, TUNING_COLUMNS)
            if rows:
                r = rows[-1]
                verlauf = " ".join(f"{x['E[Reward]']:+.4f}" for x in rows)
                print(f"fertig: {r['Embedding']} | {r['Strategie']} | Seed {r['Seed']} | lr {r['lr']} | nu {r['nu']} | "
                      f"E[Reward] je Epoche {verlauf}", flush=True)


def main(mode, arg=None):
    if mode == "tuning":
        return tuning(arg or "1")
    done = finished()
    candidates = grid_jobs(arg or REWARD_DESIGN) if mode == "raster" else {"tests": test_jobs, "architekturen": architektur_jobs}[mode]()
    jobs = [job for job in candidates if job_key(job) not in done]
    print(f"{len(jobs)} Läufe offen ({mode} {arg or ''}), {WORKERS} parallel -> {ERGEBNISSE}", flush=True)

    # maxtasksperchild=1: jeder Lauf in einem frischen Prozess, damit der Speicher danach frei wird
    ctx = mp.get_context("fork")
    with ctx.Pool(WORKERS, initializer=init_worker, initargs=(ctx.Lock(),), maxtasksperchild=1) as pool:
        for rows in pool.imap_unordered(run_job, jobs):
            append(rows)
            for r in rows:
                print(f"fertig: {r['Aktionsraum']} | {r['Rewards']} | {r['Nutzersimulation']} | {r['Embedding']} | {r['Strategie']} | "
                      f"Seed {r['Seed']} | Ø Reward {r['Ø Reward']:+.4f} | Training {r['Dauer Training (s)'] / 60:.1f} min",
                      flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "raster", sys.argv[2] if len(sys.argv) > 2 else None)
