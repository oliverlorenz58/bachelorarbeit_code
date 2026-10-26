import random
from collections import Counter

import numpy as np
import torch

from models.Evaluator import metrics
from models.Explorer import Greedy, LinTS, LinUCB, NeuralTS, NeuralUCB
from models.RLModel import DQN, LinearModel, RLModel
from simulation.Simulator import Simulation

EXPLORERS = {"LinUCB": LinUCB, "LinTS": LinTS, "NeuralUCB": NeuralUCB, "NeuralTS": NeuralTS, "Greedy": Greedy}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_embeddings(path, state='embedding'):
    e = torch.load(path, weights_only=False, mmap=True)
    return {
        'state': e['state'] if state == 'embedding' else torch.sigmoid(e['logit'].float()).unsqueeze(1),
        'ground_truth': e['ground_truth'].long().numpy(),
        'reason': e['reason'].numpy(),
        'fold': e['fold'].numpy(),
        'edge_id': e['edge_id'].numpy(),
    }


def prepare_states(train, test, pca_dim=None):
    mean, std = train['state'].mean(0), train['state'].std(0)
    std[std == 0] = 1.0
    for data in (train, test):
        data['state'].sub_(mean).div_(std)

    if pca_dim is None:
        return 1.0
    cov = train['state'].T @ train['state'] / (len(train['state']) - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    components = eigenvectors[:, -pca_dim:].flip(1)
    for data in (train, test):
        data['state'] = data['state'] @ components
    return (eigenvalues[-pca_dim:].sum() / eigenvalues.sum()).item()


def make_agent(explorer, explorer_params, state_dim, num_actions, simulator,
               hidden_dim=128, num_layers=2, lr=1e-3, ridge_lambda=1.0, batch_size=64):
    if explorer in ("LinUCB", "LinTS"):
        model = LinearModel(input_dim=state_dim, output_dim=num_actions, lamda=ridge_lambda)
    else:
        model = DQN(input_dim=state_dim, output_dim=num_actions, hidden_dim=hidden_dim, num_layers=num_layers)
    return RLAgent(net_model=model, explorer=EXPLORERS[explorer](model, num_actions, **explorer_params),
                   simulator=simulator, lr=lr, batch_size=batch_size)


class RLAgent:
    def __init__(self,
                 net_model,
                 explorer,
                 simulator: Simulation,
                 lr: float = 1e-3,
                 batch_size: int = 64):

        self.model = RLModel(
            explorer=explorer,
            model=net_model,
            lr=lr,
            batch_size=batch_size)

        self.simulator = simulator

    def generateAction(self, state, mode) -> int:
        select = {
            'train': self.model.select_action,
            'best': self.model.select_best_action,
            'random': self.model.select_random_action,
            'always_nothing': self.model.select_always_nothing,
            'always_warn': self.model.select_always_warn,
            'threshold': self.model.select_threshold,
        }
        return select[mode](state)

    def evaluateAction(self, truth, action, context):
        return self.simulator.evaluate(truth, action, context)

    def train(self, data, epochs, label=""):
        self.model.train()
        n, batch_size = len(data['ground_truth']), self.model.batch_size
        context = data.get('context', data['reason'])
        batch_rewards = []
        for epoch in range(1, epochs + 1):
            order = np.random.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                self.model.prepare(data['state'][idx])
                actions, rewards = [], []
                for i in idx:
                    action = self.generateAction(data['state'][i], 'train')
                    reward, _ = self.evaluateAction(data['ground_truth'][i], action, context[i])
                    actions.append(action)
                    rewards.append(reward)
                self.model.train_on_batch(data['state'][idx], actions, rewards)
                batch_rewards.append(np.mean(rewards))

            epoch_rewards = batch_rewards[-len(range(0, n, batch_size)):]
            print(f"{label}Epoche {epoch}/{epochs} | mittlerer Reward im Training {np.mean(epoch_rewards):.4f}",
                  flush=True)
        return batch_rewards

    @torch.no_grad()
    def evaluate(self, data, mode):
        self.model.eval()
        counts, total_reward = Counter(), 0.0
        for state, truth, context in zip(data['state'], data['ground_truth'], data.get('context', data['reason'])):
            action = self.generateAction(state, mode)
            reward, outcome = self.evaluateAction(truth, action, context)
            counts[outcome] += 1
            total_reward += reward
        return {'Ø Reward': total_reward / len(data['ground_truth']), **metrics(counts)}
