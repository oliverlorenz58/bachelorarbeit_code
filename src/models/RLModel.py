import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.Explorer import Explorer


class RLModel:
    def __init__(
        self,
        model: nn.Module,
        explorer: Explorer,
        lr: float = 1e-3,
        batch_size: int = 64,
    ):
        self.model = model
        self.explorer = explorer
        self.batch_size = batch_size
        self.action_dim = model.out_dim

        self.optimizer = None if isinstance(model, LinearModel) else optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.loss_history = []

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def prepare(self, states: torch.Tensor):
        if hasattr(self.explorer, 'prepare'):
            self.explorer.prepare(states)

    def select_action(self, state: torch.Tensor) -> int:
        return self.explorer.select_action(state)

    @torch.no_grad()
    def select_best_action(self, state: torch.Tensor) -> int:
        return int(self.model(state.unsqueeze(0)).argmax())

    def select_random_action(self, state) -> int:
        return int(np.random.randint(self.action_dim))

    def select_always_nothing(self, state) -> int:
        return 0

    def select_always_warn(self, state) -> int:
        return int(np.random.randint(1, self.action_dim))

    def select_threshold(self, state, threshold=0.5) -> int:
        # state ist hier die nicht standardisierte Prediction
        return int(state[0] >= threshold)

    def train_on_batch(self, states: torch.Tensor, actions, rewards):
        actions = torch.as_tensor(actions, dtype=torch.long)
        rewards = torch.as_tensor(rewards, dtype=torch.float)

        if isinstance(self.model, LinearModel):
            loss = self.model.fit(states, actions, rewards)
        else:
            q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(-1)
            loss = self.criterion(q_values, rewards)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            loss = loss.item()

        self.loss_history.append(loss)


class LinearModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, lamda: float = 1.0):
        super().__init__()
        self.out_dim = output_dim
        d = input_dim + 1
        # float64: stabile Inversen bei stark korrelierten Embedding-Dimensionen
        eye = torch.eye(d, dtype=torch.float64).repeat(output_dim, 1, 1)
        self.register_buffer('A', lamda * eye)
        self.register_buffer('A_inv', eye / lamda)
        self.register_buffer('b', torch.zeros(output_dim, d, dtype=torch.float64))
        self.register_buffer('theta', torch.zeros(output_dim, d, dtype=torch.float64))

    @staticmethod
    def _features(states):
        states = states.to(torch.float64)
        return torch.cat([states, torch.ones(*states.shape[:-1], 1, dtype=torch.float64)], dim=-1)

    def forward(self, states):
        return (self._features(states) @ self.theta.T).float()

    def reward_and_sigma(self, state):
        x = self._features(state)
        return self.theta @ x, torch.sqrt((self.A_inv @ x) @ x)

    @torch.no_grad()
    def fit(self, states, actions, rewards):
        x, r = self._features(states), rewards.to(torch.float64)
        loss = ((x * self.theta[actions]).sum(1) - r).pow(2).mean().item()
        for a in actions.unique():
            xa, ra = x[actions == a], r[actions == a]
            self.A[a] += xa.T @ xa
            self.b[a] += xa.T @ ra
            self.A_inv[a] = torch.linalg.inv(self.A[a])
            self.theta[a] = self.A_inv[a] @ self.b[a]
        return loss


class DQN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2
    ):
        super().__init__()
        self.out_dim = output_dim

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
