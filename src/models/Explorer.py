from typing import Protocol

import numpy as np
import torch
import torch.nn as nn


class Explorer(Protocol):
    def select_action(self, state: torch.Tensor) -> int:
        ...


class LinUCB:
    def __init__(self, model, action_dim: int, alpha: float = 1.0):
        self.model = model
        self.action_dim = action_dim
        self.alpha = alpha

    def select_action(self, state: torch.Tensor) -> int:
        reward, sigma = self.model.reward_and_sigma(state)
        return int(torch.argmax(self._score(reward, sigma)))

    def _score(self, reward, sigma):
        return reward + self.alpha * sigma


class LinTS(LinUCB):
    def __init__(self, model, action_dim: int, v: float = 1.0):
        super().__init__(model, action_dim)
        self.v = v

    def _score(self, reward, sigma):
        return torch.normal(reward, self.v * sigma)


class NeuralUCB:
    def __init__(
        self,
        model: torch.nn.Module,
        action_dim: int,
        m: int,
        lamda: float = 1.0,
        nu: float = 0.1,
    ):
        self.model = model
        self.action_dim = action_dim
        self.p = sum(q.numel() for q in model.parameters() if q.requires_grad)
        self.m = m
        self.lamda = lamda
        self.nu = nu

        self.linears = [layer for layer in model.net if isinstance(layer, nn.Linear)]
        assert all(isinstance(layer, (nn.Linear, nn.ReLU)) for layer in model.net), "Netz wie RLModel.DQN erwartet"

        # m * Z und 1 / (m * Z) als Vektoren, dazu Sichten (Gewichte, Bias) je Schicht
        self.mz = torch.full((self.p,), float(m * lamda))
        self.inv_mz = torch.empty(self.p)
        self.layers, offset = [], 0
        for layer in self.linears:
            nw, nb = layer.weight.numel(), layer.bias.numel()
            self.layers.append((self.mz[offset:offset + nw].view_as(layer.weight),
                                self.mz[offset + nw:offset + nw + nb],
                                self.inv_mz[offset:offset + nw].view_as(layer.weight),
                                self.inv_mz[offset + nw:offset + nw + nb]))
            offset += nw + nb
        self.batch = None

    @torch.no_grad()
    def prepare(self, states: torch.Tensor):
        inputs = [states]
        for layer in self.linears[:-1]:
            inputs.append(torch.relu(layer(inputs[-1])))
        q = self.linears[-1](inputs[-1])
        n, a = q.shape
        delta = torch.eye(a).expand(n, a, a)
        delta_sq = [None] * len(self.linears)
        for k in reversed(range(len(self.linears))):
            delta_sq[k] = delta ** 2
            if k > 0:
                delta = (delta @ self.linears[k].weight) * (inputs[k] > 0).unsqueeze(1)
        self.batch = dict(states=states, rewards=q.tolist(), h_sq=[h ** 2 for h in inputs],
                          delta_sq=delta_sq, pos=0)

    @torch.no_grad()
    def select_action(self, state: torch.Tensor) -> int:
        b = self.batch
        if b is None or b['pos'] >= len(b['states']) or not torch.equal(b['states'][b['pos']], state):
            self.prepare(state.unsqueeze(0))
            b = self.batch
        i = b['pos']
        b['pos'] += 1

        torch.reciprocal(self.mz, out=self.inv_mz)
        sigma_sq = torch.zeros(self.action_dim)
        for (_, _, inv_w, inv_b), h_sq, delta_sq in zip(self.layers, b['h_sq'], b['delta_sq']):
            sigma_sq += delta_sq[i] @ (inv_w @ h_sq[i] + inv_b)
        sigma = torch.sqrt(torch.clamp(sigma_sq, min=1e-8)).tolist()

        scores = [self._score(f_val, s) for f_val, s in zip(b['rewards'][i], sigma)]
        action = int(np.argmax(scores))

        for (mz_w, mz_b, _, _), h_sq, delta_sq in zip(self.layers, b['h_sq'], b['delta_sq']):
            mz_w.addr_(delta_sq[i, action], h_sq[i])
            mz_b += delta_sq[i, action]
        return action

    def _score(self, f_val, sigma):
        return f_val + self.nu * sigma


class NeuralTS(NeuralUCB):
    def _score(self, f_val, sigma):
        return np.random.normal(f_val, self.nu * np.sqrt(self.lamda) * sigma)


class Greedy:
    def __init__(
        self,
        model: torch.nn.Module,
        action_dim: int,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.model = model
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def select_action(self, state: torch.Tensor) -> int:
        if np.random.rand() < self.epsilon:
            action = int(np.random.randint(self.action_dim))
        else:
            with torch.no_grad():
                q_values = self.model(state.unsqueeze(0))
            action = int(torch.argmax(q_values).item())

        self._decay_epsilon()
        return action

    def _decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
