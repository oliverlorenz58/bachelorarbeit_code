from typing import Protocol


class Simulation(Protocol):
    def evaluate(self, truth, action, context) -> tuple[float, str]:
        ...
