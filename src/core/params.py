from dataclasses import dataclass


@dataclass(frozen=True)
class OpinionParams:
    alpha: float
    beta: float
    gamma: float
