from dataclasses import dataclass, field
from src.core.agent import Agent


@dataclass
class SimulationState:
    agents: list[Agent] = field(default_factory=list)
    current_turn: int = 0
    governance_risk_score: float = 0.0
    governance_warning: str = ""
