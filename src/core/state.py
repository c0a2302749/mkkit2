from dataclasses import dataclass, field
from src.core.agent import Agent
from src.core.proposal import Proposal


@dataclass
class SimulationState:
    agents: list[Agent] = field(default_factory=list)
    current_turn: int = 0
    total_turns: int = 0
    governance_risk_score: float = 0.0
    governance_warning: str = ""
    proposals: list[Proposal] = field(default_factory=list)
