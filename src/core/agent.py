from dataclasses import dataclass, field
from src.core.action import ActionType


@dataclass
class Action:
    action_type: ActionType
    content: str = ""
    rationale: str = ""
    proposal_id: int | None = None
    votes: list[dict] | None = None


@dataclass
class Agent:
    agent_id: int
    persona_name: str
    alpha: float
    beta: float
    gamma: float
    lambda_: float
    bias: float
    w_sys: float
    w_sns: float

    opinion: float = 0.5
    initial_opinion: float = 0.5
    risk_perception: float = 0.5

    action_history: list[Action] = field(default_factory=list)
