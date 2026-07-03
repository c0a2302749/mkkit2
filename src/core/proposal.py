from dataclasses import dataclass, field
from enum import Enum, auto


class ProposalStatus(Enum):
    OPEN = auto()
    PASSED = auto()
    FAILED = auto()


@dataclass
class Proposal:
    proposal_id: int
    agent_id: int
    content: str
    turn_created: int
    status: ProposalStatus = ProposalStatus.OPEN
    votes_for: list[int] = field(default_factory=list)
    votes_against: list[int] = field(default_factory=list)
    turn_resolved: int | None = None
