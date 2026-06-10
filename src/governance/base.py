from abc import ABC, abstractmethod
from typing import Any


class GovernanceLayer(ABC):
    @abstractmethod
    def compute_risk_score(
        self,
        state: Any,
        social_graph: Any,
        opinion_distribution: Any,
        exposure_summary: Any,
    ) -> float:
        ...

    @abstractmethod
    def build_system_warning(self, risk_score: float, details: Any = None) -> str:
        ...
