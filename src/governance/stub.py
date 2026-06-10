from src.governance.base import GovernanceLayer


class GovernanceStub(GovernanceLayer):
    def compute_risk_score(self, state, social_graph, opinion_distribution, exposure_summary) -> float:
        return 0.0

    def build_system_warning(self, risk_score: float, details=None) -> str:
        return ""
