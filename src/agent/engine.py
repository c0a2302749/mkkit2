import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.agent import Agent
    from src.social.graph import SocialGraph


class OpinionDynamicsEngine:
    @staticmethod
    def compute_social_signals(
        agent: "Agent", social_graph: "SocialGraph"
    ) -> tuple[float, float]:
        followees = social_graph.get_followees(agent.agent_id)
        if not followees:
            return (0.0, 0.0)

        n = len(followees)
        sum_o = 0.0
        sum_r = 0.0
        for fid in followees:
            f_agent = social_graph.get_agent(fid)
            if f_agent:
                sum_o += f_agent.opinion
                sum_r += f_agent.risk_perception

        return (sum_o / n, sum_r / n)

    @staticmethod
    def update_risk_perception(
        R: float,
        W: float,
        S_R: float,
        lambda_i: float,
        w_sys: float,
        w_sns: float,
        b: float,
    ) -> float:
        R_tilde = w_sys * W + w_sns * S_R + b
        sigmoid = 1.0 / (1.0 + np.exp(-R_tilde))
        return (1 - lambda_i) * R + lambda_i * sigmoid

    @staticmethod
    def compute_opinion_update(
        O: float,
        S_O: float,
        R: float,
        alpha: float,
        beta: float,
        gamma: float,
        O_0: float,
    ) -> float:
        delta = alpha * S_O - beta * R - gamma * (O - O_0)
        return np.clip(O + delta, 0.0, 1.0)
