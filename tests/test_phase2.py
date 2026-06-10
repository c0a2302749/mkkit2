import pytest
import numpy as np
from src.agent.engine import OpinionDynamicsEngine
from src.config.personas import create_agent
from src.social.graph import SocialGraph


engine = OpinionDynamicsEngine()


class TestComputeSocialSignals:
    def test_no_followees_returns_zero(self):
        agent = create_agent(0, "rational")
        graph = SocialGraph()
        graph.add_agent(agent)
        s_o, s_r = engine.compute_social_signals(agent, graph)
        assert s_o == 0.0
        assert s_r == 0.0

    def test_followees_average(self):
        agent = create_agent(0, "rational")
        f1 = create_agent(1, "conformist")
        f2 = create_agent(2, "skeptic")
        f1.opinion = 0.8
        f1.risk_perception = 0.3
        f2.opinion = 0.4
        f2.risk_perception = 0.7

        graph = SocialGraph()
        graph.add_agent(agent)
        graph.add_agent(f1)
        graph.add_agent(f2)
        graph.add_follow(0, 1)
        graph.add_follow(0, 2)

        s_o, s_r = engine.compute_social_signals(agent, graph)
        assert s_o == pytest.approx(0.6)
        assert s_r == pytest.approx(0.5)


class TestUpdateRiskPerception:
    def test_basic_update(self):
        result = engine.update_risk_perception(
            R=0.5, W=0.0, S_R=0.0, lambda_i=0.5, w_sys=0.5, w_sns=0.5, b=0.0
        )
        assert 0.0 <= result <= 1.0

    def test_high_lambda_fast_convergence(self):
        r1 = engine.update_risk_perception(0.5, 1.0, 0.5, 0.9, 0.5, 0.5, 0.0)
        r2 = engine.update_risk_perception(0.5, 1.0, 0.5, 0.1, 0.5, 0.5, 0.0)
        assert abs(r1 - 0.5) > abs(r2 - 0.5)

    def test_sigmoid_range(self):
        for b in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            r = engine.update_risk_perception(0.5, 0.0, 0.0, 1.0, 0.5, 0.5, b)
            assert 0.0 <= r <= 1.0


class TestComputeOpinionUpdate:
    def test_no_change_when_all_zero(self):
        result = engine.compute_opinion_update(
            O=0.5, S_O=0.0, R=0.0, alpha=0.0, beta=0.0, gamma=0.0, O_0=0.5
        )
        assert result == 0.5

    def test_alpha_drives_toward_social(self):
        result = engine.compute_opinion_update(
            O=0.5, S_O=1.0, R=0.0, alpha=1.0, beta=0.0, gamma=0.0, O_0=0.5
        )
        assert result > 0.5

    def test_beta_drives_away_from_risk(self):
        result = engine.compute_opinion_update(
            O=0.5, S_O=0.0, R=1.0, alpha=0.0, beta=1.0, gamma=0.0, O_0=0.5
        )
        assert result < 0.5

    def test_gamma_pulls_back_to_initial(self):
        result_with_gamma = engine.compute_opinion_update(
            O=0.8, S_O=0.0, R=0.0, alpha=0.0, beta=0.0, gamma=1.0, O_0=0.5
        )
        result_without_gamma = engine.compute_opinion_update(
            O=0.8, S_O=0.0, R=0.0, alpha=0.0, beta=0.0, gamma=0.0, O_0=0.5
        )
        assert result_with_gamma < result_without_gamma

    def test_clip_range(self):
        result = engine.compute_opinion_update(
            O=0.0, S_O=-1.0, R=1.0, alpha=1.0, beta=1.0, gamma=1.0, O_0=0.5
        )
        assert result >= 0.0
        result = engine.compute_opinion_update(
            O=1.0, S_O=1.0, R=-1.0, alpha=1.0, beta=1.0, gamma=-1.0, O_0=0.5
        )
        assert result <= 1.0
