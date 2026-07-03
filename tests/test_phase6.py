import pytest
import numpy as np
from src.analysis.metrics import Statistics
from src.social.graph import SocialGraph
from src.core.agent import Agent
from src.core.agent import Action
from src.core.action import ActionType


def _make_agent(aid: int, opinion: float, initial: float | None = None) -> Agent:
    return Agent(
        agent_id=aid,
        persona_name="rational",
        alpha=0.2, beta=0.4, gamma=0.6,
        lambda_=0.3, bias=-0.1, w_sys=0.5, w_sns=0.5,
        opinion=opinion,
        initial_opinion=initial if initial is not None else opinion,
    )


class TestStatistics:
    def test_opinion_distribution(self):
        agents = [_make_agent(i, 0.5 + i * 0.1) for i in range(5)]
        dist = Statistics.compute_opinion_distribution(agents)
        assert dist["mean"] == pytest.approx(0.7)
        assert dist["min"] == 0.5
        assert dist["max"] == 0.9

    def test_echo_chamber_single_community_returns_zero(self):
        agents = [_make_agent(i, 0.5) for i in range(4)]
        graph = SocialGraph()
        for a in agents:
            graph.add_agent(a)
        graph.generate_complete(4)
        ratio = Statistics.compute_echo_chamber_ratio(agents, graph)
        assert ratio == 0.0

    def test_echo_chamber_two_clusters(self):
        agents = [_make_agent(i, 0.1) for i in range(3)] + [_make_agent(i + 3, 0.9) for i in range(3)]
        graph = SocialGraph()
        for a in agents:
            graph.add_agent(a)
        for i in range(3):
            for j in range(3):
                if i != j:
                    graph.add_follow(i, j)
        for i in range(3, 6):
            for j in range(3, 6):
                if i != j:
                    graph.add_follow(i, j)
        ratio = Statistics.compute_echo_chamber_ratio(agents, graph)
        assert ratio > 0

    def test_conformity_rate(self):
        agents = [_make_agent(0, 0.6, initial=0.5), _make_agent(1, 0.7, initial=0.5)]
        graph = SocialGraph()
        for a in agents:
            graph.add_agent(a)
        graph.add_follow(0, 1)
        rate = Statistics.compute_conformity_rate(agents, graph)
        assert 0 <= rate <= 1

    def test_private_info_neglect_rate_no_conflict(self):
        agents = [_make_agent(0, 0.5, initial=0.5), _make_agent(1, 0.5, initial=0.5)]
        graph = SocialGraph()
        for a in agents:
            graph.add_agent(a)
        graph.add_follow(0, 1)
        rate = Statistics.compute_private_info_neglect_rate(agents, graph)
        assert rate == 0.0

    def test_inter_run_variance(self):
        run1 = [_make_agent(i, 0.5) for i in range(4)]
        run2 = [_make_agent(i, 0.7) for i in range(4)]
        var = Statistics.compute_inter_run_variance([run1, run2])
        assert var > 0


class TestDecisionMetrics:
    def test_decision_summary_no_proposals(self):
        from src.core.proposal import Proposal, ProposalStatus
        summary = Statistics.compute_decision_summary([])
        assert summary["total_proposals"] == 0
        assert summary["resolved"] == 0

    def test_decision_summary_with_resolved(self):
        from src.core.proposal import Proposal, ProposalStatus
        proposals = [
            Proposal(proposal_id=1, agent_id=0, content="a", turn_created=1,
                     status=ProposalStatus.PASSED, votes_for=[0, 1], votes_against=[2],
                     turn_resolved=3),
            Proposal(proposal_id=2, agent_id=1, content="b", turn_created=2,
                     status=ProposalStatus.OPEN),
        ]
        summary = Statistics.compute_decision_summary(proposals)
        assert summary["total_proposals"] == 2
        assert summary["resolved"] == 1
        assert summary["passed"] == 1
        assert summary["failed"] == 0
        assert summary["min_turn_resolved"] == 3

    def test_agreement_rate(self):
        from src.core.proposal import Proposal, ProposalStatus
        prop = Proposal(proposal_id=1, agent_id=0, content="x", turn_created=1,
                        status=ProposalStatus.PASSED, votes_for=[0, 1], votes_against=[2])
        rate = Statistics.compute_agreement_rate(prop)
        assert rate == pytest.approx(2/3)

    def test_participation_rate(self):
        from src.core.proposal import Proposal, ProposalStatus
        proposals = [
            Proposal(proposal_id=1, agent_id=0, content="x", turn_created=1,
                     status=ProposalStatus.PASSED, votes_for=[0, 1], votes_against=[2]),
        ]
        agents = [_make_agent(i, 0.5) for i in range(5)]
        rate = Statistics.compute_participation_rate(agents, proposals)
        assert rate == pytest.approx(3/5)

    def test_vote_opinion_alignment(self):
        from src.core.proposal import Proposal, ProposalStatus
        agents = [_make_agent(0, 0.8), _make_agent(1, 0.7), _make_agent(2, 0.2)]
        proposals = [
            Proposal(proposal_id=1, agent_id=0, content="x", turn_created=1,
                     status=ProposalStatus.PASSED, votes_for=[0, 1], votes_against=[2]),
        ]
        alignment = Statistics.compute_vote_opinion_alignment(agents, proposals)
        assert alignment == 1.0
