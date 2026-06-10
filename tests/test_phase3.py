import pytest
import numpy as np
import networkx as nx
from src.social.graph import SocialGraph
from src.social.timeline import TimelineManager
from src.core.agent import Agent, Action
from src.core.action import ActionType


def _make_agent(aid: int) -> Agent:
    return Agent(
        agent_id=aid,
        persona_name="rational",
        alpha=0.2, beta=0.4, gamma=0.6,
        lambda_=0.3, bias=-0.1, w_sys=0.5, w_sns=0.5,
    )


class TestSocialGraphTopologies:
    N = 50
    SEED = 42

    def _make_graph(self, topology: str) -> SocialGraph:
        g = SocialGraph()
        for i in range(self.N):
            g._agents[i] = _make_agent(i)
        g.generate_by_topology(topology, self.N, self.SEED)
        return g

    def _compute_metrics(self, g: SocialGraph) -> tuple[float, float]:
        ug = g.graph.to_undirected()
        if ug.number_of_edges() == 0:
            return (0.0, 0.0)
        cc = nx.average_clustering(ug)
        try:
            apl = nx.average_shortest_path_length(ug)
        except nx.NetworkXError:
            apl = float("inf")
        return (cc, apl)

    def test_random_topology(self):
        g = self._make_graph("random")
        assert g.graph.number_of_nodes() == self.N
        cc, apl = self._compute_metrics(g)
        assert 0 < cc < 1
        assert apl < float("inf")

    def test_small_world_topology(self):
        g = self._make_graph("small-world")
        assert g.graph.number_of_nodes() == self.N
        cc, apl = self._compute_metrics(g)
        assert cc > 0.2
        assert apl < 10

    def test_scale_free_topology(self):
        g = self._make_graph("scale-free")
        assert g.graph.number_of_nodes() == self.N
        cc, apl = self._compute_metrics(g)
        assert 0 < cc < 1
        assert apl < float("inf")

    def test_community_topology(self):
        g = self._make_graph("community")
        assert g.graph.number_of_nodes() == self.N
        cc, apl = self._compute_metrics(g)
        assert cc > 0.1

    def test_complete_topology(self):
        g = self._make_graph("complete")
        assert g.graph.number_of_nodes() == self.N
        cc, apl = self._compute_metrics(g)
        assert cc == 1.0
        assert apl == 1.0

    def test_isolated_topology(self):
        g = self._make_graph("isolated")
        assert g.graph.number_of_nodes() == self.N
        assert g.graph.number_of_edges() == 0


class TestSocialGraphBasic:
    def test_add_follow_and_get_followees(self):
        g = SocialGraph()
        a0, a1 = _make_agent(0), _make_agent(1)
        g.add_agent(a0)
        g.add_agent(a1)
        g.add_follow(0, 1)
        assert g.get_followees(0) == [1]
        assert g.get_followers(1) == [0]

    def test_remove_follow(self):
        g = SocialGraph()
        a0, a1 = _make_agent(0), _make_agent(1)
        g.add_agent(a0)
        g.add_agent(a1)
        g.add_follow(0, 1)
        g.remove_follow(0, 1)
        assert g.get_followees(0) == []

    def test_get_agent(self):
        g = SocialGraph()
        a0 = _make_agent(0)
        g.add_agent(a0)
        assert g.get_agent(0) is a0
        assert g.get_agent(999) is None


class TestTimelineManager:
    def test_add_propose_assigns_id(self):
        tl = TimelineManager()
        action = Action(ActionType.PROPOSE, "test")
        tl.add_post(0, 1, action)
        assert action.proposal_id == 1

    def test_incrementing_proposal_ids(self):
        tl = TimelineManager()
        a1 = Action(ActionType.PROPOSE, "first")
        a2 = Action(ActionType.PROPOSE, "second")
        tl.add_post(0, 1, a1)
        tl.add_post(0, 2, a2)
        assert a1.proposal_id == 1
        assert a2.proposal_id == 2

    def test_do_nothing_not_posted(self):
        tl = TimelineManager()
        action = Action(ActionType.DO_NOTHING)
        tl.add_post(0, 1, action)
        proposals = tl.get_proposals_with_ids([0], 1)
        discussions = tl.get_recent_discussion([0], 1)
        assert proposals == []
        assert discussions == []

    def test_get_proposals_with_ids(self):
        tl = TimelineManager()
        tl.add_post(0, 1, Action(ActionType.PROPOSE, "content"))
        result = tl.get_proposals_with_ids([0], 1)
        assert len(result) == 1
        aid, pid, action = result[0]
        assert aid == 0
        assert pid == 1
        assert action.content == "content"

    def test_window_filter(self):
        tl = TimelineManager()
        tl.add_post(0, 1, Action(ActionType.PROPOSE, "old"))
        tl.add_post(0, 5, Action(ActionType.PROPOSE, "new"))
        result = tl.get_proposals_with_ids([0], 5, window=2)
        assert len(result) == 1
        assert result[0][2].content == "new"

    def test_followee_filter(self):
        tl = TimelineManager()
        tl.add_post(0, 1, Action(ActionType.PROPOSE, "from 0"))
        tl.add_post(1, 1, Action(ActionType.PROPOSE, "from 1"))
        result = tl.get_proposals_with_ids([0], 1)
        assert len(result) == 1
        assert result[0][0] == 0

    def test_discussion_filter(self):
        tl = TimelineManager()
        tl.add_post(0, 1, Action(ActionType.COMMENT, "comment", proposal_id=1))
        tl.add_post(0, 1, Action(ActionType.SUPPORT, "support", proposal_id=1))
        tl.add_post(0, 1, Action(ActionType.OPPOSE, "oppose", proposal_id=1))
        tl.add_post(0, 1, Action(ActionType.VOTE, "vote", proposal_id=1))
        discussions = tl.get_recent_discussion([0], 1)
        assert len(discussions) == 3
        assert all(a.action_type in (ActionType.COMMENT, ActionType.SUPPORT, ActionType.OPPOSE) for a in discussions)

    def test_proposals_not_in_discussion(self):
        tl = TimelineManager()
        tl.add_post(0, 1, Action(ActionType.PROPOSE, "proposal"))
        assert tl.get_recent_discussion([0], 1) == []

    def test_discussion_not_in_proposals(self):
        tl = TimelineManager()
        tl.add_post(0, 1, Action(ActionType.COMMENT, "comment", proposal_id=1))
        assert tl.get_proposals_with_ids([0], 1) == []

    def test_get_timeline_format(self):
        tl = TimelineManager()
        agent = _make_agent(0)
        tl.add_post(1, 1, Action(ActionType.PROPOSE, "my idea"))
        tl.add_post(2, 1, Action(ActionType.SUPPORT, "good point", proposal_id=1))
        timeline = tl.get_timeline(agent, [1, 2], 1)
        assert "#1" in timeline
        assert "\"my idea\"" in timeline
        assert "SUPPORT on #1" in timeline

    def test_clear(self):
        tl = TimelineManager()
        tl.add_post(0, 1, Action(ActionType.PROPOSE, "test"))
        tl.clear()
        assert tl.get_proposals_with_ids([0], 1) == []
        assert tl._next_proposal_id == 1
