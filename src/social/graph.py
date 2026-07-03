import networkx as nx
import numpy as np
from typing import Optional
from src.core.agent import Agent, Action
from src.core.action import ActionType
from src.social.timeline import TimelineManager


class SocialGraph:
    def __init__(self):
        self._graph: nx.DiGraph = nx.DiGraph()
        self._agents: dict[int, Agent] = {}

    def add_agent(self, agent: Agent) -> None:
        self._graph.add_node(agent.agent_id)
        self._agents[agent.agent_id] = agent

    def add_follow(self, from_id: int, to_id: int) -> None:
        self._graph.add_edge(from_id, to_id)

    def remove_follow(self, from_id: int, to_id: int) -> None:
        if self._graph.has_edge(from_id, to_id):
            self._graph.remove_edge(from_id, to_id)

    def get_followees(self, agent_id: int) -> list[int]:
        return list(self._graph.successors(agent_id))

    def get_followers(self, agent_id: int) -> list[int]:
        return list(self._graph.predecessors(agent_id))

    def get_agent(self, agent_id: int) -> Optional[Agent]:
        return self._agents.get(agent_id)

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @staticmethod
    def _to_directed(undirected: nx.Graph) -> nx.DiGraph:
        directed = nx.DiGraph()
        directed.add_nodes_from(undirected.nodes())
        directed.add_edges_from(undirected.edges())
        directed.add_edges_from((v, u) for u, v in undirected.edges())
        return directed

    def generate_random(self, n_agents: int, seed: int) -> None:
        g = nx.gnp_random_graph(n_agents, 0.3, seed=seed)
        self._graph = self._to_directed(g)

    def generate_small_world(self, n_agents: int, seed: int, k: int = 4, p: float = 0.2) -> None:
        g = nx.watts_strogatz_graph(n_agents, k, p, seed=seed)
        self._graph = self._to_directed(g)

    def generate_scale_free(self, n_agents: int, seed: int, m: int = 2) -> None:
        g = nx.barabasi_albert_graph(n_agents, m, seed=seed)
        self._graph = self._to_directed(g)

    def generate_community(self, n_agents: int, seed: int, n_communities: int = 2) -> None:
        sizes = [n_agents // n_communities] * n_communities
        remainder = n_agents - sum(sizes)
        if remainder > 0:
            sizes[-1] += remainder
        p_in = 0.6
        p_out = 0.05
        g = nx.stochastic_block_model(sizes, p_in * np.eye(n_communities) + p_out * (1 - np.eye(n_communities)), seed=seed)
        self._graph = self._to_directed(g)

    def generate_complete(self, n_agents: int) -> None:
        g = nx.complete_graph(n_agents)
        self._graph = self._to_directed(g)

    def generate_isolated(self, n_agents: int) -> None:
        self._graph = nx.DiGraph()
        self._graph.add_nodes_from(range(n_agents))

    def update_network(self, action: Action, actor_id: int, timeline: TimelineManager) -> None:
        if action.action_type in (
            ActionType.SUPPORT, ActionType.OPPOSE, ActionType.COMMENT, ActionType.VOTE,
        ) and action.proposal_id is not None:
            author = timeline.get_proposal_author(action.proposal_id)
            if author is not None and author != actor_id:
                self.add_follow(actor_id, author)

    def generate_by_topology(self, topology: str, n_agents: int, seed: int) -> None:
        generators = {
            "random": self.generate_random,
            "small-world": self.generate_small_world,
            "scale-free": self.generate_scale_free,
            "community": self.generate_community,
            "complete": self.generate_complete,
            "isolated": self.generate_isolated,
        }
        if topology not in generators:
            raise ValueError(f"Unknown topology: {topology}. Choose from {list(generators.keys())}")
        if topology == "complete":
            generators[topology](n_agents)
        elif topology == "isolated":
            generators[topology](n_agents)
        else:
            generators[topology](n_agents, seed)
