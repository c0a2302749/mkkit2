import numpy as np
import networkx as nx
from typing import Optional
from src.core.agent import Agent
from src.social.graph import SocialGraph


class Statistics:
    @staticmethod
    def compute_opinion_distribution(agents: list[Agent]) -> dict:
        opinions = [a.opinion for a in agents]
        return {
            "mean": float(np.mean(opinions)),
            "std": float(np.std(opinions)),
            "min": float(np.min(opinions)),
            "max": float(np.max(opinions)),
            "q25": float(np.percentile(opinions, 25)),
            "q75": float(np.percentile(opinions, 75)),
        }

    @staticmethod
    def compute_echo_chamber_ratio(
        agents: list[Agent], graph: SocialGraph
    ) -> float:
        dig = graph.graph.to_undirected()
        communities = list(nx.community.greedy_modularity_communities(dig))
        if len(communities) < 2:
            return 0.0

        intra_variances = []
        inter_distances = []
        agent_map = {a.agent_id: a.opinion for a in agents}

        for comm in communities:
            members = [m for m in comm if m in agent_map]
            if len(members) < 2:
                continue
            opinions = [agent_map[m] for m in members]
            intra_variances.append(float(np.var(opinions)))

        comm_list = list(communities)
        for i in range(len(comm_list)):
            for j in range(i + 1, len(comm_list)):
                mi = [m for m in comm_list[i] if m in agent_map]
                mj = [m for m in comm_list[j] if m in agent_map]
                if mi and mj:
                    oi = np.mean([agent_map[m] for m in mi])
                    oj = np.mean([agent_map[m] for m in mj])
                    inter_distances.append(float(abs(oi - oj)))

        if not intra_variances or not inter_distances:
            return 0.0

        return float(np.mean(intra_variances) / np.mean(inter_distances))

    @staticmethod
    def compute_conformity_rate(
        agents: list[Agent], graph: SocialGraph
    ) -> float:
        count = 0
        total = 0
        for agent in agents:
            followees = graph.get_followees(agent.agent_id)
            if not followees:
                continue
            followee_opinions = []
            for fid in followees:
                f = graph.get_agent(fid)
                if f:
                    followee_opinions.append(f.opinion)
            if not followee_opinions:
                continue
            tl_mean = np.mean(followee_opinions)
            if len(agent.action_history) < 2:
                continue
            prev_o = agent.initial_opinion
            for action in agent.action_history:
                pass
            direction = agent.opinion - prev_o
            tl_direction = tl_mean - prev_o
            if abs(tl_direction) > 0.01:
                if direction * tl_direction > 0:
                    count += 1
                total += 1
        return count / total if total > 0 else 0.0

    @staticmethod
    def compute_private_info_neglect_rate(
        agents: list[Agent], graph: SocialGraph
    ) -> float:
        count = 0
        total = 0
        for agent in agents:
            followees = graph.get_followees(agent.agent_id)
            if not followees:
                continue
            followee_opinions = []
            for fid in followees:
                f = graph.get_agent(fid)
                if f:
                    followee_opinions.append(f.opinion)
            if not followee_opinions:
                continue
            tl_mean = np.mean(followee_opinions)
            initial = agent.initial_opinion
            if abs(tl_mean - initial) <= 0.1:
                continue
            if agent.opinion == initial:
                continue
            tl_opposes = (tl_mean > initial + 0.1) or (tl_mean < initial - 0.1)
            if not tl_opposes:
                continue
            moved_toward_tl = (agent.opinion - initial) * (tl_mean - initial) > 0
            if moved_toward_tl:
                count += 1
            total += 1
        return count / total if total > 0 else 0.0

    @staticmethod
    def compute_inter_run_variance(run_results: list[list[Agent]]) -> float:
        final_means = [np.mean([a.opinion for a in run]) for run in run_results]
        return float(np.var(final_means))

    @staticmethod
    def compute_decision_summary(proposals: list) -> dict:
        resolved = [p for p in proposals if p.status.name != "OPEN"]
        passed = [p for p in resolved if p.status.name == "PASSED"]
        failed = [p for p in resolved if p.status.name == "FAILED"]
        return {
            "total_proposals": len(proposals),
            "resolved": len(resolved),
            "passed": len(passed),
            "failed": len(failed),
            "avg_turns_to_resolution": (
                float(np.mean([p.turn_resolved for p in resolved]))
                if resolved else None
            ),
            "min_turn_resolved": min(p.turn_resolved for p in resolved) if resolved else None,
        }

    @staticmethod
    def compute_agreement_rate(proposal) -> float:
        total = len(proposal.votes_for) + len(proposal.votes_against)
        if total == 0:
            return 0.0
        return max(len(proposal.votes_for), len(proposal.votes_against)) / total

    @staticmethod
    def compute_participation_rate(agents: list, proposals: list) -> float:
        voter_ids: set[int] = set()
        for p in proposals:
            voter_ids.update(p.votes_for)
            voter_ids.update(p.votes_against)
        return len(voter_ids) / len(agents) if agents else 0.0

    @staticmethod
    def compute_vote_opinion_alignment(agents: list, proposals: list) -> float:
        aligned = 0
        total = 0
        for p in proposals:
            if p.status.name == "OPEN":
                continue
            agent_map = {a.agent_id: a for a in agents}
            for aid in p.votes_for:
                agent = agent_map.get(aid)
                if agent and agent.opinion >= 0.5:
                    aligned += 1
                total += 1
            for aid in p.votes_against:
                agent = agent_map.get(aid)
                if agent and agent.opinion < 0.5:
                    aligned += 1
                total += 1
        return aligned / total if total > 0 else 0.0

