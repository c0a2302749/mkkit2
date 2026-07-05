import random
import re
from src.core.state import SimulationState
from src.core.agent import Agent
from src.core.action import ActionType
from src.core.proposal import Proposal, ProposalStatus
from src.agent.engine import OpinionDynamicsEngine
from src.agent.manager import AgentManager
from src.social.graph import SocialGraph
from src.social.timeline import TimelineManager
from src.governance.base import GovernanceLayer


POSITIVE_WORDS = ("support", "yes", "for", "approve", "in favour", "in favor", "favorable", "vote yes")
NEGATIVE_WORDS = ("oppose", "no", "against", "reject", "cannot support", "can't support", "vote no")


def _extract_vote_from_rationale(rationale: str, proposal_id: int) -> str | None:
    """Extract yes/no vote for a specific proposal from free-text rationale."""
    if not rationale:
        return None
    # Patterns: "Proposal #N ... support/oppose" or "#N ... support/oppose"
    patterns = [
        rf'(?:[Pp]roposal\s*)?#\s*{proposal_id}\D{{0,200}}?({"|".join(POSITIVE_WORDS)})',
        rf'(?:[Pp]roposal\s*)?#\s*{proposal_id}\D{{0,200}}?({"|".join(NEGATIVE_WORDS)})',
    ]
    # Check negative first (to catch "cannot support")
    m = re.search(patterns[1], rationale, re.IGNORECASE | re.DOTALL)
    if m:
        return "no"
    m = re.search(patterns[0], rationale, re.IGNORECASE | re.DOTALL)
    if m:
        return "yes"
    return None


class SimulationEngine:
    def __init__(
        self,
        state: SimulationState,
        social_graph: SocialGraph,
        timeline: TimelineManager,
        agent_manager: AgentManager,
        governance: GovernanceLayer,
        engine: OpinionDynamicsEngine | None = None,
        scenario_context: str = "",
    ):
        self.state = state
        self.social_graph = social_graph
        self.timeline = timeline
        self.agent_manager = agent_manager
        self.governance = governance
        self.ode = engine or OpinionDynamicsEngine()
        self.scenario_context = scenario_context

    async def run_turn(self) -> dict:
        self.state.current_turn += 1
        turn = self.state.current_turn

        # Step 1: Governance → W, warning
        W = self.governance.compute_risk_score(
            self.state, self.social_graph, None, None
        )
        warning = self.governance.build_system_warning(W)
        self.state.governance_risk_score = W
        self.state.governance_warning = warning

        # Step 2: OpinionDynamicsEngine → O, R update
        social_signals: dict[int, tuple[float, float]] = {}
        for agent in self.state.agents:
            s_o, s_r = self.ode.compute_social_signals(agent, self.social_graph)
            social_signals[agent.agent_id] = (s_o, s_r)
            new_r = self.ode.update_risk_perception(
                R=agent.risk_perception,
                W=W,
                S_R=s_r,
                lambda_i=agent.lambda_,
                w_sys=agent.w_sys,
                w_sns=agent.w_sns,
                b=agent.bias,
            )
            new_o = self.ode.compute_opinion_update(
                O=agent.opinion,
                S_O=s_o,
                R=new_r,
                alpha=agent.alpha,
                beta=agent.beta,
                gamma=agent.gamma,
                O_0=agent.initial_opinion,
                noise_std=0.03,
            )
            agent.risk_perception = new_r
            agent.opinion = new_o

        # Step 3: AgentManager → LLM行動決定 (shuffle order)
        shuffled = list(self.state.agents)
        random.shuffle(shuffled)
        turn_actions: list[dict] = []
        for agent in shuffled:
            followees = self.social_graph.get_followees(agent.agent_id)
            timeline_str = self.timeline.get_timeline(agent, followees, turn)
            action = await self.agent_manager.decide_action(
                agent, timeline_str, warning, turn, self.state.total_turns,
                proposals=self.state.proposals,
                scenario_context=self.scenario_context,
            )
            agent.action_history.append(action)
            self.timeline.add_post(agent.agent_id, turn, action)

            # Create Proposal object for PROPOSE actions
            if action.action_type == ActionType.PROPOSE and action.proposal_id is not None:
                self.state.proposals.append(Proposal(
                    proposal_id=action.proposal_id,
                    agent_id=agent.agent_id,
                    content=action.content,
                    turn_created=turn,
                ))

            if action.action_type == ActionType.VOTE and action.votes is None:
                print(f"  [WARN] Agent {agent.agent_id} VOTE returned no structured votes")

            turn_actions.append({
                "turn": turn,
                "agent_id": agent.agent_id,
                "persona": agent.persona_name,
                "action": action.action_type.name if action.action_type else "NONE",
                "content": action.content,
                "rationale": action.rationale,
                "proposal_id": action.proposal_id,
                "votes": action.votes,
                "opinion": round(agent.opinion, 4),
                "risk_perception": round(agent.risk_perception, 4),
            })

        # Step 4: Vote tallying — resolve only on last turn
        is_last_turn = self.state.total_turns > 0 and turn >= self.state.total_turns
        vote_stats = {"structured_votes": 0, "fallback_votes": 0, "no_vote": 0}
        for proposal in self.state.proposals:
            if proposal.status != ProposalStatus.OPEN:
                continue
            last_votes: dict[int, str] = {}
            vote_source: dict[int, str] = {}
            for agent in self.state.agents:
                for act in reversed(agent.action_history):
                    if act.action_type != ActionType.VOTE:
                        continue
                    # Multi-vote format (structured)
                    if act.votes:
                        for v in act.votes:
                            if v.get("proposal_id") == proposal.proposal_id:
                                last_votes[agent.agent_id] = v.get("vote", "").strip().lower()
                                vote_source[agent.agent_id] = "structured"
                                break
                    # Single-vote fallback
                    elif act.proposal_id == proposal.proposal_id:
                        last_votes[agent.agent_id] = act.content.strip().lower()
                        vote_source[agent.agent_id] = "structured"
                    # Rationale-based fallback
                    else:
                        extracted = _extract_vote_from_rationale(act.rationale, proposal.proposal_id)
                        if extracted:
                            last_votes[agent.agent_id] = extracted
                            vote_source[agent.agent_id] = "fallback"
                    if agent.agent_id in last_votes:
                        break
            if is_last_turn:
                for aid, vote in last_votes.items():
                    if vote in ("yes", "yea", "approve", "true", "for", "1"):
                        proposal.votes_for.append(aid)
                    else:
                        proposal.votes_against.append(aid)
                    src = vote_source.get(aid, "fallback")
                    if src == "structured":
                        vote_stats["structured_votes"] += 1
                    else:
                        vote_stats["fallback_votes"] += 1
                proposal.turn_resolved = turn
                for_count = len(proposal.votes_for)
                against_count = len(proposal.votes_against)
                proposal.status = ProposalStatus.PASSED if for_count > against_count else ProposalStatus.FAILED

        # Count agents who didn't vote on any proposal
        all_voter_ids: set[int] = set()
        for p in self.state.proposals:
            all_voter_ids.update(p.votes_for)
            all_voter_ids.update(p.votes_against)
        n_agents = len(self.state.agents)
        vote_stats["no_vote"] = n_agents - len(all_voter_ids)

        return {
            "turn": turn,
            "actions": turn_actions,
            "social_signals": social_signals,
            "vote_stats": vote_stats if is_last_turn else {},
        }

    async def run(self, turns: int) -> list[dict]:
        turn_data_list = []
        for _ in range(turns):
            data = await self.run_turn()
            turn_data_list.append(data)
        return turn_data_list
