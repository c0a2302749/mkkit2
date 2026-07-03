import random
from src.core.state import SimulationState
from src.core.agent import Agent
from src.core.action import ActionType
from src.agent.engine import OpinionDynamicsEngine
from src.agent.manager import AgentManager
from src.social.graph import SocialGraph
from src.social.timeline import TimelineManager
from src.governance.base import GovernanceLayer


class SimulationEngine:
    def __init__(
        self,
        state: SimulationState,
        social_graph: SocialGraph,
        timeline: TimelineManager,
        agent_manager: AgentManager,
        governance: GovernanceLayer,
        engine: OpinionDynamicsEngine | None = None,
    ):
        self.state = state
        self.social_graph = social_graph
        self.timeline = timeline
        self.agent_manager = agent_manager
        self.governance = governance
        self.ode = engine or OpinionDynamicsEngine()

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
                agent, timeline_str, warning
            )
            agent.action_history.append(action)
            self.timeline.add_post(agent.agent_id, turn, action)
            self.social_graph.update_network(action, agent.agent_id, self.timeline)
            turn_actions.append({
                "turn": turn,
                "agent_id": agent.agent_id,
                "persona": agent.persona_name,
                "action": action.action_type.name if action.action_type else "NONE",
                "content": action.content,
                "rationale": action.rationale,
                "proposal_id": action.proposal_id,
                "opinion": round(agent.opinion, 4),
                "risk_perception": round(agent.risk_perception, 4),
            })

        return {
            "turn": turn,
            "actions": turn_actions,
            "social_signals": social_signals,
        }

    async def run(self, turns: int) -> list[dict]:
        turn_data_list = []
        for _ in range(turns):
            data = await self.run_turn()
            turn_data_list.append(data)
        return turn_data_list
