import asyncio
import argparse
import random
from src.core.state import SimulationState
from src.core.agent import Action
from src.core.action import ActionType
from src.config.personas import create_agent, PERSONAS
from src.social.graph import SocialGraph
from src.social.timeline import TimelineManager
from src.agent.engine import OpinionDynamicsEngine
from src.agent.manager import AgentManager
from src.agent.llm import LLMProvider
from src.governance.stub import GovernanceStub
from src.engine.simulation import SimulationEngine
from src.analysis.output import OutputManager


PERSONA_NAMES = list(PERSONAS.keys())


class StubProvider(LLMProvider):
    async def invoke(self, prompt: str, system_prompt: str = "") -> str:
        return '{"action": "DO_NOTHING", "content": "", "rationale": "stub"}'


def setup_null_engine(n_agents: int, seed: int) -> SimulationEngine:
    random.seed(seed)
    state = SimulationState()
    persona_cycle = [PERSONA_NAMES[i % len(PERSONA_NAMES)] for i in range(n_agents)]
    for i in range(n_agents):
        agent = create_agent(i, persona_cycle[i])
        state.agents.append(agent)

    graph = SocialGraph()
    for agent in state.agents:
        graph.add_agent(agent)
    graph.generate_isolated(n_agents)

    timeline = TimelineManager()
    ode = OpinionDynamicsEngine()

    class NullManager:
        def __init__(self):
            self._ode = ode

        async def decide_action(self, agent, timeline_str, warning="", current_turn=0, total_turns=0):
            s_o, s_r = self._ode.compute_social_signals(agent, graph)
            new_r = self._ode.update_risk_perception(
                R=agent.risk_perception, W=0.0, S_R=0.0,
                lambda_i=agent.lambda_, w_sys=agent.w_sys,
                w_sns=0.0, b=agent.bias,
            )
            new_o = self._ode.compute_opinion_update(
                O=agent.opinion, S_O=0.0, R=new_r,
                alpha=agent.alpha, beta=agent.beta,
                gamma=agent.gamma, O_0=agent.initial_opinion,
            )
            agent.risk_perception = new_r
            agent.opinion = new_o
            return Action(action_type=ActionType.DO_NOTHING)

    governance = GovernanceStub()
    engine = SimulationEngine(state, graph, timeline, NullManager(), governance, ode)
    engine.ode = ode
    return engine


def make_agent_manager(llm_backend: str) -> AgentManager:
    if llm_backend == "stub":
        return AgentManager(StubProvider())
    try:
        from src.agent.azure_provider import AzureProvider
        return AgentManager(AzureProvider())
    except KeyError as e:
        raise RuntimeError(
            f"Azure OpenAI not configured: {e}. "
            f"Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
            f"and AZURE_OPENAI_DEPLOYMENT env vars, or use --llm stub."
        )


def setup_engine(n_agents: int, topology: str, seed: int, is_null: bool, llm_backend: str) -> SimulationEngine:
    if is_null:
        return setup_null_engine(n_agents, seed)

    random.seed(seed)
    state = SimulationState()
    persona_cycle = [PERSONA_NAMES[i % len(PERSONA_NAMES)] for i in range(n_agents)]
    for i in range(n_agents):
        agent = create_agent(i, persona_cycle[i])
        state.agents.append(agent)

    graph = SocialGraph()
    for agent in state.agents:
        graph.add_agent(agent)
    graph.generate_by_topology(topology, n_agents, seed)

    timeline = TimelineManager()
    ode = OpinionDynamicsEngine()
    governance = GovernanceStub()
    agent_manager = make_agent_manager(llm_backend)

    return SimulationEngine(state, graph, timeline, agent_manager, governance, ode)


async def main():
    parser = argparse.ArgumentParser(description="mkkit2 simulation")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--topology", type=str, default="random",
                        choices=["random", "small-world", "scale-free", "community", "complete"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--null", action="store_true", help="Run null model")
    parser.add_argument("--llm", type=str, default="azure", choices=["azure", "stub"],
                        help="LLM backend (azure or stub)")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Output directory for logs (default: logs/)")
    args = parser.parse_args()

    config = {
        "turns": args.turns,
        "agents": args.agents,
        "topology": args.topology,
        "seed": args.seed,
        "null": args.null,
        "llm": args.llm,
    }

    engine = setup_engine(args.agents, args.topology, args.seed, args.null)
    engine.state.total_turns = args.turns
    output = OutputManager(base_dir=args.log_dir)

    turn_data_list = []
    for turn_num in range(1, args.turns + 1):
        data = await engine.run_turn()
        turn_data_list.append(data)
        output.record_turn(
            turn=data["turn"],
            agents=engine.state.agents,
            turn_actions=data["actions"],
            social_signals=data["social_signals"],
            config=config,
        )
        resolved = sum(1 for p in engine.state.proposals if p.status.name != "OPEN")
        n_actions = len(data["actions"])
        print(f"\rTurn {turn_num}/{args.turns} | actions={n_actions} proposals={len(engine.state.proposals)} resolved={resolved}", end="", flush=True)
    print()

    run_dir = output.save_all(config, engine.state.proposals)

    print(f"Results saved to: {run_dir}")
    for agent in engine.state.agents:
        print(f"Agent {agent.agent_id} ({agent.persona_name}): "
              f"O={agent.opinion:.3f}, R={agent.risk_perception:.3f}, "
              f"actions={len(agent.action_history)}")


if __name__ == "__main__":
    asyncio.run(main())
