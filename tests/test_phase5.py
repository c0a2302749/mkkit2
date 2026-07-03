import pytest
from src.core.state import SimulationState
from src.config.personas import create_agent
from src.social.graph import SocialGraph
from src.social.timeline import TimelineManager
from src.agent.engine import OpinionDynamicsEngine
from src.agent.manager import AgentManager
from src.agent.llm import LLMProvider
from src.governance.stub import GovernanceStub
from src.engine.simulation import SimulationEngine


class StubProvider(LLMProvider):
    def __init__(self):
        self.call_count = 0

    async def invoke(self, prompt: str, system_prompt: str = "") -> str:
        self.call_count += 1
        return '{"action": "DO_NOTHING", "content": "", "rationale": "stub"}'


class TestGovernanceStub:
    def test_always_zero(self):
        stub = GovernanceStub()
        assert stub.compute_risk_score(None, None, None, None) == 0.0
        assert stub.build_system_warning(0.0) == ""


class TestSimulationEngine:
    @pytest.mark.asyncio
    async def test_run_turn_updates_state(self):
        state = SimulationState()
        agent = create_agent(0, "rational")
        agent2 = create_agent(1, "conformist")
        state.agents = [agent, agent2]

        graph = SocialGraph()
        graph.add_agent(agent)
        graph.add_agent(agent2)
        graph.add_follow(0, 1)
        graph.add_follow(1, 0)

        tl = TimelineManager()
        ode = OpinionDynamicsEngine()
        stub = GovernanceStub()
        llm = StubProvider()
        mgr = AgentManager(llm)
        engine = SimulationEngine(state, graph, tl, mgr, stub, ode)

        initial_o = agent.opinion
        await engine.run_turn()

        assert engine.state.current_turn == 1
        assert agent.opinion != initial_o
        assert len(agent.action_history) >= 0

    @pytest.mark.asyncio
    async def test_multi_turn_runs_without_error(self):
        state = SimulationState()
        agents = [create_agent(i, "rational") for i in range(4)]
        state.agents = agents

        graph = SocialGraph()
        for a in agents:
            graph.add_agent(a)
        graph.generate_complete(4)

        tl = TimelineManager()
        ode = OpinionDynamicsEngine()
        stub = GovernanceStub()
        llm = StubProvider()
        mgr = AgentManager(llm)
        engine = SimulationEngine(state, graph, tl, mgr, stub, ode)

        await engine.run(turns=3)
        assert engine.state.current_turn == 3
