import pytest
from src.core.state import SimulationState
from src.config.personas import create_agent
from src.social.graph import SocialGraph
from src.social.timeline import TimelineManager
from src.agent.engine import OpinionDynamicsEngine
from src.agent.manager import AgentManager
from src.agent.llm import LLMProvider
from src.governance.stub import GovernanceStub
from src.core.action import ActionType
from src.core.proposal import Proposal, ProposalStatus
from src.core.agent import Action
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

    @pytest.mark.asyncio
    async def test_vote_proposal_creation_on_propose(self):
        state = SimulationState()
        agents = [create_agent(i, "rational") for i in range(3)]
        state.agents = agents

        graph = SocialGraph()
        for a in agents:
            graph.add_agent(a)
        graph.generate_complete(3)

        tl = TimelineManager()
        ode = OpinionDynamicsEngine()
        stub = GovernanceStub()

        class ProposeProvider(LLMProvider):
            def __init__(self):
                self.call_count = 0
            async def invoke(self, prompt: str) -> str:
                self.call_count += 1
                return '{"action": "PROPOSE", "content": "test proposal", "rationale": "stub"}'

        mgr = AgentManager(ProposeProvider())
        engine = SimulationEngine(state, graph, tl, mgr, stub, ode)
        await engine.run_turn()

        assert len(engine.state.proposals) == 3
        for p in engine.state.proposals:
            assert p.content == "test proposal"
            assert p.status.name == "OPEN"

    @pytest.mark.asyncio
    async def test_vote_tally_resolves_proposal(self):
        state = SimulationState()
        agents = [create_agent(i, "rational") for i in range(3)]
        state.agents = agents

        graph = SocialGraph()
        for a in agents:
            graph.add_agent(a)
        graph.generate_complete(3)

        tl = TimelineManager()
        ode = OpinionDynamicsEngine()
        stub = GovernanceStub()
        llm = StubProvider()  # returns DO_NOTHING

        engine = SimulationEngine(state, graph, tl, AgentManager(llm), stub, ode)
        engine.state.total_turns = 1  # resolve on turn 1

        prop = Proposal(proposal_id=1, agent_id=0, content="test", turn_created=0)
        engine.state.proposals.append(prop)

        agents[0].action_history.append(
            Action(ActionType.VOTE, content="yes", proposal_id=1)
        )
        agents[1].action_history.append(
            Action(ActionType.VOTE, content="yes", proposal_id=1)
        )
        agents[2].action_history.append(
            Action(ActionType.VOTE, content="no", proposal_id=1)
        )

        await engine.run_turn()

        assert prop.status == ProposalStatus.PASSED
        assert prop.turn_resolved == 1

    @pytest.mark.asyncio
    async def test_multi_vote_resolves_all_proposals(self):
        state = SimulationState()
        agents = [create_agent(i, "rational") for i in range(3)]
        state.agents = agents

        graph = SocialGraph()
        for a in agents:
            graph.add_agent(a)
        graph.generate_complete(3)

        tl = TimelineManager()
        ode = OpinionDynamicsEngine()
        stub = GovernanceStub()
        llm = StubProvider()

        engine = SimulationEngine(state, graph, tl, AgentManager(llm), stub, ode)
        engine.state.total_turns = 1

        prop1 = Proposal(proposal_id=1, agent_id=0, content="prop1", turn_created=0)
        prop2 = Proposal(proposal_id=2, agent_id=1, content="prop2", turn_created=0)
        engine.state.proposals.extend([prop1, prop2])

        agents[0].action_history.append(
            Action(ActionType.VOTE, votes=[{"proposal_id": 1, "vote": "yes"}, {"proposal_id": 2, "vote": "no"}])
        )
        agents[1].action_history.append(
            Action(ActionType.VOTE, votes=[{"proposal_id": 1, "vote": "yes"}, {"proposal_id": 2, "vote": "yes"}])
        )
        agents[2].action_history.append(
            Action(ActionType.VOTE, votes=[{"proposal_id": 1, "vote": "no"}, {"proposal_id": 2, "vote": "yes"}])
        )

        await engine.run_turn()

        assert prop1.status == ProposalStatus.PASSED, f"prop1: {prop1.status} (for={prop1.votes_for}, against={prop1.votes_against})"
        assert prop1.turn_resolved == 1
        assert prop2.status == ProposalStatus.PASSED, f"prop2: {prop2.status} (for={prop2.votes_for}, against={prop2.votes_against})"
        assert prop2.turn_resolved == 1
        assert sorted(prop1.votes_for) == [0, 1]
        assert sorted(prop1.votes_against) == [2]
