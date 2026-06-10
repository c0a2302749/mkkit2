import pytest
from src.core.action import ActionType
from src.core.agent import Agent, Action
from src.config.personas import PERSONAS, create_agent


class TestActionType:
    def test_enum_values(self):
        assert len(ActionType) == 6
        assert ActionType.PROPOSE.value is not None
        assert ActionType.DO_NOTHING.value is not None


class TestAgent:
    def test_create_agent(self):
        agent = create_agent(0, "rational", initial_opinion=0.5)
        assert agent.agent_id == 0
        assert agent.persona_name == "rational"
        assert agent.alpha == 0.2
        assert agent.beta == 0.4
        assert agent.gamma == 0.6
        assert agent.opinion == 0.5
        assert agent.initial_opinion == 0.5
        assert agent.risk_perception == 0.5
        assert agent.action_history == []

    def test_all_personas_creatable(self):
        for persona_name in PERSONAS:
            agent = create_agent(0, persona_name)
            assert agent.persona_name == persona_name

    def test_action_history(self):
        agent = create_agent(0, "rational")
        action = Action(action_type=ActionType.PROPOSE, content="test", rationale="reason")
        agent.action_history.append(action)
        assert len(agent.action_history) == 1
        assert agent.action_history[0].action_type == ActionType.PROPOSE
        assert agent.action_history[0].content == "test"

    def test_persona_params_match_table(self):
        assert PERSONAS["rational"]["alpha"] == 0.2
        assert PERSONAS["rational"]["beta"] == 0.4
        assert PERSONAS["rational"]["gamma"] == 0.6
        assert PERSONAS["conformist"]["alpha"] == 0.7
        assert PERSONAS["conformist"]["beta"] == 0.4
        assert PERSONAS["conformist"]["gamma"] == 0.1
        assert PERSONAS["information_seeker"]["alpha"] == 0.4
        assert PERSONAS["information_seeker"]["beta"] == 0.35
        assert PERSONAS["information_seeker"]["gamma"] == 0.3
        assert PERSONAS["risk_overestimator"]["alpha"] == 0.4
        assert PERSONAS["risk_overestimator"]["beta"] == 0.7
        assert PERSONAS["risk_overestimator"]["gamma"] == 0.4
        assert PERSONAS["skeptic"]["alpha"] == 0.15
        assert PERSONAS["skeptic"]["beta"] == 0.15
        assert PERSONAS["skeptic"]["gamma"] == 0.6
        assert PERSONAS["agitator"]["alpha"] == 0.2
        assert PERSONAS["agitator"]["beta"] == 0.1
        assert PERSONAS["agitator"]["gamma"] == 0.6
