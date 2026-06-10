import pytest
from src.core.agent import Agent, Action
from src.core.action import ActionType
from src.agent.manager import AgentManager
from src.agent.llm import LLMProvider


class StubProvider(LLMProvider):
    def __init__(self, response: str = '{"action": "DO_NOTHING", "content": "", "rationale": "stub"}'):
        self._response = response

    async def invoke(self, prompt: str) -> str:
        return self._response


def _make_agent() -> Agent:
    return Agent(
        agent_id=0, persona_name="rational",
        alpha=0.2, beta=0.4, gamma=0.6,
        lambda_=0.3, bias=-0.1, w_sys=0.5, w_sns=0.5,
    )


class TestAgentManagerParse:
    def test_parse_valid_json(self):
        response = '{"action": "PROPOSE", "content": "test", "rationale": "reason"}'
        action = AgentManager._parse_response(response)
        assert action.action_type == ActionType.PROPOSE
        assert action.content == "test"
        assert action.rationale == "reason"

    def test_parse_invalid_json_returns_do_nothing(self):
        action = AgentManager._parse_response("not json")
        assert action.action_type == ActionType.DO_NOTHING

    def test_parse_missing_action_returns_do_nothing(self):
        response = '{"content": "test"}'
        action = AgentManager._parse_response(response)
        assert action.action_type == ActionType.DO_NOTHING

    def test_parse_all_action_types(self):
        for at in ActionType:
            response = f'{{"action": "{at.name}", "content": "", "rationale": ""}}'
            action = AgentManager._parse_response(response)
            assert action.action_type == at

    def test_parse_proposal_id(self):
        response = '{"action": "SUPPORT", "content": "agree", "proposal_id": 3, "rationale": "good"}'
        action = AgentManager._parse_response(response)
        assert action.action_type == ActionType.SUPPORT
        assert action.proposal_id == 3

    def test_parse_missing_proposal_id(self):
        response = '{"action": "PROPOSE", "content": "new idea", "rationale": "needed"}'
        action = AgentManager._parse_response(response)
        assert action.proposal_id is None


class TestAgentManagerBuildPrompt:
    def test_prompt_contains_agent_info(self):
        manager = AgentManager(StubProvider())
        agent = _make_agent()
        prompt = manager._build_prompt(agent, "timeline", "warning")
        assert str(agent.agent_id) in prompt
        assert agent.persona_name in prompt
        assert "alpha=0.2" in prompt
        assert "beta=0.4" in prompt
        assert "gamma=0.6" in prompt
        assert "timeline" in prompt
        assert ActionType.PROPOSE.name in prompt

    def test_prompt_contains_proposal_id_instruction(self):
        manager = AgentManager(StubProvider())
        agent = _make_agent()
        prompt = manager._build_prompt(agent, "timeline", "warning")
        assert "proposal_id" in prompt

    def test_prompt_contains_propose_and_other_formats(self):
        manager = AgentManager(StubProvider())
        agent = _make_agent()
        prompt = manager._build_prompt(agent, "timeline", "warning")
        assert '"action": "PROPOSE"' in prompt
        assert '"proposal_id":' in prompt


@pytest.mark.asyncio
async def test_decide_action_with_stub():
    manager = AgentManager(StubProvider())
    agent = _make_agent()
    action = await manager.decide_action(agent, "timeline")
    assert action.action_type == ActionType.DO_NOTHING
