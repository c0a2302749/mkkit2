import json
from typing import Optional
from src.core.agent import Agent, Action
from src.core.action import ActionType
from src.agent.llm import LLMProvider


ACTION_DESCRIPTIONS = {
    ActionType.PROPOSE: "Propose a new idea or policy",
    ActionType.COMMENT: "Comment on a proposal (specify proposal_id)",
    ActionType.SUPPORT: "Express support for a proposal (specify proposal_id)",
    ActionType.OPPOSE: "Express opposition to a proposal (specify proposal_id)",
    ActionType.VOTE: "Vote on a proposal (specify proposal_id)",
    ActionType.DO_NOTHING: "Do nothing this turn",
}


class AgentManager:
    def __init__(self, llm_provider: LLMProvider):
        self._llm = llm_provider

    def _build_prompt(
        self,
        agent: Agent,
        timeline: str,
        warning: str,
    ) -> str:
        action_options = "\n".join(
            f"- {a.name}: {ACTION_DESCRIPTIONS[a]}"
            for a in ActionType
        )
        return f"""You are Agent {agent.agent_id} ({agent.persona_name}).
Your opinion: {agent.opinion:.2f}
Your risk perception: {agent.risk_perception:.2f}
Personality (alpha={agent.alpha}, beta={agent.beta}, gamma={agent.gamma}):
- alpha: tendency to follow social opinion
- beta: sensitivity to risk
- gamma: resistance to changing initial opinion

Warning: {warning}

Your timeline:
{timeline}

Choose an action from:
{action_options}

Respond in JSON format:
For PROPOSE: {{"action": "PROPOSE", "content": "your proposal", "rationale": "reasoning"}}
For other actions: {{"action": "ACTION_NAME", "content": "message", "proposal_id": TARGET_ID, "rationale": "reasoning"}}"""

    async def decide_action(
        self,
        agent: Agent,
        timeline: str,
        warning: str = "",
    ) -> Action:
        prompt = self._build_prompt(agent, timeline, warning)
        response = await self._llm.invoke(prompt)
        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: str) -> Action:
        try:
            data = json.loads(response)
            action_type = ActionType[data["action"].upper()]
            return Action(
                action_type=action_type,
                content=data.get("content", ""),
                rationale=data.get("rationale", ""),
                proposal_id=data.get("proposal_id"),
            )
        except (json.JSONDecodeError, KeyError):
            return Action(ActionType.DO_NOTHING, content="", rationale="Parse error")
