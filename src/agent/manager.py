import json
from src.core.agent import Agent, Action
from src.core.action import ActionType
from src.agent.llm import LLMProvider
from src.config.personas import SYSTEM_PROMPTS


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

    def _get_system_prompt(self, agent: Agent) -> str:
        base = SYSTEM_PROMPTS.get(agent.persona_name, "")
        return (
            f"{base}\n\n"
            f"Your opinion: {agent.opinion:.2f}\n"
            f"Your risk perception: {agent.risk_perception:.2f}\n"
            f"Personality: alpha={agent.alpha} (conformity), "
            f"beta={agent.beta} (risk sensitivity), "
            f"gamma={agent.gamma} (stance persistence)."
        )

    def _build_user_prompt(self, timeline: str, warning: str) -> str:
        action_options = "\n".join(
            f"- {a.name}: {ACTION_DESCRIPTIONS[a]}"
            for a in ActionType
        )
        parts = []
        if warning:
            parts.append(f"System warning: {warning}")
        parts.append(f"Your timeline:\n{timeline}")
        parts.append("Choose an action from:\n" + action_options)
        parts.append(
            "Respond in JSON format:\n"
            'For PROPOSE: {"action": "PROPOSE", "content": "your proposal", "rationale": "reasoning"}\n'
            'For other actions: {"action": "ACTION_NAME", "content": "message", '
            '"proposal_id": TARGET_ID, "rationale": "reasoning"}'
        )
        return "\n\n".join(parts)

    async def decide_action(
        self,
        agent: Agent,
        timeline: str,
        warning: str = "",
    ) -> Action:
        system_prompt = self._get_system_prompt(agent)
        user_prompt = self._build_user_prompt(timeline, warning)
        response = await self._llm.invoke(user_prompt, system_prompt=system_prompt)
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
