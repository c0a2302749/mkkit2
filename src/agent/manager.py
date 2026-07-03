import json
from src.core.agent import Agent, Action
from src.core.action import ActionType
from src.agent.llm import LLMProvider
from src.config.personas import SYSTEM_PROMPTS


ACTION_DESCRIPTIONS = {
    ActionType.PROPOSE: "Propose a new idea or policy for the group to decide on",
    ActionType.COMMENT: "Comment on a proposal (specify proposal_id)",
    ActionType.SUPPORT: "Express informal support for a proposal (discussion phase)",
    ActionType.OPPOSE: "Express informal opposition to a proposal (discussion phase)",
    ActionType.VOTE: "Cast a formal yes/no vote on a proposal (decision phase, content: yes/no)",
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
        current_turn: int = 0,
        total_turns: int = 0,
        proposals: list | None = None,
    ) -> Action:
        system_prompt = self._get_system_prompt(agent)
        user_prompt = self._build_user_prompt(timeline, warning)
        response = await self._llm.invoke(user_prompt, system_prompt=system_prompt)
        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: str) -> Action:
        text = re.sub(r"```(?:json)?\s*", "", response).strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                data = data[0] if data else {}
        except (json.JSONDecodeError, ValueError):
            try:
                m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
                if not m:
                    return Action(ActionType.DO_NOTHING, "", "No JSON found")
                data = json.loads(m.group())
                if isinstance(data, list):
                    data = data[0] if data else {}
            except (json.JSONDecodeError, KeyError, IndexError, ValueError):
                return Action(ActionType.DO_NOTHING, "", "Parse error")
        try:
            action_type = ActionType[data["action"].upper()]
            return Action(
                action_type=action_type,
                content=data.get("content", ""),
                rationale=data.get("rationale", ""),
                proposal_id=data.get("proposal_id"),
                votes=data.get("votes"),
            )
        except KeyError:
            return Action(ActionType.DO_NOTHING, "", "Parse error")
