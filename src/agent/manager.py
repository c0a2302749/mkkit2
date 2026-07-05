import json
import re
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

    def _phase_label(self, current_turn: int, total_turns: int) -> str:
        if current_turn == 1:
            return "Ideation Phase"
        if total_turns > 0 and current_turn >= total_turns:
            return "Voting Phase"
        return "Discussion Phase"

    def _get_system_prompt(self, agent: Agent, current_turn: int = 0, total_turns: int = 0) -> str:
        base = SYSTEM_PROMPTS.get(agent.persona_name, "")
        phase = self._phase_label(current_turn, total_turns)
        phase_instruction = {
            "Ideation Phase": (
                "This is the Ideation Phase (Turn 1). "
                "You MUST submit one original proposal. "
                "Do NOT support or oppose existing ideas."
            ),
            "Discussion Phase": (
                "This is the Discussion Phase. "
                "You may comment on, support, or oppose proposals, "
                "or submit new proposals."
            ),
            "Voting Phase": (
                "This is the Voting Phase (final turn). "
                "You must cast your vote on open proposals. "
                "Vote on each proposal based on your opinion."
            ),
        }.get(phase, "")
        return (
            f"{base}\n\n"
            f"{phase_instruction}\n\n"
            f"Your opinion: {agent.opinion:.2f}\n"
            f"Your risk perception: {agent.risk_perception:.2f}\n"
            f"Personality: alpha={agent.alpha} (conformity), "
            f"beta={agent.beta} (risk sensitivity), "
            f"gamma={agent.gamma} (stance persistence)."
        )

    def _build_user_prompt(self, timeline: str, warning: str, current_turn: int = 0, total_turns: int = 0, proposals: list | None = None, scenario_context: str = "") -> str:
        phase = self._phase_label(current_turn, total_turns)
        open_list = ""

        if phase == "Ideation Phase":
            action_options = "- PROPOSE: Propose a new idea or policy for the group to decide on"
            json_help = (
                'Respond in JSON format:\n'
                '{"action": "PROPOSE", "content": "your proposal", "rationale": "reasoning"}'
            )
        elif phase == "Voting Phase":
            action_options = "- VOTE: Cast a formal yes/no vote on all open proposals (use votes list)"
            open_list = ""
            if proposals:
                open_props = [p for p in proposals if p.status.name == "OPEN"]
                if open_props:
                    lines = ["[Open proposals to vote on]"]
                    for p in open_props:
                        short = p.content[:80] + "..." if len(p.content) > 80 else p.content
                        lines.append(f"  #{p.proposal_id} (by Agent {p.agent_id}): \"{short}\"")
                    open_list = "\n".join(lines)
            json_help = (
                'Respond in JSON format with a "votes" array containing ALL open proposals:\n'
                '{"action": "VOTE", "votes": [{"proposal_id": 1, "vote": "yes"}, '
                '{"proposal_id": 2, "vote": "no"}], "rationale": "reasoning"}\n'
                'You MUST include EVERY open proposal_id in the votes array. Do not omit any.'
            )
        else:
            action_options = "\n".join(
                f"- {a.name}: {ACTION_DESCRIPTIONS[a]}"
                for a in ActionType if a != ActionType.VOTE
            )
            json_help = (
                "Respond in JSON format:\n"
                'For PROPOSE: {"action": "PROPOSE", "content": "your proposal", "rationale": "reasoning"}\n'
                'For other actions: {"action": "ACTION_NAME", "content": "message", '
                '"proposal_id": TARGET_ID, "rationale": "reasoning"}'
            )

        parts = []
        if scenario_context:
            parts.append(f"Scenario background: {scenario_context}")
        if warning:
            parts.append(f"System warning: {warning}")
        parts.append(f"Your timeline:\n{timeline}")
        if open_list:
            parts.append(open_list)
        parts.append("Choose an action from:\n" + action_options)
        parts.append(json_help)
        return "\n\n".join(parts)

    async def decide_action(
        self,
        agent: Agent,
        timeline: str,
        warning: str = "",
        current_turn: int = 0,
        total_turns: int = 0,
        proposals: list | None = None,
        scenario_context: str = "The city council is considering whether to approve \
the construction of a large-scale data center. Key concerns include \
environmental impact, energy consumption, local economic benefits, \
and noise pollution.",
    ) -> Action:
        system_prompt = self._get_system_prompt(agent, current_turn, total_turns)
        user_prompt = self._build_user_prompt(timeline, warning, current_turn, total_turns, proposals, scenario_context)
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
