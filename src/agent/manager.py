import json
import re
from typing import Optional
from src.core.agent import Agent, Action
from src.core.action import ActionType
from src.agent.llm import LLMProvider


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

    def _build_prompt(
        self,
        agent: Agent,
        timeline: str,
        warning: str = "",
        current_turn: int = 0,
        total_turns: int = 0,
        proposals: list | None = None,
    ) -> str:
        action_options = "\n".join(
            f"- {a.name}: {ACTION_DESCRIPTIONS[a]}"
            for a in ActionType
        )

        phase_text = ""
        if total_turns > 0 and current_turn == 1:
            phase_text = (
                "This is the Ideation Phase (Turn 1). "
                "DO NOT simply support or oppose existing ideas. "
                "Based on your unique persona and perspective, "
                "you MUST propose an original idea or approach."
            )
        elif total_turns > 0 and current_turn < total_turns:
            phase_text = f"(Turn {current_turn}/{total_turns} — discussion phase. Use SUPPORT/OPPOSE to debate proposals.)"
        elif total_turns > 0 and current_turn >= total_turns:
            phase_text = (
                f"(Turn {current_turn}/{total_turns} — FINAL decision phase. "
                "You MUST vote on ALL open proposals below.)"
            )

        # Build proposal list for final turn
        proposals_block = ""
        if total_turns > 0 and current_turn >= total_turns and proposals:
            open_proposals = [p for p in proposals if p.status.name == "OPEN"]
            if open_proposals:
                lines = ["=== PROPOSALS TO VOTE ON ==="]
                for p in open_proposals:
                    lines.append(
                        f"  #{p.proposal_id} by Agent {p.agent_id}: "
                        f"\"{p.content[:80]}\""
                    )
                proposals_block = "\n".join(lines)

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

{proposals_block}

Choose an action from:
{action_options}

{phase_text}

Respond in JSON format:
For PROPOSE: {{"action": "PROPOSE", "content": "your proposal", "rationale": "reasoning"}}
For VOTE (single): {{"action": "VOTE", "content": "yes/no", "proposal_id": TARGET_ID, "rationale": "reasoning"}}
For VOTE (multiple): {{"action": "VOTE", "votes": [{{"proposal_id": 1, "vote": "yes/no"}}, ...], "rationale": "reasoning"}}
For COMMENT/SUPPORT/OPPOSE: {{"action": "ACTION_NAME", "content": "message", "proposal_id": TARGET_ID, "rationale": "reasoning"}}"""

    async def decide_action(
        self,
        agent: Agent,
        timeline: str,
        warning: str = "",
        current_turn: int = 0,
        total_turns: int = 0,
        proposals: list | None = None,
    ) -> Action:
        prompt = self._build_prompt(agent, timeline, warning, current_turn, total_turns, proposals)
        response = await self._llm.invoke(prompt)
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
