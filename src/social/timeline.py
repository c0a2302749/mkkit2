from src.core.agent import Agent, Action
from src.core.action import ActionType


class TimelineManager:
    def __init__(self):
        self._posts: list[tuple[int, int, Action]] = []
        self._next_proposal_id: int = 1

    def add_post(self, agent_id: int, turn: int, action: Action) -> None:
        if action.action_type not in (
            ActionType.PROPOSE, ActionType.COMMENT,
            ActionType.SUPPORT, ActionType.OPPOSE, ActionType.VOTE,
        ):
            return
        if action.action_type == ActionType.PROPOSE:
            action.proposal_id = self._next_proposal_id
            self._next_proposal_id += 1
        self._posts.append((agent_id, turn, action))

    def get_proposals_with_ids(self, followees: list[int], turn: int, window: int = 3) -> list[tuple[int, int, Action]]:
        return [
            (aid, pid, action)
            for aid, t, action in self._posts
            if action.action_type == ActionType.PROPOSE
            and action.proposal_id is not None
            and aid in followees and t >= turn - window
            for pid in [action.proposal_id]
        ]

    def get_recent_discussion(self, followees: list[int], turn: int, window: int = 3) -> list[Action]:
        return [
            action
            for aid, t, action in self._posts
            if action.action_type in (ActionType.COMMENT, ActionType.SUPPORT, ActionType.OPPOSE)
            and aid in followees and t >= turn - window
        ]

    def get_timeline(self, agent: Agent, followees: list[int], turn: int) -> str:
        proposals = self.get_proposals_with_ids(followees, turn)
        discussions = self.get_recent_discussion(followees, turn)

        lines = []
        if proposals:
            lines.append("[Recent proposals]")
            for aid, pid, action in proposals:
                lines.append(f"  #{pid} (by Agent {aid}): \"{action.content}\"")
        if discussions:
            lines.append("[Recent discussion]")
            for action in discussions:
                target = f" on #{action.proposal_id}" if action.proposal_id else ""
                lines.append(f"  [{action.action_type.name}{target}]: \"{action.content}\"")
        return "\n".join(lines) if lines else "No recent posts."

    def clear(self) -> None:
        self._posts.clear()
        self._next_proposal_id = 1
