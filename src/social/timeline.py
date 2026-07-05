from src.core.agent import Agent, Action
from src.core.action import ActionType


class TimelineManager:
    def __init__(self):
        self._posts: list[tuple[int, int, Action]] = []
        self._next_proposal_id: int = 1
        self._proposal_author: dict[int, int] = {}

    def add_post(self, agent_id: int, turn: int, action: Action) -> None:
        if action.action_type not in (
            ActionType.PROPOSE, ActionType.COMMENT,
            ActionType.SUPPORT, ActionType.OPPOSE, ActionType.VOTE,
        ):
            return
        if action.action_type == ActionType.PROPOSE:
            action.proposal_id = self._next_proposal_id
            self._proposal_author[self._next_proposal_id] = agent_id
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
            if action.action_type == ActionType.COMMENT
            and aid in followees and t >= turn - window
        ]

    def get_proposal_opinions(self, followees: list[int], turn: int, window: int = 3) -> dict[int, list[tuple[int, Action]]]:
        opinions: dict[int, list[tuple[int, Action]]] = {}
        for aid, t, action in self._posts:
            if action.action_type not in (ActionType.SUPPORT, ActionType.OPPOSE):
                continue
            if aid not in followees or t < turn - window:
                continue
            pid = action.proposal_id
            if pid is None:
                continue
            opinions.setdefault(pid, []).append((aid, action))
        return opinions

    def get_all_proposals(self, turn: int, window: int = 3) -> list[tuple[int, int, Action]]:
        return [
            (aid, pid, action)
            for aid, t, action in self._posts
            if action.action_type == ActionType.PROPOSE
            and action.proposal_id is not None
            and t >= turn - window
            for pid in [action.proposal_id]
        ]

    def get_timeline(self, agent: Agent, followees: list[int], turn: int) -> str:
        proposals = self.get_all_proposals(turn)
        opinions = self.get_proposal_opinions(followees, turn)
        discussions = self.get_recent_discussion(followees, turn)

        lines = []
        if proposals:
            lines.append("[Proposals]")
            for aid, pid, action in proposals:
                lines.append(f"  #{pid} (by Agent {aid}): \"{action.content}\"")
        if opinions:
            lines.append("[What people you follow think]")
            for pid in sorted(opinions):
                for aid, action in opinions[pid]:
                    lines.append(f"  #{pid} - Agent {aid} {action.action_type.name}: \"{action.content}\"")
        if discussions:
            lines.append("[Discussion]")
            for action in discussions:
                target = f" on #{action.proposal_id}" if action.proposal_id else ""
                lines.append(f"  [COMMENT{target}]: \"{action.content}\"")
        return "\n".join(lines) if lines else "No recent posts."

    def get_proposal_author(self, proposal_id: int) -> int | None:
        return self._proposal_author.get(proposal_id)

    def clear(self) -> None:
        self._posts.clear()
        self._next_proposal_id = 1
        self._proposal_author.clear()
