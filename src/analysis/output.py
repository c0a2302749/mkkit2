import csv
import json
import os
from datetime import datetime


class OutputManager:
    def __init__(self, base_dir: str = "logs"):
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        self._run_dir = os.path.join(base_dir, timestamp)
        os.makedirs(self._run_dir, exist_ok=True)

        self._actions: list[dict] = []
        self._trajectories: list[dict] = []
        self._statistics: list[dict] = []

    @property
    def run_dir(self) -> str:
        return self._run_dir

    def record_turn(
        self,
        turn: int,
        agents: list,
        turn_actions: list[dict],
        social_signals: dict[int, tuple[float, float]],
        config: dict,
    ) -> None:
        for action in turn_actions:
            self._actions.append(action)

        for agent in agents:
            s_o, s_r = social_signals.get(agent.agent_id, (0.0, 0.0))
            self._trajectories.append({
                "turn": turn,
                "agent_id": agent.agent_id,
                "persona": agent.persona_name,
                "opinion": round(agent.opinion, 4),
                "risk_perception": round(agent.risk_perception, 4),
                "social_opinion": round(s_o, 4),
                "social_risk": round(s_r, 4),
            })

        opinions = [a.opinion for a in agents]
        risks = [a.risk_perception for a in agents]
        n = len(agents)
        self._statistics.append({
            "turn": turn,
            "n_agents": n,
            "avg_opinion": round(sum(opinions) / n, 4) if n else 0.0,
            "std_opinion": round(
                (sum((o - sum(opinions) / n) ** 2 for o in opinions) / n) ** 0.5, 4
            ) if n else 0.0,
            "avg_risk": round(sum(risks) / n, 4) if n else 0.0,
        })

    def save_all(self, config: dict) -> str:
        actions_path = os.path.join(self._run_dir, "actions.json")
        with open(actions_path, "w", encoding="utf-8") as f:
            json.dump(self._actions, f, ensure_ascii=False, indent=2)

        traj_path = os.path.join(self._run_dir, "opinion_trajectory.csv")
        if self._trajectories:
            with open(traj_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self._trajectories[0].keys())
                w.writeheader()
                w.writerows(self._trajectories)

        stats_path = os.path.join(self._run_dir, "statistics.csv")
        if self._statistics:
            with open(stats_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self._statistics[0].keys())
                w.writeheader()
                w.writerows(self._statistics)

        summary_path = os.path.join(self._run_dir, "summary.json")
        summary = {
            "config": config,
            "timestamp": datetime.now().isoformat(),
            "n_actions": len(self._actions),
            "n_turns": max(s["turn"] for s in self._statistics) if self._statistics else 0,
            "final_statistics": self._statistics[-1] if self._statistics else {},
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return self._run_dir
