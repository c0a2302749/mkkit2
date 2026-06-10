import os
import json
import tempfile
import pytest
from src.analysis.output import OutputManager
from src.core.agent import Agent
from src.config.personas import create_agent


@pytest.fixture
def tmp_log_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestOutputManager:
    def test_creates_run_directory(self, tmp_log_dir):
        om = OutputManager(base_dir=tmp_log_dir)
        assert os.path.isdir(om.run_dir)
        assert om.run_dir.startswith(tmp_log_dir)

    def test_record_turn_and_save_actions_json(self, tmp_log_dir):
        om = OutputManager(base_dir=tmp_log_dir)
        agents = [create_agent(0, "rational"), create_agent(1, "conformist")]
        turn_actions = [
            {"turn": 1, "agent_id": 0, "persona": "rational",
             "action": "DO_NOTHING", "content": "", "rationale": "stub",
             "opinion": 0.5, "risk_perception": 0.5},
            {"turn": 1, "agent_id": 1, "persona": "conformist",
             "action": "PROPOSE", "content": "test", "rationale": "reason",
             "opinion": 0.6, "risk_perception": 0.4},
        ]
        om.record_turn(1, agents, turn_actions, {0: (0.0, 0.0), 1: (0.0, 0.0)}, {})
        om.save_all({"turns": 1})

        actions_path = os.path.join(om.run_dir, "actions.json")
        assert os.path.isfile(actions_path)
        with open(actions_path) as f:
            actions = json.load(f)
        assert len(actions) == 2
        assert actions[0]["rationale"] == "stub"
        assert actions[1]["rationale"] == "reason"

    def test_save_opinion_trajectory_csv(self, tmp_log_dir):
        om = OutputManager(base_dir=tmp_log_dir)
        agents = [create_agent(0, "rational"), create_agent(1, "conformist")]
        agents[0].opinion = 0.3
        agents[1].opinion = 0.7
        om.record_turn(1, agents, [], {0: (0.0, 0.0), 1: (0.0, 0.0)}, {})
        om.save_all({})

        traj_path = os.path.join(om.run_dir, "opinion_trajectory.csv")
        assert os.path.isfile(traj_path)
        with open(traj_path) as f:
            lines = f.readlines()
        assert len(lines) == 3  # header + 2 agents
        assert "rational,0.3" in lines[1]
        assert "conformist,0.7" in lines[2]

    def test_save_statistics_csv(self, tmp_log_dir):
        om = OutputManager(base_dir=tmp_log_dir)
        agents = [create_agent(0, "rational"), create_agent(1, "conformist")]
        agents[0].opinion = 0.2
        agents[1].opinion = 0.8
        om.record_turn(1, agents, [], {0: (0.0, 0.0), 1: (0.0, 0.0)}, {})
        om.save_all({})

        stats_path = os.path.join(om.run_dir, "statistics.csv")
        assert os.path.isfile(stats_path)
        with open(stats_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        parts = lines[1].strip().split(",")
        assert parts[2] == "0.5"  # avg_opinion

    def test_save_summary_json(self, tmp_log_dir):
        om = OutputManager(base_dir=tmp_log_dir)
        agents = [create_agent(0, "rational")]
        om.record_turn(1, agents, [], {0: (0.0, 0.0)}, {"turns": 3})
        om.save_all({"turns": 3})

        summary_path = os.path.join(om.run_dir, "summary.json")
        assert os.path.isfile(summary_path)
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["config"]["turns"] == 3
        assert summary["n_turns"] == 1

    def test_multi_turn_accumulation(self, tmp_log_dir):
        om = OutputManager(base_dir=tmp_log_dir)
        agent = create_agent(0, "rational")
        for t in range(1, 4):
            actions = [{"turn": t, "agent_id": 0, "persona": "rational",
                        "action": "DO_NOTHING", "content": "", "rationale": f"turn{t}",
                        "opinion": 0.5, "risk_perception": 0.5}]
            om.record_turn(t, [agent], actions, {0: (0.0, 0.0)}, {})
        om.save_all({})

        with open(os.path.join(om.run_dir, "actions.json")) as f:
            actions = json.load(f)
        assert len(actions) == 3
        assert actions[0]["rationale"] == "turn1"
        assert actions[2]["rationale"] == "turn3"
