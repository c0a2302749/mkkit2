import asyncio
import argparse
import math
import os
import random
import sys
from dataclasses import dataclass, field
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.core.state import SimulationState
from src.core.agent import Action
from src.core.action import ActionType
from src.config.personas import create_agent, PERSONAS
from src.social.graph import SocialGraph
from src.social.timeline import TimelineManager
from src.agent.engine import OpinionDynamicsEngine
from src.agent.manager import AgentManager
from src.agent.llm import LLMProvider
from src.governance.stub import GovernanceStub
from src.engine.simulation import SimulationEngine


class StubProvider(LLMProvider):
    async def invoke(self, prompt: str, system_prompt: str = "") -> str:
        return '{"action": "DO_NOTHING", "content": "", "rationale": "stub"}'


@dataclass
class RunResult:
    persona: str
    condition: str
    seed: int
    opinion_changes: list[float] = field(default_factory=list)
    conformity_rate: float = 0.0
    do_nothing_rate: float = 0.0
    risk_perception_changes: list[float] = field(default_factory=list)
    invariance_rate: float = 0.0
    posting_freq: float = 0.0
    action_counts: dict[str, int] = field(default_factory=lambda: {a.name: 0 for a in ActionType})


def _setup_engine(persona: str, n_agents: int, turns: int, seed: int,
                  sns_on: bool, llm: str) -> SimulationEngine:
    random.seed(seed)
    state = SimulationState()
    for i in range(n_agents):
        agent = create_agent(i, persona, initial_opinion=random.uniform(0.3, 0.7))
        state.agents.append(agent)

    graph = SocialGraph()
    for agent in state.agents:
        graph.add_agent(agent)

    if sns_on:
        graph.generate_random(n_agents, seed)
    else:
        graph.generate_isolated(n_agents)

    timeline = TimelineManager()
    ode = OpinionDynamicsEngine()

    if sns_on:
        if llm == "stub":
            am = AgentManager(StubProvider())
        else:
            from src.agent.azure_provider import AzureProvider
            am = AgentManager(AzureProvider())
        governance = GovernanceStub()
        engine = SimulationEngine(state, graph, timeline, am, governance, ode)
        engine.ode = ode
    else:
        class NullManager:
            def __init__(self):
                self._ode = ode
            async def decide_action(self, agent, tl_str, warning=""):
                s_o, s_r = self._ode.compute_social_signals(agent, graph)
                new_r = self._ode.update_risk_perception(
                    R=agent.risk_perception, W=0.0, S_R=0.0,
                    lambda_i=agent.lambda_, w_sys=agent.w_sys,
                    w_sns=0.0, b=agent.bias,
                )
                new_o = self._ode.compute_opinion_update(
                    O=agent.opinion, S_O=0.0, R=new_r,
                    alpha=agent.alpha, beta=agent.beta,
                    gamma=agent.gamma, O_0=agent.initial_opinion,
                )
                agent.risk_perception = new_r
                agent.opinion = new_o
                return Action(action_type=ActionType.DO_NOTHING)
        governance = GovernanceStub()
        engine = SimulationEngine(state, graph, timeline, NullManager(), governance, ode)
        engine.ode = ode

    return engine


def _extract_metrics(engine: SimulationEngine, initial_opinions: list[float],
                     initial_rps: list[float]) -> RunResult:
    agents = engine.state.agents
    n = len(agents)

    opinion_changes = [abs(a.opinion - init) for a, init in zip(agents, initial_opinions)]
    rp_changes = [abs(a.risk_perception - init) for a, init in zip(agents, initial_rps)]

    total_actions = sum(len(a.action_history) for a in agents)
    do_nothing_count = sum(
        1 for a in agents for act in a.action_history
        if act.action_type == ActionType.DO_NOTHING
    )
    do_nothing_rate = do_nothing_count / total_actions if total_actions else 0.0

    invariance_rate = sum(1 for c in opinion_changes if c < 0.05) / n if n else 0.0

    posting_count = sum(
        1 for a in agents for act in a.action_history
        if act.action_type in (ActionType.PROPOSE, ActionType.COMMENT,
                                ActionType.SUPPORT, ActionType.OPPOSE)
    )
    posting_freq = posting_count / (n * max(1, engine.state.current_turn))

    action_counts = {at.name: 0 for at in ActionType}
    for a in agents:
        for act in a.action_history:
            action_counts[act.action_type.name] += 1

    # Conformity: not calculated for null model (no S_O)
    return RunResult(
        persona=agents[0].persona_name,
        condition="",
        seed=0,
        opinion_changes=opinion_changes,
        conformity_rate=0.0,
        do_nothing_rate=do_nothing_rate,
        risk_perception_changes=rp_changes,
        invariance_rate=invariance_rate,
        posting_freq=posting_freq,
        action_counts=action_counts,
    )


def _run_chi_square(results: list[RunResult]) -> dict:
    from scipy.stats import chi2_contingency

    personas = sorted(set(r.persona for r in results))
    action_types = sorted(set(at.name for at in ActionType))

    contingency = []
    for p in personas:
        row = [sum(r.action_counts[at] for r in results if r.persona == p) for at in action_types]
        if sum(row) == 0:
            return {"chi2": 0.0, "p_value": 1.0, "dof": 0, "cramers_v": 0.0}
        contingency.append(row)

    # Drop columns with all zeros
    col_sums = [sum(row[c] for row in contingency) for c in range(len(action_types))]
    keep = [c for c in range(len(action_types)) if col_sums[c] > 0]
    if len(keep) < 2:
        return {"chi2": 0.0, "p_value": 1.0, "dof": 0, "cramers_v": 0.0}
    filtered = [[row[c] for c in keep] for row in contingency]

    chi2, p_val, dof, expected = chi2_contingency(filtered)
    n_total = sum(sum(row) for row in filtered)
    min_dim = min(len(filtered), len(filtered[0])) if filtered else 1
    cramer_v = math.sqrt(chi2 / (n_total * (min_dim - 1))) if n_total > 0 and min_dim > 1 else 0.0
    return {"chi2": chi2, "p_value": p_val, "dof": dof, "cramers_v": cramer_v}


async def run_trial(persona: str, sns_on: bool, seed: int,
                    n_agents: int, turns: int, llm: str) -> RunResult:
    engine = _setup_engine(persona, n_agents, turns, seed, sns_on, llm)
    initial_opinions = [a.opinion for a in engine.state.agents]
    initial_rps = [a.risk_perception for a in engine.state.agents]
    await engine.run(turns)
    result = _extract_metrics(engine, initial_opinions, initial_rps)
    result.persona = persona
    result.condition = "sns_on" if sns_on else "sns_off"
    result.seed = seed
    return result


def _make_condition_list(arg: str) -> list[bool]:
    if arg == "both":
        return [True, False]
    return [arg == "sns_on"]


async def run_experiment(trials: int, n_agents: int, turns: int,
                         seed_offset: int, llm: str, condition_flags: list[bool]):
    all_results: list[RunResult] = []
    persona_list = list(PERSONAS.keys())
    total = len(persona_list) * len(condition_flags) * trials
    done = 0

    for persona in persona_list:
        for sns_on in condition_flags:
            sns_str = "SNS_on" if sns_on else "SNS_off"
            print(f"  {persona} / {sns_str}: ", end="", flush=True)
            batch = []
            for t in range(trials):
                seed = seed_offset + t + hash(persona) % 10000
                result = await run_trial(persona, sns_on, seed, n_agents, turns, llm)
                batch.append(result)
                done += 1
            all_results.extend(batch)
            changes = [c for r in batch for c in r.opinion_changes]
            dn_rate = sum(r.do_nothing_rate for r in batch) / len(batch)
            pf = sum(r.posting_freq for r in batch) / len(batch)
            print(f"|ΔO|={sum(changes)/len(changes):.3f} "
                  f"DN={dn_rate:.2f} post={pf:.2f} ({done}/{total})")

    return all_results


def print_results(results: list[RunResult]):
    print()
    print("=" * 80)
    print("Phase A: ペルソナ-パラメータ連動検証 結果")
    print("=" * 80)
    print(f"{'Persona':<20} {'Cond':<8} {'|ΔO|':>6} {'DN%':>6} {'Post':>6} "
          f"{'Inv%':>6} {'Prop':>5} {'Supp':>5} {'Opp':>5} {'Comm':>5} {'Vote':>5} {'DN':>5}")
    print("-" * 80)

    cond_labels = {True: "SNS_on", False: "SNS_off"}
    conds_present = sorted(set(r.condition for r in results), key=lambda c: (c != "sns_on", c))
    persona_list = list(PERSONAS.keys())

    for persona in persona_list:
        for condition in conds_present:
            subset = [r for r in results if r.persona == persona and r.condition == condition]
            if not subset:
                continue
            n = len(subset)
            avg_oc = sum(c for r in subset for c in r.opinion_changes) / max(1, n * 8)
            avg_dn = sum(r.do_nothing_rate for r in subset) / n
            avg_pf = sum(r.posting_freq for r in subset) / n
            avg_inv = sum(r.invariance_rate for r in subset) / n
            avg_ac = {at: sum(r.action_counts[at] for r in subset) / n for at in [a.name for a in ActionType]}
            p, s, o, c, v, dn = (avg_ac["PROPOSE"], avg_ac["SUPPORT"],
                                  avg_ac["OPPOSE"], avg_ac["COMMENT"],
                                  avg_ac["VOTE"], avg_ac["DO_NOTHING"])
            print(f"{persona:<20} {cond_labels.get(condition, condition):<8} "
                  f"{avg_oc:>6.3f} {avg_dn:>6.2%} {avg_pf:>6.2f} "
                  f"{avg_inv:>6.2%} "
                  f"{p:>5.1f} {s:>5.1f} {o:>5.1f} {c:>5.1f} {v:>5.1f} {dn:>5.1f}")

    for condition in conds_present:
        subset = [r for r in results if r.condition == condition]
        if not subset:
            continue
        total_acts = sum(sum(r.action_counts.values()) for r in subset)
        if total_acts == 0:
            cond_str = cond_labels.get(condition, condition)
            print(f"\n[χ² test - {cond_str}]")
            print("  No actions recorded (all DO_NOTHING)")
            continue
        chi2_result = _run_chi_square(subset)
        cond_str = cond_labels.get(condition, condition)
        print(f"\n[χ² test - {cond_str}]")
        print(f"  χ²={chi2_result['chi2']:.2f}, p={chi2_result['p_value']:.4f}, "
              f"Cramér V={chi2_result['cramers_v']:.3f}")
        sig = "SIGNIFICANT" if chi2_result['p_value'] < 0.05 else "NOT significant"
        print(f"  → {sig} (p<0.05)")

    print("=" * 80)


async def main():
    parser = argparse.ArgumentParser(description="Phase A: Persona-parameter linkage validation")
    parser.add_argument("--llm", type=str, default="stub", choices=["azure", "stub"])
    parser.add_argument("--trials", type=int, default=3, help="Trials per condition (default: 3, full: 20)")
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=1000)
    parser.add_argument("--condition", type=str, default="both",
                        choices=["both", "sns_on", "sns_off"])
    args = parser.parse_args()

    condition_flags = _make_condition_list(args.condition)
    print(f"Phase A: {len(list(PERSONAS.keys()))} personas × {len(condition_flags)} conditions × {args.trials} trials")
    print(f"  LLM={args.llm}, agents={args.agents}, turns={args.turns}")

    results = await run_experiment(args.trials, args.agents, args.turns,
                                   args.seed_offset, args.llm, condition_flags)
    print_results(results)


if __name__ == "__main__":
    asyncio.run(main())
