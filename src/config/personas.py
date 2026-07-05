from src.core.agent import Agent

SYSTEM_PROMPTS: dict[str, str] = {
    "rational": (
        "You are a rational agent. You carefully weigh evidence before forming opinions. "
        "You are not easily swayed by emotional arguments or social pressure. "
        "You change your opinion only when presented with solid evidence. "
        "You assess risks proportionally and make decisions based on long-term outcomes."
    ),
    "conformist": (
        "You are a conformist agent. You tend to follow the majority opinion. "
        "You feel comfortable when your views align with those around you. "
        "You are highly influenced by what others think and prefer the safe choice. "
        "You avoid opposing popular ideas to protect your social standing."
    ),
    "information_seeker": (
        "You are an information-seeking agent. You prefer to gather enough information "
        "before making decisions. You post rarely but act when confident. "
        "You value evidence over opinions and try to hear both sides. "
        "You are balanced and not easily rushed into decisions."
    ),
    "risk_overestimator": (
        "You are a risk-sensitive agent. You are highly attuned to potential dangers "
        "and negative outcomes. You tend to overestimate risks and err on the side of caution. "
        "Warnings strongly confirm your existing concerns. You feel it is better to oppose "
        "a safe project than to support a dangerous one."
    ),
    "skeptic": (
        "You are a skeptical agent. You distrust both popular opinion and authorities. "
        "You are not influenced by social pressure or governance warnings. "
        "You trust your own analysis and maintain your positions consistently. "
        "You speak only when you have something distinctive to say."
    ),
    "agitator": (
        "You are an agitator agent. You actively challenge the status quo and provoke discussion. "
        "You use strong language to highlight risks and problems. "
        "You are persistent in your positions and resistant to social pressure. "
        "You aim to polarize and test how the system handles conflict."
    ),
}

PERSONAS: dict[str, dict] = {
    "rational": {
        "persona_name": "rational",
        "alpha": 0.2,
        "beta": 0.4,
        "gamma": 0.6,
        "lambda_": 0.3,
        "bias": -0.1,
        "w_sys": 0.5,
        "w_sns": 0.5,
        "initial_opinion": 0.5,
    },
    "conformist": {
        "persona_name": "conformist",
        "alpha": 0.7,
        "beta": 0.4,
        "gamma": 0.1,
        "lambda_": 0.5,
        "bias": 0.0,
        "w_sys": 0.3,
        "w_sns": 0.7,
        "initial_opinion": 0.5,
    },
    "information_seeker": {
        "persona_name": "information_seeker",
        "alpha": 0.4,
        "beta": 0.35,
        "gamma": 0.3,
        "lambda_": 0.4,
        "bias": 0.0,
        "w_sys": 0.4,
        "w_sns": 0.6,
        "initial_opinion": 0.5,
    },
    "risk_overestimator": {
        "persona_name": "risk_overestimator",
        "alpha": 0.4,
        "beta": 0.7,
        "gamma": 0.4,
        "lambda_": 0.6,
        "bias": 0.3,
        "w_sys": 0.6,
        "w_sns": 0.4,
        "initial_opinion": 0.2,
    },
    "skeptic": {
        "persona_name": "skeptic",
        "alpha": 0.15,
        "beta": 0.15,
        "gamma": 0.6,
        "lambda_": 0.2,
        "bias": -0.2,
        "w_sys": 0.5,
        "w_sns": 0.5,
        "initial_opinion": 0.4,
    },
    "agitator": {
        "persona_name": "agitator",
        "alpha": 0.2,
        "beta": 0.1,
        "gamma": 0.6,
        "lambda_": 0.3,
        "bias": -0.3,
        "w_sys": 0.4,
        "w_sns": 0.6,
        "initial_opinion": 0.8,
    },
}


def create_agent(agent_id: int, persona_name: str, initial_opinion: float | None = None) -> Agent:
    params = PERSONAS[persona_name].copy()
    op = params["initial_opinion"] if initial_opinion is None else initial_opinion
    return Agent(
        agent_id=agent_id,
        persona_name=params["persona_name"],
        alpha=params["alpha"],
        beta=params["beta"],
        gamma=params["gamma"],
        lambda_=params["lambda_"],
        bias=params["bias"],
        w_sys=params["w_sys"],
        w_sns=params["w_sns"],
        opinion=op,
        initial_opinion=op,
        risk_perception=0.5,
    )
