from src.core.agent import Agent

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
    },
}


def create_agent(agent_id: int, persona_name: str, initial_opinion: float = 0.5) -> Agent:
    params = PERSONAS[persona_name].copy()
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
        opinion=initial_opinion,
        initial_opinion=initial_opinion,
        risk_perception=0.5,
    )
