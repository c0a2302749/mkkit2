import os
from openai import AsyncAzureOpenAI
from src.agent.llm import LLMProvider


class AzureProvider(LLMProvider):
    def __init__(self, seed: int | None = None, temperature: float = 0.0):
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise KeyError("AZURE_OPENAI_ENDPOINT is not set")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise KeyError("AZURE_OPENAI_API_KEY is not set")
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        if not deployment:
            raise KeyError("AZURE_OPENAI_DEPLOYMENT is not set")

        self._client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        )
        self._deployment = deployment
        self._seed = seed
        self._temperature = temperature
        self._last_fingerprint: str | None = None

    @property
    def last_fingerprint(self) -> str | None:
        return self._last_fingerprint

    async def invoke(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs = dict(
            model=self._deployment,
            messages=messages,
            temperature=self._temperature,
            max_tokens=500,
        )
        if self._seed is not None:
            kwargs["seed"] = self._seed
        response = await self._client.chat.completions.create(**kwargs)
        self._last_fingerprint = getattr(response, "system_fingerprint", None)
        return response.choices[0].message.content or ""
