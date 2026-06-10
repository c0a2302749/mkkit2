import os
from openai import AsyncAzureOpenAI
from src.agent.llm import LLMProvider


class AzureProvider(LLMProvider):
    def __init__(self):
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

    async def invoke(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = await self._client.chat.completions.create(
            model=self._deployment,
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content or ""
