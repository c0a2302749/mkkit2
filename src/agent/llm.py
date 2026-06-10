from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    async def invoke(self, prompt: str, system_prompt: str = "") -> str:
        ...
