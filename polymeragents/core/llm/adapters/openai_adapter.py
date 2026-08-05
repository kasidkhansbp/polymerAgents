from typing import List, Dict, Any, AsyncGenerator
from openai import AsyncOpenAI
from polymeragents.core.llm.base import BaseLLM, LLMResponse

class OpenAIAdapter(BaseLLM):
    """Adapter for OpenAI's LLMs, providing a unified interface."""

    def __init__(self, config: Any):
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=config.api_key)

    async def generate(self, message: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Invoke the OpenAI LLM with a given prompt and return a normalized response."""
        model = kwargs.pop("model", self.config.model)
        temperature = kwargs.pop("temperature", self.config.temperature)

        response = await self.client.chat.completions.create(
            model=model,
            messages=message,
            temperature=temperature,
            **kwargs
        )
        return LLMResponse(
            content=response.choices[0].message["content"] or "",
            raw_response=response.model_dump(),
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
            model_name=model
        )

    async def generate_stream(self, message: List[Dict[str, str]], **kwargs) -> AsyncGenerator[LLMResponse, None]:
        """Asynchronously invoke the OpenAI LLM with a given prompt and yield normalized responses as they are generated."""
        model = kwargs.pop("model", self.config.model)
        temperature = kwargs.pop("temperature", self.config.temperature)

        response = await self.client.chat.completions.create(
            model=model,
            messages=message,
            stream=True,
            temperature=temperature,
            **kwargs
        )
        async for chunk in response:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            content = delta.content or ""
            yield LLMResponse(
                content=content,
                raw_response=chunk.model_dump(),
                prompt_tokens=None,  # Not available in streaming mode
                completion_tokens=None,  # Not available in streaming mode
                model_name=model
            )