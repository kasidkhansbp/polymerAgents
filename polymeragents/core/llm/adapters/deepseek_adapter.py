from typing import List, Dict, Any, AsyncGenerator
from polymeragents.core.llm.base import BaseLLM, LLMResponse

# DeepSeek exposes an OpenAI-compatible API, so this adapter can reuse the
# AsyncOpenAI client pointed at DeepSeek's base_url (e.g. "https://api.deepseek.com").


class DeepSeekAdapter(BaseLLM):
    """Adapter for DeepSeek's LLMs, providing a unified interface."""

    def __init__(self, config: Any):
        super().__init__(config)
        # TODO: initialize the client (AsyncOpenAI with base_url + api_key)
        self.client = None

    async def generate(self, message: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Invoke the DeepSeek LLM with a given prompt and return a normalized response."""
        model = kwargs.pop("model", self.config.model)
        temperature = kwargs.pop("temperature", self.config.temperature)

        # TODO: call the client and map the result into LLMResponse
        raise NotImplementedError

    async def generate_stream(self, message: List[Dict[str, str]], **kwargs) -> AsyncGenerator[LLMResponse, None]:
        """Asynchronously invoke the DeepSeek LLM with a given prompt and yield normalized responses as they are generated."""
        model = kwargs.pop("model", self.config.model)
        temperature = kwargs.pop("temperature", self.config.temperature)

        # TODO: open a streaming call and yield LLMResponse per chunk
        raise NotImplementedError
        yield  # keeps this an async generator until implemented
