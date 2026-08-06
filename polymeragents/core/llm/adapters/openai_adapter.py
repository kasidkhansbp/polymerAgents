
from typing import List, Dict, Any, AsyncGenerator, Union
from openai import AsyncOpenAI
from polymeragents.core.llm.base import BaseLLM, LLMResponse

class OpenAIAdapter(BaseLLM):
    """Adapter for OpenAI's LLMs, providing a unified interface."""

    def __init__(self, config: Any):
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=config.api_key)

    async def generate(self, messages: Union[str,List[Dict[str, str]]], **kwargs) -> LLMResponse:
        """Invoke the OpenAI LLM with a given prompt and return a normalized response."""
        model = kwargs.pop("model", self.config.model)
        temperature = kwargs.pop("temperature", self.config.temperature)

        formatted_messages = (
            [{"role": "user", "content": messages}]
            if isinstance(messages, str)
            else messages
        )

        response = await self.client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            temperature=temperature,
            **kwargs
        )
        return LLMResponse(
            content=response.choices[0].message.content or "",
            raw_response=response.model_dump(),
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
            model_name=model
        )

    async def generate_stream(self, messages: Union[str, List[Dict[str, str]]], **kwargs) -> AsyncGenerator[LLMResponse, None]:
        """Asynchronously invoke the OpenAI LLM with a given prompt and yield normalized responses as they are generated."""
        model = kwargs.pop("model", self.config.model)
        temperature = kwargs.pop("temperature", self.config.temperature)
        kwargs.pop("stream", None)          # discard — streaming is forced below
        kwargs.pop("stream_options", None)  # discard — we set our own

        # 1. Format string or list input into OpenAI message list format
        formatted_messages = (
            [{"role": "user", "content": messages}]
            if isinstance(messages, str)
            else messages
        )
        response = await self.client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            stream=True,
            stream_options={"include_usage": True},
            temperature=temperature,
            **kwargs
        )
        async for chunk in response:
            # 1. Properly indent 'content = ""' under 'async for'
            content = ""
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content

            # 2. Extract usage (populated on OpenAI's final trailing chunk)
            input_tokens = chunk.usage.prompt_tokens if chunk.usage else None
            output_tokens = chunk.usage.completion_tokens if chunk.usage else None

            # 3. Yield when text arrives OR when final token counts arrive
            if content or chunk.usage:
                yield LLMResponse(
                    content=content,
                    raw_response=chunk.model_dump(),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model_name=model
                )