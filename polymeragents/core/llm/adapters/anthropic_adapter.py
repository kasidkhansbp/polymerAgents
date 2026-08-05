from typing import AsyncGenerator, Dict, Any, List, Union
from urllib import response
from polymeragents.core.llm.base import BaseLLM, LLMResponse
from anthropic import AsyncAnthropic

class AnthropicAdapter(BaseLLM):
    def __init__(self, config: Any):
        super().__init__(config)
        self.client = AsyncAnthropic(api_key=config.api_key)

    async def generate(self, messages: Union[str, List[Dict[str, str]]], **kwargs) -> LLMResponse:
        """Invoke the Anthropic LLM with a given prompt and return a normalized response."""
        model = kwargs.pop("model", self.config.model)
        temperature = kwargs.pop("temperature", getattr(self.config, "temperature", 1.0))
        max_tokens = kwargs.pop("max_tokens", getattr(self.config, "max_tokens", 2048))

        system_prompt = kwargs.pop("system", None)
        formatted_messages = []

        if isinstance(messages, str):
            formatted_messages = [{"role": "user", "content": messages}]
        else:
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    formatted_messages.append({ "role": msg["role"], "content": msg["content"] })


        response = await self.client.messages.create(
            model=model,
            messages=formatted_messages,
            system=system_prompt if system_prompt else None,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        text_content  = ""
        for block in response.content:
            text_content += block.text
            
        return LLMResponse(
            content=text_content,
            raw_response=response.model_dump(),
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
            model_name=model
        )

    async def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[LLMResponse, None]:
        """Asynchronously invoke the Anthropic LLM with a given prompt and yield normalized responses as they are generated."""
        model = kwargs.pop("model", self.config.model)
        temperature = kwargs.pop("temperature", self.config.temperature)
        max_tokens = kwargs.pop("max_tokens", self.config.max_tokens)

        system_prompt = kwargs.pop("system_prompt", None)
        formatted_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                formatted_messages.append(f"{msg['role']}: {msg['content']}")

        async with self.client.messages.stream(
            model=model,
            messages = formatted_messages,
            system_prompt=system_prompt if system_prompt else None,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        ) as stream:
            async for text_delta in stream.text_deltas():
                yield LLMResponse(
                    content=text_delta,
                    raw_response={},
                    prompt_tokens=None,
                    completion_tokens=None,
                    model_name=model
                )