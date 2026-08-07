# LLMFactory for dynamic adapter instantiation

from typing import Dict, Type, Any
from polymeragents.core.llm.base import BaseLLM
from polymeragents.core.llm.adapters.openai_adapter import OpenAIAdapter
from polymeragents.core.llm.adapters.anthropic_adapter import AnthropicAdapter

class LLMFactory:
    _adapters: Dict[str, Type[BaseLLM]] = {
        "openai": OpenAIAdapter,
        "anthropic": AnthropicAdapter,
    }

    @classmethod
    def register_adapter(cls, provider_name: str, adapter_cls: Type[BaseLLM]) -> None:
        """Register a new LLM adapter class for a given provider name."""
        cls._adapters[provider_name.lower()] = adapter_cls

    @classmethod
    def get_llm(cls, config: Any) -> BaseLLM:
        # Resolve string representation whether enum or raw str is passed
        provider_str = config.provider.value if hasattr(config.provider, "value") else str(config.provider)
        provider_key = provider_str.lower()
        
        adapter_cls = cls._adapters.get(provider_key)
        if not adapter_cls:
            raise ValueError(
                f"Unsupported provider '{provider_key}'. Registered providers: {list(cls._adapters.keys())}"
            )

        return adapter_cls(config)