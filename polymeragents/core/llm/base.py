# Abstract Base Class (BaseLLM) defining unified interface

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncGenerator
from pydantic import BaseModel, Field

class LLMResponse(BaseModel):
    """Normalized output structure across all LLM providers"""
    content: str
    raw_response: Dict[str, Any] = Field(default=None, description="The raw response from the LLM provider, if available.")
    input_tokens: Optional[int] = Field(default=None, description="Number of tokens used in the prompt.")
    output_tokens: Optional[int] = Field(default=None, description="Number of tokens used in the completion.")
    model_name: Optional[str] = Field(default=None, description="The name of the model used for the response.")

class BaseLLM(ABC):
    """Abstract base class for LLMs, providing a unified interface for different providers."""

    def __init__(self, config: Any):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Invoke the LLM with a given prompt and return a normalized response."""
        pass

    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[LLMResponse, None]:
        """Asynchronously invoke the LLM with a given prompt and yield normalized responses as they are generated."""
        pass
