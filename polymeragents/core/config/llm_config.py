from enum import Enum
from typing import Optional
from pydantic import SecretStr, Field
from polymeragents.core.config.config import BaseComponentConfig

class LLMProvider(str, Enum):
    """Enumeration of supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    COHERE = "cohere"
    AI21 = "ai21"
    HUGGINGFACE = "huggingface"

class LLMConfig(BaseComponentConfig):
    """Configuration class for LLM components."""
    provider: LLMProvider = Field(..., description="The LLM provider to use.")
    api_key: Optional[SecretStr] = Field(default=None, description="API key for the LLM provider.")
    model_name: str = Field(..., description="The name of the model to use from the provider.")

    # Generation parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    seed: Optional[int] = Field(default=None)

    # Operational controls
    timeout: float = Field(default=60.0, gt=0.0, description="API call timeout in seconds")
    max_retries: int = Field(default=3, ge=0)
    api_key: Optional[SecretStr] = Field(default=None, description="Falls back to standard env vars if None")

    # Advanced provider pass-throughs
    extra_headers: dict[str, str] = Field(default_factory=dict)