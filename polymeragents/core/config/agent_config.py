# Agent behavior (max_iterations, system_prompt, memory_config)

from typing import List, Optional
from pydantic import BaseModel, Field
from polymeragents.core.config.config import BaseComponentConfig
from polymeragents.core.config.llm_config import LLMConfig

class MemoryConfig(BaseModel):
    """Configuration for memory management in the agent."""
    enabled: bool = Field(default=True, description="Enable or disable memory management.")
    memory_type: str = Field(default="buffer", description="Type of memory to use (e.g., buffer, vector, or summary).")
    max_memory_size: Optional[int] = Field(default=None, description="Maximum size of memory (in bytes) if applicable.")
    max_history_messages: int = Field(default=10, gt=0)
    vector_store_uri: Optional[str] = Field(default=None, description="URI for the vector store if using a database-backed memory.")

class AgentConfig(BaseComponentConfig):
    """Configuration for the agent's behavior and settings."""
    agent_id: str = Field(..., description="Unique identifier for the agent.")
    system_prompt: str = Field(..., description="The system prompt that defines the agent's behavior.")

    max_iterations: int = Field(default=10, gt=0, description="Maximum number of iterations the agent can perform.")
    early_stopping_method: str = Field(default="force", description="'force' or 'generate'")

    memory_config: MemoryConfig = Field(default_factory=MemoryConfig, description="Configuration for the agent's memory management.")
    enabled_tools: List[str] = Field(default_factory=list, description="List of enabled tools for the agent.")

    llm_config: LLMConfig = Field(..., description="Configuration for the LLM used by the agent.")