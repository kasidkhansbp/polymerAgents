# Base configuration schemas (Pydantic / Dataclasses)
# Purpose: Shared abstractions, reusable mixins, custom types, and the root system setting that loads environment variables (.env).

from enum import Enum
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Environment(str, Enum):
    """The environment in which the application is running."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"

class BaseAppConfig(BaseSettings):
    """Base configuration class for the application."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False)
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="The environment in which the application is running.")
    debug: bool = Field(default=True, description="Enable or disable debug mode.")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO", description="Logging level for the application.")

class BaseComponentConfig(BaseModel):
    """Base configuration class for individual components."""
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, extra="forbid")
    enabled: bool = Field(default=True, description="Enable or disable the component.")
    timeout: Optional[float] = Field(default=None, gt=0.0, description="Timeout in seconds for component operations.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary user metadata")