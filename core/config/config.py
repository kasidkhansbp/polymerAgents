"""
Agent configuration for polymerAgents.

Design decisions embodied here:
  1. This is a LIBRARY, so config is PASSED IN CODE by the caller. Nothing here
     reads a .env, a config file, or os.environ. The user's app owns its env.
  2. We use pydantic BaseModel (NOT BaseSettings). BaseModel reads nothing on
     its own; every value is supplied explicitly by the caller.
  3. API keys are NOT in this config. The provider SDK (via its LangChain
     integration) reads its own standard env var. Config carries provider +
     model only; it never touches secrets.
  4. A field is mandatory when no default can be *safely* chosen for the caller.
  5. Defaults that VARY BY PROVIDER cannot be static field defaults (those are
     fixed at class-definition time, before the provider is known). They are
     filled by a model_validator(mode="after"), which runs once the provider
     is available.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class Provider(str, Enum):
    """The LLM providers polymerAgents supports.

    A str-Enum so the value round-trips cleanly as a plain string ("anthropic")
    while still giving us a closed set that pydantic validates for free — pass
    an unknown provider and construction fails loudly, right where the mistake
    is, instead of surfacing later as a cryptic SDK error.
    """

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"


# --- Decision 5: per-provider defaults ---------------------------------------
# Values for fields whose sensible default depends on the chosen provider.
# This table is DATA; the model_validator below is the MECHANISM that applies
# it — and only for fields the caller left unset.
#
# To make another field provider-varying: give it a `None` default in the model
# and add its per-provider values to each row here. To make it universal again,
# give it a static default in the model and drop it from these rows.
_PROVIDER_DEFAULTS: dict[Provider, dict[str, object]] = {
    Provider.ANTHROPIC: {"max_tokens": 4096},
    Provider.OPENAI: {"max_tokens": 4096},
    Provider.GEMINI: {"max_tokens": 8192},
    Provider.DEEPSEEK: {"max_tokens": 4096},
}


class AgentConfig(BaseModel):
    """What the caller constructs and hands to a polymerAgents factory.

    Example:
        config = AgentConfig(provider="anthropic", model="claude-sonnet-4-5")
        # temperature, timeout, max_retries take universal defaults;
        # max_tokens is filled from the per-provider table above.
    """

    # extra="forbid": reject unknown kwargs so a typo like `temprature=0.2`
    # fails at construction instead of being silently ignored.
    model_config = {"extra": "forbid"}

    # --- Mandatory: Decision 4 — no default is safe to pick for the caller ---
    # provider: no universal right answer; a wrong guess surfaces as a confusing
    # auth error far from the actual mistake.
    provider: Provider
    # model: a per-provider default is *technically* possible, but model choice
    # is a cost/capability decision. We force the caller to own it rather than
    # silently spend their money on a model we picked.
    model: str = Field(min_length=1)

    # --- Optional with UNIVERSAL defaults (same value serves everyone) --------
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout: float = Field(default=60.0, gt=0.0)  # seconds
    max_retries: int = Field(default=2, ge=0)

    # --- Optional, PROVIDER-VARYING (Decision 5) -----------------------------
    # Default is None as a sentinel meaning "caller did not set this". The
    # validator replaces None with the provider-specific value. It stays a real,
    # validated int (gt=0) if the caller does supply one.
    max_tokens: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _apply_provider_defaults(self) -> "AgentConfig":
        """Fill provider-varying fields the caller left unset.

        `self.model_fields_set` is exactly the set of fields the caller passed
        explicitly, so we only fill in defaults for the ones they omitted — an
        explicit `max_tokens=100` is never overwritten.
        """
        for field, value in _PROVIDER_DEFAULTS.get(self.provider, {}).items():
            if field not in self.model_fields_set:
                setattr(self, field, value)
        return self
