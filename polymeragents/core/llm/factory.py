"""
LLM factory for polymerAgents.

Decision 6 (flow): the caller constructs an AgentConfig -> hands it to a factory
-> the factory reads config.provider and builds the matching LangChain chat
model. Providers are swapped by changing config.provider IN CODE, never by an
env var (Decision 1).

On API keys (Decision 3): we pass provider + model + tuning params only. Each
LangChain integration reads its provider's standard env var for the key
(ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY / DEEPSEEK_API_KEY). The
factory never sees a secret.

Provider SDKs are imported lazily, inside each branch, so a user who installs
only the integration they use (e.g. langchain-anthropic) is not forced to have
the others installed.
"""

from __future__ import annotations

from polymeragents.core.config import AgentConfig, Provider


def build_llm(config: AgentConfig):
    """Build a LangChain chat model from `config`.

    Returns a BaseChatModel; call the usual LangChain methods on it
    (`.invoke(...)`, `.with_structured_output(...)`, etc.).
    """
    # Params whose name is identical across every integration.
    common = {
        "model": config.model,
        "temperature": config.temperature,
        "timeout": config.timeout,
        "max_retries": config.max_retries,
    }

    if config.provider is Provider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(max_tokens=config.max_tokens, **common)

    if config.provider is Provider.OPENAI:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(max_tokens=config.max_tokens, **common)

    if config.provider is Provider.GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Gemini names this param differently — exactly the kind of per-provider
        # difference the factory exists to absorb.
        return ChatGoogleGenerativeAI(max_output_tokens=config.max_tokens, **common)

    if config.provider is Provider.DEEPSEEK:
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(max_tokens=config.max_tokens, **common)

    # Unreachable while Provider and this function agree; kept as a guard so a
    # newly added provider that someone forgets to wire fails loudly here.
    raise ValueError(f"Unsupported provider: {config.provider!r}")


def build_agent(config: AgentConfig):
    """Factory that wires an agent from config (Decision 6).

    This is the seam the 8 patterns plug into: an agent receives its config
    through a factory function rather than constructing its own client. Each
    pattern will build its LLM here via `build_llm(config)` and assemble its
    graph/tools around it.

    Left as the intended shape until the first pattern is implemented.
    """
    llm = build_llm(config)
    raise NotImplementedError(
        "build_agent() is the wiring seam for the 8 patterns; "
        "no pattern is implemented yet."
    )
