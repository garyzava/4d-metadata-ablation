"""Multi-provider LLM client supporting OpenAI, Anthropic, Google Gemini, Vertex AI, and HuggingFace.

Configure via environment variables to switch between providers:
- OpenAI (default): Also works with Ollama, LM Studio, vLLM, Azure OpenAI
- Anthropic: Claude models
- Gemini: Google AI Studio (consumer API)
- Vertex: Google Cloud Vertex AI (enterprise)
- HuggingFace: Serverless Inference API (OpenAI-compatible)
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from typing import Any, Literal

from opentelemetry import trace
from pydantic import BaseModel

_logger = logging.getLogger(__name__)

# Provider type
Provider = Literal["openai", "anthropic", "gemini", "vertex", "huggingface"]
PROVIDERS: tuple[Provider, ...] = ("openai", "anthropic", "gemini", "vertex", "huggingface")

# Reasoning effort levels
ReasoningEffort = Literal["none", "low", "medium", "high"]

# Thinking-token budgets per reasoning effort level (for providers that need explicit budgets)
_REASONING_BUDGETS: dict[ReasoningEffort, int] = {
    "none": 0,
    "low": 4096,
    "medium": 8192,
    "high": 16384,
}

# Per-level defaults for the answer-token budget (max_tokens) when the user has not set
# GYZASQL_LLM_MAX_TOKENS explicitly. Scales with reasoning effort: deeper reasoning typically
# needs a larger answer budget too. An explicit GYZASQL_LLM_MAX_TOKENS always wins.
_DEFAULT_ANSWER_TOKENS: dict[ReasoningEffort, int] = {
    "none": 1024,
    "low": 2048,
    "medium": 4096,
    "high": 8192,
}
_LEGACY_MAX_TOKENS_DEFAULT = 1024  # used when reasoning is off and no env var is set


class LLMConfig(BaseModel, frozen=True):
    """Configuration for the LLM client."""

    provider: Provider = "openai"
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key: str | None = None
    seed: int = 42
    temperature: float = 0.0
    # None means "user did not set GYZASQL_LLM_MAX_TOKENS"; effective value is resolved
    # by _effective_max_tokens() against reasoning_effort. Set to an int for explicit override.
    max_tokens: int | None = None
    top_p: float = 1.0
    presence_penalty: float = 0.0
    top_k: int | None = None  # Ollama/llama.cpp/Anthropic (not in OpenAI spec)
    reasoning_effort: ReasoningEffort | None = None
    # Vertex AI specific
    project_id: str | None = None
    location: str = "us-central1"


def get_config(**overrides: Any) -> LLMConfig:
    """Build LLMConfig from environment variables with optional overrides.

    Environment variables:
        GYZASQL_LLM_PROVIDER        — provider: openai, anthropic, gemini, vertex, huggingface
        GYZASQL_MODEL               — model name (default: gpt-4o-mini)
        GYZASQL_LLM_BASE_URL        — base URL for OpenAI-compatible API
        GYZASQL_LLM_API_KEY         — API key (falls back to provider-specific keys)
        GYZASQL_LLM_SEED            — seed for reproducibility (default: 42)
        GYZASQL_LLM_TEMPERATURE     — temperature (default: 0)
        GYZASQL_LLM_MAX_TOKENS      — max answer tokens. If unset, defaults scale with
                                     reasoning_effort: none=1024, low=2048, medium=4096, high=8192.
                                     An explicit value always wins.
        GYZASQL_LLM_TOP_P           — top_p (default: 1.0)
        GYZASQL_LLM_PRESENCE_PENALTY — presence penalty (default: 0.0)
        GYZASQL_LLM_TOP_K           — top_k for Ollama/Anthropic (default: None)
        GYZASQL_LLM_REASONING_EFFORT — reasoning effort: none, low, medium, high (default: None)

    Backward compatibility (deprecated):
        GYZASQL_LLM_THINKING        — DEPRECATED. Maps to reasoning_effort=medium when truthy.
                                     Emits DeprecationWarning. Will be removed in a future release;
                                     migrate to GYZASQL_LLM_REASONING_EFFORT.
        GYZASQL_LLM_THINKING_BUDGET — ignored (use GYZASQL_LLM_REASONING_EFFORT instead)

    Vertex AI specific (requires ADC, not API keys):
        GOOGLE_CLOUD_PROJECT       — GCP project ID (required)
        GOOGLE_CLOUD_LOCATION      — GCP location (default: us-central1)

    Provider-specific API key fallbacks:
        - openai: OPENAI_API_KEY
        - anthropic: ANTHROPIC_API_KEY
        - gemini: GOOGLE_API_KEY
        - huggingface: HUGGINGFACE_API_KEY
        - vertex: Uses ADC (gcloud auth application-default login)
    """
    env_values: dict[str, Any] = {}

    # Determine provider first
    provider = os.environ.get("GYZASQL_LLM_PROVIDER", "openai").lower()
    if provider in PROVIDERS:
        env_values["provider"] = provider

    if v := os.environ.get("GYZASQL_MODEL"):
        env_values["model"] = v
    if v := os.environ.get("GYZASQL_LLM_BASE_URL"):
        env_values["base_url"] = v

    # API key resolution with provider-specific fallbacks
    if v := os.environ.get("GYZASQL_LLM_API_KEY"):
        env_values["api_key"] = v
    else:
        # Provider-specific fallbacks
        effective_provider = env_values.get("provider", "openai")
        if effective_provider == "anthropic":
            if v := os.environ.get("ANTHROPIC_API_KEY"):
                env_values["api_key"] = v
        elif effective_provider == "gemini":
            if v := os.environ.get("GOOGLE_API_KEY"):
                env_values["api_key"] = v
        elif effective_provider == "huggingface":
            if v := os.environ.get("HUGGINGFACE_API_KEY"):
                env_values["api_key"] = v
        elif effective_provider == "vertex":
            # Vertex AI uses ADC, not API keys
            pass
        else:  # openai
            if v := os.environ.get("OPENAI_API_KEY"):
                env_values["api_key"] = v

    # Vertex AI specific config
    if v := os.environ.get("GOOGLE_CLOUD_PROJECT"):
        env_values["project_id"] = v
    if v := os.environ.get("GOOGLE_CLOUD_LOCATION"):
        env_values["location"] = v

    if v := os.environ.get("GYZASQL_LLM_SEED"):
        env_values["seed"] = int(v)
    if v := os.environ.get("GYZASQL_LLM_TEMPERATURE"):
        env_values["temperature"] = float(v)
    if v := os.environ.get("GYZASQL_LLM_MAX_TOKENS"):
        env_values["max_tokens"] = int(v)
    if v := os.environ.get("GYZASQL_LLM_TOP_P"):
        env_values["top_p"] = float(v)
    if v := os.environ.get("GYZASQL_LLM_PRESENCE_PENALTY"):
        env_values["presence_penalty"] = float(v)
    if v := os.environ.get("GYZASQL_LLM_TOP_K"):
        env_values["top_k"] = int(v)

    # Reasoning effort (new) with backward compatibility for GYZASQL_LLM_THINKING
    if v := os.environ.get("GYZASQL_LLM_REASONING_EFFORT"):
        env_values["reasoning_effort"] = v.lower()
    elif v := os.environ.get("GYZASQL_LLM_THINKING"):
        if v.lower() in ("1", "true", "yes"):
            warnings.warn(
                "GYZASQL_LLM_THINKING is deprecated and will be removed in a future release. "
                "Use GYZASQL_LLM_REASONING_EFFORT=medium (or low/high) instead. "
                "This flag now maps to reasoning_effort=medium (was high).",
                DeprecationWarning,
                stacklevel=2,
            )
            env_values["reasoning_effort"] = "medium"

    # Overrides take precedence over env vars
    env_values.update({k: v for k, v in overrides.items() if v is not None})
    return LLMConfig(**env_values)


class ChatResult(BaseModel, frozen=True):
    """Result from a chat completion call."""

    text: str
    thinking: str | None = None


def _extract_think_blocks(text: str) -> str | None:
    """Extract concatenated content from <think>...</think> blocks, or None if absent."""
    matches = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if not matches:
        return None
    joined = "\n".join(m.strip() for m in matches if m.strip())
    return joined or None


def _strip_think_blocks(text: str) -> str:
    """Strip <think>...</think> blocks from model output (e.g. Qwen3 thinking mode)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _reasoning_enabled(config: LLMConfig) -> bool:
    """Check if reasoning/thinking mode is active."""
    return config.reasoning_effort is not None and config.reasoning_effort != "none"


def _effective_max_tokens(config: LLMConfig) -> int:
    """Resolve the answer-token budget.

    An explicit `config.max_tokens` (set via GYZASQL_LLM_MAX_TOKENS) always wins.
    Otherwise the default scales with reasoning effort via _DEFAULT_ANSWER_TOKENS.
    """
    if config.max_tokens is not None:
        return config.max_tokens
    if _reasoning_enabled(config) and config.reasoning_effort:
        return _DEFAULT_ANSWER_TOKENS.get(config.reasoning_effort, _LEGACY_MAX_TOKENS_DEFAULT)
    return _LEGACY_MAX_TOKENS_DEFAULT


def chat_completion(
    messages: list[dict[str, str]],
    config: LLMConfig | None = None,
    **kwargs: Any,
) -> ChatResult:
    """Run a chat completion with the configured provider.

    Returns a ChatResult with the response text and optional thinking content.
    """
    config = config or get_config()

    if config.provider == "anthropic":
        content, reasoning = _chat_anthropic(messages, config, **kwargs)
    elif config.provider == "gemini":
        content, reasoning = _chat_gemini(messages, config, **kwargs)
    elif config.provider == "vertex":
        content, reasoning = _chat_vertex(messages, config, **kwargs)
    elif config.provider == "huggingface":
        content, reasoning = _chat_huggingface(messages, config, **kwargs)
    else:  # openai (default, covers Ollama/vLLM/LM Studio)
        content, reasoning = _chat_openai(messages, config, **kwargs)

    thinking = _extract_think_blocks(content) or reasoning
    text = _strip_think_blocks(content)

    if thinking:
        _logger.debug("LLM thinking: %s", thinking[:500])
        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("gyzasql.llm.thinking", thinking[:2000])
            span.set_attribute("gyzasql.llm.thinking_tokens_approx", len(thinking.split()))

    return ChatResult(text=text, thinking=thinking)


def _chat_openai(
    messages: list[dict[str, str]],
    config: LLMConfig,
    **kwargs: Any,
) -> tuple[str, str | None]:
    """OpenAI and OpenAI-compatible API implementation."""
    from openai import OpenAI

    client_kwargs: dict[str, Any] = {}
    if config.base_url:
        client_kwargs["base_url"] = config.base_url
    if config.api_key:
        client_kwargs["api_key"] = config.api_key

    client = OpenAI(**client_kwargs)

    call_kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": _effective_max_tokens(config),
        "seed": config.seed,
        "top_p": config.top_p,
        "presence_penalty": config.presence_penalty,
    }

    # Reasoning mode for Ollama/vLLM (enable_thinking flag)
    if _reasoning_enabled(config):
        budget = _REASONING_BUDGETS.get(config.reasoning_effort, 8192)
        if budget > 0:
            call_kwargs["extra_body"] = {
                "enable_thinking": True,
                "thinking_budget": budget,
            }

    # top_k is not in the OpenAI spec — pass via Ollama's extra_body.options
    if config.top_k is not None:
        existing_extra = call_kwargs.get("extra_body", {})
        options = existing_extra.get("options", {})
        options["top_k"] = config.top_k
        existing_extra["options"] = options
        call_kwargs["extra_body"] = existing_extra

    call_kwargs.update(kwargs)

    response = client.chat.completions.create(**call_kwargs)
    content = response.choices[0].message.content.strip()
    raw = getattr(response.choices[0].message, "reasoning", None) or getattr(
        response.choices[0].message, "reasoning_content", None
    )
    reasoning = raw.strip() if isinstance(raw, str) and raw.strip() else None
    return content, reasoning


def _chat_huggingface(
    messages: list[dict[str, str]],
    config: LLMConfig,
    **kwargs: Any,
) -> tuple[str, str | None]:
    """HuggingFace Inference API (OpenAI-compatible).

    Uses router.huggingface.co/v1 with model in the request body.
    Sends only standard OpenAI fields — no Ollama-specific extensions.
    """
    from openai import OpenAI

    base_url = config.base_url or "https://router.huggingface.co/v1"
    client = OpenAI(base_url=base_url, api_key=config.api_key)

    # Standard OpenAI fields only — no extra_body, no options.top_k
    call_kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": _effective_max_tokens(config),
        "seed": config.seed,
        "top_p": config.top_p,
        "presence_penalty": config.presence_penalty,
    }

    # Pass reasoning_effort through, but on the HF router it is best-effort: thinking
    # is not reliably disableable there, so the real lever is the token budget
    # (max_tokens). See config/experiment-parameters.yaml -> thinking_rules.huggingface_qwen.
    if _reasoning_enabled(config):
        call_kwargs["reasoning_effort"] = config.reasoning_effort

    call_kwargs.update(kwargs)

    response = client.chat.completions.create(**call_kwargs)
    content = response.choices[0].message.content.strip()
    raw = getattr(response.choices[0].message, "reasoning", None) or getattr(
        response.choices[0].message, "reasoning_content", None
    )
    reasoning = raw.strip() if isinstance(raw, str) and raw.strip() else None
    return content, reasoning


def _chat_anthropic(
    messages: list[dict[str, str]],
    config: LLMConfig,
    **kwargs: Any,
) -> tuple[str, str | None]:
    """Anthropic Claude API implementation."""
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise ImportError("anthropic package not installed. Install with: uv add anthropic") from e

    client = Anthropic(api_key=config.api_key)

    # Extract system message (Anthropic uses separate system parameter)
    system_content = None
    user_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            user_messages.append(msg)

    # Two API-compatibility quirks for Claude Opus 4.7+ (see src/gyzasql/_VENDORED.md):
    #   (a) `temperature`, `top_p`, `top_k` are deprecated — sending any of them
    #       returns HTTP 400 "X is deprecated for this model".
    #   (b) `thinking={"type": "enabled", "budget_tokens": N}` is no longer
    #       supported — the API requires `{"type": "adaptive", ...}` plus an
    #       `output_config.effort` field, which gyzasql doesn't yet emit. The
    #       simplest correct behaviour is to NOT enable extended thinking for
    #       Opus 4.7+; the model is strong enough to answer without explicit
    #       thinking, and skipping it avoids both API-shape drift and a ~3x
    #       output-cost reduction.
    _NO_SAMPLING_PARAMS_PREFIXES = ("claude-opus-4-7",)
    _NO_LEGACY_THINKING_PREFIXES = ("claude-opus-4-7",)
    _supports_sampling_params = not config.model.startswith(_NO_SAMPLING_PARAMS_PREFIXES)
    _supports_legacy_thinking = not config.model.startswith(_NO_LEGACY_THINKING_PREFIXES)

    call_kwargs: dict[str, Any] = {
        "model": config.model,
        "max_tokens": _effective_max_tokens(config),
        "messages": user_messages,
    }
    if _supports_sampling_params:
        call_kwargs["temperature"] = config.temperature
        call_kwargs["top_p"] = config.top_p

    if system_content:
        call_kwargs["system"] = system_content

    # Anthropic supports top_k natively (when the model accepts sampling params)
    if _supports_sampling_params and config.top_k is not None:
        call_kwargs["top_k"] = config.top_k

    # Reasoning mode for Anthropic (extended thinking) — legacy enabled-budget API
    if _reasoning_enabled(config) and _supports_legacy_thinking:
        budget = _REASONING_BUDGETS.get(config.reasoning_effort, 8192)
        call_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

    # Anthropic doesn't support seed, so we don't pass it
    call_kwargs.update(kwargs)

    response = client.messages.create(**call_kwargs)
    return response.content[0].text.strip(), None


def _chat_gemini(
    messages: list[dict[str, str]],
    config: LLMConfig,
    **kwargs: Any,
) -> tuple[str, str | None]:
    """Google Gemini / Gemma via the google-genai SDK (Google AI Studio).

    Thinking is controlled with ``thinking_config`` (the legacy
    ``google-generativeai`` SDK could not set it, which is why earlier gemma runs
    used the slow, uncontrolled provider default). Gemma-4 exposes only
    ``thinking_level`` MINIMAL or HIGH (no budget, no LOW/MEDIUM), so we map the
    engine's reasoning_effort -> level: medium|high -> HIGH, low -> MINIMAL. With
    reasoning off, no thinking_config is sent. Gemini 2.5+ accept the same field.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise ImportError("google-genai package not installed. Install with: uv add google-genai") from e

    client = genai.Client(api_key=config.api_key)

    # Convert OpenAI-style messages -> google-genai contents; system -> system_instruction.
    system_content = None
    contents = []
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "system":
            system_content = content
        elif role == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part(text=content)]))
        else:  # user
            contents.append(types.Content(role="user", parts=[types.Part(text=content)]))

    cfg: dict[str, Any] = {
        "temperature": config.temperature,
        "max_output_tokens": _effective_max_tokens(config),
        "top_p": config.top_p,
    }
    if config.top_k is not None:
        cfg["top_k"] = config.top_k
    if system_content:
        cfg["system_instruction"] = system_content
    if _reasoning_enabled(config):
        level = "HIGH" if config.reasoning_effort in ("medium", "high") else "MINIMAL"
        cfg["thinking_config"] = types.ThinkingConfig(thinking_level=level)

    response = client.models.generate_content(
        model=config.model,
        contents=contents,
        config=types.GenerateContentConfig(**cfg),
    )

    # Return the answer text, excluding any thought parts (gemma-4 still returns
    # thought parts even when includeThoughts is off — a known provider no-op).
    cand = (response.candidates or [None])[0]
    parts = (cand.content.parts if cand and cand.content else None) or []
    answer = "".join(
        p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False)
    ).strip()
    if not answer:
        answer = (response.text or "").strip()
    return answer, None


def _chat_vertex(
    messages: list[dict[str, str]],
    config: LLMConfig,
    **kwargs: Any,
) -> tuple[str, str | None]:
    """Google Cloud Vertex AI implementation.

    Uses Application Default Credentials (ADC) for authentication.
    Set up with: gcloud auth application-default login

    Note: Vertex AI does not support API keys. For API key auth, use the gemini provider instead.
    """
    try:
        import vertexai
        from vertexai.generative_models import Content, GenerativeModel, Part
    except ImportError as e:
        raise ImportError(
            "google-cloud-aiplatform package not installed. Install with: uv add google-cloud-aiplatform"
        ) from e

    if not config.project_id:
        raise ValueError(
            "Vertex AI requires GOOGLE_CLOUD_PROJECT environment variable or project_id config. "
            "For API key authentication, use GYZASQL_LLM_PROVIDER=gemini instead."
        )

    vertexai.init(project=config.project_id, location=config.location)

    # Convert OpenAI-style messages to Vertex AI format
    system_instruction = None
    vertex_contents = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            system_instruction = content
        elif role == "assistant":
            vertex_contents.append(Content(role="model", parts=[Part.from_text(content)]))
        else:  # user
            vertex_contents.append(Content(role="user", parts=[Part.from_text(content)]))

    # Create model with optional system instruction
    model_kwargs: dict[str, Any] = {"model_name": config.model}
    if system_instruction:
        model_kwargs["system_instruction"] = system_instruction

    model = GenerativeModel(**model_kwargs)

    # Generation config
    generation_config = {
        "temperature": config.temperature,
        "max_output_tokens": _effective_max_tokens(config),
        "top_p": config.top_p,
    }

    response = model.generate_content(vertex_contents, generation_config=generation_config)
    return response.text.strip(), None


# Legacy function for backwards compatibility
def get_client(config: LLMConfig | None = None):
    """Return a configured OpenAI client (for backwards compatibility).

    Note: This only works with the openai provider. For other providers,
    use chat_completion() directly.
    """
    from openai import OpenAI

    config = config or get_config()
    kwargs: dict[str, Any] = {}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return OpenAI(**kwargs)
