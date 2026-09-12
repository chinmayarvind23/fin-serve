"""Expose an alternative engine through the shared request and accounting contract."""

from finserve.engines.openai_adapter import OpenAICompletionEngine


class SGLangEngine(OpenAICompletionEngine):
    """Use a separately deployed SGLang server without changing the benchmark protocol."""
