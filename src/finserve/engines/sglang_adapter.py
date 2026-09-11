"""Expose the measured alternative through exactly the same request and accounting path."""

from finserve.engines.openai_adapter import OpenAICompletionEngine


class SGLangEngine(OpenAICompletionEngine):
    """Use a separately deployed SGLang server without changing the benchmark protocol."""
