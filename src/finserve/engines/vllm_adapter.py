"""Keep the deployment identity explicit while sharing the tested completion protocol."""

from finserve.engines.openai_adapter import OpenAICompletionEngine


class VLLMEngine(OpenAICompletionEngine):
    """Use a separately deployed vLLM server, which owns continuous batching and GPU memory."""
