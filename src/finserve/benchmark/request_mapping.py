"""Opt-in offline ChatML role extraction; model answers and evaluator data never enter mapping."""

START = "<|im_start|>system\n"
SEPARATOR = "<|im_end|>\n<|im_start|>user\n"
END = "<|im_end|>\n<|im_start|>assistant\n"
MAXIMUM_PROMPT_BYTES = 131072

FORMAT_INSTRUCTION_V1 = (
    "Follow the output format requested by the user. Return only the requested answer, without "
    "Markdown, code fences, labels, or explanations. For numeric-only answers, use the shortest "
    "ordinary decimal representation: no exponent, separators, leading plus, unnecessary leading "
    "zeros, or trailing fractional zeros. Keep one zero before the decimal point when needed. "
    "An explicit request for a particular precision or notation takes priority over these numeric "
    "defaults. If the request asks for yes or no, use exactly lowercase yes or no. For JSON, "
    "return only the requested object, preserve the requested value types, and include no "
    "unrequested keys."
)


def has_control(value: str) -> bool:
    """Reject complete and truncated control prefixes rather than guessing at malformed roles."""
    return "<|im_" in value or "<|endoftext" in value


def chatml_roles(prompt: str, instruction: str) -> list[dict[str, str]]:
    """Interpret only exact system->user->empty-assistant ChatML in this selected offline arm.

    Plain user content is preserved, including whitespace. This is never applied implicitly
    to gateway text. The instruction is explicit run configuration, not an inferred answer.
    """
    if len(prompt.encode("utf-8")) > MAXIMUM_PROMPT_BYTES or len(instruction) > 16384:
        raise ValueError("request mapping exceeds bounded input size")
    if has_control(instruction):
        raise ValueError("mapping instruction contains control markers")
    source_system: str | None = None
    user = prompt
    if has_control(prompt):
        if not prompt.startswith(START) or not prompt.endswith(END):
            raise ValueError("unsupported or malformed ChatML prompt")
        content = prompt[len(START) : -len(END)]
        if content.count(SEPARATOR) != 1:
            raise ValueError("ChatML prompt requires exactly one system/user boundary")
        source_system, user = content.split(SEPARATOR)
        if has_control(source_system) or has_control(user):
            raise ValueError("nested or ambiguous ChatML content")
        if START + source_system + SEPARATOR + user + END != prompt:
            raise ValueError("ChatML prompt does not round trip exactly")
    messages: list[dict[str, str]] = []
    if source_system is not None:
        system = source_system + ("\n\n" + instruction if instruction else "")
        messages.append({"role": "system", "content": system})
    elif instruction:
        messages.append({"role": "system", "content": instruction})
    messages.append({"role": "user", "content": user})
    return messages
