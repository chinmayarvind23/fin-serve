"""Carry only bounded W3C trace identity across private inference boundaries."""

import re
from collections.abc import AsyncGenerator, AsyncIterator

from opentelemetry.context import Context
from opentelemetry.trace import Span, StatusCode, Tracer, use_span
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

_PARENT = re.compile(r"00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}\Z")
_PROPAGATOR = TraceContextTextMapPropagator()


def parent_context(value: str | None) -> Context:
    """Ignore unsupported/oversized trace metadata and never import baggage or trace state."""
    if value is None or len(value) != 55 or _PARENT.fullmatch(value) is None:
        return Context()
    return _PROPAGATOR.extract({"traceparent": value}, context=Context())


def trace_headers() -> dict[str, str]:
    """Emit identity only; process-global baggage propagators cannot forward request content."""
    carrier: dict[str, str] = {}
    _PROPAGATOR.inject(carrier)
    value = carrier.get("traceparent")
    return {"traceparent": value} if value and _PARENT.fullmatch(value) else {}


async def next_in_span[T](iterator: AsyncIterator[T], span: Span) -> T:
    """Attach only during an await, so a suspended generator never leaks context to its caller."""
    with use_span(span, end_on_exit=False, record_exception=False, set_status_on_exception=False):
        return await anext(iterator)


async def traced_stream[T](
    output: AsyncGenerator[T], tracer: Tracer, name: str, parent: str | None
) -> AsyncGenerator[T]:
    """Span ownership includes upstream close; cancellation records metadata without exceptions."""
    span = tracer.start_span(name, context=parent_context(parent))
    outcome = "cancelled"
    try:
        while True:
            try:
                item = await next_in_span(output, span)
            except StopAsyncIteration:
                outcome = "success"
                break
            yield item
    except Exception:
        outcome = "failed"
        raise
    finally:
        try:
            with use_span(
                span, end_on_exit=False, record_exception=False, set_status_on_exception=False
            ):
                await output.aclose()
        except BaseException:
            outcome = "failed"
            raise
        finally:
            span.set_attribute("finserve.outcome", outcome)
            span.set_status(StatusCode.OK if outcome == "success" else StatusCode.ERROR)
            span.end()
