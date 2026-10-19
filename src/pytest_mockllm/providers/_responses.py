"""OpenAI Responses SSE events for text and function-call outputs."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def response_events(completed: Any, configured: Any) -> Iterator[Any]:
    from openai.types import responses as types

    sequence = 0

    def event(cls: str, kind: str, **fields: Any) -> Any:
        nonlocal sequence
        result = getattr(types, cls)(type=kind, sequence_number=sequence, **fields)
        sequence += 1
        return result

    yield event(
        "ResponseCreatedEvent",
        "response.created",
        response=completed.model_copy(
            update={"output": [], "status": "in_progress", "usage": None}
        ),
    )
    for index, item in enumerate(completed.output):
        is_tool = item.type == "function_call"
        updates: dict[str, Any] = {"status": "in_progress"}
        if is_tool:
            updates["arguments"] = ""
        else:
            updates["content"] = []
        initial = item.model_copy(update=updates)
        yield event(
            "ResponseOutputItemAddedEvent",
            "response.output_item.added",
            output_index=index,
            item=initial,
        )
        common = {"output_index": index, "item_id": item.id}
        if is_tool:
            fragments = configured.tool_call_chunks if index == 0 else None
            for fragment in fragments or [item.arguments]:
                yield event(
                    "ResponseFunctionCallArgumentsDeltaEvent",
                    "response.function_call_arguments.delta",
                    delta=fragment,
                    **common,
                )
            yield event(
                "ResponseFunctionCallArgumentsDoneEvent",
                "response.function_call_arguments.done",
                arguments=item.arguments,
                name=item.name,
                **common,
            )
        else:
            part = item.content[0]
            yield event(
                "ResponseContentPartAddedEvent",
                "response.content_part.added",
                content_index=0,
                part=part.model_copy(update={"text": ""}),
                **common,
            )
            for fragment in configured.stream_chunks or [part.text]:
                yield event(
                    "ResponseTextDeltaEvent",
                    "response.output_text.delta",
                    content_index=0,
                    delta=fragment,
                    logprobs=[],
                    **common,
                )
            yield event(
                "ResponseTextDoneEvent",
                "response.output_text.done",
                content_index=0,
                text=part.text,
                logprobs=[],
                **common,
            )
            yield event(
                "ResponseContentPartDoneEvent",
                "response.content_part.done",
                content_index=0,
                part=part,
                **common,
            )
        yield event(
            "ResponseOutputItemDoneEvent",
            "response.output_item.done",
            output_index=index,
            item=item,
        )
    yield event("ResponseCompletedEvent", "response.completed", response=completed)
