"""Record through real SDK parsers; replay must never reach the HTTP transport."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import yaml

from pytest_mockllm.recording import CassetteMissError, LLMRecorder, RecordingError


def http_module(provider):
    sdk = importlib.import_module(f"{provider}._base_client")
    return getattr(sdk, "httpx2", getattr(sdk, "httpx", None))


def payload(provider, text="hello"):
    if provider == "openai":
        return {
            "id": "chat_1",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "test-model",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }


def streaming_body(provider, tool=False):
    if provider == "openai":
        chunks = []
        for index, text in enumerate(["hel", "lo"] if not tool else ['{"city":', '"Paris"}']):
            delta = {"content": text}
            if tool:
                delta = {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1" if index == 0 else None,
                            "type": "function",
                            "function": {
                                "name": "weather" if index == 0 else None,
                                "arguments": text,
                            },
                        }
                    ]
                }
            chunks.append(
                {
                    "id": "chat_1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": ("tool_calls" if tool else "stop") if index else None,
                        }
                    ],
                }
            )
        return "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
    message = payload(provider)
    message.update(content=[], stop_reason=None)
    events = [
        {"type": "message_start", "message": message},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "tool_1", "name": "weather", "input": {}}
            if tool
            else {"type": "text", "text": ""},
        },
    ]
    for text in ['{"city":', '"Paris"}'] if tool else ["hel", "lo"]:
        events.append(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": text}
                if tool
                else {"type": "text_delta", "text": text},
            }
        )
    events += [
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use" if tool else "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 2},
        },
        {"type": "message_stop"},
    ]
    return "".join("event: " + e["type"] + "\ndata: " + json.dumps(e) + "\n\n" for e in events)


def make_client(provider, handler, asynchronous=False, key="private-record-key", base_url=None):
    http = http_module(provider)
    sdk = importlib.import_module(provider)
    cls = getattr(
        sdk, ("Async" if asynchronous else "") + ("OpenAI" if provider == "openai" else "Anthropic")
    )
    http_client = (http.AsyncClient if asynchronous else http.Client)(
        transport=http.MockTransport(handler)
    )
    return cls(
        api_key=key,
        http_client=http_client,
        max_retries=0,
        base_url=base_url or "https://provider.invalid/v1",
    )


def call(client, provider, prompt="hi", **kwargs):
    params = {"model": "test-model", "messages": [{"role": "user", "content": prompt}], **kwargs}
    if provider == "anthropic":
        return client.messages.create(max_tokens=64, **params)
    return client.chat.completions.create(**params)


def dump(result):
    return result.model_dump(mode="json")


def forbidden(request):
    pytest.fail("Replay reached HTTP transport")


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("mode", ["once", "new_episodes", "all"])
def test_sync_record_replay_offline(tmp_path, provider, mode):
    path = tmp_path / "test.yaml"
    requests = []
    http = http_module(provider)

    def handle(request):
        requests.append(request)
        return http.Response(
            200,
            json=payload(provider),
            headers={"x-request-id" if provider == "openai" else "request-id": "req_1"},
        )

    with make_client(provider, handle) as client, LLMRecorder(path, record_mode=mode):
        original = call(client, provider)
    assert len(requests) == 1
    assert "private-record-key" not in path.read_text()
    assert "hello" in path.read_text()
    with make_client(provider, forbidden, key="different-offline-key") as client, LLMRecorder(path):
        replay = call(client, provider)
    assert type(original) is type(replay)
    assert dump(original) == dump(replay)
    assert replay._request_id == "req_1"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("tool", [False, True])
@pytest.mark.asyncio
async def test_async_stream_roundtrip(tmp_path, provider, tool):
    path = tmp_path / "stream.yaml"
    http = http_module(provider)

    def handle(request):
        return http.Response(
            200,
            content=streaming_body(provider, tool),
            headers={"content-type": "text/event-stream"},
        )

    async with make_client(provider, handle, asynchronous=True) as client:
        with LLMRecorder(path, record_mode="once"):
            stream = await call(client, provider, stream=True)
            original = [dump(event) async for event in stream]
            await stream.close()
    async with make_client(provider, forbidden, asynchronous=True) as client:
        with LLMRecorder(path):
            stream = await call(client, provider, stream=True)
            replay = [dump(event) async for event in stream]
            await stream.close()
    assert original == replay
    assert len(replay) >= 2
    assert yaml.safe_load(path.read_text())["interactions"][0]["response"]["encoding"] == "text"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("tool", [False, True])
def test_sync_stream_roundtrip(tmp_path, provider, tool):
    path = tmp_path / "stream.yaml"
    http = http_module(provider)

    def handle(request):
        return http.Response(
            200,
            content=streaming_body(provider, tool),
            headers={"content-type": "text/event-stream"},
        )

    with make_client(provider, handle) as client, LLMRecorder(path, record_mode="once"):
        with call(client, provider, stream=True) as stream:
            original = [dump(event) for event in stream]
    with make_client(provider, forbidden) as client, LLMRecorder(path):
        with call(client, provider, stream=True) as stream:
            assert [dump(event) for event in stream] == original


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.asyncio
async def test_async_nonstream_roundtrip(tmp_path, provider):
    http = http_module(provider)
    path = tmp_path / "async.yaml"
    async with make_client(
        provider, lambda r: http.Response(200, json=payload(provider)), True
    ) as client:
        with LLMRecorder(path, record_mode="once"):
            original = await call(client, provider)
    async with make_client(provider, forbidden, True) as client:
        with LLMRecorder(path):
            assert dump(await call(client, provider)) == dump(original)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_modes_matching_and_repeated_calls(tmp_path, provider):
    path = tmp_path / "modes.yaml"
    http = http_module(provider)
    requests = []

    def handle(request):
        requests.append(request)
        return http.Response(200, json=payload(provider, str(len(requests))))

    with make_client(provider, handle) as client:
        with LLMRecorder(path, record_mode="once"):
            first = dump(call(client, provider, extra_body={"temperature": 0.1}))
            second = dump(call(client, provider, extra_body={"temperature": 0.1}))
        before = path.read_bytes()
        with LLMRecorder(path, record_mode="once"):
            assert dump(call(client, provider, extra_body={"temperature": 0.1})) == first
            assert dump(call(client, provider, extra_body={"temperature": 0.1})) == second
            with pytest.raises(CassetteMissError):
                call(client, provider, extra_body={"temperature": 0.1})
            with pytest.raises(CassetteMissError):
                call(client, provider, extra_body={"temperature": 0.2})
        assert path.read_bytes() == before
        assert len(requests) == 2
        with LLMRecorder(path, record_mode="new_episodes"):
            assert dump(call(client, provider, extra_body={"temperature": 0.1})) == first
            call(client, provider, prompt="new")
        assert len(requests) == 3
        assert len(yaml.safe_load(path.read_text())["interactions"]) == 3
        with LLMRecorder(path, record_mode="all"):
            call(client, provider, prompt="replacement")
        assert len(requests) == 4
        assert len(yaml.safe_load(path.read_text())["interactions"]) == 1


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_missing_mismatched_and_unsupported_fail_closed(tmp_path, provider):
    path = tmp_path / "missing.yaml"
    with make_client(provider, forbidden) as client, LLMRecorder(path):
        with pytest.raises(CassetteMissError, match="Network access refused"):
            call(client, provider)
        with pytest.raises(RecordingError, match="Unsupported"):
            client.models.list()
    assert not path.exists()


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_errors_replay_as_official_sdk_errors(tmp_path, provider):
    http = http_module(provider)
    sdk = importlib.import_module(provider)
    path = tmp_path / "error.yaml"

    def handle(request):
        return http.Response(
            429,
            json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}},
            headers={"retry-after": "0"},
        )

    with (
        make_client(provider, handle) as client,
        LLMRecorder(path, record_mode="once"),
        pytest.raises(sdk.RateLimitError) as original,
    ):
        call(client, provider)
    with (
        make_client(provider, forbidden) as client,
        LLMRecorder(path),
        pytest.raises(sdk.RateLimitError) as replay,
    ):
        call(client, provider)
    assert replay.value.status_code == original.value.status_code == 429
    assert replay.value.body == original.value.body


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_redaction_in_headers_query_nested_body_and_echo(tmp_path, provider):
    http = http_module(provider)
    path = tmp_path / "secret.yaml"

    def handle(request):
        data = payload(provider, "echo private-record-key and Bearer short-token")
        data["metadata"] = {
            "api_key": "response-secret",
            "nested": [{"Authorization": "another-secret"}],
        }
        return http.Response(
            200,
            json=data,
            headers={"Set-Cookie": "session=private", "authorization": "response-auth"},
        )

    with make_client(provider, handle, base_url="https://provider.invalid/v1") as client:
        # Query params on SDK base URLs vary; explicit extra_query is stable.
        with LLMRecorder(path, record_mode="once"):
            call(
                client,
                provider,
                extra_query={"key": "query-secret"},
                extra_body={"metadata": {"api_key": "body-secret"}},
                extra_headers={"X-Api-Key": "custom-secret"},
            )
    text = path.read_text()
    for secret in [
        "private-record-key",
        "short-token",
        "response-secret",
        "another-secret",
        "session=private",
        "response-auth",
        "query-secret",
        "body-secret",
        "custom-secret",
    ]:
        assert secret not in text
    assert "[REDACTED]" in text


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_failed_stream_does_not_create_cassette(tmp_path, provider):
    http = http_module(provider)
    path = tmp_path / "incomplete.yaml"

    class Broken(http.SyncByteStream):
        def __iter__(self):
            yield b"data: partial\n\n"
            raise http.ReadError("disconnected")

    def handle(request):
        return http.Response(200, stream=Broken(), headers={"content-type": "text/event-stream"})

    with (
        make_client(provider, handle) as client,
        LLMRecorder(path, record_mode="once"),
        pytest.raises(Exception),
    ):
        call(client, provider, stream=True)
    assert not path.exists()


def test_cleanup_nested_context_and_legacy_modes(tmp_path):
    import openai

    original = openai._base_client.SyncAPIClient.request
    with pytest.raises(ValueError, match="body failed"), LLMRecorder(tmp_path / "empty.yaml"):
        with pytest.raises(RecordingError, match="Nested"):
            with LLMRecorder(tmp_path / "nested.yaml"):
                pass
        raise ValueError("body failed")
    assert openai._base_client.SyncAPIClient.request is original
    with (
        pytest.warns(DeprecationWarning),
        LLMRecorder(tmp_path / "legacy.yaml", mode="none") as recorder,
    ):
        assert not recorder.is_recording
        assert openai._base_client.SyncAPIClient.request is original
    for old, new in [("auto", "once"), ("record", "all"), ("replay", "none")]:
        with pytest.warns(DeprecationWarning):
            assert LLMRecorder(tmp_path / "legacy.yaml", mode=old).record_mode == new
    with pytest.raises(ValueError):
        LLMRecorder(tmp_path / "bad.yaml", record_mode="typo")


def test_corrupt_file_does_not_enable_network_and_restores_patches(tmp_path):
    path = tmp_path / "corrupt.yaml"
    path.write_text('name: old\nversion: "1.0"\ninteractions: []\n')
    with pytest.raises(RecordingError, match="version"), LLMRecorder(path, record_mode="once"):
        pytest.fail("must not enter")
    with LLMRecorder(tmp_path / "other.yaml"):
        pass


def test_unrelated_http_passes_through(tmp_path):
    import httpx

    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text="unrelated"))
    ) as client:
        with LLMRecorder(tmp_path / "empty.yaml"):
            assert client.get("https://unrelated.invalid").text == "unrelated"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_sdk_retry_sequence_roundtrips(tmp_path, provider):
    http = http_module(provider)
    path = tmp_path / "retry.yaml"
    count = 0

    def handle(request):
        nonlocal count
        count += 1
        if count == 1:
            return http.Response(
                429,
                json={"type": "error", "error": {"type": "rate_limit_error", "message": "retry"}},
                headers={"retry-after-ms": "1"},
            )
        return http.Response(200, json=payload(provider))

    with (
        make_client(provider, handle).with_options(max_retries=1) as client,
        LLMRecorder(path, record_mode="once"),
    ):
        original = call(client, provider)
    assert count == 2
    with make_client(provider, forbidden).with_options(max_retries=1) as client, LLMRecorder(path):
        assert dump(call(client, provider)) == dump(original)
    assert len(yaml.safe_load(path.read_text())["interactions"]) == 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_anthropic_stream_manager(tmp_path, asynchronous):
    http = http_module("anthropic")
    path = tmp_path / "manager.yaml"

    def handle(request):
        return http.Response(
            200, content=streaming_body("anthropic"), headers={"content-type": "text/event-stream"}
        )

    results = []
    for mode, handler in [("once", handle), ("none", forbidden)]:
        client = make_client("anthropic", handler, asynchronous)
        with LLMRecorder(path, record_mode=mode):
            manager = client.messages.stream(
                model="test-model", messages=[{"role": "user", "content": "hi"}], max_tokens=64
            )
            if asynchronous:
                async with manager as stream:
                    text = "".join([part async for part in stream.text_stream])
                    message = await stream.get_final_message()
                await client.close()
            else:
                with manager as stream:
                    text = "".join(stream.text_stream)
                    message = stream.get_final_message()
                client.close()
            results.append((text, dump(message)))
    assert results[0] == results[1]
    assert results[0][0] == "hello"


def response_payload(text="hello"):
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "test-model",
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "temperature": 1,
        "top_p": 1,
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_openai_responses_parse_roundtrip(tmp_path, asynchronous):
    from pydantic import BaseModel

    class Answer(BaseModel):
        answer: str

    path = tmp_path / "responses.yaml"
    http = http_module("openai")
    results = []
    for mode, handler in [
        ("once", lambda r: http.Response(200, json=response_payload('{"answer":"hello"}'))),
        ("none", forbidden),
    ]:
        client = make_client("openai", handler, asynchronous)
        with LLMRecorder(path, record_mode=mode):
            response = client.responses.parse(model="test-model", input="hi", text_format=Answer)
            if asynchronous:
                response = await response
            assert response.output_parsed == Answer(answer="hello")
            results.append(dump(response))
        if asynchronous:
            await client.close()
        else:
            client.close()
    assert results[0] == results[1]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_openai_responses_sse_roundtrip(tmp_path, asynchronous):
    path = tmp_path / "responses-stream.yaml"
    http = http_module("openai")
    events = [
        {
            "type": "response.output_text.delta",
            "sequence_number": 0,
            "output_index": 0,
            "content_index": 0,
            "item_id": "msg_1",
            "delta": "hello",
            "logprobs": [],
        },
        {"type": "response.completed", "sequence_number": 1, "response": response_payload()},
    ]

    def handle(request):
        return http.Response(
            200,
            content="".join(
                "event: " + e["type"] + "\ndata: " + json.dumps(e) + "\n\n" for e in events
            ),
            headers={"content-type": "text/event-stream"},
        )

    results = []
    for mode, handler in [("once", handle), ("none", forbidden)]:
        client = make_client("openai", handler, asynchronous)
        with LLMRecorder(path, record_mode=mode):
            stream = client.responses.create(model="test-model", input="hi", stream=True)
            if asynchronous:
                stream = await stream
                result = [dump(e) async for e in stream]
                await client.close()
            else:
                result = [dump(e) for e in stream]
                client.close()
            results.append(result)
    assert results[0] == results[1]
    assert results[1][0]["delta"] == "hello"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_actual_local_http_roundtrip(tmp_path, provider, monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    import socket

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps(payload(provider)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    sdk = importlib.import_module(provider)
    cls = getattr(sdk, "OpenAI" if provider == "openai" else "Anthropic")
    path = tmp_path / "local.yaml"
    try:
        with cls(api_key="local-only", base_url=base_url, max_retries=0) as client:
            with LLMRecorder(path, record_mode="once"):
                original = call(client, provider)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert len(requests) == 1

    def no_connect(*args, **kwargs):
        pytest.fail("Replay attempted a socket connection")

    monkeypatch.setattr(socket.socket, "connect", no_connect)
    with cls(api_key="offline-only", base_url=base_url, max_retries=0) as client:
        with LLMRecorder(path):
            assert dump(call(client, provider)) == dump(original)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_custom_send_override_fails_before_network(tmp_path, provider):
    http = http_module(provider)
    sdk = importlib.import_module(provider)

    class Custom(http.Client):
        def send(self, *args, **kwargs):
            pytest.fail("custom send bypassed the recorder")

    cls = getattr(sdk, "OpenAI" if provider == "openai" else "Anthropic")
    with cls(api_key="fake", http_client=Custom(), max_retries=0) as client:
        with LLMRecorder(tmp_path / "unsupported.yaml", record_mode="once"):
            with pytest.raises(RecordingError, match="overriding send"):
                call(client, provider)


def test_atomic_save_preserves_previous_file_on_failure(tmp_path, monkeypatch):
    import pytest_mockllm.recording as recording

    path = tmp_path / "existing.yaml"
    path.write_text("previous contents")
    cassette = recording.Cassette(name="test")

    def failure(*args):
        raise OSError("replacement failed")

    monkeypatch.setattr(recording.os, "replace", failure)
    with pytest.raises(OSError, match="replacement failed"):
        cassette.save(path)
    assert path.read_text() == "previous contents"
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_stream_credentials_are_redacted(tmp_path, provider):
    http = http_module(provider)
    path = tmp_path / "secret-stream.yaml"
    body = streaming_body(provider).replace('"hel"', '"private-record-key"')

    def handle(request):
        return http.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream", "authorization": "response-secret"},
        )

    with make_client(provider, handle) as client, LLMRecorder(path, record_mode="once"):
        list(call(client, provider, stream=True))
    contents = path.read_text()
    assert "private-record-key" not in contents
    assert "response-secret" not in contents
    with make_client(provider, forbidden) as client, LLMRecorder(path):
        events = list(call(client, provider, stream=True))
    assert "[REDACTED]" in json.dumps([dump(e) for e in events])


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_embedding_record_replay(tmp_path, asynchronous):
    http = http_module("openai")
    path = tmp_path / "embedding.yaml"
    body = {
        "object": "list",
        "model": "text-embedding-3-small",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
            {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
        ],
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }
    results = []
    for mode, handler in [("once", lambda r: http.Response(200, json=body)), ("none", forbidden)]:
        client = make_client("openai", handler, asynchronous)
        with LLMRecorder(path, record_mode=mode):
            result = client.embeddings.create(
                model="text-embedding-3-small", input=["one", "two"], dimensions=2
            )
            if asynchronous:
                result = await result
            results.append(dump(result))
        if asynchronous:
            await client.close()
        else:
            client.close()
    assert results[0] == results[1]
    assert results[1]["data"][1]["embedding"] == [0.3, 0.4]
