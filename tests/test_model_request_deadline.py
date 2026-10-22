"""Exercise the request budget through the pinned SDK and real local transport."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from openhands.sdk import LLM
from openhands.sdk.conversation import ConversationExecutionStatus
from openhands.sdk.llm import Message, TextContent

import senpai_agent.openhands_runner as runner
from senpai_agent.controller import _provider_failure
from senpai_agent.inference_heartbeat import ModelRequestDeadlineExceeded
from openhands_support import isolate_agent_discovery, runtime_config


@pytest.fixture
def local_provider(request):
    response = request.param
    accepted = []
    stopping = threading.Event()

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            accepted.append(time.monotonic())
            try:
                if response == "rate_limit" or (
                    response == "previous_rate_limit" and len(accepted) == 1
                ):
                    body = json.dumps({
                        "type": "error",
                        "error": {"type": "rate_limit_error", "message": "retry later"},
                    }).encode()
                    self.send_response(429)
                    self.send_header("Retry-After", "300")
                elif response == "previous_rate_limit" and len(accepted) == 2:
                    body = json.dumps({
                        "id": "local-success", "type": "message", "role": "assistant",
                        "model": "claude-sonnet-4-5", "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": "done"}],
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }).encode()
                    self.send_response(200)
                elif response == "trickle":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", "1000000")
                    self.end_headers()
                    # Stay inside the transport read timeout without completing.
                    while not stopping.is_set():
                        self.wfile.write(b" ")
                        self.wfile.flush()
                        stopping.wait(0.02)
                    return
                else:
                    stopping.wait(8)
                    return
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except ConnectionError:
                pass

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield response, f"http://127.0.0.1:{server.server_port}", accepted
    finally:
        stopping.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize(
    "local_provider", ["silent", "trickle", "rate_limit", "previous_rate_limit"],
    indirect=True,
)
def test_total_deadline_bounds_real_sdk_transport_and_retries(
    local_provider, tmp_path, monkeypatch,
):
    response, base_url, accepted = local_provider
    states = []
    interrupts = []

    class LocalLLM(LLM):
        def __init__(self, **kwargs):
            super().__init__(**{
                **kwargs, "base_url": base_url, "timeout": 1,
                "retry_min_wait": 8 if response == "rate_limit" else 0,
                "retry_max_wait": 8 if response == "rate_limit" else 0,
            })

    class Conversation:
        def __init__(self, agent, **kwargs):
            self.agent = agent
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.RUNNING,
            )

        def send_message(self, _prompt):
            pass

        async def arun(self):
            messages = [Message(role="user", content=[TextContent(text="local test")])]
            if response == "previous_rate_limit":
                await self.agent.llm.acompletion(messages)
            # An enclosing request scope models nested auth/fallback calls. The
            # earliest expired scope must retain the inner request's guidance.
            with self.agent.llm._request_scope():
                await self.agent.llm.acompletion(messages)

        def interrupt(self):
            interrupts.append(True)

        def close(self):
            pass

    monkeypatch.setattr(runner, "LLM", LocalLLM)
    monkeypatch.setattr(runner, "LocalConversation", Conversation)
    isolate_agent_discovery(monkeypatch, runner)
    started = time.monotonic()
    with pytest.raises(ModelRequestDeadlineExceeded) as caught:
        runner.run_openhands(
            "local test", runtime_config(
                tmp_path, model="anthropic/claude-sonnet-4-5",
                llm_timeout_seconds=2, llm_num_retries=10, timeout_seconds=6,
            ),
            on_inference_state=lambda *state: states.append(state),
        )
    assert interrupts == [True]
    assert time.monotonic() - started < 5
    assert len(accepted) >= (2 if response == "silent" else 1)
    request_deadline = max(state[2] for state in states if state[2])
    assert all(at < request_deadline for at in accepted)
    attempts_after_cancellation = len(accepted)
    time.sleep(0.1)
    assert len(accepted) == attempts_after_cancellation
    assert states[-1] == (None, None, None)

    failure = _provider_failure(caught.value, time.time())
    assert failure is not None
    assert not failure.refund_inference_attempt
    if response == "rate_limit":
        assert len(accepted) == 1
        assert 295 < failure.retry_after <= 300
    else:
        # A completed earlier request's hint must not delay this request.
        assert failure.retry_after == 0
