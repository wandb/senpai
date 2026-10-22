"""Exercise the request budget through the pinned SDK and real local transport."""

import asyncio
import time
from contextlib import suppress

import pytest
from openhands.sdk import LLM
from openhands.sdk.llm import Message, TextContent
from pydantic import SecretStr

from senpai_agent.inference_heartbeat import (
    InferenceHeartbeat,
    ModelRequestDeadlineExceeded,
)
from senpai_agent.openhands_runner import arun_conversation


@pytest.mark.parametrize("response", ["silent", "trickle"])
def test_total_deadline_bounds_real_sdk_transport_and_retries(response):
    async def run():
        accepted = []
        handlers = set()
        states = []

        async def provider(reader, writer):
            handlers.add(asyncio.current_task())
            accepted.append(time.monotonic())
            try:
                headers = await reader.readuntil(b"\r\n\r\n")
                length = next(
                    int(line.split(b":", 1)[1])
                    for line in headers.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                )
                await reader.readexactly(length)
                if response == "silent":
                    await reader.read()
                else:
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                        b"Content-Length: 1000000\r\n\r\n"
                    )
                    # Bytes keep arriving inside the transport read timeout,
                    # but the provider never completes its response.
                    while True:
                        writer.write(b" ")
                        await writer.drain()
                        await asyncio.sleep(0.02)
            except (ConnectionError, asyncio.IncompleteReadError):
                pass
            finally:
                writer.close()
                with suppress(ConnectionError):
                    await writer.wait_closed()
                handlers.discard(asyncio.current_task())

        server = await asyncio.start_server(provider, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        heartbeat = InferenceHeartbeat(
            lambda *state: states.append(state), timeout_seconds=2.5,
        )
        llm = LLM(
            model="anthropic/claude-sonnet-4-5",
            api_key=SecretStr("local-test-key"),
            base_url=f"http://127.0.0.1:{port}",
            timeout=1,
            num_retries=10,
            retry_min_wait=0,
            retry_max_wait=0,
        )
        llm.set_request_scope(heartbeat.request)

        class Conversation:
            interrupts = 0
            task = None

            async def arun(self):
                self.task = asyncio.current_task()
                await llm.acompletion([
                    Message(role="user", content=[TextContent(text="local test")]),
                ])

            def interrupt(self):
                self.interrupts += 1

        conversation = Conversation()
        started = time.monotonic()
        try:
            with pytest.raises(ModelRequestDeadlineExceeded):
                await asyncio.wait_for(
                    arun_conversation(
                        conversation,
                        3600,
                        request_deadline=heartbeat.deadline,
                    ),
                    timeout=6,
                )
            assert conversation.interrupts == 1
            assert time.monotonic() - started < 5
            assert len(accepted) >= (2 if response == "silent" else 1)
            request_deadline = next(state[2] for state in states if state[2])
            assert all(at < request_deadline for at in accepted)
            attempts_after_cancellation = len(accepted)
            await asyncio.sleep(0.1)
            assert len(accepted) == attempts_after_cancellation
        finally:
            if conversation.task is not None and not conversation.task.done():
                conversation.task.cancel()
                await asyncio.gather(conversation.task, return_exceptions=True)
            heartbeat.close()
            server.close()
            await server.wait_closed()
            pending = tuple(handlers)
            for handler in pending:
                handler.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        assert states[-1] == (None, None, None)

    asyncio.run(run())
