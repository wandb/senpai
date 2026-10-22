import asyncio
import threading
import time

import pytest
from litellm.exceptions import APIConnectionError
from openhands.sdk import LLM
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.llm.exceptions import LLMServiceUnavailableError
from pydantic import SecretStr

from senpai_agent.inference_heartbeat import InferenceHeartbeat


def wait_until(predicate, timeout: float = 2) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise TimeoutError("inference heartbeat condition was not met")


def test_overlapping_requests_share_a_live_inference_heartbeat():
    entered = {name: threading.Event() for name in ("first", "second")}
    release = {name: threading.Event() for name in ("first", "second")}
    states = []
    publisher_threads = set()
    state_lock = threading.Lock()

    def callback(started_at, heartbeat_at, deadline):
        with state_lock:
            states.append((started_at, heartbeat_at, deadline))
            publisher_threads.add(threading.get_ident())

    def state_snapshot():
        with state_lock:
            return len(states), states[-1] if states else None

    heartbeat = InferenceHeartbeat(
        callback, timeout_seconds=60, interval_seconds=0.01,
    )
    results = []

    def request(name):
        with heartbeat.request():
            entered[name].set()
            assert release[name].wait(2)
            results.append(name)

    first = threading.Thread(
        target=request,
        args=("first",),
    )
    second = threading.Thread(
        target=request,
        args=("second",),
    )

    first.start()
    assert entered["first"].wait(2)
    second.start()
    assert entered["second"].wait(2)
    wait_until(lambda: state_snapshot()[1] is not None)
    with state_lock:
        first_started_at = states[0][0]
        state_count = len(states)
        previous_heartbeat = states[-1][1]
    wait_until(
        lambda: (
            (snapshot := state_snapshot())[0] > state_count
            and (state := snapshot[1]) is not None
            and state[1] is not None
            and state[1] > previous_heartbeat
        )
    )
    with state_lock:
        active_states = [state for state in states if state[0] is not None]
    assert len({state[0] for state in active_states}) == 1
    assert len({state[2] for state in active_states}) == 1
    first_deadline = active_states[0][2]
    assert time.monotonic() < first_deadline <= time.monotonic() + 60

    release["first"].set()
    first.join(2)
    wait_until(
        lambda: (
            (state := state_snapshot()[1]) is not None
            and state[0] is not None
            and state[0] != first_started_at
        )
    )
    assert state_snapshot()[1][2] > first_deadline

    release["second"].set()
    second.join(2)
    heartbeat.close()
    with state_lock:
        assert states[-1] == (None, None, None)
        assert len(publisher_threads) == 1
        assert not publisher_threads.intersection({first.ident, second.ident})
    assert sorted(results) == ["first", "second"]


def test_slow_publication_does_not_block_request_tracking():
    publishing = threading.Event()
    release_publication = threading.Event()
    entered = {name: threading.Event() for name in ("first", "second")}
    states = []

    def publish(*state):
        states.append(state)
        publishing.set()
        assert release_publication.wait(2)

    heartbeat = InferenceHeartbeat(publish, timeout_seconds=60)

    def request(name):
        with heartbeat.request():
            entered[name].set()

    first = threading.Thread(target=request, args=("first",))
    second = threading.Thread(target=request, args=("second",))
    first.start()
    assert entered["first"].wait(2)
    assert publishing.wait(2)

    try:
        second.start()
        assert entered["second"].wait(2)
        first.join(2)
        second.join(2)
    finally:
        release_publication.set()

    heartbeat.close()
    assert states[-1] == (None, None, None)


def test_async_inference_failure_clears_the_heartbeat():
    states = []
    heartbeat = InferenceHeartbeat(lambda *state: states.append(state), timeout_seconds=60)

    async def fail():
        with heartbeat.request():
            raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        asyncio.run(fail())

    heartbeat.close()
    assert states[0][0] is not None
    assert states[-1] == (None, None, None)


def test_request_completed_before_expiry_is_observed_does_not_arm_cleanup():
    states = []
    heartbeat = InferenceHeartbeat(
        lambda *state: states.append(state), timeout_seconds=0.01,
    )
    with heartbeat.request():
        time.sleep(0.02)
    assert heartbeat.deadline() is None
    heartbeat.close()
    assert states[-1] == (None, None, None)


@pytest.mark.parametrize("deep", [False, True])
def test_canonical_llm_copies_share_the_heartbeat_scope(monkeypatch, deep):
    states = []
    heartbeat = InferenceHeartbeat(lambda *state: states.append(state), timeout_seconds=60)
    llm = LLM(
        model="anthropic/claude-opus-4-8",
        api_key=SecretStr("test-key"),
    )
    llm.set_request_scope(heartbeat.request)
    copied_llm = llm.model_copy(deep=deep)

    def fail(*_args, **_kwargs):
        raise RuntimeError("stop before transport")

    monkeypatch.setattr(LLM, "_prepare_completion_params", fail)

    with pytest.raises(RuntimeError, match="stop before transport"):
        copied_llm.completion([])

    heartbeat.close()
    assert states[0][0] is not None
    assert states[-1] == (None, None, None)


def test_close_reports_a_publisher_failure():
    def fail(*_state):
        raise RuntimeError("lease publication failed")

    heartbeat = InferenceHeartbeat(fail, timeout_seconds=60)
    with heartbeat.request():
        pass

    with pytest.raises(RuntimeError, match="lease publication failed"):
        heartbeat.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_model_retries_keep_the_original_request_deadline(monkeypatch, asynchronous):
    """SDK retries stay inside one request budget across both transport APIs."""
    states = []
    attempts = []
    heartbeat = InferenceHeartbeat(
        lambda *state: states.append(state), timeout_seconds=60,
    )
    llm = LLM(
        model="openai/gpt-4o-mini",
        api_key=SecretStr("test-key"),
        num_retries=2,
        retry_min_wait=0,
        retry_max_wait=0,
    )
    llm.set_request_scope(heartbeat.request)

    def fail_transport(**_kwargs):
        wait_until(lambda: bool(states))
        attempts.append(states[-1])
        raise APIConnectionError(
            message="test transport failed", llm_provider="openai", model=llm.model,
        )

    async def fail_async_transport(**kwargs):
        return fail_transport(**kwargs)

    monkeypatch.setattr(
        "openhands.sdk.llm.llm.litellm_completion", fail_transport,
    )
    monkeypatch.setattr(
        "openhands.sdk.llm.llm.litellm_acompletion", fail_async_transport,
    )
    messages = [Message(role="user", content=[TextContent(text="test")])]
    try:
        with pytest.raises(LLMServiceUnavailableError):
            if asynchronous:
                asyncio.run(llm.acompletion(messages))
            else:
                llm.completion(messages)
    finally:
        heartbeat.close()

    assert len(attempts) == 2
    assert attempts[0][0] is not None
    assert attempts[0][2] is not None
    assert attempts[1][0] == attempts[0][0]
    assert attempts[1][2] == attempts[0][2]
    assert states[-1] == (None, None, None)
