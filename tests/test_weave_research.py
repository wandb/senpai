"""Research workflows exercised through the published SDK HTTP contracts."""

import base64
import json
import os
import shlex
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from openhands.sdk.llm import Message
from pydantic import SecretStr, ValidationError
from weave.trace_server import trace_server_interface as tsi

from senpai_agent.hooks import terminal_policy
from senpai_agent.weave_research import (
    WandbReportDraftAction,
    WandbViewsAction,
    WandbViewsExecutor,
    WeaveResearchAction,
    WeaveResearchExecutor,
    configure_weave_credentials,
)


@pytest.fixture
def research_service(tmp_path):
    key = "controller-research-sentinel"
    calls = []
    state = SimpleNamespace(requests=calls, route=None)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            auth = self.headers.get("Authorization")
            calls.append((self.path, payload, auth))
            if auth != "Basic " + base64.b64encode(f"api:{key}".encode()).decode():
                status, response, content_type = 401, b"private project", "text/plain"
            else:
                status, response, content_type = state.route(self.path, payload)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(response)))
            if status == 302:
                self.send_header("Location", state.origin + "/credential-sink")
            self.end_headers()
            self.wfile.write(response)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.origin = f"http://127.0.0.1:{server.server_port}"
    state.workspace = tmp_path / "target"
    state.workspace.mkdir()
    state.directory = tmp_path / "state"
    state.weave = WeaveResearchExecutor(state.directory, state.workspace)
    state.views = WandbViewsExecutor(state.directory, state.workspace)
    configure_weave_credentials(
        SecretStr(key), trace_base_url=state.origin, wandb_base_url=state.origin
    )
    try:
        yield state
    finally:
        configure_weave_credentials(None)
        server.shutdown()
        server.server_close()
        thread.join()


def _json(value):
    return 200, json.dumps(value).encode(), "application/json"


def _rows(observation):
    return [
        json.loads(line) for line in Path(observation.path).read_text().splitlines()
    ]


def test_private_eval_calls_preserve_full_results_and_sdk_query_contract(
    research_service, monkeypatch
):
    service = research_service
    monkeypatch.setenv("WANDB_API_KEY", "unrelated-training-key")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("WEAVE_DEBUG_HTTP", "1")
    original_environment = dict(os.environ)
    records = [
        {
            "id": f"call-{i}",
            "inputs": {"dataset_row": i},
            "output": {"scores": {"quality": i / 2501}},
            "summary": {"usage": {"model": {"total_tokens": i}}},
        }
        for i in range(2501)
    ]

    def route(path, payload):
        assert path == "/calls/stream_query"
        request = tsi.CallsQueryReq.model_validate(payload)
        assert request.project_id == "other-team/private-evals"
        assert request.filter.parent_ids == ["evaluation-root"]
        assert request.query.model_dump(by_alias=True)["$expr"]["$gt"][1] == {
            "$literal": 5
        }
        assert request.columns == ["inputs", "output", "summary"]
        assert request.include_costs and request.include_feedback
        assert request.sort_by[0].field == "started_at"
        selected = records[
            request.offset : None
            if request.limit is None
            else request.offset + request.limit
        ]
        return (
            200,
            b"\n".join(json.dumps(row).encode() for row in selected),
            "application/x-ndjson",
        )

    service.route = route
    request = {
        "op": "calls",
        "project": "other-team/private-evals",
        "filter": {"parent_ids": ["evaluation-root"]},
        "query": {
            "$expr": {
                "$gt": [
                    {"$getField": "summary.usage.model.total_tokens"},
                    {"$literal": 5},
                ]
            }
        },
        "columns": ["inputs", "output", "summary"],
        "sort_by": [{"field": "started_at", "direction": "asc"}],
        "include_costs": True,
        "include_feedback": True,
    }
    complete = service.weave(WeaveResearchAction(request=request))
    assert complete.count == 2501
    assert _rows(complete) == records
    assert [payload["offset"] for _, payload, _ in service.requests] == [0, 1000, 2000]
    page = service.weave(
        WeaveResearchAction(request={**request, "offset": 2499, "limit": 2})
    )
    assert _rows(page) == records[-2:]
    assert page.next_offset == 2501
    assert dict(os.environ) == original_environment
    assert "controller-research-sentinel" not in Path(complete.path).read_text()
    assert len(complete.to_llm_content[0].text) < 1000
    assert not Path(complete.path).is_relative_to(service.workspace)

    # The provider gets a compact file receipt; all evidence remains retrievable
    # through bounded terminal reads beyond the provider's tool-text cap.
    assert Path(complete.path).stat().st_size > 50_000
    message = Message(
        role="tool",
        name="weave_research",
        tool_call_id="calls-export",
        content=complete.to_llm_content,
    )
    payloads = [
        message.to_chat_dict(
            cache_enabled=False,
            vision_enabled=False,
            function_calling_enabled=True,
            force_string_serializer=force_string,
            send_reasoning_content=False,
        )
        for force_string in (False, True)
    ]
    payloads.append(message.to_responses_dict(vision_enabled=False))
    for payload in payloads:
        serialized = json.dumps(payload)
        assert complete.path in serialized
        assert len(serialized) < 2_000
    command = f"sed -n '2501p' {shlex.quote(complete.path)}"
    assert terminal_policy(command, "student", service.workspace).allowed
    last_row = subprocess.check_output(
        ["sed", "-n", "2501p", complete.path], cwd=service.workspace, text=True
    )
    assert json.loads(last_row) == records[-1]


def test_call_costs_and_stats_use_server_side_queries(research_service):
    service = research_service

    def route(path, payload):
        if path == "/calls/query_stats":
            request = tsi.CallsQueryStatsReq.model_validate(payload)
            assert request.filter.trace_roots_only
            return _json({"count": 1800000})
        request = tsi.CallsQueryReq.model_validate(payload)
        assert request.filter.call_ids == ["eval"]
        assert request.include_costs
        return (
            200,
            json.dumps(
                {
                    "id": "eval",
                    "summary": {"weave": {"costs": {"model": {"total_cost": 1.25}}}},
                }
            ).encode(),
            "application/x-ndjson",
        )

    service.route = route
    call = service.weave(
        WeaveResearchAction(
            request={
                "op": "call",
                "project": "team/project",
                "call_id": "eval",
                "include_costs": True,
            }
        )
    )
    assert _rows(call)[0]["summary"]["weave"]["costs"]["model"]["total_cost"] == 1.25
    stats = service.weave(
        WeaveResearchAction(
            request={
                "op": "calls_stats",
                "project": "team/project",
                "filter": {"trace_roots_only": True},
            }
        )
    )
    assert stats.details["matching_calls"] == 1800000
    assert len(service.requests) == 2


def test_self_hosted_trace_path_prefix_preserves_authenticated_queries(
    research_service,
):
    service = research_service
    configure_weave_credentials(
        SecretStr("controller-research-sentinel"),
        trace_base_url=service.origin + "/traces/",
        wandb_base_url=service.origin,
    )

    def route(path, payload):
        assert path == "/traces/calls/query_stats"
        assert (
            tsi.CallsQueryStatsReq.model_validate(payload).project_id == "team/private"
        )
        return _json({"count": 42})

    service.route = route
    result = service.weave(
        WeaveResearchAction(
            request={
                "op": "calls_stats",
                "project": "team/private",
            }
        )
    )
    assert result.details["matching_calls"] == 42


def test_refs_objects_and_dataset_tables_remain_plain_data(research_service):
    service = research_service
    value = {
        "_type": "CustomScorer",
        "_class_name": "ExecutableLookingObject",
        "score": {"passed": True},
        "source": "raise RuntimeError('must stay data')",
        "accidental_key": "controller-research-sentinel",
    }
    sanitized_value = {**value, "accidental_key": "[REDACTED]"}

    def route(path, payload):
        if path == "/refs/read_batch":
            tsi.RefsReadBatchReq.model_validate(payload)
            return _json({"vals": [value]})
        if path == "/obj/read":
            request = tsi.ObjReadReq.model_validate(payload)
            assert request.digest == "v2"
            return _json({"obj": {"object_id": "scorer", "val": value}})
        request = tsi.TableQueryReq.model_validate(payload)
        assert request.offset == 10 and request.limit == 3
        return _json(
            {
                "rows": [
                    {
                        "digest": "row-10",
                        "val": {"question": "private row", "answer": 42},
                    }
                ]
            }
        )

    service.route = route
    ref = "weave:///team/private/object/scorer:v2"
    refs = service.weave(WeaveResearchAction(request={"op": "refs", "refs": [ref]}))
    assert _rows(refs) == [{"ref": ref, "value": sanitized_value}]
    obj = service.weave(
        WeaveResearchAction(
            request={
                "op": "object",
                "project": "team/private",
                "object_id": "scorer",
                "digest": "v2",
            }
        )
    )
    assert _rows(obj)[0]["val"] == sanitized_value
    table = service.weave(
        WeaveResearchAction(
            request={
                "op": "table",
                "project": "team/private",
                "digest": "data",
                "offset": 10,
                "limit": 3,
            }
        )
    )
    assert _rows(table)[0]["val"]["answer"] == 42
    assert table.next_offset is None


def test_dataset_table_exports_every_sdk_page(research_service):
    service = research_service
    records = [{"digest": f"row-{i}", "val": {"answer": i}} for i in range(7)]

    def route(path, payload):
        assert path == "/table/query"
        request = tsi.TableQueryReq.model_validate(payload)
        return _json({"rows": records[request.offset : request.offset + request.limit]})

    service.route = route
    result = service.weave(
        WeaveResearchAction(
            request={
                "op": "table",
                "project": "team/private",
                "digest": "data",
                "page_size": 3,
            }
        )
    )
    assert _rows(result) == records
    assert [payload["offset"] for _, payload, _ in service.requests] == [0, 3, 6]


def test_workspace_specs_preserve_panels_and_discover_step_axes(research_service):
    service = research_service
    spec = {
        "section": {
            "settings": {"xAxis": "train/tokens"},
            "panels": [{"config": {"x": "epoch", "y": ["loss"]}}],
            "accidental_key": {"xAxis": "controller-research-sentinel"},
        }
    }

    def route(path, payload):
        assert path == "/graphql"
        assert payload["variables"]["entityName"] == "another-team"
        assert payload["variables"]["viewType"] == "project-view"
        return _json(
            {
                "data": {
                    "project": {
                        "allViews": {
                            "edges": [
                                {
                                    "node": {
                                        "id": "workspace-id",
                                        "spec": json.dumps(spec),
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        )

    service.route = route
    result = service.views(
        WandbViewsAction(request={"op": "views", "project": "another-team/private"})
    )
    sanitized_spec = json.loads(
        json.dumps(spec).replace("controller-research-sentinel", "[REDACTED]")
    )
    assert _rows(result)[0]["spec"] == sanitized_spec
    assert result.details["step_axis_candidates"] == [
        "[REDACTED]",
        "epoch",
        "train/tokens",
    ]
    assert "controller-research-sentinel" not in result.to_llm_content[0].text


@pytest.mark.parametrize("readback_fails", [False, True])
def test_arbitrary_report_draft_preserves_spec_and_receipt(
    research_service, readback_fails
):
    service = research_service
    spec = {
        "version": 5,
        "width": "fluid",
        "blocks": [
            {"type": "heading", "level": 1, "children": [{"text": "Study"}]},
            {
                "type": "panel-grid",
                "runSets": [
                    {
                        "entity": "other-team",
                        "project": "comparison",
                        "filters": {"key": "config.lr", "value": 0.02},
                    }
                ],
                "panels": [
                    {"viewType": "Vega2", "config": {"customChart": "custom-panel"}}
                ],
            },
            {"type": "image", "url": "https://example.com/figure.png"},
        ],
        "panelSettings": {"custom": True},
    }

    def route(path, payload):
        assert path == "/graphql"
        if "mutation ResearchDraft" in payload["query"]:
            assert 'type: "runs/draft"' in payload["query"]
            assert "id" not in payload["variables"]
            assert json.loads(payload["variables"]["spec"]) == spec
            return _json(
                {
                    "data": {
                        "upsertView": {"view": {"id": "new-draft"}, "inserted": True}
                    }
                }
            )
        assert payload["variables"] == {"reportId": "new-draft"}
        if readback_fails:
            return 503, b"temporary unavailable", "text/plain"
        return _json(
            {
                "data": {
                    "view": {
                        "id": "new-draft",
                        "type": "runs/draft",
                        "spec": json.dumps(spec),
                        "displayName": "Study",
                        "description": "",
                        "project": {"name": "private", "entityName": "team"},
                    }
                }
            }
        )

    service.route = route
    result = service.views(
        WandbReportDraftAction(
            request={
                "op": "create_report_draft",
                "project": "team/private",
                "title": "Study",
                "spec": spec,
            }
        )
    )
    assert _rows(result)[0]["spec"] == spec
    assert result.details["view_id"] == "new-draft"
    assert result.details["verified"] is not readback_fails
    assert result.is_error is readback_fails
    assert len(service.requests) == 2


def test_uncertain_report_creation_does_not_retry_mutation(research_service):
    service = research_service
    service.route = lambda path, payload: (503, b"temporary unavailable", "text/plain")
    with pytest.raises(RuntimeError, match="may have succeeded") as caught:
        service.views(
            WandbReportDraftAction(
                request={
                    "op": "create_report_draft",
                    "project": "team/private",
                    "title": "Research",
                    "spec": {"version": 5, "blocks": []},
                }
            )
        )
    assert len(service.requests) == 1
    assert service.requests[0][1]["variables"]["name"] in str(caught.value)
    assert "Do not retry" in str(caught.value)
    assert not list(service.directory.rglob("*.jsonl"))


@pytest.mark.parametrize("failure", ["redirect", "error", "partial"])
def test_failed_transports_do_not_follow_redirects_or_leave_partial_exports(
    research_service, failure
):
    service = research_service

    def route(path, payload):
        if failure == "redirect":
            return 302, b"", "text/plain"
        if failure == "error":
            return 401, b"controller-research-sentinel", "text/plain"
        return 200, b'{"id":"first"}\ninvalid-json', "application/x-ndjson"

    service.route = route
    with pytest.raises((RuntimeError, ValueError)) as caught:
        service.weave(
            WeaveResearchAction(request={"op": "calls", "project": "team/private"})
        )
    assert "controller-research-sentinel" not in str(caught.value)
    assert len(service.requests) == 1
    assert not list(service.directory.rglob("*.jsonl"))


def test_invalid_sdk_queries_fail_before_network_or_export(research_service):
    service = research_service
    with pytest.raises(ValidationError):
        service.weave(
            WeaveResearchAction(
                request={
                    "op": "calls",
                    "project": "team/private",
                    "filter": {"unrecognized_identity_field": "anything"},
                }
            )
        )
    assert not service.requests
    assert not list(service.directory.rglob("*.jsonl"))


def test_tools_reject_target_workspace_exports(tmp_path):
    for executor in (WeaveResearchExecutor, WandbViewsExecutor):
        with pytest.raises(ValueError, match="outside the target workspace"):
            executor(tmp_path / "target" / "state", tmp_path / "target")
