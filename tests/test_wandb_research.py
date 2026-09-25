import base64
import hashlib
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import wandb
from pydantic import SecretStr, ValidationError
from wandb.apis.public.service_api import ServiceApi
from wandb.proto import wandb_api_pb2 as pb
from wandb.sdk import wandb_login

from senpai_agent.wandb_research import (
    WandbResearchAction,
    WandbResearchTool,
    configure_wandb_credentials,
)

RESEARCH_KEY = "a" * 40


def run_node(name="r1"):
    return {
        "id": f"storage-{name}",
        "name": name,
        "displayName": f"Experiment {name}",
        "state": "finished",
        "config": '{"lr":{"value":0.01},"_wandb":{"value":{"version":1}}}',
        "summaryMetrics": '{"loss":0.125,"nested":{"accuracy":0.98}}',
        "systemMetrics": '{"gpu.memory":42}',
        "tags": ["research"],
        "createdAt": "2026-09-01T00:00:00Z",
        "heartbeatAt": "2026-09-01T01:00:00Z",
        "group": "baseline",
        "jobType": "train",
        "notes": "private evidence",
        "commit": "abc123",
        "description": "Baseline run",
        "sweepName": None,
    }


def connection(nodes, *, more=False, cursor=None):
    return {
        "edges": [
            {"node": node, "cursor": str(index)} for index, node in enumerate(nodes)
        ],
        "pageInfo": {"endCursor": cursor, "hasNextPage": more},
    }


@pytest.fixture
def research(tmp_path, monkeypatch):
    """Replace only SDK network/core transport, keeping SDK models and pagination."""
    monkeypatch.setenv("WANDB_API_KEY", "ambient-key-must-not-be-used")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "sdk-run-directories"))
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "false")
    verified = []
    monkeypatch.setattr(
        wandb_login,
        "_verify_login",
        lambda key, base_url: verified.append((key, base_url)),
    )
    monkeypatch.setattr(wandb.Api, "_configure_sentry", lambda self: None)
    calls = []
    state = SimpleNamespace(handler=None, requests=[], files=[])

    def execute(service, query, variables=None, **kwargs):
        assert service._settings.api_key == RESEARCH_KEY
        calls.append((query, dict(variables or {})))
        if state.handler:
            answer = state.handler(query, variables or {})
            if answer is not None:
                return answer
        if "query Run(" in query:
            return {"project": {"run": run_node(variables["name"])}}
        if "query RunFiles(" in query:
            return {
                "project": {
                    "internalId": 123,
                    "run": {
                        "fileCount": len(state.files),
                        "files": connection(state.files),
                    },
                }
            }
        if "ServerFeaturesQuery" in query:
            return {"serverInfo": {"features": []}}
        raise AssertionError(f"Unexpected SDK query: {query}")

    monkeypatch.setattr(ServiceApi, "execute_graphql", execute)
    monkeypatch.setattr(ServiceApi, "finalize", lambda *_: lambda: None)
    configure_wandb_credentials(SecretStr(RESEARCH_KEY))
    workspace = tmp_path / "target"
    workspace.mkdir()
    conv_state = SimpleNamespace(workspace=SimpleNamespace(working_dir=workspace))
    tool = WandbResearchTool.create(conv_state, state_dir=tmp_path / "state")[0]
    state.tool, state.calls, state.verified = tool, calls, verified
    yield state
    configure_wandb_credentials(None)


def invoke(research, **request):
    observation = research.tool(WandbResearchAction(request=request))
    rows = [
        json.loads(line) for line in Path(observation.path).read_text().splitlines()
    ]
    return observation, rows


def test_private_runs_follow_real_sdk_pages_and_explicit_limits(research):
    def respond(query, variables):
        if "query Runs(" in query:
            assert json.loads(variables["filters"]) == {"config.model": "candidate"}
            assert variables["order"] == "+summary_metrics.loss"
            assert variables["entity"] == "other-team"
            assert variables["project"] == "private-project"
            cursor = variables.get("cursor")
            if cursor is None:
                return {
                    "project": {
                        "runCount": 3,
                        "runs": connection([run_node("r1"), run_node("r2")], more=True),
                    }
                }
            assert cursor == "1"
            return {"project": {"runCount": 3, "runs": connection([run_node("r3")])}}

    research.handler = respond
    request = {
        "op": "runs",
        "path": "other-team/private-project",
        "filters": {"config.model": "candidate"},
        "order": "+summary_metrics.loss",
        "page_size": 2,
    }
    observation, rows = invoke(research, **request)
    assert [row["id"] for row in rows] == ["r1", "r2", "r3"]
    assert rows[0]["config"] == {"lr": 0.01}
    assert rows[0]["summary"] == {"loss": 0.125, "nested": {"accuracy": 0.98}}
    assert observation.count == 3 and observation.next_offset is None
    assert research.verified == [(RESEARCH_KEY, "https://api.wandb.ai")]
    assert RESEARCH_KEY not in observation.model_dump_json()
    assert RESEARCH_KEY not in research.tool.model_dump_json()
    assert "ambient-key-must-not-be-used" not in Path(observation.path).read_text()

    first, rows = invoke(research, **request, limit=2)
    assert [row["id"] for row in rows] == ["r1", "r2"]
    assert first.next_offset == 2
    last, rows = invoke(research, **request, offset=first.next_offset, limit=2)
    assert [row["id"] for row in rows] == ["r3"]
    assert last.next_offset is None


def test_full_history_uses_sdk_scan_across_empty_pages_and_keeps_sparse_rows(
    research, monkeypatch
):
    research.handler = lambda query, variables: (
        {"project": {"run": {"historyKeys": {"lastStep": 6}}}}
        if "RunHistoryKeys" in query
        else None
    )
    ranges = []

    def send(service, request, **kwargs):
        assert service._settings.api_key == RESEARCH_KEY
        read = request.read_run_history_request
        response = pb.ApiResponse()
        if read.HasField("scan_run_history_init"):
            assert list(read.scan_run_history_init.keys) == []
            assert read.scan_run_history_init.use_cache is False
            response.read_run_history_response.scan_run_history_init.request_id = 12
            return response
        scan = read.scan_run_history
        ranges.append((scan.min_step, scan.max_step))
        rows = {
            0: [{"_step": 0, "loss": 9}, {"_step": 1, "lr": 0.02}],
            2: [],
            4: [{"_step": 4, "loss": 1}],
            6: [{"_step": 6, "loss": 0.5}],
        }
        for row in rows[scan.min_step]:
            history_row = (
                response.read_run_history_response.run_history.history_rows.add()
            )
            for key, value in row.items():
                history_row.history_items.add(key=key, value_json=json.dumps(value))
        return response

    monkeypatch.setattr(ServiceApi, "send_api_request", send)
    observation, rows = invoke(
        research, op="history", path="other/private/r1", keys=["loss"], page_size=2
    )
    assert rows == [
        {"_step": 0, "loss": 9},
        {"_step": 1},
        {"_step": 4, "loss": 1},
        {"_step": 6, "loss": 0.5},
    ]
    assert ranges == [(0, 2), (2, 4), (4, 6), (6, 7)]
    assert observation.count == 4
    assert observation.details["sampled"] is False


@pytest.mark.parametrize("stream", ["default", "system"])
def test_sampled_histories_use_correct_sdk_stream_and_axis(research, stream):
    def respond(query, variables):
        if "RunSampledHistory" in query:
            assert json.loads(variables["specs"][0]) == {
                "keys": ["epoch", "loss"],
                "samples": 21,
            }
            return {
                "project": {"run": {"sampledHistory": [[{"epoch": 4, "loss": 0.5}]]}}
            }
        if "RunFullHistory" in query:
            assert "events(samples:" in query and variables["samples"] == 21
            return {
                "project": {
                    "run": {"events": ['{"_timestamp":123,"loss":0.5,"other":7}']}
                }
            }

    research.handler = respond
    observation, rows = invoke(
        research,
        op="history",
        path="other/private/r1",
        sampled=True,
        stream=stream,
        keys=["loss"],
        samples=21,
        x_axis="epoch",
    )
    assert rows == (
        [{"epoch": 4, "loss": 0.5}]
        if stream == "default"
        else [{"_timestamp": 123, "loss": 0.5}]
    )
    assert observation.details["sampled"] is True


@contextmanager
def download_server(routes):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers.get("Authorization")))
            status, headers, body = routes[self.path]
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def file_node(name, body, direct_url):
    return {
        "id": "file-1",
        "name": name,
        "url": "https://api.wandb.ai/run-file",
        "directUrl": direct_url,
        "sizeBytes": len(body),
        "md5": base64.b64encode(
            hashlib.md5(body, usedforsecurity=False).digest()
        ).decode(),
        "storagePath": "storage",
        "mimetype": "application/octet-stream",
        "updatedAt": "2026-09-01T00:00:00Z",
        "digest": "digest",
    }


def artifact_membership():
    collection = {
        "__typename": "ArtifactSequence",
        "name": "weights",
        "project": {"name": "private", "entity": {"name": "other"}},
    }
    return {
        "id": "membership",
        "versionIndex": 2,
        "aliases": [{"id": "alias", "alias": "latest"}],
        "artifactCollection": collection,
        "artifact": {
            "id": "artifact",
            "artifactSequence": collection,
            "versionIndex": 2,
            "artifactType": {"name": "model"},
            "description": "private weights",
            "metadata": '{"accuracy":0.98}',
            "ttlDurationSeconds": 0,
            "ttlIsInherited": True,
            "tags": [],
            "historyStep": 100,
            "state": "COMMITTED",
            "size": 12,
            "digest": "artifact-digest",
            "commitHash": "abc",
            "fileCount": 2,
            "createdAt": "2026-09-01T00:00:00Z",
            "updatedAt": None,
        },
    }


def test_artifact_private_download_keeps_auth_on_origin_and_bytes_on_generated_paths(
    research, tmp_path, monkeypatch
):
    body = b"private model bytes"
    with (
        download_server({"/blob": (200, {}, body)}) as (storage, storage_seen),
        download_server({"/private": (302, {"Location": storage + "/blob"}, b"")}) as (
            api_origin,
            api_seen,
        ),
    ):
        configure_wandb_credentials(SecretStr(RESEARCH_KEY), base_url=api_origin)
        netrc = tmp_path / "netrc"
        netrc.write_text(
            "machine 127.0.0.1 login unwanted password ambient-storage-secret\n"
        )
        monkeypatch.setenv("NETRC", str(netrc))
        nodes = [
            file_node("weights/../../outside.py", body, api_origin + "/private"),
            file_node("logs/output.txt", b"skip", storage + "/unused"),
        ]

        def respond(query, variables):
            if "ArtifactMembershipByName" in query:
                return {
                    "project": {"artifactCollectionMembership": artifact_membership()}
                }
            if "ArtifactMembershipFiles" in query:
                return {
                    "project": {
                        "artifactCollection": {
                            "__typename": "ArtifactSequence",
                            "artifactMembership": {"files": connection(nodes)},
                        }
                    }
                }
            if "ArtifactCreatedBy" in query:
                return {
                    "artifact": {
                        "createdBy": {
                            "__typename": "Run",
                            "id": "s1",
                            "name": "r1",
                            "project": {
                                "name": "private",
                                "entity": {"name": "other"},
                            },
                        }
                    }
                }
            if "ArtifactUsedBy" in query:
                return {
                    "artifact": {
                        "usedBy": connection(
                            [
                                {
                                    "__typename": "Run",
                                    "id": "s2",
                                    "name": "r2",
                                    "project": {
                                        "name": "consumer",
                                        "entity": {"name": "team-two"},
                                    },
                                }
                            ]
                        )
                    }
                }

        research.handler = respond
        observation, rows = invoke(
            research,
            op="artifact",
            path="other/private/weights:latest",
            download=True,
            path_prefix="weights/",
        )
    assert len(rows) == 1 and rows[0]["name"] == "weights/../../outside.py"
    exported = Path(rows[0]["local_path"])
    assert exported.read_bytes() == body
    assert exported.parent == Path(observation.path).parent
    assert exported.suffix == ".bin" and exported.stat().st_mode & 0o777 == 0o600
    assert not (tmp_path / "outside.py").exists()
    assert api_seen == [
        (
            "/private",
            "Basic " + base64.b64encode(f"api:{RESEARCH_KEY}".encode()).decode(),
        )
    ]
    assert storage_seen == [("/blob", None)]
    metadata = json.loads(Path(observation.details["metadata_path"]).read_text())
    assert metadata["metadata"] == {"accuracy": 0.98}
    assert metadata["logged_by"] == "other/private/r1"
    assert metadata["used_by"] == ["team-two/consumer/r2"]


def test_full_system_export_requires_raw_file_and_never_claims_sample_completeness(
    research,
):
    with pytest.raises(RuntimeError, match="system-history API provides samples only"):
        invoke(research, op="history", path="other/private/r1", stream="system")
    body = b'{"_timestamp":1,"cpu":2}\n{"_timestamp":2,"cpu":3}\n'
    with download_server({"/events": (200, {}, body)}) as (storage, seen):
        research.files = [file_node("wandb-events.jsonl", body, storage + "/events")]
        observation, rows = invoke(
            research, op="history", path="other/private/r1", stream="system"
        )
    assert rows == [{"_timestamp": 1, "cpu": 2}, {"_timestamp": 2, "cpu": 3}]
    assert observation.details["source_file"] == "wandb-events.jsonl"
    assert observation.details["sampled"] is False
    assert seen == [("/events", None)]


def test_exports_reject_symlink_output_parent_and_remove_partial_results(
    research, tmp_path
):
    output = tmp_path / "state" / "wandb-research"
    output.parent.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    output.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(RuntimeError, match="(NotADirectoryError|OSError)"):
        invoke(research, op="run", path="other/private/r1")
    assert list(elsewhere.iterdir()) == []
    output.unlink()
    with pytest.raises(RuntimeError, match="system-history API"):
        invoke(research, op="history", path="other/private/r1", stream="system")
    assert list(output.iterdir()) == []


@pytest.mark.parametrize(
    "path", ["../project", "team/..", "https://attacker/path", "team\\x/project"]
)
def test_research_paths_cannot_select_urls_or_parent_directories(path):
    with pytest.raises(ValidationError):
        WandbResearchAction(request={"op": "runs", "path": path})


def test_research_requires_explicit_credentials_even_when_ambient_auth_exists(research):
    configure_wandb_credentials(None)
    with pytest.raises(RuntimeError, match="credentials are not configured"):
        invoke(research, op="run", path="other/private/r1")
    assert research.verified == []


def test_private_project_discovery_and_artifact_existence_use_sdk_responses(research):
    def respond(query, variables):
        if "GetProjects" in query:
            assert variables["entity"] == "other"
            node = {
                "id": "project-id",
                "name": "private",
                "entityName": "other",
                "createdAt": "2026-09-01T00:00:00Z",
                "isBenchmark": False,
                "user": None,
            }
            return {"models": connection([node])}
        if "ArtifactMembershipByName" in query:
            value = (
                artifact_membership() if variables["name"] == "weights:latest" else None
            )
            return {"project": {"artifactCollectionMembership": value}}

    research.handler = respond
    observation, rows = invoke(research, op="projects", path="other")
    assert rows == [{"path": "other/private", "name": "private"}]
    assert observation.count == 1
    _, present = invoke(
        research, op="artifact_exists", path="other/private/weights:latest"
    )
    _, absent = invoke(
        research, op="artifact_exists", path="other/private/weights:missing"
    )
    assert present[0]["exists"] is True
    assert absent[0]["exists"] is False


def test_metadata_and_error_outputs_redact_configured_key(research):
    encoded_auth = base64.b64encode(f"api:{RESEARCH_KEY}".encode()).decode()

    def respond(query, variables):
        if "query Run(" in query:
            node = run_node()
            node["summaryMetrics"] = json.dumps(
                {
                    "nested": {
                        "accidental-key": RESEARCH_KEY,
                        "authorization": f"Basic {encoded_auth}",
                    }
                }
            )
            return {"project": {"run": node}}

    research.handler = respond
    observation, rows = invoke(research, op="run", path="other/private/r1")
    assert rows[0]["summary"] == {
        "nested": {
            "accidental-key": "[REDACTED]",
            "authorization": "Basic [REDACTED]",
        }
    }
    assert RESEARCH_KEY not in Path(observation.path).read_text()
    assert encoded_auth not in Path(observation.path).read_text()

    def fail(*_):
        raise ValueError(f"rejected authorization {RESEARCH_KEY}; Basic {encoded_auth}")

    research.handler = fail
    with pytest.raises(RuntimeError) as raised:
        invoke(research, op="run", path="other/private/r1")
    assert RESEARCH_KEY not in str(raised.value)
    assert encoded_auth not in str(raised.value)
    assert "rejected authorization [REDACTED]; Basic [REDACTED]" in str(raised.value)


def test_trusted_self_hosted_path_prefix_reaches_actual_sdk(research):
    base_url = "https://selfhost.example/platform/wandb"
    configure_wandb_credentials(SecretStr(RESEARCH_KEY), base_url=base_url + "/")
    _, rows = invoke(research, op="run", path="other/private/r1")
    assert research.verified == [(RESEARCH_KEY, base_url)]
    assert rows[0]["url"] == base_url + "/other/private/runs/r1"


@pytest.mark.parametrize("failure", ["digest", "external_reference"])
def test_failed_download_publishes_no_bytes_or_manifest(research, tmp_path, failure):
    body = b"downloaded evidence"
    with download_server({"/file": (200, {}, body)}) as (storage, seen):
        node = file_node("wandb-metadata.json", body, storage + "/file")
        if failure == "digest":
            node["md5"] = "wrong-digest"
        else:
            node["url"] = node["directUrl"]
        research.files = [node]
        expected = (
            "digest does not match"
            if failure == "digest"
            else "External artifact references"
        )
        with pytest.raises(RuntimeError, match=expected):
            invoke(
                research,
                op="run_files",
                path="other/private/r1",
                names=["wandb-metadata.json"],
                download=True,
            )
    assert seen == ([("/file", None)] if failure == "digest" else [])
    assert list((tmp_path / "state" / "wandb-research").iterdir()) == []
