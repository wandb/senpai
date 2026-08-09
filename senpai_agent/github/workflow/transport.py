"""Default HTTP transport for GitHub workflow requests."""

import json
from urllib import request
from urllib.error import HTTPError, URLError

from senpai_agent.github.workflow.errors import GitHubTransportError
from senpai_agent.github.workflow.responses import HttpResponse


class UrllibTransport:
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: object | None = None,
    ) -> HttpResponse:
        data = (
            None
            if json_body is None
            else json.dumps(
                json_body,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        )
        github_request = request.Request(
            url,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with request.urlopen(github_request, timeout=30) as response:
                return HttpResponse(
                    status_code=response.status,
                    json_body=_decode_json(response.read()),
                    headers=tuple(response.headers.items()),
                )
        except HTTPError as error:
            return HttpResponse(
                status_code=error.code,
                json_body=_decode_json(error.read()),
                headers=tuple(error.headers.items()) if error.headers else (),
            )
        except (URLError, TimeoutError) as error:
            raise GitHubTransportError(method, url) from error


def _decode_json(body: bytes) -> object | None:
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body.decode(errors="replace")
