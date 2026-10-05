"""Review and merge one experiment while its advisor continues working."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from functools import partial

from pydantic import SecretStr

from senpai_agent.openhands_runner import (
    RunnerConfig,
    parse_runner_args,
    resolve_config,
    scrub_model_credentials,
)
from senpai_agent.delegation import record_delegated_task_result
from senpai_agent.github.code_review import review_code_quality
from senpai_agent.github.tools.contracts import MergeExperimentAction
from senpai_agent.github.workflow import GitHubWorkflow, WorkflowPreconditionError
from senpai_agent.secrets import (
    consume_model_credential_fd,
    scrub_github_credentials,
    set_process_nondumpable,
)
from senpai_agent.weave_monitoring import (
    finish_weave_monitoring,
    register_trace_secret,
)


def merge_with_review(
    action: MergeExperimentAction,
    *,
    config: RunnerConfig,
    token: SecretStr,
) -> dict[str, object]:
    assignment = action.assignment
    workflow = GitHubWorkflow(
        config.github_repo,
        token,
        role="advisor",
        trusted_actor=config.github_trusted_actor,
        mutation_lock_path=(config.delegation_root_state_dir or config.state_dir)
        / "github" / "assignment-mutations.lock",
    )
    try:
        result = workflow.merge_experiment(
            assignment.pr_number,
            expected_head_sha=assignment.expected_pr_head_sha,
            assignment_id=assignment.assignment_id,
            current_revision_id=assignment.revision_id,
            expected_current_base_sha=action.expected_current_base_sha,
            merge_method=action.merge_method,
            review_code=partial(
                review_code_quality,
                base_sha=action.expected_current_base_sha,
                token=token,
                config=config,
            ),
        )
        return asdict(result)
    except Exception as error:
        blocked = isinstance(error, WorkflowPreconditionError)
        reason = f"{type(error).__name__}: {error}"
        outcome: dict[str, object] = {
            "state": "merge_blocked" if blocked else "merge_failed",
            "pr_number": assignment.pr_number,
            "head_sha": assignment.expected_pr_head_sha,
            "base_sha": action.expected_current_base_sha,
            "reason": reason,
        }
        conclusion = "blocked" if blocked else "could not complete"
        comment = (
            f"Merge gate {conclusion} for PR #{assignment.pr_number} at "
            f"`{assignment.expected_pr_head_sha}` against research base "
            f"`{action.expected_current_base_sha}`.\n\n{reason}\n\n"
            "The advisor must resolve this issue before requesting another merge."
        )
        try:
            feedback = workflow.send_assignment_feedback(
                assignment.pr_number,
                assignment_id=assignment.assignment_id,
                revision_id=assignment.revision_id,
                expected_head_sha=assignment.expected_pr_head_sha,
                feedback_id=f"merge-review:{config.delegation_task_id}",
                comment=comment,
            )
            outcome["feedback_url"] = feedback.resource_url
        except Exception as feedback_error:
            outcome["feedback_error"] = (
                f"{type(feedback_error).__name__}: {feedback_error}"
            )
        return outcome


def main(argv: Sequence[str] | None = None) -> int:
    try:
        try:
            set_process_nondumpable()
            credentials = consume_model_credential_fd(os.environ)
            for credential in credentials.values():
                register_trace_secret(credential)
            token_value = credentials.pop("GITHUB_TOKEN", None)
            if not token_value:
                raise RuntimeError("merge worker requires private GitHub credentials")
            token = SecretStr(token_value)
            del token_value
            scrub_github_credentials(os.environ)
            runtime_environment = {**os.environ, **credentials}
            args = parse_runner_args(argv)
            if not args.child or args.agent != "supervisor":
                raise RuntimeError("merge worker requires a supervisor child agent")
            config = resolve_config(args, runtime_environment)
            del runtime_environment, credentials
            scrub_model_credentials(os.environ, config)
            if config.role != "advisor" or not config.delegation_task_id:
                raise RuntimeError("merge worker requires an advisor delegation task")
            action = MergeExperimentAction.model_validate_json(
                os.environ.pop("SENPAI_MERGE_REQUEST_JSON")
            )
            outcome = merge_with_review(
                action,
                config=config,
                token=token,
            )
            result = json.dumps(outcome, sort_keys=True)
            record_delegated_task_result(config.delegation_task_id, result=result)
            print(
                "OPENHANDS_RESULT " + json.dumps({
                    "conversation_id": str(config.conversation_id),
                    "status": "finished",
                    "result": result,
                }, sort_keys=True),
                flush=True,
            )
            return 0
        except BaseException as error:
            if task_id := os.environ.get("SENPAI_DELEGATION_TASK_ID"):
                record_delegated_task_result(
                    task_id,
                    error=f"{type(error).__name__}: {error}",
                )
            raise
    finally:
        finish_weave_monitoring()


if __name__ == "__main__":
    raise SystemExit(main())
