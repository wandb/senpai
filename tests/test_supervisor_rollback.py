import copy
import json
import os
import stat
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from k8s import supervisor_rollback


def completed(argv, *, stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr=stderr)


class FakeKubernetes:
    """Small stateful kubectl boundary used by rollback contract tests."""

    _RESOURCE_BY_KIND = {
        "NetworkPolicy": "networkpolicy.networking.k8s.io",
        "ServiceAccount": "serviceaccount",
        "Role": "role.rbac.authorization.k8s.io",
        "RoleBinding": "rolebinding.rbac.authorization.k8s.io",
        "Deployment": "deployment.apps",
        "Lease": "lease.coordination.k8s.io",
    }

    def __init__(self, *, context="cluster-a", namespace="research-a"):
        self.context = context
        self.namespace = namespace
        self.cluster_uid = "cluster-uid"
        self.namespace_uid = "namespace-uid"
        self.objects = {}
        self.calls = []
        self.before = None
        self._revision = 0

    def _next_revision(self):
        self._revision += 1
        return str(self._revision)

    def put(self, document):
        document = copy.deepcopy(document)
        metadata = document.setdefault("metadata", {})
        metadata.setdefault("namespace", self.namespace)
        metadata.setdefault("resourceVersion", self._next_revision())
        resource = self._RESOURCE_BY_KIND[document["kind"]]
        self.objects[f'{resource}/{metadata["name"]}'] = document

    @property
    def lease_name(self):
        return "lease.coordination.k8s.io/senpai-supervisor-release-campaign-a"

    @property
    def lease(self):
        return self.objects.get(self.lease_name)

    def expire_lease(self):
        self.lease["spec"]["renewTime"] = (
            datetime.now(UTC) - timedelta(seconds=10)
        ).isoformat().replace("+00:00", "Z")
        self.lease["spec"]["leaseDurationSeconds"] = 1

    def run(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.before is not None:
            response = self.before(argv, kwargs)
            if response is not None:
                return response

        if argv == ["kubectl", "config", "current-context"]:
            return completed(argv, stdout=f"{self.context}\n")

        action = next(
            (
                candidate
                for candidate in ("get", "create", "replace", "delete", "rollout")
                if candidate in argv
            ),
            None,
        )
        if action == "get":
            resource = argv[argv.index("get") + 1]
            if resource == "namespace/kube-system":
                return completed(
                    argv,
                    stdout=json.dumps({"metadata": {"uid": self.cluster_uid}}),
                )
            if resource == f"namespace/{self.namespace}":
                return completed(
                    argv,
                    stdout=json.dumps({"metadata": {"uid": self.namespace_uid}}),
                )
            document = self.objects.get(resource)
            return completed(
                argv,
                stdout="" if document is None else json.dumps(document),
            )

        if action in {"create", "replace"}:
            document = json.loads(kwargs["input"])
            metadata = document.setdefault("metadata", {})
            resource = self._RESOURCE_BY_KIND[document["kind"]]
            name = f'{resource}/{metadata["name"]}'
            current = self.objects.get(name)
            if action == "create" and current is not None:
                return completed(argv, returncode=1, stderr="AlreadyExists")
            if action == "replace" and (
                current is None
                or metadata.get("resourceVersion")
                != current.get("metadata", {}).get("resourceVersion")
            ):
                return completed(argv, returncode=1, stderr="Conflict")
            metadata["namespace"] = self.namespace
            metadata["resourceVersion"] = self._next_revision()
            if document["kind"] == "Lease":
                metadata["uid"] = (
                    current["metadata"]["uid"]
                    if current is not None
                    else "lease-uid"
                )
            self.objects[name] = document
            return completed(argv, stdout=json.dumps(document))

        if action == "delete":
            resource = argv[argv.index("delete") + 1]
            self.objects.pop(resource, None)
            return completed(argv, stdout="deleted")
        if action == "rollout":
            return completed(argv, stdout="ready")
        raise AssertionError(f"unexpected kubectl command: {argv}")


def install_fake(monkeypatch, fake):
    monkeypatch.setattr(supervisor_rollback.subprocess, "run", fake.run)


def old_deployment():
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "senpai-supervisor-campaign-a",
            "namespace": "research-a",
            "labels": {"release": "old"},
            "annotations": {"owner": "research"},
        },
        "spec": {"replicas": 1, "selector": {"matchLabels": {"app": "old"}}},
    }


def old_service_account():
    return {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {
            "name": "senpai-supervisor-campaign-a",
            "namespace": "research-a",
            "labels": {"release": "old"},
        },
        "automountServiceAccountToken": False,
    }


def old_network_policy():
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {
            "name": "senpai-supervisor-egress-campaign-a",
            "namespace": "research-a",
            "labels": {"release": "old"},
        },
        "spec": {"podSelector": {}, "policyTypes": ["Egress"]},
    }


def safety_network_policy():
    policy = old_network_policy()
    policy["metadata"]["labels"] = {"release": "first-launch-safety"}
    policy["spec"] = {
        "podSelector": {
            "matchLabels": {
                "research-tag": "campaign-a",
                "senpai-supervisor-access": "true",
            }
        },
        "policyTypes": ["Egress"],
        "egress": [
            {
                "to": [
                    {
                        "ipBlock": {
                            "cidr": "0.0.0.0/0",
                            "except": ["169.254.0.0/16"],
                        }
                    }
                ]
            }
        ],
    }
    return policy


def capture(monkeypatch, tmp_path, fake, **kwargs):
    install_fake(monkeypatch, fake)
    return supervisor_rollback.SupervisorRollback.capture(
        tag="campaign-a",
        kube_context=kwargs.pop("kube_context", fake.context),
        namespace=fake.namespace,
        directory=tmp_path,
        **kwargs,
    )


def test_capture_pins_scope_lineage_and_exact_mutable_resources(monkeypatch, tmp_path):
    fake = FakeKubernetes()
    deployment = old_deployment()
    deployment["metadata"].update(
        {
            "resourceVersion": "1234",
            "uid": "server-only",
            "managedFields": [{"manager": "kubectl"}],
            "creationTimestamp": "2026-08-10T10:00:00Z",
            "generation": 7,
        }
    )
    deployment["status"] = {"readyReplicas": 1}
    fake.put(deployment)

    rollback = capture(monkeypatch, tmp_path, fake, kube_context="")

    assert stat.S_IMODE(rollback.path.stat().st_mode) == 0o600
    bundle = json.loads(rollback.path.read_text())
    assert bundle["schema"] == "senpai-supervisor-rollback/v2"
    assert bundle["status"] == "captured"
    assert bundle["kube_context"] == fake.context
    assert bundle["kube_system_uid"] == fake.cluster_uid
    assert bundle["namespace_uid"] == fake.namespace_uid
    assert bundle["lease_uid"] == "lease-uid"
    assert bundle["lease_transitions"] == 0
    assert "manual_restore_argv" not in bundle
    assert bundle["persistent_state_rolled_back"] is False
    assert "SQLite" in bundle["operator_notice"]
    assert len(bundle["resources"]) == 5
    saved = next(item for item in bundle["resources"] if item["kind"] == "Deployment")
    assert saved["manifest"]["metadata"] == {
        "name": "senpai-supervisor-campaign-a",
        "namespace": "research-a",
        "labels": {"release": "old"},
        "annotations": {"owner": "research"},
    }
    assert "status" not in saved["manifest"]
    assert "Secret" not in {item["kind"] for item in bundle["resources"]}
    assert all(call[1].get("timeout") for call in fake.calls)


def test_first_launch_recovery_preserves_the_metadata_egress_policy(
    monkeypatch,
    tmp_path,
):
    fake = FakeKubernetes()
    safety_policy = safety_network_policy()
    rollback = capture(
        monkeypatch,
        tmp_path,
        fake,
        network_policy_safety_manifest=safety_policy,
    )
    bundle = json.loads(rollback.path.read_text())
    saved_policy = next(
        item for item in bundle["resources"] if item["kind"] == "NetworkPolicy"
    )
    assert saved_policy["present"] is False
    assert saved_policy["manifest"] is None
    assert saved_policy["safety_manifest"] == safety_policy

    rollback.mark_mutation_started()
    fake.put(safety_policy)
    fake.put(old_service_account())
    fake.put(old_deployment())
    rollback._lease.release()

    recovered = supervisor_rollback.SupervisorRollback(rollback.path)
    recovered.restore(timeout_seconds=10)

    restored_policy = fake.objects[
        "networkpolicy.networking.k8s.io/senpai-supervisor-egress-campaign-a"
    ]
    assert supervisor_rollback._canonical_manifest(
        restored_policy,
        supervisor_rollback._targets("campaign-a")[0],
        namespace=fake.namespace,
    ) == supervisor_rollback._canonical_manifest(
        safety_policy,
        supervisor_rollback._targets("campaign-a")[0],
        namespace=fake.namespace,
    )
    assert "serviceaccount/senpai-supervisor-campaign-a" not in fake.objects
    assert "deployment.apps/senpai-supervisor-campaign-a" not in fake.objects


def test_restore_quiesces_new_deployment_then_restores_security_then_old_deployment(
    monkeypatch,
    tmp_path,
):
    fake = FakeKubernetes()
    fake.put(old_service_account())
    fake.put(old_deployment())
    rollback = capture(monkeypatch, tmp_path, fake)
    rollback.mark_mutation_started()

    fake.objects = {
        name: document
        for name, document in fake.objects.items()
        if document["kind"] == "Lease"
    }
    new_deployment = old_deployment()
    new_deployment["metadata"]["labels"] = {"release": "new"}
    fake.put(new_deployment)
    calls_before_restore = len(fake.calls)

    rollback.restore(timeout_seconds=47)

    restore_calls = fake.calls[calls_before_restore:]
    mutations = [
        argv
        for argv, kwargs in restore_calls
        if any(action in argv for action in ("create", "replace", "delete"))
        and not (
            kwargs.get("input")
            and json.loads(kwargs["input"]).get("kind") == "Lease"
        )
    ]
    assert "deployment.apps/senpai-supervisor-campaign-a" in mutations[0]
    restored_documents = [
        json.loads(kwargs["input"])
        for argv, kwargs in restore_calls
        if any(action in argv for action in ("create", "replace"))
        and kwargs.get("input")
        and json.loads(kwargs["input"]).get("kind") != "Lease"
    ]
    assert [document["kind"] for document in restored_documents] == [
        "ServiceAccount",
        "Deployment",
    ]
    assert fake.objects[
        "deployment.apps/senpai-supervisor-campaign-a"
    ]["metadata"]["labels"] == {"release": "old"}
    assert any(
        "rollout" in argv and "--timeout=47s" in argv
        for argv, _kwargs in restore_calls
    )
    assert json.loads(rollback.path.read_text())["status"] == "restored"


def test_deployment_quiescence_failure_stops_before_security_changes(monkeypatch, tmp_path):
    fake = FakeKubernetes()
    rollback = capture(monkeypatch, tmp_path, fake)
    rollback.mark_mutation_started()
    fake.put(old_deployment())
    calls_before_restore = len(fake.calls)

    def fail_deployment_delete(argv, _kwargs):
        if "delete" in argv and "deployment.apps/senpai-supervisor-campaign-a" in argv:
            return completed(argv, returncode=1, stderr="finalizer blocked deletion")
        return None

    fake.before = fail_deployment_delete
    with pytest.raises(supervisor_rollback.RollbackError, match="finalizer"):
        rollback.restore(timeout_seconds=10)

    deletes = [argv for argv, _kwargs in fake.calls[calls_before_restore:] if "delete" in argv]
    assert len(deletes) == 1
    assert "deployment.apps/senpai-supervisor-campaign-a" in deletes[0]
    assert "--cascade=foreground" in deletes[0]
    assert "--wait=true" in deletes[0]


def test_restore_does_not_recreate_deployment_when_a_security_postcondition_fails(
    monkeypatch,
    tmp_path,
):
    fake = FakeKubernetes()
    fake.put(old_service_account())
    fake.put(old_deployment())
    rollback = capture(monkeypatch, tmp_path, fake)
    rollback.mark_mutation_started()
    fake.objects.pop("serviceaccount/senpai-supervisor-campaign-a")
    calls_before_restore = len(fake.calls)
    service_account_gets = 0

    def drift_after_create(argv, _kwargs):
        nonlocal service_account_gets
        if "get" in argv and "serviceaccount/senpai-supervisor-campaign-a" in argv:
            service_account_gets += 1
            if service_account_gets == 2:
                drifted = old_service_account()
                drifted["metadata"]["labels"] = {"release": "drifted"}
                drifted["metadata"]["resourceVersion"] = "drift"
                return completed(argv, stdout=json.dumps(drifted))
        return None

    fake.before = drift_after_create
    with pytest.raises(supervisor_rollback.RollbackError, match="postcondition"):
        rollback.restore(timeout_seconds=10)

    documents = [
        json.loads(kwargs["input"])
        for argv, kwargs in fake.calls[calls_before_restore:]
        if any(action in argv for action in ("create", "replace")) and kwargs.get("input")
    ]
    assert not any(document.get("kind") == "Deployment" for document in documents)


@pytest.mark.parametrize(
    ("drift_get", "deployment_was_recreated"),
    ((3, False), (4, True)),
)
def test_whole_security_gates_keep_deployment_quiesced_on_drift(
    monkeypatch,
    tmp_path,
    drift_get,
    deployment_was_recreated,
):
    fake = FakeKubernetes()
    fake.put(old_network_policy())
    fake.put(old_deployment())
    rollback = capture(monkeypatch, tmp_path, fake)
    rollback.mark_mutation_started()
    policy_gets = 0
    calls_before_restore = len(fake.calls)

    def drift_policy(argv, _kwargs):
        nonlocal policy_gets
        if "get" in argv and "networkpolicy.networking.k8s.io/" in " ".join(argv):
            policy_gets += 1
            if policy_gets == drift_get:
                drifted = old_network_policy()
                drifted["metadata"]["labels"] = {"release": "drifted"}
                drifted["metadata"]["resourceVersion"] = "drift"
                return completed(argv, stdout=json.dumps(drifted))
        return None

    fake.before = drift_policy
    with pytest.raises(supervisor_rollback.RollbackError, match="postcondition"):
        rollback.restore(timeout_seconds=10)

    restore_calls = fake.calls[calls_before_restore:]
    deployment_creates = [
        json.loads(kwargs["input"])
        for argv, kwargs in restore_calls
        if "create" in argv
        and kwargs.get("input")
        and json.loads(kwargs["input"]).get("kind") == "Deployment"
    ]
    assert bool(deployment_creates) is deployment_was_recreated
    assert "deployment.apps/senpai-supervisor-campaign-a" not in fake.objects


def test_restore_rejects_changed_cluster_identity_before_mutation(monkeypatch, tmp_path):
    fake = FakeKubernetes()
    rollback = capture(monkeypatch, tmp_path, fake)
    fake.cluster_uid = "replacement-cluster"
    calls_before_restore = len(fake.calls)

    with pytest.raises(supervisor_rollback.RollbackError, match="cluster identity"):
        rollback.restore(timeout_seconds=10)

    assert not any(
        any(action in argv for action in ("create", "replace", "delete"))
        for argv, _kwargs in fake.calls[calls_before_restore:]
    )


def test_manual_restore_reclaims_the_exact_expired_transaction(monkeypatch, tmp_path):
    fake = FakeKubernetes()
    rollback = capture(monkeypatch, tmp_path, fake)
    rollback.mark_mutation_started()
    transaction_id = json.loads(rollback.path.read_text())["transaction_id"]
    rollback._lease.release()

    recovered = supervisor_rollback.SupervisorRollback(
        rollback.path,
        timeout_seconds=rollback.timeout_seconds,
    )
    recovered.restore(timeout_seconds=10)

    assert json.loads(recovered.path.read_text())["status"] == "restored"
    assert fake.lease["spec"]["holderIdentity"] == transaction_id
    assert fake.lease["spec"]["leaseTransitions"] == 0


def test_manual_restore_refuses_to_race_a_live_launch(monkeypatch, tmp_path):
    fake = FakeKubernetes()
    rollback = capture(monkeypatch, tmp_path, fake)
    rollback.mark_mutation_started()

    with pytest.raises(supervisor_rollback.RollbackError, match="still active"):
        supervisor_rollback.SupervisorRollback(rollback.path).restore(
            timeout_seconds=10
        )


def test_only_one_manual_recovery_attempt_can_hold_the_transaction(monkeypatch, tmp_path):
    fake = FakeKubernetes()
    rollback = capture(monkeypatch, tmp_path, fake)
    rollback.mark_mutation_started()
    rollback._lease.release()
    bundle, scope, tag, transaction_id, lease_uid, transitions, _plan = (
        supervisor_rollback.SupervisorRollback(rollback.path)._validated_bundle()
    )
    first = supervisor_rollback.SupervisorRollback(rollback.path)
    first._ensure_lease(
        scope,
        tag,
        transaction_id,
        lease_uid,
        transitions,
        10,
    )

    with pytest.raises(supervisor_rollback.RollbackError, match="still active"):
        supervisor_rollback.SupervisorRollback(rollback.path).restore(
            timeout_seconds=10
        )

    assert bundle["status"] == "mutating"


def test_old_bundle_cannot_restore_after_a_newer_lease_epoch(monkeypatch, tmp_path):
    fake = FakeKubernetes()
    rollback = capture(monkeypatch, tmp_path, fake)
    rollback.mark_mutation_started()
    stale_bundle = rollback.path.read_bytes()
    rollback.commit()
    stale_path = tmp_path / "stale-rollback.json"
    stale_path.write_bytes(stale_bundle)
    stale_path.chmod(0o600)

    scope = supervisor_rollback._Scope(
        fake.context,
        fake.namespace,
        fake.cluster_uid,
        fake.namespace_uid,
    )
    newer = supervisor_rollback._CampaignLease.acquire(
        scope,
        "campaign-a",
        timeout_seconds=10,
        holder_identity="senpai-launch-newer",
    )
    assert newer.lease_transitions == 1
    newer.release()

    with pytest.raises(supervisor_rollback.RollbackError, match="lineage changed"):
        supervisor_rollback.SupervisorRollback(stale_path).restore(
            timeout_seconds=10
        )


def test_old_bundle_cannot_restore_after_lease_recreation(monkeypatch, tmp_path):
    fake = FakeKubernetes()
    rollback = capture(monkeypatch, tmp_path, fake)
    rollback.mark_mutation_started()
    rollback._lease.release()
    replacement = copy.deepcopy(fake.lease)
    replacement["metadata"].pop("resourceVersion")
    replacement["metadata"]["uid"] = "replacement-lease-uid"
    fake.objects.pop(fake.lease_name)
    fake.put(replacement)

    with pytest.raises(supervisor_rollback.RollbackError, match="lineage changed"):
        supervisor_rollback.SupervisorRollback(rollback.path).restore(
            timeout_seconds=10
        )


def test_active_other_transaction_blocks_capture(monkeypatch, tmp_path):
    fake = FakeKubernetes()
    scope = supervisor_rollback._Scope(
        fake.context,
        fake.namespace,
        fake.cluster_uid,
        fake.namespace_uid,
    )
    install_fake(monkeypatch, fake)
    supervisor_rollback._CampaignLease.acquire(
        scope,
        "campaign-a",
        timeout_seconds=10,
        holder_identity="senpai-launch-other",
    )

    with pytest.raises(supervisor_rollback.RollbackError, match="transaction is active"):
        supervisor_rollback.SupervisorRollback.capture(
            tag="campaign-a",
            kube_context=fake.context,
            namespace=fake.namespace,
            directory=tmp_path,
        )


def test_unfinished_journal_blocks_a_new_release(monkeypatch, tmp_path):
    fake = FakeKubernetes()
    rollback = capture(monkeypatch, tmp_path, fake)
    rollback.mark_mutation_started()
    rollback._lease.release()
    transition = fake.lease["spec"]["leaseTransitions"]

    with pytest.raises(RuntimeError, match="unfinished operational-supervisor rollback"):
        supervisor_rollback.SupervisorRollback.capture(
            tag="campaign-a",
            kube_context=fake.context,
            namespace=fake.namespace,
            directory=tmp_path,
        )

    assert fake.lease["spec"]["leaseTransitions"] == transition
    supervisor_rollback.SupervisorRollback(rollback.path).restore(
        timeout_seconds=10
    )
    assert json.loads(rollback.path.read_text())["status"] == "restored"


def test_context_alias_and_other_directory_cannot_bypass_unfinished_transaction(
    monkeypatch,
    tmp_path,
):
    fake = FakeKubernetes()
    rollback = capture(monkeypatch, tmp_path, fake)
    rollback.mark_mutation_started()
    rollback._lease.release()
    transition = fake.lease["spec"]["leaseTransitions"]

    with pytest.raises(RuntimeError, match="unfinished operational-supervisor rollback"):
        supervisor_rollback.SupervisorRollback.capture(
            tag="campaign-a",
            kube_context="same-cluster-alias",
            namespace=fake.namespace,
            directory=tmp_path,
        )
    with pytest.raises(
        supervisor_rollback.RollbackError,
        match="requires recovery",
    ):
        supervisor_rollback.SupervisorRollback.capture(
            tag="campaign-a",
            kube_context=fake.context,
            namespace=fake.namespace,
            directory=tmp_path / "other-host-state",
        )

    assert fake.lease["spec"]["leaseTransitions"] == transition


def test_commit_marks_journal_safe_before_release_and_deletion(monkeypatch, tmp_path):
    fake = FakeKubernetes()
    rollback = capture(monkeypatch, tmp_path, fake)
    rollback.mark_mutation_started()
    status_at_release = []

    def observe_release(argv, kwargs):
        if "replace" in argv and kwargs.get("input"):
            document = json.loads(kwargs["input"])
            if document.get("kind") == "Lease" and document["spec"]["leaseDurationSeconds"] == 1:
                status_at_release.append(json.loads(rollback.path.read_text())["status"])
        return None

    fake.before = observe_release
    rollback.commit()

    assert status_at_release == ["committed"]
    assert not rollback.path.exists()


def test_committed_journal_is_retained_but_not_recoverable_when_unlink_fails(
    monkeypatch,
    tmp_path,
):
    fake = FakeKubernetes()
    rollback = capture(monkeypatch, tmp_path, fake)
    rollback.mark_mutation_started()
    monkeypatch.setattr(
        supervisor_rollback,
        "_remove_bundle",
        lambda _path: (_ for _ in ()).throw(OSError("disk failure")),
    )

    with pytest.raises(OSError, match="disk failure"):
        rollback.commit()

    assert json.loads(rollback.path.read_text())["status"] == "committed"
    with pytest.raises(supervisor_rollback.RollbackError, match="finalized"):
        supervisor_rollback.SupervisorRollback(rollback.path).restore(
            timeout_seconds=10
        )


def test_commit_crash_between_cluster_and_local_finality_reconciles_on_next_capture(
    monkeypatch,
    tmp_path,
):
    fake = FakeKubernetes()
    rollback = capture(monkeypatch, tmp_path, fake)
    rollback.mark_mutation_started()
    update_status = rollback._update_status

    def fail_committed_status(status):
        if status == "committed":
            raise OSError("journal fsync failed")
        update_status(status)

    monkeypatch.setattr(rollback, "_update_status", fail_committed_status)
    with pytest.raises(OSError, match="journal fsync"):
        rollback.commit()

    assert supervisor_rollback._lease_phase(fake.lease["metadata"]) == "committed"
    assert json.loads(rollback.path.read_text())["status"] == "mutating"
    replacement = supervisor_rollback.SupervisorRollback.capture(
        tag="campaign-a",
        kube_context=fake.context,
        namespace=fake.namespace,
        directory=tmp_path,
        timeout_seconds=10,
    )
    assert replacement.path != rollback.path
    assert json.loads(rollback.path.read_text())["status"] == "committed"


def test_restore_crash_between_cluster_and_local_finality_reconciles_on_next_capture(
    monkeypatch,
    tmp_path,
):
    fake = FakeKubernetes()
    rollback = capture(monkeypatch, tmp_path, fake)
    rollback.mark_mutation_started()
    update_status = rollback._update_status

    def fail_restored_status(status):
        if status == "restored":
            raise OSError("journal fsync failed")
        update_status(status)

    monkeypatch.setattr(rollback, "_update_status", fail_restored_status)
    with pytest.raises(OSError, match="journal fsync"):
        rollback.restore(timeout_seconds=10)

    assert supervisor_rollback._lease_phase(fake.lease["metadata"]) == "restored"
    assert json.loads(rollback.path.read_text())["status"] == "recovery_required"
    supervisor_rollback.SupervisorRollback.capture(
        tag="campaign-a",
        kube_context=fake.context,
        namespace=fake.namespace,
        directory=tmp_path,
        timeout_seconds=10,
    )
    assert json.loads(rollback.path.read_text())["status"] == "restored"


def test_explicit_finalize_repairs_local_final_cluster_nonfinal_pair(
    monkeypatch,
    tmp_path,
):
    fake = FakeKubernetes()
    rollback = capture(monkeypatch, tmp_path, fake)
    rollback.mark_mutation_started()
    rollback._update_status("committed")
    rollback._lease.release()

    supervisor_rollback.SupervisorRollback(rollback.path).finalize(
        outcome="committed",
        timeout_seconds=10,
    )

    assert supervisor_rollback._lease_phase(fake.lease["metadata"]) == "committed"
    assert not rollback.path.exists()


def test_lease_phase_timeout_is_reconciled_from_authoritative_readback(
    monkeypatch,
    tmp_path,
):
    fake = FakeKubernetes()
    rollback = capture(monkeypatch, tmp_path, fake)
    timed_out_phases = []

    def apply_then_timeout(argv, kwargs):
        if "replace" not in argv or not kwargs.get("input"):
            return None
        document = json.loads(kwargs["input"])
        if document.get("kind") != "Lease":
            return None
        if document["spec"].get("leaseDurationSeconds") == 1:
            return None
        phase = document["metadata"].get("annotations", {}).get(
            supervisor_rollback.LEASE_PHASE_ANNOTATION
        )
        if phase not in {"mutating", "committed"}:
            return None
        fake.before = None
        try:
            applied = fake.run(argv, **kwargs)
        finally:
            fake.before = apply_then_timeout
        assert applied.returncode == 0
        timed_out_phases.append(phase)
        return completed(argv, returncode=124, stderr="response timed out")

    fake.before = apply_then_timeout
    rollback.mark_mutation_started()
    rollback.commit()

    assert timed_out_phases == ["mutating", "committed"]
    assert not rollback.path.exists()


def test_maximum_timeout_lease_has_rollout_cushion_and_is_renewable(monkeypatch, tmp_path):
    fake = FakeKubernetes()
    rollback = capture(
        monkeypatch,
        tmp_path,
        fake,
        timeout_seconds=supervisor_rollback.MAX_ROLLBACK_TIMEOUT_SECONDS,
    )
    renew_time = fake.lease["spec"]["renewTime"]

    assert fake.lease["spec"]["leaseDurationSeconds"] == (
        supervisor_rollback.MAX_ROLLBACK_TIMEOUT_SECONDS
        + supervisor_rollback.LEASE_CUSHION_SECONDS
    )
    rollback.renew_lease()
    assert fake.lease["spec"]["renewTime"] >= renew_time


def test_manual_recovery_command_is_derived_not_loaded_from_bundle(tmp_path):
    bundle = tmp_path / "rollback.json"
    bundle.write_text(json.dumps({"manual_restore_argv": ["/bin/sh", "-c", "malicious"]}))
    bundle.chmod(0o600)

    argv = supervisor_rollback.SupervisorRollback(bundle).manual_restore_argv()

    assert argv[0] == os.fspath(Path(supervisor_rollback.sys.executable))
    assert argv[1] == os.fspath(Path(supervisor_rollback.__file__).resolve())
    assert argv[2:4] == ["restore", os.fspath(bundle)]
    assert "malicious" not in argv


def test_restore_rejects_a_tampered_api_identity_before_cluster_mutation(
    monkeypatch,
    tmp_path,
):
    fake = FakeKubernetes()
    rollback = capture(monkeypatch, tmp_path, fake)
    rollback._lease.release()
    bundle = json.loads(rollback.path.read_text())
    bundle["resources"][0]["api_version"] = "networking.k8s.io/v1beta1"
    rollback.path.write_text(json.dumps(bundle))
    calls_before_restore = len(fake.calls)

    with pytest.raises(supervisor_rollback.RollbackError, match="unexpected resources"):
        supervisor_rollback.SupervisorRollback(rollback.path).restore(
            timeout_seconds=10
        )

    assert fake.calls[calls_before_restore:] == []


def test_restore_rejects_non_object_symlink_and_public_files(tmp_path):
    non_object = tmp_path / "non-object.json"
    non_object.write_text("[]")
    non_object.chmod(0o600)
    with pytest.raises(supervisor_rollback.RollbackError, match="invalid rollback"):
        supervisor_rollback.SupervisorRollback(non_object).restore(timeout_seconds=10)

    target = tmp_path / "target.json"
    target.write_text("{}")
    target.chmod(0o600)
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    with pytest.raises(supervisor_rollback.RollbackError, match="regular private"):
        supervisor_rollback.SupervisorRollback(symlink).restore(timeout_seconds=10)

    public = tmp_path / "public.json"
    public.write_text("{}")
    public.chmod(0o644)
    with pytest.raises(supervisor_rollback.RollbackError, match="regular private"):
        supervisor_rollback.SupervisorRollback(public).restore(timeout_seconds=10)


def test_capture_refuses_public_or_symlinked_rollback_directory(monkeypatch, tmp_path):
    fake = FakeKubernetes()
    install_fake(monkeypatch, fake)
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    public.chmod(0o755)
    with pytest.raises(RuntimeError, match="private rollback directory"):
        supervisor_rollback.SupervisorRollback.capture(
            tag="campaign-a",
            kube_context=fake.context,
            namespace=fake.namespace,
            directory=public,
        )

    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlinks"):
        supervisor_rollback.SupervisorRollback.capture(
            tag="campaign-a",
            kube_context=fake.context,
            namespace=fake.namespace,
            directory=linked,
        )


def test_capture_failure_creates_no_bundle_and_releases_lease(monkeypatch, tmp_path):
    fake = FakeKubernetes()

    def fail_resource_get(argv, _kwargs):
        if "get" in argv and "serviceaccount/senpai-supervisor-campaign-a" in argv:
            return completed(argv, returncode=1, stderr="cluster unavailable")
        return None

    fake.before = fail_resource_get
    install_fake(monkeypatch, fake)
    with pytest.raises(RuntimeError, match="cluster unavailable"):
        supervisor_rollback.SupervisorRollback.capture(
            tag="campaign-a",
            kube_context=fake.context,
            namespace=fake.namespace,
            directory=tmp_path,
        )

    assert not list(tmp_path.glob("*.json"))
    assert fake.lease["spec"]["leaseDurationSeconds"] == 1


def test_kubectl_process_timeout_is_a_typed_failure(monkeypatch):
    def timeout(argv, **_kwargs):
        raise subprocess.TimeoutExpired(argv, 45)

    monkeypatch.setattr(supervisor_rollback.subprocess, "run", timeout)
    result = supervisor_rollback._run_global("config", "current-context")

    assert result.returncode == 124
    assert "process deadline" in result.stderr
