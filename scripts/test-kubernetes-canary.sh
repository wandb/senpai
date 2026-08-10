#!/usr/bin/env bash

# Exercise the production Kubernetes topology without any live credential.

set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

KIND_BIN=${KIND_BIN:-kind}
KUBECTL_BIN=${KUBECTL_BIN:-kubectl}
DOCKER_BIN=${DOCKER_BIN:-docker}
SOURCE_REVISION=${SOURCE_REVISION:-$(git rev-parse HEAD)}
BASE_IMAGE=${BASE_IMAGE:?BASE_IMAGE must name the locally loaded advisor image}
ADVISOR_CANARY_IMAGE=${ADVISOR_CANARY_IMAGE:-senpai-kubernetes-canary-advisor:sha-$SOURCE_REVISION}
STUDENT_CANARY_IMAGE=${STUDENT_CANARY_IMAGE:-senpai-kubernetes-canary-student:sha-$SOURCE_REVISION}
CANARY_ID=${SENPAI_CANARY_ID:-local-$$}
CANARY_ID=$(printf '%s' "$CANARY_ID" \
  | tr '[:upper:]' '[:lower:]' \
  | tr -cd '[:alnum:]-' \
  | cut -c1-30)
[[ -n "$CANARY_ID" ]] || { echo "SENPAI_CANARY_ID has no safe characters" >&2; exit 2; }
[[ "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "SOURCE_REVISION must be a full lowercase commit SHA" >&2; exit 2; }
CLUSTER="senpai-ci-$CANARY_ID"
CONTEXT="kind-$CLUSTER"
NAMESPACE="senpai-ci-$CANARY_ID"
OTHER_NAMESPACE="$NAMESPACE-other"
TAG="ci-$CANARY_ID"
PV="senpai-ci-$TAG"
STATE_PV="$PV-supervisor-state"
NODE="$CLUSTER-control-plane"
DIAGNOSTICS_DIR=${DIAGNOSTICS_DIR:-$ROOT/.canary-diagnostics}
PYTHON_BIN=${PYTHON_BIN:-python3}
ROLE_SHELL=/usr/local/bin/senpai-role-shell
WORK_DIR=$(mktemp -d)
INITIAL_MANIFEST="$WORK_DIR/initial.yaml"
UPGRADE_MANIFEST="$WORK_DIR/broken-upgrade.yaml"
CLUSTER_CREATED=false

kubectl_canary() {
  "$KUBECTL_BIN" --context "$CONTEXT" "$@"
}

verify_sha256() {
  "$PYTHON_BIN" -c '
import hashlib
import pathlib
import sys

expected, path = sys.argv[1:]
actual = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"SHA-256 mismatch for {path}: {actual} != {expected}")
' "$1" "$2"
}

collect_role_container_logs() {
  local pods="$DIAGNOSTICS_DIR/role-pods.json"
  local inventory="$DIAGNOSTICS_DIR/container-restarts.tsv"
  if ! kubectl_canary get pods -n "$NAMESPACE" -o json > "$pods" 2>&1; then
    return
  fi
  if ! "$PYTHON_BIN" -c '
import json
import pathlib
import sys

document = json.loads(pathlib.Path(sys.argv[1]).read_text())
print("pod\trole\tcontainer\trestart_count\tprevious_exit_code\tprevious_reason\tprevious_finished_at")
for pod in document.get("items", []):
    metadata = pod.get("metadata", {})
    role = metadata.get("labels", {}).get("role")
    if role not in {"advisor", "student", "supervisor"}:
        continue
    pod_name = metadata.get("name")
    if not isinstance(pod_name, str):
        continue
    for status in pod.get("status", {}).get("containerStatuses", []):
        container = status.get("name")
        restarts = status.get("restartCount")
        if not isinstance(container, str) or not isinstance(restarts, int):
            continue
        terminated = status.get("lastState", {}).get("terminated", {})
        fields = (
            pod_name,
            role,
            container,
            str(restarts),
            str(terminated.get("exitCode", "")),
            str(terminated.get("reason", "")),
            str(terminated.get("finishedAt", "")),
        )
        print("\t".join(value.replace("\t", " ").replace("\n", " ") for value in fields))
' "$pods" > "$inventory" 2> "$DIAGNOSTICS_DIR/container-restarts.error.txt"; then
    return
  fi

  while IFS=$'\t' read -r pod role container restarts _exit _reason _finished; do
    [[ "$pod" != pod ]] || continue
    local prefix="$DIAGNOSTICS_DIR/$pod.$container"
    kubectl_canary logs -n "$NAMESPACE" "pod/$pod" -c "$container" \
      --timestamps=true --tail=2000 > "$prefix.current.log" 2>&1 || true
    if (( restarts > 0 )); then
      kubectl_canary logs -n "$NAMESPACE" "pod/$pod" -c "$container" \
        --previous --timestamps=true --tail=2000 \
        > "$prefix.previous.log" 2>&1 || true
    fi
  done < "$inventory"
}

collect_diagnostics() {
  mkdir -p "$DIAGNOSTICS_DIR"
  if [[ "$CLUSTER_CREATED" == true ]]; then
    kubectl_canary get namespace,pv,pvc,deploy,pod,cm,secret,sa,role,rolebinding,networkpolicy \
      -A -o yaml > "$DIAGNOSTICS_DIR/resources.yaml" 2>&1 || true
    kubectl_canary get events -A --sort-by=.lastTimestamp \
      > "$DIAGNOSTICS_DIR/events.txt" 2>&1 || true
    kubectl_canary describe deployment,pod -n "$NAMESPACE" \
      > "$DIAGNOSTICS_DIR/describe.txt" 2>&1 || true
    collect_role_container_logs || true
    "$DOCKER_BIN" logs "$NODE" > "$DIAGNOSTICS_DIR/kind-node.log" 2>&1 || true
  fi
}

cleanup() {
  local status=$?
  trap - EXIT
  if (( status != 0 )); then
    collect_diagnostics
  fi
  if [[ "$CLUSTER_CREATED" == true ]]; then
    # Drain namespaced users of the static volume before deleting the claim.
    # Each namespace is unique to this canary, so --all remains exact here.
    kubectl_canary delete deployment,pod --all -n "$NAMESPACE" \
      --ignore-not-found --wait=true --timeout=60s >/dev/null 2>&1 || true
    kubectl_canary delete pod --all -n "$OTHER_NAMESPACE" \
      --ignore-not-found --wait=true --timeout=60s >/dev/null 2>&1 || true
    kubectl_canary delete pvc "$PV" -n "$NAMESPACE" \
      --ignore-not-found --wait=true --timeout=60s >/dev/null 2>&1 || true
    kubectl_canary delete pvc "$STATE_PV" -n "$NAMESPACE" \
      --ignore-not-found --wait=true --timeout=60s >/dev/null 2>&1 || true
    kubectl_canary delete namespace "$NAMESPACE" "$OTHER_NAMESPACE" \
      --ignore-not-found --wait=true --timeout=60s >/dev/null 2>&1 || true
    kubectl_canary delete pv "$PV" --ignore-not-found --wait=true \
      --timeout=60s >/dev/null 2>&1 || true
    kubectl_canary delete pv "$STATE_PV" --ignore-not-found --wait=true \
      --timeout=60s >/dev/null 2>&1 || true
    kubectl_canary delete pod senpai-metadata-decoy -n kube-system \
      --ignore-not-found --wait=true --timeout=60s >/dev/null 2>&1 || true
    if kubectl_canary get namespace "$NAMESPACE" >/dev/null 2>&1 \
      || kubectl_canary get namespace "$OTHER_NAMESPACE" >/dev/null 2>&1 \
      || kubectl_canary get pv "$PV" >/dev/null 2>&1 \
      || kubectl_canary get pv "$STATE_PV" >/dev/null 2>&1; then
      echo "canary cleanup left exact Kubernetes resources behind" >&2
      status=1
    fi
    "$KIND_BIN" delete cluster --name "$CLUSTER" >/dev/null 2>&1 || status=1
  fi
  rm -rf "$WORK_DIR"
  exit "$status"
}
trap cleanup EXIT

[[ "$ADVISOR_CANARY_IMAGE" != "$STUDENT_CANARY_IMAGE" ]] \
  || { echo "advisor and student canary image names must differ" >&2; exit 2; }
echo "Building thin canary images from $BASE_IMAGE"
"$DOCKER_BIN" build \
  --file tests/kubernetes/Dockerfile \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --tag "$ADVISOR_CANARY_IMAGE" \
  .
"$DOCKER_BIN" tag "$ADVISOR_CANARY_IMAGE" "$STUDENT_CANARY_IMAGE"

cat > "$WORK_DIR/kind.yaml" <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
networking:
  disableDefaultCNI: true
  podSubnet: 192.168.0.0/16
EOF

CLUSTER_CREATED=true
"$KIND_BIN" create cluster \
  --name "$CLUSTER" \
  --image "kindest/node:v1.34.8@sha256:02722c2dedddcfc00febf5d27fbeb9b7b2c14294c82109ff4a85d89ac9ba3256" \
  --config "$WORK_DIR/kind.yaml"

# Calico's released manifest is checksum-pinned, then its three image tags are
# replaced by the released multi-architecture manifest-list digests. This is
# the enforcing CNI required by the production NetworkPolicy canary.
CALICO_VERSION=v3.32.1
CALICO_MANIFEST="$WORK_DIR/calico-$CALICO_VERSION.yaml"
curl --proto '=https' --tlsv1.2 -fsSLo "$CALICO_MANIFEST" \
  "https://raw.githubusercontent.com/projectcalico/calico/$CALICO_VERSION/manifests/calico.yaml"
verify_sha256 \
  a1df919d9721cf667accdc3e72848911b0cb25cfab7d2478ad0c996302c95744 \
  "$CALICO_MANIFEST"
sed -i.bak \
  's#quay.io/calico/cni:v3.32.1#quay.io/calico/cni@sha256:bb1567e3ed81e2e8414e9a68f186e1f7ffd4067a4871a9ae90896793af0190dd#g; s#quay.io/calico/kube-controllers:v3.32.1#quay.io/calico/kube-controllers@sha256:18008f781c869376dbbc4dfb1ffe3afb46f7897887d4f20e080c420ac44a6612#g; s#quay.io/calico/node:v3.32.1#quay.io/calico/node@sha256:7f874b3f0b540c2b523aea9961ef5e2f43b0af9056a47874c916d6cf348168d3#g' \
  "$CALICO_MANIFEST"
rm -f "$CALICO_MANIFEST.bak"
! grep -q 'quay.io/calico/.*:v3.32.1' "$CALICO_MANIFEST"
kubectl_canary apply -f "$CALICO_MANIFEST"
kubectl_canary rollout status -n kube-system daemonset/calico-node --timeout=240s
kubectl_canary rollout status -n kube-system deployment/calico-kube-controllers \
  --timeout=240s
kubectl_canary wait --for=condition=Ready node --all --timeout=120s

"$DOCKER_BIN" exec "$NODE" mkdir -p "/var/senpai-ci/$TAG/dataset"
"$DOCKER_BIN" exec "$NODE" mkdir -p "/var/senpai-ci/$TAG/supervisor-state"
"$DOCKER_BIN" exec "$NODE" chmod 0777 "/var/senpai-ci/$TAG/dataset"
"$DOCKER_BIN" exec "$NODE" chmod 0777 "/var/senpai-ci/$TAG/supervisor-state"
"$KIND_BIN" load docker-image --name "$CLUSTER" "$ADVISOR_CANARY_IMAGE"
"$KIND_BIN" load docker-image --name "$CLUSTER" "$STUDENT_CANARY_IMAGE"

# Place an HTTP decoy on the exact IPv4 IMDS address. The unrestricted probe
# must reach it, while every supervisor-capable pod must be denied by Calico.
"$DOCKER_BIN" exec "$NODE" ip address add 169.254.169.254/32 dev lo
cat > "$WORK_DIR/metadata-decoy.yaml" <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: senpai-metadata-decoy
  namespace: kube-system
spec:
  hostNetwork: true
  dnsPolicy: ClusterFirstWithHostNet
  automountServiceAccountToken: false
  containers:
  - name: server
    image: $ADVISOR_CANARY_IMAGE
    imagePullPolicy: IfNotPresent
    command: ["python", "-m", "http.server", "80", "--bind", "169.254.169.254"]
    securityContext:
      runAsUser: 0
      runAsNonRoot: false
      allowPrivilegeEscalation: false
      capabilities:
        drop: ["ALL"]
        add: ["NET_BIND_SERVICE"]
    readinessProbe:
      tcpSocket:
        host: 169.254.169.254
        port: 80
      periodSeconds: 1
      failureThreshold: 30
EOF
kubectl_canary apply -f "$WORK_DIR/metadata-decoy.yaml"
kubectl_canary wait -n kube-system --for=condition=Ready \
  pod/senpai-metadata-decoy --timeout=90s

render() {
  local phase=$1
  local destination=$2
  "$DOCKER_BIN" run --rm --entrypoint python "$ADVISOR_CANARY_IMAGE" \
    /opt/senpai/tests/kubernetes/canary.py render \
    --phase "$phase" \
    --namespace "$NAMESPACE" \
    --other-namespace "$OTHER_NAMESPACE" \
    --tag "$TAG" \
    --advisor-image "$ADVISOR_CANARY_IMAGE" \
    --student-image "$STUDENT_CANARY_IMAGE" \
    --revision "$SOURCE_REVISION" > "$destination"
}

capture_supervisor_rollback() {
  local tag=$1
  local directory=$2
  "$PYTHON_BIN" -c '
import sys
from pathlib import Path

from k8s.supervisor_rollback import SupervisorRollback

rollback = SupervisorRollback.capture(
    tag=sys.argv[1],
    kube_context=sys.argv[2],
    namespace=sys.argv[3],
    directory=Path(sys.argv[4]),
    timeout_seconds=120,
)
print(rollback.path)
' "$tag" "$CONTEXT" "$NAMESPACE" "$directory"
}

expire_supervisor_rollback_lease() {
  local tag=$1
  kubectl_canary patch lease -n "$NAMESPACE" \
    "senpai-supervisor-release-$tag" --type=merge \
    -p '{"spec":{"leaseDurationSeconds":1,"renewTime":"1970-01-01T00:00:00Z"}}' \
    >/dev/null
}

render initial "$INITIAL_MANIFEST"
kubectl_canary apply -f "$INITIAL_MANIFEST"
kubectl_canary wait -n "$OTHER_NAMESPACE" --for=condition=Ready \
  "pod/senpai-decoy-$TAG" --timeout=90s
kubectl_canary rollout status -n "$NAMESPACE" \
  "deployment/senpai-advisor-$TAG" --timeout=120s
kubectl_canary rollout status -n "$NAMESPACE" \
  "deployment/senpai-$TAG-fern" --timeout=120s
kubectl_canary rollout status -n "$NAMESPACE" \
  "deployment/senpai-supervisor-$TAG" --timeout=120s

SUPERVISOR="deployment/senpai-supervisor-$TAG"
ADVISOR="deployment/senpai-advisor-$TAG"
STUDENT="deployment/senpai-$TAG-fern"
ROLE_SECRET="senpai-launch-secrets-$TAG"

role_listener_generation() {
  kubectl_canary exec -n "$NAMESPACE" "$1" -c "$2" -- \
    curl --connect-timeout 1 --max-time 2 -fsS http://127.0.0.1:18765/
}

wait_for_role_listener_advance() {
  local deployment=$1
  local container=$2
  local previous=$3
  local observed=""
  for _ in $(seq 1 60); do
    observed=$(role_listener_generation "$deployment" "$container" 2>/dev/null || true)
    if [[ "$observed" =~ ^[0-9]+$ ]] && (( observed > previous )); then
      printf '%s' "$observed"
      return 0
    fi
    sleep 0.5
  done
  echo "role listener did not return on a replacement generation: $deployment" >&2
  return 1
}

# Trusted controller code and dependencies are image-installed and immutable to
# the role UID. A malicious target cwd/PATH/PYTHONPATH cannot affect liveness.
for role_spec in \
  "$ADVISOR:advisor:/workspace/target:/var/lib/senpai/$TAG/advisor/openhands_state/controller-lease.json" \
  "$STUDENT:student:/workspace/target:/var/lib/senpai/openhands_state/controller-lease.json"; do
  IFS=: read -r deployment container target lease <<EOF
$role_spec
EOF
  kubectl_canary exec -n "$NAMESPACE" "$deployment" -c "$container" -- \
    /bin/sh -c '
      set -eu
      test ! -w /opt/senpai-venv/bin/python
      package=$(/opt/senpai-venv/bin/python -I -c \
        "import pathlib,senpai_agent; print(pathlib.Path(senpai_agent.__file__).parent)")
      case "$package" in /opt/senpai-venv/*) ;; *) exit 1 ;; esac
      test ! -w "$package"
      test ! -e /run/senpai-repair-executor
      cd "$1"
      PATH="$1:$PATH" PYTHONPATH="$1" \
        /usr/local/bin/senpai-container-health "$2"
      test ! -e /tmp/senpai-path-poisoned
      test ! -e /tmp/senpai-sitecustomize-poisoned
    ' -- "$target" "$lease"
done

# The executor socket is a filesystem capability mounted only into the repair
# sidecar. Resolve its path from the rendered workload instead of assuming a
# transport address, then prove the live executor answers on that exact path.
repair_socket_for() {
  kubectl_canary get -n "$NAMESPACE" "$1" -o json | "$PYTHON_BIN" -c '
import json
import sys

deployment = json.load(sys.stdin)
repair = next(
    container
    for container in deployment["spec"]["template"]["spec"]["containers"]
    if container["name"] == "repair"
)
command = repair["command"]
print(command[command.index("--socket") + 1])
'
}
for deployment in "$ADVISOR" "$STUDENT"; do
  repair_socket=$(repair_socket_for "$deployment")
  [[ "$repair_socket" == /run/senpai-repair-executor/* ]]
  kubectl_canary exec -n "$NAMESPACE" "$deployment" -c repair -- \
    /opt/senpai-venv/bin/python -I /usr/local/bin/senpai-repair-executor \
    health --socket "$repair_socket"
done
kubectl_canary exec -n "$NAMESPACE" "$ADVISOR" -c advisor -- \
  /bin/sh -c '
    test "$SENPAI_OPENHANDS_ROLE_FILE" = \
      "$SENPAI_IMMUTABLE_ADVISOR_GUIDANCE_FILE"
    test -f "$SENPAI_OPENHANDS_ROLE_FILE"
    test ! -w "$SENPAI_OPENHANDS_ROLE_FILE"
    grep -q "# Research Advisor" "$SENPAI_OPENHANDS_ROLE_FILE"
  '

# Prove the decoy itself is reachable from an ordinary, unselected pod. If
# this fails, later denials would not demonstrate policy enforcement.
kubectl_canary exec -n "$OTHER_NAMESPACE" "pod/senpai-decoy-$TAG" -- \
  curl --connect-timeout 2 --max-time 5 -fsS \
  http://169.254.169.254/ >/dev/null

# The model-facing shell has neither credentials nor Kubernetes authority.
kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" -c supervisor-shell -- \
  /bin/sh -c '! env | grep -q SENPAI_CI_DUMMY_'
kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" -c supervisor-shell -- \
  test ! -e /var/run/secrets/kubernetes.io/serviceaccount/token

# Its typed repair bridge may execute in the exact secret-free role sidecar.
ADVISOR_LISTENER_BEFORE=$(role_listener_generation "$ADVISOR" advisor)
REPAIR_OUTPUT=$(kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" \
  -c supervisor-shell -- "$ROLE_SHELL" \
  --operation-id canary-advisor-workspace --role advisor --command \
  'test "$(pwd)" = /repair/workspace && test "$(git rev-parse --show-toplevel)" = /repair/workspace && test "$(git rev-parse --abbrev-ref HEAD)" = main && grep -qx advisor-target-workspace canary-target-marker && test ! -e /workspace/senpai && ! curl --connect-timeout 1 --max-time 2 -fsS http://127.0.0.1:18765/ >/dev/null 2>&1 && printf canary-repair-ok')
[[ "$REPAIR_OUTPUT" == "canary-repair-ok" ]]
ADVISOR_LISTENER_AFTER=$(wait_for_role_listener_advance \
  "$ADVISOR" advisor "$ADVISOR_LISTENER_BEFORE")
[[ "$ADVISOR_LISTENER_AFTER" =~ ^[0-9]+$ ]]
STUDENT_LISTENER_BEFORE=$(role_listener_generation "$STUDENT" student)
STUDENT_REPAIR_OUTPUT=$(kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" \
  -c supervisor-shell -- "$ROLE_SHELL" \
  --operation-id canary-student-workspace --role student --student fern --command \
  'test "$(pwd)" = /repair/workspace && test "$(git rev-parse --show-toplevel)" = /repair/workspace && test "$(git rev-parse --abbrev-ref HEAD)" = main && grep -qx student-target-workspace canary-target-marker && test ! -e /workspace/senpai && ! curl --connect-timeout 1 --max-time 2 -fsS http://127.0.0.1:18765/ >/dev/null 2>&1 && printf canary-student-repair-ok')
[[ "$STUDENT_REPAIR_OUTPUT" == "canary-student-repair-ok" ]]
STUDENT_LISTENER_AFTER=$(wait_for_role_listener_advance \
  "$STUDENT" student "$STUDENT_LISTENER_BEFORE")
[[ "$STUDENT_LISTENER_AFTER" =~ ^[0-9]+$ ]]
if kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" -c supervisor-shell -- \
  "$ROLE_SHELL" \
  --operation-id canary-unconfigured-student \
  --role student --student unconfigured --command true; then
  echo "typed repair accepted a target outside the campaign inventory" >&2
  exit 1
fi

# Stable operation IDs make a lost reply safe: exact replay returns the first
# receipt, while changing the payload is rejected and cannot execute twice.
REPLAY_COMMAND='counter=/repair/scratch/canary-replay-count; value=0; test ! -f "$counter" || value=$(cat "$counter"); value=$((value + 1)); printf "%s" "$value" > "$counter"; printf "%s" "$value"'
REPLAY_OUTPUT=$(kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" \
  -c supervisor-shell -- "$ROLE_SHELL" \
  --operation-id canary-repair-replay --role advisor --timeout 10 \
  --command "$REPLAY_COMMAND")
[[ "$REPLAY_OUTPUT" == "1" ]]
REPLAY_STATUS=$(kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" \
  -c supervisor-shell -- "$ROLE_SHELL" \
  --status canary-repair-replay)
[[ "$REPLAY_STATUS" == *'"status":"completed"'* ]]
[[ "$(kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" \
  -c supervisor-shell -- "$ROLE_SHELL" \
  --operation-id canary-repair-replay --role advisor --timeout 10 \
  --command "$REPLAY_COMMAND")" == "1" ]]
[[ "$(kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" \
  -c supervisor-shell -- "$ROLE_SHELL" \
  --operation-id canary-replay-count-check --role advisor \
  --command 'cat /repair/scratch/canary-replay-count')" == "1" ]] \
  || { echo "repair replay executed twice" >&2; exit 1; }
if kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" -c supervisor-shell -- \
  "$ROLE_SHELL" \
  --operation-id canary-repair-replay --role advisor --command 'printf changed'; then
  echo "same operation ID accepted a different command" >&2
  exit 1
fi

# The socket client deliberately drops its first reply, then resolves the
# durable receipt and exact replay without incrementing the role-side counter.
kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" -c supervisor-shell -- \
  python /opt/senpai/tests/kubernetes/canary.py drop-repair-reply \
  --socket /run/senpai-repair/repair.sock --tag "$TAG" \
  --operation-id canary-lost-repair-reply
[[ "$(kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" \
  -c supervisor-shell -- "$ROLE_SHELL" \
  --operation-id canary-lost-reply-count-check --role advisor \
  --command 'cat /repair/scratch/canary-lost-reply-count')" == "1" ]]

# Every supervisor wake gets a fresh shell process tree, HOME, TMP, cwd, and
# environment while its explicit workspace survives. Retired wakes stay stale.
kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" -c supervisor-control -- \
  python /opt/senpai/tests/kubernetes/canary.py probe-terminal-wakes

# A previous unrestricted command may leave a permission-poisoned volatile
# root behind if its container exits at the wrong instant. Restart the real
# sidecars with exact stale fixtures and prove startup removes them rather than
# entering a permanent liveness loop.
SUPERVISOR_POD=$(kubectl_canary get pod -n "$NAMESPACE" \
  -l "app=senpai,role=supervisor,research-tag=$TAG" \
  -o jsonpath='{.items[0].metadata.name}')
ADVISOR_POD=$(kubectl_canary get pod -n "$NAMESPACE" \
  -l "app=senpai,role=advisor,research-tag=$TAG" \
  -o jsonpath='{.items[0].metadata.name}')

container_restart_count() {
  kubectl_canary get pod -n "$NAMESPACE" "$1" \
    -o "jsonpath={.status.containerStatuses[?(@.name==\"$2\")].restartCount}"
}

wait_for_container_restart() {
  local pod=$1
  local container=$2
  local previous=$3
  local count=""
  local ready=""
  for _ in $(seq 1 90); do
    count=$(container_restart_count "$pod" "$container" 2>/dev/null || true)
    ready=$(kubectl_canary get pod -n "$NAMESPACE" "$pod" \
      -o "jsonpath={.status.containerStatuses[?(@.name==\"$container\")].ready}" \
      2>/dev/null || true)
    if [[ "$count" =~ ^[0-9]+$ ]] && (( count > previous )) \
      && [[ "$ready" == true ]]; then
      return 0
    fi
    sleep 1
  done
  echo "container did not recover from the exact stale-root fixture: $pod/$container" >&2
  return 1
}

SHELL_RESTARTS=$(container_restart_count "$SUPERVISOR_POD" supervisor-shell)
kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR_POD" -c supervisor-shell -- \
  /bin/sh -ceu '
    root=/tmp/senpai-terminal-wakes/wake-stale456
    mkdir -p "$root/home/locked"
    printf junk > "$root/home/locked/junk"
    printf "{\"pid\":999999999,\"start_token\":null}" > \
      "$root/.senpai-owner.json"
    chmod 000 "$root/home/locked" "$root"
  '
kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR_POD" -c supervisor-shell -- \
  kill -TERM 1 || true
wait_for_container_restart "$SUPERVISOR_POD" supervisor-shell "$SHELL_RESTARTS"
kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR_POD" -c supervisor-shell -- \
  /bin/sh -c 'test ! -e /tmp/senpai-terminal-wakes/wake-stale456'
kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR_POD" -c supervisor-shell -- \
  /opt/senpai-venv/bin/python -I -m senpai_agent.isolated_terminal_health \
  --socket '@senpai-isolated-terminal'

REPAIR_RESTARTS=$(container_restart_count "$ADVISOR_POD" repair)
kubectl_canary exec -n "$NAMESPACE" "$ADVISOR_POD" -c repair -- \
  /bin/sh -ceu '
    root=/tmp/senpai-repair-operations/operation-stale456
    mkdir -p "$root/home/locked"
    printf junk > "$root/home/locked/junk"
    printf "{\"pid\":999999999,\"start_token\":null}" > \
      "$root/.senpai-owner.json"
    chmod 000 "$root/home/locked" "$root"
  '
kubectl_canary exec -n "$NAMESPACE" "$ADVISOR_POD" -c repair -- \
  kill -TERM 1 || true
wait_for_container_restart "$ADVISOR_POD" repair "$REPAIR_RESTARTS"
kubectl_canary exec -n "$NAMESPACE" "$ADVISOR_POD" -c repair -- \
  /bin/sh -c 'test ! -e /tmp/senpai-repair-operations/operation-stale456'
ADVISOR_REPAIR_SOCKET=$(repair_socket_for "$ADVISOR")
kubectl_canary exec -n "$NAMESPACE" "$ADVISOR_POD" -c repair -- \
  /opt/senpai-venv/bin/python -I /usr/local/bin/senpai-repair-executor \
  health --socket "$ADVISOR_REPAIR_SOCKET"

# Neither the unrestricted model shell nor typed role repairs may reach IMDS.
if kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" -c supervisor-shell -- \
  curl --connect-timeout 1 --max-time 3 -fsS http://169.254.169.254/; then
  echo "supervisor shell reached the metadata decoy" >&2
  exit 1
fi
for role_spec in \
  "canary-imds-advisor:--role advisor" \
  "canary-imds-student:--role student --student fern"; do
  IFS=: read -r operation_id role_args <<EOF
$role_spec
EOF
  # shellcheck disable=SC2086
  if kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" \
    -c supervisor-shell -- "$ROLE_SHELL" \
    --operation-id "$operation_id" $role_args --timeout 5 --command \
    'curl --connect-timeout 1 --max-time 3 -fsS http://169.254.169.254/'; then
    echo "repair sidecar reached the metadata decoy: $role_args" >&2
    exit 1
  fi
done

# The control process receives an explicitly projected token. Its namespaced
# RBAC works in this campaign and is denied in the decoy namespace.
kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" -c supervisor-control -- \
  kubectl get pods -n "$NAMESPACE" --request-timeout=5s >/dev/null
if kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" -c supervisor-control -- \
  kubectl get pods -n "$OTHER_NAMESPACE" --request-timeout=5s >/dev/null; then
  echo "supervisor control crossed its namespace RBAC boundary" >&2
  exit 1
fi

ADVISOR_UID=$(kubectl_canary get pod -n "$NAMESPACE" \
  -l "app=senpai,role=advisor,research-tag=$TAG" \
  -o jsonpath='{.items[0].metadata.uid}')
STUDENT_UID=$(kubectl_canary get pod -n "$NAMESPACE" \
  -l "app=senpai,role=student,student=fern,research-tag=$TAG" \
  -o jsonpath='{.items[0].metadata.uid}')
ROLE_SECRET_VERSION=$(kubectl_canary get secret -n "$NAMESPACE" "$ROLE_SECRET" \
  -o jsonpath='{.metadata.resourceVersion}')
kubectl_canary exec -n "$NAMESPACE" "$ADVISOR" -c advisor -- \
  grep -qx owner-state-preserved \
  "/var/lib/senpai/$TAG/advisor/openhands_state/canary-state-marker"
kubectl_canary exec -n "$NAMESPACE" "$STUDENT" -c student -- \
  grep -qx owner-state-preserved \
  "/var/lib/senpai/openhands_state/canary-state-marker"

# Exercise the real typed kubectl transport and role-owned worker replacement.
kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" -c supervisor-control -- \
  python /opt/senpai/tests/kubernetes/canary.py probe-control \
  --namespace "$NAMESPACE" --tag "$TAG" --timeout 30
[[ "$(kubectl_canary get pod -n "$NAMESPACE" \
  -l "app=senpai,role=advisor,research-tag=$TAG" \
  -o jsonpath='{.items[0].metadata.uid}')" == "$ADVISOR_UID" ]]
[[ "$(kubectl_canary get pod -n "$NAMESPACE" \
  -l "app=senpai,role=student,student=fern,research-tag=$TAG" \
  -o jsonpath='{.items[0].metadata.uid}')" == "$STUDENT_UID" ]]
kubectl_canary exec -n "$NAMESPACE" "$ADVISOR" -c advisor -- \
  grep -qx owner-state-preserved \
  "/var/lib/senpai/$TAG/advisor/openhands_state/canary-state-marker"
kubectl_canary exec -n "$NAMESPACE" "$STUDENT" -c student -- \
  grep -qx owner-state-preserved \
  "/var/lib/senpai/openhands_state/canary-state-marker"

FIRST_SECRET=$(kubectl_canary get deployment -n "$NAMESPACE" \
  "senpai-supervisor-$TAG" -o jsonpath='{.spec.template.spec.containers[?(@.name=="supervisor-control")].env[0].valueFrom.secretKeyRef.name}')
FIRST_CONFIG=$(kubectl_canary get deployment -n "$NAMESPACE" \
  "senpai-supervisor-$TAG" -o jsonpath='{.spec.template.spec.containers[?(@.name=="supervisor-control")].envFrom[0].configMapRef.name}')
SUPERVISOR_STATE_SENTINEL="state-$SOURCE_REVISION-$CANARY_ID"
SUPERVISOR_STATE_MARKER="/var/lib/senpai/$TAG/operational-supervisor/canary-state-marker"
kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" -c supervisor-control -- \
  /bin/sh -c 'printf "%s\n" "$1" > "$2"' -- \
  "$SUPERVISOR_STATE_SENTINEL" "$SUPERVISOR_STATE_MARKER"
INTERRUPTED_OPERATION_KEY=canary-interrupted-general-operation
INTERRUPTED_REPAIR_ID=canary-interrupted-repair-operation
kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" -c supervisor-control -- \
  python /opt/senpai/tests/kubernetes/canary.py seed-interrupted-operations \
  --tag "$TAG" \
  --operation-key "$INTERRUPTED_OPERATION_KEY" \
  --repair-operation-id "$INTERRUPTED_REPAIR_ID"

# Capture the exact mutable supervisor release before attempting an upgrade.
# All production targets are present; a second empty-tag bundle proves that a
# resource introduced after an absent snapshot is removed during restoration.
ROLLBACK_DIR="$WORK_DIR/rollback"
ROLLBACK_BUNDLE=$(capture_supervisor_rollback "$TAG" "$ROLLBACK_DIR")
"$PYTHON_BIN" -c '
import json
import pathlib
import sys

bundle = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert all(record["present"] for record in bundle["resources"])
assert bundle["persistent_state_rolled_back"] is False
' "$ROLLBACK_BUNDLE"

ABSENT_TAG="$TAG-absent"
ABSENT_ROLLBACK_BUNDLE=$(capture_supervisor_rollback \
  "$ABSENT_TAG" "$ROLLBACK_DIR")
ABSENT_SERVICE_ACCOUNT="senpai-supervisor-$ABSENT_TAG"
kubectl_canary create serviceaccount -n "$NAMESPACE" "$ABSENT_SERVICE_ACCOUNT"
# The capture process deliberately exited without releasing its transaction.
# Expire that exact Lease epoch before exercising cross-process recovery.
expire_supervisor_rollback_lease "$ABSENT_TAG"
"$PYTHON_BIN" k8s/supervisor_rollback.py restore \
  "$ABSENT_ROLLBACK_BUNDLE" --timeout-seconds 120
if kubectl_canary get serviceaccount -n "$NAMESPACE" \
  "$ABSENT_SERVICE_ACCOUNT" >/dev/null 2>&1; then
  echo "rollback retained a resource captured as absent" >&2
  exit 1
fi

# A failed supervisor-only release must be observable and exactly reversible
# from the durable snapshot, without rolling back its SQLite state PVC.
render broken-upgrade "$UPGRADE_MANIFEST"
kubectl_canary apply -f "$UPGRADE_MANIFEST"
SECOND_SECRET=$(kubectl_canary get deployment -n "$NAMESPACE" \
  "senpai-supervisor-$TAG" -o jsonpath='{.spec.template.spec.containers[?(@.name=="supervisor-control")].env[0].valueFrom.secretKeyRef.name}')
SECOND_CONFIG=$(kubectl_canary get deployment -n "$NAMESPACE" \
  "senpai-supervisor-$TAG" -o jsonpath='{.spec.template.spec.containers[?(@.name=="supervisor-control")].envFrom[0].configMapRef.name}')
[[ "$FIRST_SECRET" != "$SECOND_SECRET" && "$FIRST_CONFIG" != "$SECOND_CONFIG" ]]
if kubectl_canary rollout status -n "$NAMESPACE" "$SUPERVISOR" --timeout=15s; then
  echo "deliberately broken supervisor release unexpectedly became ready" >&2
  exit 1
fi
expire_supervisor_rollback_lease "$TAG"
"$PYTHON_BIN" k8s/supervisor_rollback.py restore \
  "$ROLLBACK_BUNDLE" --timeout-seconds 120
test -f "$ROLLBACK_BUNDLE"

[[ "$(kubectl_canary get deployment -n "$NAMESPACE" \
  "senpai-supervisor-$TAG" -o jsonpath='{.spec.template.spec.containers[?(@.name=="supervisor-control")].env[0].valueFrom.secretKeyRef.name}')" == "$FIRST_SECRET" ]]
[[ "$(kubectl_canary get deployment -n "$NAMESPACE" \
  "senpai-supervisor-$TAG" -o jsonpath='{.spec.template.spec.containers[?(@.name=="supervisor-control")].envFrom[0].configMapRef.name}')" == "$FIRST_CONFIG" ]]
kubectl_canary get secret -n "$NAMESPACE" "$FIRST_SECRET" "$SECOND_SECRET" >/dev/null
kubectl_canary get configmap -n "$NAMESPACE" "$FIRST_CONFIG" "$SECOND_CONFIG" >/dev/null
[[ "$(kubectl_canary get secret -n "$NAMESPACE" "$ROLE_SECRET" \
  -o jsonpath='{.metadata.resourceVersion}')" == "$ROLE_SECRET_VERSION" ]]
[[ "$(kubectl_canary get pod -n "$NAMESPACE" \
  -l "app=senpai,role=advisor,research-tag=$TAG" \
  -o jsonpath='{.items[0].metadata.uid}')" == "$ADVISOR_UID" ]]
[[ "$(kubectl_canary get pod -n "$NAMESPACE" \
  -l "app=senpai,role=student,student=fern,research-tag=$TAG" \
  -o jsonpath='{.items[0].metadata.uid}')" == "$STUDENT_UID" ]]
kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" -c supervisor-control -- \
  grep -qx "$SUPERVISOR_STATE_SENTINEL" "$SUPERVISOR_STATE_MARKER"
kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" -c supervisor-control -- \
  python /opt/senpai/tests/kubernetes/canary.py assert-interrupted-operations \
  --socket /run/senpai-repair/repair.sock \
  --tag "$TAG" \
  --operation-key "$INTERRUPTED_OPERATION_KEY" \
  --repair-operation-id "$INTERRUPTED_REPAIR_ID"
[[ "$(kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" \
  -c supervisor-shell -- "$ROLE_SHELL" \
  --operation-id canary-interrupted-marker-check --role advisor \
  --command 'test ! -e /repair/scratch/canary-interrupted-repair-ran && printf absent')" == "absent" ]]

echo "Kubernetes production canary passed: enforcing metadata isolation, wake isolation, durable typed repair, owner restart, exact snapshot rollback, and dedicated SQLite state"
