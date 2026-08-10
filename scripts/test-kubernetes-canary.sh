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
CANARY_IMAGE=${CANARY_IMAGE:-senpai-kubernetes-canary:sha-$SOURCE_REVISION}
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
WORK_DIR=$(mktemp -d)
INITIAL_MANIFEST="$WORK_DIR/initial.yaml"
UPGRADE_MANIFEST="$WORK_DIR/broken-upgrade.yaml"
CLUSTER_CREATED=false

kubectl_canary() {
  "$KUBECTL_BIN" --context "$CONTEXT" "$@"
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
    for deployment in \
      "senpai-advisor-$TAG" \
      "senpai-$TAG-fern" \
      "senpai-supervisor-$TAG"; do
      kubectl_canary logs -n "$NAMESPACE" "deployment/$deployment" \
        --all-containers --prefix --tail=500 \
        > "$DIAGNOSTICS_DIR/${deployment}.log" 2>&1 || true
    done
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

echo "Building thin canary image $CANARY_IMAGE from $BASE_IMAGE"
"$DOCKER_BIN" build \
  --file tests/kubernetes/Dockerfile \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --tag "$CANARY_IMAGE" \
  .

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
printf '%s  %s\n' \
  a1df919d9721cf667accdc3e72848911b0cb25cfab7d2478ad0c996302c95744 \
  "$CALICO_MANIFEST" | sha256sum --check
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
"$KIND_BIN" load docker-image --name "$CLUSTER" "$CANARY_IMAGE"

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
    image: $CANARY_IMAGE
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
  "$DOCKER_BIN" run --rm --entrypoint python "$CANARY_IMAGE" \
    /opt/senpai/tests/kubernetes/canary.py render \
    --phase "$phase" \
    --namespace "$NAMESPACE" \
    --other-namespace "$OTHER_NAMESPACE" \
    --tag "$TAG" \
    --image "$CANARY_IMAGE" \
    --revision "$SOURCE_REVISION" > "$destination"
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
      cd "$1"
      PATH="$1:$PATH" PYTHONPATH="$1" \
        /usr/local/bin/senpai-container-health "$2"
      test ! -e /tmp/senpai-path-poisoned
      test ! -e /tmp/senpai-sitecustomize-poisoned
    ' -- "$target" "$lease"
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
REPAIR_OUTPUT=$(kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" \
  -c supervisor-shell -- /home/senpai/.local/bin/senpai-role-shell \
  --role advisor --command \
  'test "$(pwd)" = /repair/workspace && test -d .git && grep -qx advisor-target-workspace canary-target-marker && test ! -e /workspace/senpai && printf canary-repair-ok')
[[ "$REPAIR_OUTPUT" == "canary-repair-ok" ]]
STUDENT_REPAIR_OUTPUT=$(kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" \
  -c supervisor-shell -- /home/senpai/.local/bin/senpai-role-shell \
  --role student --student fern --command \
  'test "$(pwd)" = /repair/workspace && test -d .git && grep -qx student-target-workspace canary-target-marker && test ! -e /workspace/senpai && printf canary-student-repair-ok')
[[ "$STUDENT_REPAIR_OUTPUT" == "canary-student-repair-ok" ]]
if kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" -c supervisor-shell -- \
  /home/senpai/.local/bin/senpai-role-shell \
  --role student --student unconfigured --command true; then
  echo "typed repair accepted a target outside the campaign inventory" >&2
  exit 1
fi

# Neither the unrestricted model shell nor typed role repairs may reach IMDS.
if kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" -c supervisor-shell -- \
  curl --connect-timeout 1 --max-time 3 -fsS http://169.254.169.254/; then
  echo "supervisor shell reached the metadata decoy" >&2
  exit 1
fi
for role_args in "--role advisor" "--role student --student fern"; do
  # shellcheck disable=SC2086
  if kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" \
    -c supervisor-shell -- /home/senpai/.local/bin/senpai-role-shell \
    $role_args --timeout 5 --command \
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

# A failed supervisor-only release must be observable and exactly reversible.
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
kubectl_canary rollout undo -n "$NAMESPACE" "$SUPERVISOR"
kubectl_canary rollout status -n "$NAMESPACE" "$SUPERVISOR" --timeout=120s

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

echo "Kubernetes production canary passed: enforcing metadata isolation, advisor/student typed repair, owner restart, rollback, and dedicated durable state"
