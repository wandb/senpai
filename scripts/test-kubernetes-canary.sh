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
CANARY_ID=$(printf '%s' "$CANARY_ID" | tr -cd '[:alnum:]-' | cut -c1-30)
CLUSTER="senpai-ci-$CANARY_ID"
CONTEXT="kind-$CLUSTER"
NAMESPACE="senpai-ci-$CANARY_ID"
OTHER_NAMESPACE="$NAMESPACE-other"
TAG="ci-$CANARY_ID"
PV="senpai-ci-$TAG"
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
    kubectl_canary get namespace,pv,pvc,deploy,pod,cm,secret,sa,role,rolebinding \
      -A -o yaml > "$DIAGNOSTICS_DIR/resources.yaml" 2>&1 || true
    kubectl_canary get events -A --sort-by=.lastTimestamp \
      > "$DIAGNOSTICS_DIR/events.txt" 2>&1 || true
    kubectl_canary describe deployment,pod -n "$NAMESPACE" \
      > "$DIAGNOSTICS_DIR/describe.txt" 2>&1 || true
    for deployment in "senpai-advisor-$TAG" "senpai-supervisor-$TAG"; do
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
    kubectl_canary delete namespace "$NAMESPACE" "$OTHER_NAMESPACE" \
      --ignore-not-found --wait=true --timeout=60s >/dev/null 2>&1 || true
    kubectl_canary delete pv "$PV" --ignore-not-found --wait=true \
      --timeout=60s >/dev/null 2>&1 || true
    if kubectl_canary get namespace "$NAMESPACE" >/dev/null 2>&1 \
      || kubectl_canary get namespace "$OTHER_NAMESPACE" >/dev/null 2>&1 \
      || kubectl_canary get pv "$PV" >/dev/null 2>&1; then
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
EOF

"$KIND_BIN" create cluster \
  --name "$CLUSTER" \
  --image "kindest/node:v1.34.8@sha256:02722c2dedddcfc00febf5d27fbeb9b7b2c14294c82109ff4a85d89ac9ba3256" \
  --config "$WORK_DIR/kind.yaml" \
  --wait 120s
CLUSTER_CREATED=true
"$DOCKER_BIN" exec "$NODE" mkdir -p "/var/senpai-ci/$TAG"
"$DOCKER_BIN" exec "$NODE" chmod 0777 "/var/senpai-ci/$TAG"
"$KIND_BIN" load docker-image --name "$CLUSTER" "$CANARY_IMAGE"

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
  "deployment/senpai-supervisor-$TAG" --timeout=120s

SUPERVISOR="deployment/senpai-supervisor-$TAG"
ADVISOR="deployment/senpai-advisor-$TAG"
ROLE_SECRET="senpai-launch-secrets-$TAG"

# The model-facing shell has neither credentials nor Kubernetes authority.
kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" -c supervisor-shell -- \
  /bin/sh -c '! env | grep -q SENPAI_CI_DUMMY_'
kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" -c supervisor-shell -- \
  test ! -e /var/run/secrets/kubernetes.io/serviceaccount/token

# Its typed repair bridge may execute in the exact secret-free role sidecar.
REPAIR_OUTPUT=$(kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" \
  -c supervisor-shell -- /home/senpai/.local/bin/senpai-role-shell \
  --role advisor --command 'printf canary-repair-ok')
[[ "$REPAIR_OUTPUT" == "canary-repair-ok" ]]
if kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" -c supervisor-shell -- \
  /home/senpai/.local/bin/senpai-role-shell \
  --research-tag "$TAG-wrong" --role advisor --command true; then
  echo "typed repair accepted a target outside the campaign inventory" >&2
  exit 1
fi

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
ROLE_SECRET_VERSION=$(kubectl_canary get secret -n "$NAMESPACE" "$ROLE_SECRET" \
  -o jsonpath='{.metadata.resourceVersion}')
kubectl_canary exec -n "$NAMESPACE" "$ADVISOR" -c advisor -- \
  grep -qx owner-state-preserved \
  "/var/lib/senpai/$TAG/advisor/openhands_state/canary-state-marker"

# Exercise the real typed kubectl transport and role-owned worker replacement.
kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" -c supervisor-control -- \
  python /opt/senpai/tests/kubernetes/canary.py probe-control \
  --namespace "$NAMESPACE" --tag "$TAG" --timeout 30
[[ "$(kubectl_canary get pod -n "$NAMESPACE" \
  -l "app=senpai,role=advisor,research-tag=$TAG" \
  -o jsonpath='{.items[0].metadata.uid}')" == "$ADVISOR_UID" ]]
kubectl_canary exec -n "$NAMESPACE" "$ADVISOR" -c advisor -- \
  grep -qx owner-state-preserved \
  "/var/lib/senpai/$TAG/advisor/openhands_state/canary-state-marker"

FIRST_SECRET=$(kubectl_canary get deployment -n "$NAMESPACE" \
  "senpai-supervisor-$TAG" -o jsonpath='{.spec.template.spec.containers[?(@.name=="supervisor-control")].env[0].valueFrom.secretKeyRef.name}')
FIRST_CONFIG=$(kubectl_canary get deployment -n "$NAMESPACE" \
  "senpai-supervisor-$TAG" -o jsonpath='{.spec.template.spec.containers[?(@.name=="supervisor-control")].envFrom[0].configMapRef.name}')

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
kubectl_canary exec -n "$NAMESPACE" "$SUPERVISOR" -c supervisor-control -- \
  grep -qx supervisor-state-preserved \
  "/var/lib/senpai/$TAG/operational-supervisor/canary-state-marker"

echo "Kubernetes production canary passed: isolation, typed repair, owner restart, rollback, and durable state"
