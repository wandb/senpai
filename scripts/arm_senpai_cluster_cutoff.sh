#!/usr/bin/env bash
# Arm a cluster-side Senpai cutoff job.
#
# The job waits inside Kubernetes until all expected Senpai pods are Ready,
# records that timestamp on the PVC, sleeps for the requested budget, then
# deletes the tagged Senpai deployments. This keeps the hard cutoff independent
# of the operator laptop staying awake. Operators clean up retained launch
# ConfigMaps and Secrets separately.

set -euo pipefail

CONTEXT="${CONTEXT:-pai-2}"
NAMESPACE="${NAMESPACE:-default}"
KUBECTL="${KUBECTL:-kubectl}"

RUN_SLUG=""
TAGS_CSV=""
EXPECTED_PODS="90"
EXPECTED_DEPLOYMENTS="90"
READINESS_TIMEOUT_MINUTES="30"
BUDGET_HOURS="48"
PVC_CLAIM_NAME="new-pvc"
PVC_MOUNT_PATH="/mnt/new-pvc"
PVC_LOG_ROOT="/mnt/new-pvc/senpai-conversation-logs"
IMAGE=""
START_GATE_PATH=""
DRY_RUN="false"

usage() {
  cat <<'USAGE'
Usage:
  scripts/arm_senpai_cluster_cutoff.sh \
    --run-slug pai2h-charlie-willow-48h \
    --tags-csv charlie-pai2h-48h-r1,...,willow-pai2h-48h-r5

Options:
  --expected-pods N           Expected Ready pod count before the 48h timer starts (default: 90)
  --expected-deployments N    Expected deployment count before the timer starts (default: 90)
  --readiness-timeout-minutes M
                              Arm even if not all pods are Ready after M minutes (default: 30)
  --budget-hours H            Fleet runtime after readiness or timeout arming (default: 48)
  --pvc-claim NAME            PVC claim mounted into cutoff job (default: new-pvc)
  --pvc-mount-path PATH       Mount path inside cutoff job (default: /mnt/new-pvc)
  --pvc-log-root PATH         PVC cutoff-state root (default: /mnt/new-pvc/senpai-conversation-logs)
  --image IMAGE               Immutable cutoff image digest or :sha-<commit> tag
                              (default: this checkout's senpai-cutoff commit tag)
  --start-gate-path PATH      Write this file after readiness or timeout, releasing gated pods
  --dry-run                   Print manifests and helper script without applying
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-slug) RUN_SLUG="$2"; shift 2 ;;
    --tags-csv) TAGS_CSV="$2"; shift 2 ;;
    --expected-pods) EXPECTED_PODS="$2"; shift 2 ;;
    --expected-deployments) EXPECTED_DEPLOYMENTS="$2"; shift 2 ;;
    --readiness-timeout-minutes) READINESS_TIMEOUT_MINUTES="$2"; shift 2 ;;
    --budget-hours) BUDGET_HOURS="$2"; shift 2 ;;
    --pvc-claim) PVC_CLAIM_NAME="$2"; shift 2 ;;
    --pvc-mount-path) PVC_MOUNT_PATH="$2"; shift 2 ;;
    --pvc-log-root) PVC_LOG_ROOT="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --start-gate-path) START_GATE_PATH="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$RUN_SLUG" ] || [ -z "$TAGS_CSV" ]; then
  usage >&2
  exit 2
fi

if [ -z "$IMAGE" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  SOURCE_REVISION="$(git -C "$SCRIPT_DIR/.." rev-parse --verify HEAD 2>/dev/null)" || {
    echo "Unable to resolve the Senpai checkout revision; pass --image." >&2
    exit 2
  }
  IMAGE="ghcr.io/wandb/senpai-cutoff:sha-${SOURCE_REVISION}"
fi
if [[ ! "$IMAGE" =~ @sha256:[0-9a-f]{64}$ && ! "$IMAGE" =~ :sha-[0-9a-f]{40}$ ]]; then
  echo "--image must be an immutable digest or :sha-<40-character-commit> tag" >&2
  exit 2
fi
if [ -n "$START_GATE_PATH" ] && ! python - "$START_GATE_PATH" "$PVC_MOUNT_PATH" <<'PY'
import posixpath
import sys

gate, mount = map(posixpath.normpath, sys.argv[1:])
valid = (
    posixpath.isabs(gate)
    and gate == sys.argv[1]
    and posixpath.isabs(mount)
    and gate.startswith(f"{mount.rstrip('/')}/")
)
if not valid:
    raise SystemExit(
        "ERROR: --start-gate-path must be an absolute normalized file path "
        "beneath the shared PVC --pvc-mount-path"
    )
PY
then
  exit 2
fi

safe_name() {
  python - "$1" <<'PY'
import re
import sys
s = re.sub(r"[^a-z0-9-]+", "-", sys.argv[1].lower()).strip("-")
print((s or "senpai-cutoff")[:45])
PY
}

SAFE_SLUG="$(safe_name "$RUN_SLUG")"
JOB_NAME="senpai-cutoff-${SAFE_SLUG}"
CONFIGMAP_NAME="${JOB_NAME}-script"
SA_NAME="senpai-cutoff"
ROLE_NAME="senpai-cutoff"
ROLEBINDING_NAME="senpai-cutoff"
ARM_ID="$(python - <<'PY'
import uuid
print(uuid.uuid4())
PY
)"
STATE_AUTH_KEY="$(python - <<'PY'
import secrets
print(secrets.token_hex(32))
PY
)"
BUDGET_SECONDS="$(python - "$BUDGET_HOURS" <<'PY'
import sys
print(int(float(sys.argv[1]) * 3600))
PY
)"
READINESS_TIMEOUT_SECONDS="$(python - "$READINESS_TIMEOUT_MINUTES" <<'PY'
import math
import sys

minutes = float(sys.argv[1])
if not math.isfinite(minutes) or minutes < 0:
    raise SystemExit("--readiness-timeout-minutes must be a non-negative number")
print(int(minutes * 60))
PY
)"
read -r ARMING_DEADLINE_EPOCH HARD_KILL_AT_EPOCH < <(
  python - "$READINESS_TIMEOUT_SECONDS" "$BUDGET_SECONDS" <<'PY'
import sys
import time

readiness, budget = map(int, sys.argv[1:])
arming_deadline = int(time.time()) + readiness
print(arming_deadline, arming_deadline + budget)
PY
)

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
CUTOFF_SCRIPT="${TMP_DIR}/cutoff-job.sh"
RBAC_MANIFEST="${TMP_DIR}/rbac.yaml"
JOB_MANIFEST="${TMP_DIR}/job.yaml"

cat > "$CUTOFF_SCRIPT" <<'JOBSCRIPT'
#!/usr/bin/env bash
set -euo pipefail
umask 077

RUN_SLUG="${RUN_SLUG:?}"
TAGS_CSV="${TAGS_CSV:?}"
EXPECTED_PODS="${EXPECTED_PODS:?}"
EXPECTED_DEPLOYMENTS="${EXPECTED_DEPLOYMENTS:?}"
READINESS_TIMEOUT_SECONDS="${READINESS_TIMEOUT_SECONDS:?}"
BUDGET_SECONDS="${BUDGET_SECONDS:?}"
ARMING_DEADLINE_EPOCH="${ARMING_DEADLINE_EPOCH:?}"
HARD_KILL_AT_EPOCH="${HARD_KILL_AT_EPOCH:?}"
REQUESTED_ARM_ID="${ARM_ID:?}"
STATE_AUTH_KEY="${STATE_AUTH_KEY:?}"
PVC_LOG_ROOT="${PVC_LOG_ROOT:?}"
START_GATE_PATH="${START_GATE_PATH:-}"
NAMESPACE="${NAMESPACE:-default}"
SELECTOR="research-tag in (${TAGS_CSV})"

RUN_DIR="${PVC_LOG_ROOT}/${RUN_SLUG}"
STATE_FILE="${RUN_DIR}/cutoff_state.json"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

if ! mkdir -p "$RUN_DIR"; then
  log "Shared cutoff telemetry directory is unavailable; continuing fail closed"
fi

utc_from_epoch() {
  python - "$1" <<'PY'
import datetime as dt
import sys
print(dt.datetime.fromtimestamp(int(sys.argv[1]), tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
PY
}

pod_counts() {
  kubectl -n "$NAMESPACE" get pods -l "$SELECTOR" -o json | python -c '
import json
import sys
data = json.load(sys.stdin)
items = data.get("items", [])
ready = 0
for item in items:
    statuses = item.get("status", {}).get("containerStatuses") or []
    if statuses and all(s.get("ready") for s in statuses):
        ready += 1
print(f"{len(items)} {ready}")
'
}

deployment_count() {
  kubectl -n "$NAMESPACE" get deployments -l "$SELECTOR" --no-headers 2>/dev/null | wc -l | tr -d ' '
}

write_state() {
  local reason="$1" now tmp persisted="true"
  now="$(date -u '+%s')"
  PERSISTED_ARM_ID="$REQUESTED_ARM_ID"
  ARMED_AT_UTC="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  ARM_REASON="$reason"
  KILL_AT_EPOCH=$((now + BUDGET_SECONDS))
  if [ "$KILL_AT_EPOCH" -gt "$HARD_KILL_AT_EPOCH" ]; then
    KILL_AT_EPOCH="$HARD_KILL_AT_EPOCH"
  fi
  KILL_AT_UTC="$(utc_from_epoch "$KILL_AT_EPOCH")"
  if ! tmp="$(mktemp "${STATE_FILE}.tmp.XXXXXX")"; then
    log "Shared cutoff telemetry is unavailable; using in-memory deadline"
    return 0
  fi
  python - \
    "$tmp" "$STATE_AUTH_KEY" "$REQUESTED_ARM_ID" "$RUN_SLUG" "$TAGS_CSV" \
    "$ARMED_AT_UTC" "$reason" "$KILL_AT_EPOCH" "$KILL_AT_UTC" "$EXPECTED_PODS" \
    "$EXPECTED_DEPLOYMENTS" "$SELECTOR" "$START_GATE_PATH" <<'PY' || persisted="false"
import hashlib
import hmac
import json
import sys

(
    path,
    key,
    arm_id,
    run_slug,
    tags_csv,
    armed_at,
    reason,
    kill_at,
    kill_at_utc,
    expected_pods,
    expected_deployments,
    selector,
    start_gate_path,
) = sys.argv[1:]
payload = {
    "PERSISTED_ARM_ID": arm_id,
    "RUN_SLUG": run_slug,
    "TAGS_CSV": tags_csv,
    "ARMED_AT_UTC": armed_at,
    "ARM_REASON": reason,
    "KILL_AT_EPOCH": int(kill_at),
    "KILL_AT_UTC": kill_at_utc,
    "EXPECTED_PODS": int(expected_pods),
    "EXPECTED_DEPLOYMENTS": int(expected_deployments),
    "SELECTOR": selector,
    "START_GATE_PATH": start_gate_path,
}
encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
document = {
    "payload": payload,
    "mac": hmac.new(key.encode(), encoded, hashlib.sha256).hexdigest(),
}
with open(path, "w", encoding="utf-8") as file:
    json.dump(document, file, sort_keys=True, separators=(",", ":"))
    file.write("\n")
PY
  if [ "$persisted" = "true" ] && ! mv "$tmp" "$STATE_FILE"; then
    persisted="false"
  fi
  if [ "$persisted" = "false" ]; then
    rm -f "$tmp" || log "Unable to remove shared cutoff telemetry temporary"
    log "Shared cutoff telemetry changed concurrently; using in-memory deadline"
  fi
}

read_state_value() {
  python - "$STATE_FILE" "$STATE_AUTH_KEY" "$1" <<'PY'
import hashlib
import hmac
import json
import os
import stat
import sys

path, key, requested = sys.argv[1:]
allowed = {
    "PERSISTED_ARM_ID",
    "RUN_SLUG",
    "TAGS_CSV",
    "ARMED_AT_UTC",
    "ARM_REASON",
    "KILL_AT_EPOCH",
    "KILL_AT_UTC",
    "EXPECTED_PODS",
    "EXPECTED_DEPLOYMENTS",
    "SELECTOR",
    "START_GATE_PATH",
}
if requested not in allowed:
    raise SystemExit("invalid cutoff state field")
fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
try:
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 16_384:
        raise SystemExit("invalid cutoff state file")
    with os.fdopen(fd, encoding="utf-8") as file:
        document = json.load(file)
    fd = -1
finally:
    if fd >= 0:
        os.close(fd)
if set(document) != {"payload", "mac"} or not isinstance(document["payload"], dict):
    raise SystemExit("invalid cutoff state document")
payload = document["payload"]
if set(payload) != allowed:
    raise SystemExit("invalid cutoff state fields")
encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
expected_mac = hmac.new(key.encode(), encoded, hashlib.sha256).hexdigest()
if not hmac.compare_digest(document["mac"], expected_mac):
    raise SystemExit("cutoff state authentication failed")
if not isinstance(payload["KILL_AT_EPOCH"], int) or payload["KILL_AT_EPOCH"] < 0:
    raise SystemExit("invalid cutoff deadline")
value = payload[requested]
if not isinstance(value, (str, int)):
    raise SystemExit("invalid cutoff state value")
print(value, end="")
PY
}

load_state() {
  PERSISTED_ARM_ID="$(read_state_value PERSISTED_ARM_ID)" &&
    ARMED_AT_UTC="$(read_state_value ARMED_AT_UTC)" &&
    KILL_AT_EPOCH="$(read_state_value KILL_AT_EPOCH)" &&
    KILL_AT_UTC="$(read_state_value KILL_AT_UTC)"
}

open_start_gate() {
  local tmp
  [ -n "$START_GATE_PATH" ] || return 0
  if ! mkdir -p "$(dirname "$START_GATE_PATH")" || \
     ! tmp="$(mktemp "${START_GATE_PATH}.tmp.XXXXXX")"; then
    log "Unable to open the shared start gate; failing so the Job restarts"
    return 1
  fi
  {
    printf 'RUN_SLUG=%q\n' "$RUN_SLUG"
    printf 'TAGS_CSV=%q\n' "$TAGS_CSV"
    printf 'OPENED_AT_UTC=%q\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'KILL_AT_UTC=%q\n' "${KILL_AT_UTC:-}"
    printf 'SELECTOR=%q\n' "$SELECTOR"
  } > "$tmp" || {
    rm -f "$tmp" || log "Unable to remove shared start-gate temporary"
    log "Unable to write the shared start gate; failing so the Job restarts"
    return 1
  }
  if ! mv "$tmp" "$START_GATE_PATH"; then
    rm -f "$tmp" || log "Unable to remove shared start-gate temporary"
    log "Unable to publish the shared start gate; failing so the Job restarts"
    return 1
  fi
  log "Opened start gate: ${START_GATE_PATH}"
}

wait_for_ready_gate() {
  local total ready deploys deadline now delay existing_arm_id
  if [ -f "$STATE_FILE" ]; then
    if existing_arm_id="$(read_state_value PERSISTED_ARM_ID 2>/dev/null)" && \
       [ "$existing_arm_id" = "$REQUESTED_ARM_ID" ] && load_state; then
      log "Loaded existing cutoff state: KILL_AT_UTC=${KILL_AT_UTC}"
      if [ "$KILL_AT_EPOCH" -gt "$HARD_KILL_AT_EPOCH" ]; then
        KILL_AT_EPOCH="$HARD_KILL_AT_EPOCH"
        KILL_AT_UTC="$(utc_from_epoch "$KILL_AT_EPOCH")"
      fi
      open_start_gate
      return 0
    fi
    log "Discarding cutoff state from an earlier arm of this run slug"
    rm -f "$STATE_FILE" || log "Unable to remove stale cutoff telemetry"
    [ -z "$START_GATE_PATH" ] || rm -f "$START_GATE_PATH" || \
      log "Unable to remove stale start gate"
  fi

  deadline="$ARMING_DEADLINE_EPOCH"
  log "Waiting for ready gate: expected pods=${EXPECTED_PODS}, deployments=${EXPECTED_DEPLOYMENTS}, timeout=${READINESS_TIMEOUT_SECONDS}s, selector=${SELECTOR}"
  while true; do
    read -r total ready < <(pod_counts)
    deploys="$(deployment_count)"
    log "Ready gate poll: ready=${ready}/${EXPECTED_PODS}, pods=${total}/${EXPECTED_PODS}, deployments=${deploys}/${EXPECTED_DEPLOYMENTS}"
    if [ "$total" = "$EXPECTED_PODS" ] && [ "$ready" = "$EXPECTED_PODS" ] && [ "$deploys" = "$EXPECTED_DEPLOYMENTS" ]; then
      write_state "all_ready"
      log "Ready gate passed: ARMED_AT_UTC=${ARMED_AT_UTC}, KILL_AT_UTC=${KILL_AT_UTC}"
      open_start_gate
      return 0
    fi
    now="$(date -u '+%s')"
    if [ "$now" -ge "$deadline" ]; then
      log "Readiness deadline reached; arming cutoff anyway"
      write_state "readiness_timeout"
      open_start_gate
      return 0
    fi
    delay=$((deadline - now))
    [ "$delay" -gt 60 ] && delay=60
    sleep "$delay"
  done
}

sleep_until() {
  local target="$1" label="$2" now delay chunk
  case "$target" in
    ''|*[!0-9]*) log "Invalid deadline for ${label}: '${target}'"; return 1 ;;
  esac
  while true; do
    now="$(date -u '+%s')"
    delay=$((target - now))
    if [ "$delay" -le 0 ]; then
      log "Deadline for ${label} reached"
      return 0
    fi
    chunk="$delay"
    [ "$chunk" -gt 600 ] && chunk=600
    log "Sleeping ${chunk}s until ${label} (${delay}s remaining)"
    sleep "$chunk"
  done
}

delete_targets() {
  local deploys
  deploys="$(deployment_count)"
  if [ "$deploys" = "0" ]; then
    log "No target deployments remain; the delete command is idempotent"
  fi
  log "Deleting deployments with selector: ${SELECTOR}"
  kubectl -n "$NAMESPACE" delete deployments -l "$SELECTOR" --ignore-not-found=true
  log "ConfigMaps and Secrets remain for operator cleanup"
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "${RUN_DIR}/delete_done_utc.txt" || \
    log "Unable to persist cutoff completion telemetry"
  log "Delete command completed"
}

main() {
  log "Cluster cutoff job starting: ${RUN_SLUG}"
  wait_for_ready_gate
  sleep_until "$KILL_AT_EPOCH" "hard cutoff delete"
  delete_targets
  log "Cluster cutoff job done: ${RUN_SLUG}"
}

main "$@"
JOBSCRIPT

chmod +x "$CUTOFF_SCRIPT"

cat > "$RBAC_MANIFEST" <<EOF_RBAC
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ${SA_NAME}
  namespace: ${NAMESPACE}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: ${ROLE_NAME}
  namespace: ${NAMESPACE}
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: ${ROLEBINDING_NAME}
  namespace: ${NAMESPACE}
subjects:
- kind: ServiceAccount
  name: ${SA_NAME}
  namespace: ${NAMESPACE}
roleRef:
  kind: Role
  name: ${ROLE_NAME}
  apiGroup: rbac.authorization.k8s.io
EOF_RBAC

"$KUBECTL" --context "$CONTEXT" -n "$NAMESPACE" create configmap "$CONFIGMAP_NAME" \
  --from-file=cutoff-job.sh="$CUTOFF_SCRIPT" \
  --dry-run=client -o yaml > "${TMP_DIR}/configmap.yaml"

cat > "$JOB_MANIFEST" <<EOF_JOB
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: senpai-cutoff
    run-slug: ${SAFE_SLUG}
spec:
  backoffLimit: 20
  template:
    metadata:
      labels:
        app: senpai-cutoff
        run-slug: ${SAFE_SLUG}
    spec:
      serviceAccountName: ${SA_NAME}
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        runAsGroup: 10001
        fsGroup: 10001
        seccompProfile:
          type: RuntimeDefault
      restartPolicy: OnFailure
      containers:
      - name: cutoff
        image: ${IMAGE}
        imagePullPolicy: Always
        command: ["/bin/bash", "/opt/senpai-cutoff/cutoff-job.sh"]
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop: ["ALL"]
        env:
        - name: RUN_SLUG
          value: "${RUN_SLUG}"
        - name: TAGS_CSV
          value: "${TAGS_CSV}"
        - name: EXPECTED_PODS
          value: "${EXPECTED_PODS}"
        - name: EXPECTED_DEPLOYMENTS
          value: "${EXPECTED_DEPLOYMENTS}"
        - name: READINESS_TIMEOUT_SECONDS
          value: "${READINESS_TIMEOUT_SECONDS}"
        - name: BUDGET_SECONDS
          value: "${BUDGET_SECONDS}"
        - name: ARMING_DEADLINE_EPOCH
          value: "${ARMING_DEADLINE_EPOCH}"
        - name: HARD_KILL_AT_EPOCH
          value: "${HARD_KILL_AT_EPOCH}"
        - name: ARM_ID
          value: "${ARM_ID}"
        - name: STATE_AUTH_KEY
          value: "${STATE_AUTH_KEY}"
        - name: PVC_LOG_ROOT
          value: "${PVC_LOG_ROOT}"
        - name: START_GATE_PATH
          value: "${START_GATE_PATH}"
        - name: NAMESPACE
          value: "${NAMESPACE}"
        volumeMounts:
        - name: cutoff-script
          mountPath: /opt/senpai-cutoff
        - name: dataset
          mountPath: ${PVC_MOUNT_PATH}
      volumes:
      - name: cutoff-script
        configMap:
          name: ${CONFIGMAP_NAME}
          defaultMode: 0755
      - name: dataset
        persistentVolumeClaim:
          claimName: ${PVC_CLAIM_NAME}
EOF_JOB

if [ "$DRY_RUN" = "true" ]; then
  echo "--- RBAC ---"
  cat "$RBAC_MANIFEST"
  echo "--- ConfigMap ---"
  cat "${TMP_DIR}/configmap.yaml"
  echo "--- Job ---"
  cat "$JOB_MANIFEST"
  exit 0
fi

"$KUBECTL" --context "$CONTEXT" apply -f "$RBAC_MANIFEST"
"$KUBECTL" --context "$CONTEXT" apply -f "${TMP_DIR}/configmap.yaml"
"$KUBECTL" --context "$CONTEXT" -n "$NAMESPACE" delete job "$JOB_NAME" --ignore-not-found=true
"$KUBECTL" --context "$CONTEXT" apply -f "$JOB_MANIFEST"

echo "Armed cluster cutoff job: ${JOB_NAME}"
echo "PVC cutoff state: ${PVC_LOG_ROOT}/${RUN_SLUG}"
echo "Monitor: kubectl --context ${CONTEXT} -n ${NAMESPACE} logs -f job/${JOB_NAME}"
