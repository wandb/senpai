#!/usr/bin/env bash
# Arm a cluster-side Senpai cutoff job.
#
# The job waits inside Kubernetes until all expected Senpai pods are Ready,
# records that timestamp on the PVC, sleeps for the requested budget, harvests
# Claude Code conversation logs from /root/.claude, writes them to the PVC, then
# deletes the tagged Senpai deployments/configmaps/secrets. This keeps the hard
# cutoff independent of the operator laptop staying awake.

set -euo pipefail

CONTEXT="${CONTEXT:-pai-2}"
NAMESPACE="${NAMESPACE:-default}"
KUBECTL="${KUBECTL:-kubectl}"
REPO_ROOT="${REPO_ROOT:-/Users/mmcguire/ML/senpai}"
CONVERSATION_LOG_DIR="${CONVERSATION_LOG_DIR:-${REPO_ROOT}/conversation_logs}"

RUN_SLUG=""
TAGS_CSV=""
EXPECTED_PODS="90"
EXPECTED_DEPLOYMENTS="90"
BUDGET_HOURS="48"
HARVEST_LEAD_SECONDS="900"
PVC_CLAIM_NAME="new-pvc"
PVC_MOUNT_PATH="/mnt/new-pvc"
PVC_LOG_ROOT="/mnt/new-pvc/senpai-conversation-logs"
IMAGE="ghcr.io/wandb/senpai:latest"
MAX_PARALLEL_COPIES="16"
START_GATE_PATH=""
START_LOCAL_PULL="true"
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
  --budget-hours H            Fleet runtime after all pods are Ready (default: 48)
  --harvest-lead-seconds S    Seconds before cutoff to harvest Claude logs (default: 900)
  --pvc-claim NAME            PVC claim mounted into cutoff job (default: new-pvc)
  --pvc-mount-path PATH       Mount path inside cutoff job (default: /mnt/new-pvc)
  --pvc-log-root PATH         PVC output root (default: /mnt/new-pvc/senpai-conversation-logs)
  --image IMAGE               Image containing bash, python, tar, and kubectl (default: ghcr.io/wandb/senpai:latest)
  --max-parallel-copies N     Concurrent pod log tar streams during harvest (default: 16)
  --start-gate-path PATH      Write this file after all pods are Ready, releasing gated pods
  --no-local-pull             Do not start the best-effort local PVC mirror process
  --dry-run                   Print manifests and helper script without applying
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-slug) RUN_SLUG="$2"; shift 2 ;;
    --tags-csv) TAGS_CSV="$2"; shift 2 ;;
    --expected-pods) EXPECTED_PODS="$2"; shift 2 ;;
    --expected-deployments) EXPECTED_DEPLOYMENTS="$2"; shift 2 ;;
    --budget-hours) BUDGET_HOURS="$2"; shift 2 ;;
    --harvest-lead-seconds) HARVEST_LEAD_SECONDS="$2"; shift 2 ;;
    --pvc-claim) PVC_CLAIM_NAME="$2"; shift 2 ;;
    --pvc-mount-path) PVC_MOUNT_PATH="$2"; shift 2 ;;
    --pvc-log-root) PVC_LOG_ROOT="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --max-parallel-copies) MAX_PARALLEL_COPIES="$2"; shift 2 ;;
    --start-gate-path) START_GATE_PATH="$2"; shift 2 ;;
    --no-local-pull) START_LOCAL_PULL="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$RUN_SLUG" ] || [ -z "$TAGS_CSV" ]; then
  usage >&2
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
BUDGET_SECONDS="$(python - "$BUDGET_HOURS" <<'PY'
import sys
print(int(float(sys.argv[1]) * 3600))
PY
)"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
CUTOFF_SCRIPT="${TMP_DIR}/cutoff-job.sh"
RBAC_MANIFEST="${TMP_DIR}/rbac.yaml"
JOB_MANIFEST="${TMP_DIR}/job.yaml"
LOCAL_PULL_SCRIPT="${CONVERSATION_LOG_DIR}/_${RUN_SLUG}_pull_from_pvc.sh"
LOCAL_PULL_LOG="${CONVERSATION_LOG_DIR}/_${RUN_SLUG}_pull_from_pvc.log"

mkdir -p "$CONVERSATION_LOG_DIR"

cat > "$CUTOFF_SCRIPT" <<'JOBSCRIPT'
#!/usr/bin/env bash
set -euo pipefail

RUN_SLUG="${RUN_SLUG:?}"
TAGS_CSV="${TAGS_CSV:?}"
EXPECTED_PODS="${EXPECTED_PODS:?}"
EXPECTED_DEPLOYMENTS="${EXPECTED_DEPLOYMENTS:?}"
BUDGET_SECONDS="${BUDGET_SECONDS:?}"
HARVEST_LEAD_SECONDS="${HARVEST_LEAD_SECONDS:?}"
PVC_LOG_ROOT="${PVC_LOG_ROOT:?}"
MAX_PARALLEL_COPIES="${MAX_PARALLEL_COPIES:?}"
START_GATE_PATH="${START_GATE_PATH:-}"
NAMESPACE="${NAMESPACE:-default}"
SELECTOR="research-tag in (${TAGS_CSV})"

RUN_DIR="${PVC_LOG_ROOT}/${RUN_SLUG}"
STATE_FILE="${RUN_DIR}/cutoff_state.env"
mkdir -p "$RUN_DIR"
exec > >(tee -a "${RUN_DIR}/cutoff-job.log") 2>&1

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

utc_from_epoch() {
  python - "$1" <<'PY'
import datetime as dt
import sys
print(dt.datetime.fromtimestamp(int(sys.argv[1]), tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
PY
}

safe_name() {
  python - "$1" <<'PY'
import re
import sys
print(re.sub(r"[^A-Za-z0-9_.=-]+", "_", sys.argv[1]).strip("_") or "pod")
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
  local now kill_at kill_at_utc tmp
  now="$(date -u '+%s')"
  kill_at=$((now + BUDGET_SECONDS))
  kill_at_utc="$(utc_from_epoch "$kill_at")"
  tmp="${STATE_FILE}.tmp"
  {
    printf 'RUN_SLUG=%q\n' "$RUN_SLUG"
    printf 'TAGS_CSV=%q\n' "$TAGS_CSV"
    printf 'LAST_READY_UTC=%q\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'KILL_AT_EPOCH=%q\n' "$kill_at"
    printf 'KILL_AT_UTC=%q\n' "$kill_at_utc"
    printf 'EXPECTED_PODS=%q\n' "$EXPECTED_PODS"
    printf 'EXPECTED_DEPLOYMENTS=%q\n' "$EXPECTED_DEPLOYMENTS"
    printf 'SELECTOR=%q\n' "$SELECTOR"
    printf 'START_GATE_PATH=%q\n' "$START_GATE_PATH"
  } > "$tmp"
  mv "$tmp" "$STATE_FILE"
}

open_start_gate() {
  local tmp
  [ -n "$START_GATE_PATH" ] || return 0
  mkdir -p "$(dirname "$START_GATE_PATH")"
  tmp="${START_GATE_PATH}.tmp"
  {
    printf 'RUN_SLUG=%q\n' "$RUN_SLUG"
    printf 'TAGS_CSV=%q\n' "$TAGS_CSV"
    printf 'OPENED_AT_UTC=%q\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'KILL_AT_UTC=%q\n' "${KILL_AT_UTC:-}"
    printf 'SELECTOR=%q\n' "$SELECTOR"
  } > "$tmp"
  mv "$tmp" "$START_GATE_PATH"
  log "Opened start gate: ${START_GATE_PATH}"
}

wait_for_ready_gate() {
  local total ready deploys
  if [ -f "$STATE_FILE" ]; then
    # shellcheck source=/dev/null
    source "$STATE_FILE"
    log "Loaded existing cutoff state: LAST_READY_UTC=${LAST_READY_UTC}, KILL_AT_UTC=${KILL_AT_UTC}"
    open_start_gate
    return 0
  fi

  log "Waiting for ready gate: expected pods=${EXPECTED_PODS}, deployments=${EXPECTED_DEPLOYMENTS}, selector=${SELECTOR}"
  while true; do
    read -r total ready < <(pod_counts)
    deploys="$(deployment_count)"
    log "Ready gate poll: ready=${ready}/${EXPECTED_PODS}, pods=${total}/${EXPECTED_PODS}, deployments=${deploys}/${EXPECTED_DEPLOYMENTS}"
    if [ "$total" = "$EXPECTED_PODS" ] && [ "$ready" = "$EXPECTED_PODS" ] && [ "$deploys" = "$EXPECTED_DEPLOYMENTS" ]; then
      write_state
      # shellcheck source=/dev/null
      source "$STATE_FILE"
      log "Ready gate passed: LAST_READY_UTC=${LAST_READY_UTC}, KILL_AT_UTC=${KILL_AT_UTC}"
      open_start_gate
      return 0
    fi
    sleep 60
  done
}

sleep_until() {
  local target="$1" label="$2" now delay chunk
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

copy_one_pod() {
  local pod="$1" tag="$2" dest="$3" pod_dir
  pod_dir="${dest}/pods/$(safe_name "$tag")/$(safe_name "$pod")"
  mkdir -p "$pod_dir"
  printf '%s\t%s\n' "$pod" "$tag" > "${pod_dir}/pod.tsv"
  log "Harvesting Claude Code logs from ${pod} (${tag})"
  if kubectl -n "$NAMESPACE" exec "$pod" -- sh -lc '
      cd /root || exit 0
      set --
      [ -d .claude/projects ] && set -- "$@" .claude/projects
      [ -d .claude/todos ] && set -- "$@" .claude/todos
      [ -f .claude.json ] && set -- "$@" .claude.json
      [ "$#" -gt 0 ] || exit 0
      tar -czf - "$@"
    ' > "${pod_dir}/claude-code-logs.tgz" 2> "${pod_dir}/harvest.err"; then
    log "Harvested ${pod}"
  else
    log "WARN: failed to harvest ${pod}; see ${pod_dir}/harvest.err"
    return 1
  fi
}

harvest_claude_logs() {
  local ts dest copy_failed job_count pod_count
  ts="$(date -u '+%Y%m%dT%H%M%SZ')"
  dest="${RUN_DIR}/${ts}_claude_logs"
  mkdir -p "${dest}/pods"
  cp "$STATE_FILE" "${dest}/schedule.env"
  log "Harvest destination on PVC: ${dest}"

  kubectl -n "$NAMESPACE" get pods -l "$SELECTOR" -o json | python -c '
import json
import sys
data = json.load(sys.stdin)
for item in data.get("items", []):
    meta = item.get("metadata", {})
    print("{}\t{}".format(meta.get("name", ""), meta.get("labels", {}).get("research-tag", "")))
' > "${dest}/pod_list.tsv"

  pod_count="$(wc -l < "${dest}/pod_list.tsv" | tr -d ' ')"
  log "Pods selected for Claude log harvest: ${pod_count}"
  if [ "$pod_count" = "0" ]; then
    log "WARN: no pods found at harvest time"
    return 0
  fi

  copy_failed=0
  while IFS="$(printf '\t')" read -r pod tag; do
    [ -n "$pod" ] || continue
    while true; do
      job_count="$(jobs -pr | wc -l | tr -d ' ')"
      [ "$job_count" -lt "$MAX_PARALLEL_COPIES" ] && break
      sleep 1
    done
    copy_one_pod "$pod" "$tag" "$dest" &
  done < "${dest}/pod_list.tsv"

  for job in $(jobs -pr); do
    if ! wait "$job"; then
      copy_failed=1
    fi
  done

  if [ "$copy_failed" -ne 0 ]; then
    log "WARN: one or more Claude log harvests failed"
  else
    log "All Claude log harvests completed"
  fi
  printf '%s\n' "$dest" > "${RUN_DIR}/latest_harvest_path.txt"
}

delete_targets() {
  local deploys
  deploys="$(deployment_count)"
  if [ "$deploys" = "0" ]; then
    log "No target deployments remain; delete already complete or launch never existed"
    return 0
  fi
  log "Deleting deployments/configmaps/secrets with selector: ${SELECTOR}"
  kubectl -n "$NAMESPACE" delete deployments,configmaps,secrets -l "$SELECTOR" --ignore-not-found=true
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "${RUN_DIR}/delete_done_utc.txt"
  log "Delete command completed"
}

main() {
  log "Cluster cutoff job starting: ${RUN_SLUG}"
  wait_for_ready_gate
  # shellcheck source=/dev/null
  source "$STATE_FILE"
  harvest_at=$((KILL_AT_EPOCH - HARVEST_LEAD_SECONDS))
  sleep_until "$harvest_at" "Claude Code log harvest"
  harvest_claude_logs
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
  verbs: ["get", "list", "watch", "delete"]
- apiGroups: [""]
  resources: ["pods/log"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods/exec"]
  verbs: ["create"]
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list", "watch", "delete"]
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
      restartPolicy: OnFailure
      containers:
      - name: cutoff
        image: ${IMAGE}
        imagePullPolicy: Always
        command: ["/bin/bash", "/opt/senpai-cutoff/cutoff-job.sh"]
        env:
        - name: RUN_SLUG
          value: "${RUN_SLUG}"
        - name: TAGS_CSV
          value: "${TAGS_CSV}"
        - name: EXPECTED_PODS
          value: "${EXPECTED_PODS}"
        - name: EXPECTED_DEPLOYMENTS
          value: "${EXPECTED_DEPLOYMENTS}"
        - name: BUDGET_SECONDS
          value: "${BUDGET_SECONDS}"
        - name: HARVEST_LEAD_SECONDS
          value: "${HARVEST_LEAD_SECONDS}"
        - name: PVC_LOG_ROOT
          value: "${PVC_LOG_ROOT}"
        - name: MAX_PARALLEL_COPIES
          value: "${MAX_PARALLEL_COPIES}"
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
"$KUBECTL" --context "$CONTEXT" delete job "$JOB_NAME" --ignore-not-found=true
"$KUBECTL" --context "$CONTEXT" apply -f "$JOB_MANIFEST"

echo "Armed cluster cutoff job: ${JOB_NAME}"
echo "PVC log root: ${PVC_LOG_ROOT}/${RUN_SLUG}"
echo "Monitor: kubectl --context ${CONTEXT} -n ${NAMESPACE} logs -f job/${JOB_NAME}"

if [ "$START_LOCAL_PULL" = "true" ]; then
  cat > "$LOCAL_PULL_SCRIPT" <<EOF_PULL
#!/usr/bin/env bash
set -euo pipefail
KUBECTL="${KUBECTL}"
CONTEXT="${CONTEXT}"
NAMESPACE="${NAMESPACE}"
JOB_NAME="${JOB_NAME}"
RUN_SLUG="${RUN_SLUG}"
PVC_CLAIM_NAME="${PVC_CLAIM_NAME}"
PVC_MOUNT_PATH="${PVC_MOUNT_PATH}"
PVC_LOG_ROOT="${PVC_LOG_ROOT}"
IMAGE="${IMAGE}"
DEST="${CONVERSATION_LOG_DIR}/\${RUN_SLUG}_pvc"
READER_POD="senpai-pvc-log-reader-${SAFE_SLUG}"

mkdir -p "\${DEST}"
"\${KUBECTL}" --context "\${CONTEXT}" -n "\${NAMESPACE}" wait --for=condition=complete "job/\${JOB_NAME}" --timeout=200h
"\${KUBECTL}" --context "\${CONTEXT}" -n "\${NAMESPACE}" delete pod "\${READER_POD}" --ignore-not-found=true
"\${KUBECTL}" --context "\${CONTEXT}" -n "\${NAMESPACE}" run "\${READER_POD}" \
  --image="\${IMAGE}" \
  --restart=Never \
  --overrides="{\"spec\":{\"containers\":[{\"name\":\"reader\",\"image\":\"\${IMAGE}\",\"command\":[\"/bin/bash\",\"-lc\",\"sleep 3600\"],\"volumeMounts\":[{\"name\":\"dataset\",\"mountPath\":\"\${PVC_MOUNT_PATH}\"}]}],\"volumes\":[{\"name\":\"dataset\",\"persistentVolumeClaim\":{\"claimName\":\"\${PVC_CLAIM_NAME}\"}}]}}"
"\${KUBECTL}" --context "\${CONTEXT}" -n "\${NAMESPACE}" wait --for=condition=Ready "pod/\${READER_POD}" --timeout=10m
"\${KUBECTL}" --context "\${CONTEXT}" -n "\${NAMESPACE}" cp "\${READER_POD}:\${PVC_LOG_ROOT}/\${RUN_SLUG}" "\${DEST}"
"\${KUBECTL}" --context "\${CONTEXT}" -n "\${NAMESPACE}" delete pod "\${READER_POD}" --ignore-not-found=true
EOF_PULL
  chmod +x "$LOCAL_PULL_SCRIPT"
  nohup "$LOCAL_PULL_SCRIPT" > "$LOCAL_PULL_LOG" 2>&1 &
  echo "Started best-effort local PVC mirror: ${LOCAL_PULL_SCRIPT}"
  echo "Local mirror log: ${LOCAL_PULL_LOG}"
fi
