#!/usr/bin/env sh

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

# Shared process ownership helpers for Senpai pods. A singleton student pod can
# use global process ownership; grouped pods should scope ownership to the
# logical student's target checkout.

senpai_realpath_dir() {
    (cd "$1" && pwd -P) 2>/dev/null || printf '%s\n' "${1%/}"
}

senpai_pid_cwd_under() {
    senpai_pid="$1"
    senpai_target="$(senpai_realpath_dir "$2")"
    senpai_cwd="$(readlink -f "/proc/$senpai_pid/cwd" 2>/dev/null || true)"
    case "$senpai_cwd" in
        "$senpai_target"|"$senpai_target"/*) return 0 ;;
        *) return 1 ;;
    esac
}

senpai_pid_descends_from() {
    senpai_pid="$1"
    senpai_ancestor="$2"
    [ -n "$senpai_pid" ] && [ -n "$senpai_ancestor" ] || return 1
    while [ -n "$senpai_pid" ] && [ "$senpai_pid" != "0" ] && [ "$senpai_pid" != "1" ]; do
        [ "$senpai_pid" = "$senpai_ancestor" ] && return 0
        senpai_pid="$(ps -o ppid= -p "$senpai_pid" 2>/dev/null | tr -d ' ')"
    done
    return 1
}

senpai_pid_belongs_to_target() {
    senpai_pid="$1"
    senpai_target="$2"
    senpai_scope="${3:-target}"
    senpai_ancestor="${4:-}"
    [ "$senpai_scope" = "global" ] && return 0
    senpai_pid_cwd_under "$senpai_pid" "$senpai_target" && return 0
    senpai_pid_descends_from "$senpai_pid" "$senpai_ancestor"
}

senpai_process_pids_for_target() {
    senpai_kind="$1"
    senpai_target="$(senpai_realpath_dir "$2")"
    senpai_scope="${3:-target}"
    senpai_ancestor="${4:-}"
    ps -eo pid=,comm=,args= |
    while read -r senpai_candidate_pid senpai_comm senpai_args; do
        case "$senpai_kind:$senpai_comm" in
            train:python|train:python[0-9.]*|train:torchrun|train:pt_elastic) ;;
            claude:claude) ;;
            claude:*) case "$senpai_args" in *" claude "*|*/claude\ *|*/claude) ;; *) continue ;; esac ;;
            *) continue ;;
        esac
        if [ "$senpai_kind" = "train" ]; then
            case "$senpai_args" in *train.py*) ;; *) continue ;; esac
        fi
        senpai_pid_belongs_to_target "$senpai_candidate_pid" "$senpai_target" "$senpai_scope" "$senpai_ancestor" &&
            printf '%s\n' "$senpai_candidate_pid"
    done
}

senpai_training_pids_for_target() {
    senpai_process_pids_for_target train "$@"
}

senpai_claude_pids_for_target() {
    senpai_process_pids_for_target claude "$@"
}

senpai_gpu_pids_for_target() {
    senpai_target="$(senpai_realpath_dir "$1")"
    senpai_scope="${2:-target}"
    if [ "$senpai_scope" = "global" ]; then
        nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
            awk 'NF {print}'
        return 0
    fi
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
    while read -r senpai_candidate_pid; do
        [ -n "$senpai_candidate_pid" ] || continue
        senpai_pid_cwd_under "$senpai_candidate_pid" "$senpai_target" && printf '%s\n' "$senpai_candidate_pid"
    done
}

senpai_count_lines() {
    awk 'NF {n++} END {print n+0}'
}

senpai_student_activity_snapshot() {
    senpai_target="$(senpai_realpath_dir "$1")"
    senpai_scope="${2:-target}"
    senpai_branch="$(git -C "$senpai_target" branch --show-current 2>/dev/null || true)"
    senpai_pytrain="$(senpai_training_pids_for_target "$senpai_target" "$senpai_scope" | senpai_count_lines)"
    senpai_gpu="$(senpai_gpu_pids_for_target "$senpai_target" "$senpai_scope" | senpai_count_lines)"
    senpai_claude="$(senpai_claude_pids_for_target "$senpai_target" "$senpai_scope" | senpai_count_lines)"
    senpai_dirty="$(git -C "$senpai_target" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
    printf '%s\t%s\t%s\t%s\t%s\n' "$senpai_branch" "$senpai_pytrain" "$senpai_gpu" "$senpai_claude" "${senpai_dirty:-0}"
}
