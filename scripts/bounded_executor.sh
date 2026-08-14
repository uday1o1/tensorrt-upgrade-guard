#!/usr/bin/env bash

# Shared GNU timeout policy for the Linux qualification runner.
UG_TIMEOUT_QUICK_SECONDS="${UG_TIMEOUT_QUICK_SECONDS:-120}"
UG_TIMEOUT_PREFLIGHT_SECONDS="${UG_TIMEOUT_PREFLIGHT_SECONDS:-300}"
UG_TIMEOUT_NETWORK_SECONDS="${UG_TIMEOUT_NETWORK_SECONDS:-1800}"
UG_TIMEOUT_BUILD_SECONDS="${UG_TIMEOUT_BUILD_SECONDS:-7200}"
UG_TIMEOUT_GPU_SECONDS="${UG_TIMEOUT_GPU_SECONDS:-3600}"
UG_TIMEOUT_AUDIT_SECONDS="${UG_TIMEOUT_AUDIT_SECONDS:-3600}"
UG_TIMEOUT_EVIDENCE_SECONDS="${UG_TIMEOUT_EVIDENCE_SECONDS:-900}"
UG_TIMEOUT_CLEANUP_SECONDS="${UG_TIMEOUT_CLEANUP_SECONDS:-30}"
UG_TIMEOUT_KILL_AFTER_SECONDS="${UG_TIMEOUT_KILL_AFTER_SECONDS:-30}"

validate_positive_seconds() {
  local name=$1
  local value=$2
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer number of seconds.\n' "${name}" >&2
    return 2
  fi
}

initialize_bounded_executor() {
  command -v timeout >/dev/null 2>&1 || {
    printf 'GNU timeout is required by the Linux qualification runner.\n' >&2
    return 4
  }
  validate_positive_seconds UG_TIMEOUT_QUICK_SECONDS "${UG_TIMEOUT_QUICK_SECONDS}" || return
  validate_positive_seconds UG_TIMEOUT_PREFLIGHT_SECONDS "${UG_TIMEOUT_PREFLIGHT_SECONDS}" || return
  validate_positive_seconds UG_TIMEOUT_NETWORK_SECONDS "${UG_TIMEOUT_NETWORK_SECONDS}" || return
  validate_positive_seconds UG_TIMEOUT_BUILD_SECONDS "${UG_TIMEOUT_BUILD_SECONDS}" || return
  validate_positive_seconds UG_TIMEOUT_GPU_SECONDS "${UG_TIMEOUT_GPU_SECONDS}" || return
  validate_positive_seconds UG_TIMEOUT_AUDIT_SECONDS "${UG_TIMEOUT_AUDIT_SECONDS}" || return
  validate_positive_seconds UG_TIMEOUT_EVIDENCE_SECONDS "${UG_TIMEOUT_EVIDENCE_SECONDS}" || return
  validate_positive_seconds UG_TIMEOUT_CLEANUP_SECONDS "${UG_TIMEOUT_CLEANUP_SECONDS}" || return
  validate_positive_seconds UG_TIMEOUT_KILL_AFTER_SECONDS "${UG_TIMEOUT_KILL_AFTER_SECONDS}" || return
}

bounded_run() {
  local timeout_class=$1
  shift
  local seconds
  case "${timeout_class}" in
    quick) seconds=${UG_TIMEOUT_QUICK_SECONDS} ;;
    preflight) seconds=${UG_TIMEOUT_PREFLIGHT_SECONDS} ;;
    network) seconds=${UG_TIMEOUT_NETWORK_SECONDS} ;;
    build) seconds=${UG_TIMEOUT_BUILD_SECONDS} ;;
    gpu) seconds=${UG_TIMEOUT_GPU_SECONDS} ;;
    audit) seconds=${UG_TIMEOUT_AUDIT_SECONDS} ;;
    evidence) seconds=${UG_TIMEOUT_EVIDENCE_SECONDS} ;;
    cleanup) seconds=${UG_TIMEOUT_CLEANUP_SECONDS} ;;
    *)
      printf 'Unknown timeout class: %s\n' "${timeout_class}" >&2
      return 2
      ;;
  esac
  if [[ $# -eq 0 ]]; then
    printf 'bounded_run requires a command.\n' >&2
    return 2
  fi
  command timeout --foreground --signal=TERM \
    --kill-after="${UG_TIMEOUT_KILL_AFTER_SECONDS}s" "${seconds}s" "$@"
}

ensure_exact_docker_image() {
  local image=$1
  if bounded_run quick docker image inspect "${image}" >/dev/null 2>&1; then
    return 0
  fi
  bounded_run network docker pull "${image}"
  bounded_run quick docker image inspect "${image}" >/dev/null
}
