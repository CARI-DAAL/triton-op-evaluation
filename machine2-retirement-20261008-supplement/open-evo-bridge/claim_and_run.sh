#!/bin/bash
# Claim one NPU from a shared pool via flock, then run the command on it.
#
# This is the sandbox-stage device-claim mechanism (run_npu_command.sh's
# acquire_and_run) reduced to its invariant: exactly one evaluation per device
# at a time, the device chosen by a non-blocking POSIX file lock over a shared
# directory, and the waiting happening *inside* the tool call rather than as a
# queue rejection surfacing to the agent.
#
# Environment:
#   EVAL_DEVICE_IDS      comma-separated pool, e.g. "4,5,6,7"
#   EVAL_LOCK_DIR        shared lock directory (default /tmp/device-locks)
#   EVAL_RETRY_INTERVAL  seconds between sweeps (default 1.0)
#   EVAL_TIMEOUT         overall wait budget in seconds (default 900)
#
# Exit codes: the command's own status, or 124 when the pool never freed up.
set -u

lock_dir="${EVAL_LOCK_DIR:-/tmp/device-locks}"
retry="${EVAL_RETRY_INTERVAL:-1.0}"
timeout="${EVAL_TIMEOUT:-900}"
devices_raw="${EVAL_DEVICE_IDS:-}"

if [[ -z "${devices_raw}" ]]; then
    echo "[npu-lock] EVAL_DEVICE_IDS is empty" >&2
    exit 125
fi
if [[ $# -eq 0 ]]; then
    echo "[npu-lock] no command given" >&2
    exit 125
fi

mkdir -p "${lock_dir}" 2>/dev/null || true
probe="${lock_dir}/.write-test.$$"
if ! ( : > "${probe}" ) 2>/dev/null; then
    echo "[npu-lock] cannot write lock directory ${lock_dir}" >&2
    exit 125
fi
rm -f "${probe}"

IFS=',' read -r -a devices <<< "${devices_raw}"
start_ts="$(date +%s)"

while true; do
    for device_id in "${devices[@]}"; do
        device_id="${device_id//[[:space:]]/}"
        [[ -n "${device_id}" ]] || continue
        lock_file="${lock_dir}/npu${device_id}.lock"
        exec {lock_fd}>"${lock_file}" || continue
        if flock -n "${lock_fd}"; then
            export ASCEND_RT_VISIBLE_DEVICES="${device_id}"
            export ALLOCATED_DEVICE_ID="${device_id}"
            echo "[npu-lock] acquired npu${device_id}" >&2
            "$@"
            status=$?
            flock -u "${lock_fd}" || true
            exec {lock_fd}>&-
            exit "${status}"
        fi
        exec {lock_fd}>&-
    done

    now="$(date +%s)"
    if (( now - start_ts >= ${timeout%.*} )); then
        echo "[npu-lock] timed out waiting for one of ${devices_raw}" >&2
        exit 124
    fi
    sleep "${retry}"
done
