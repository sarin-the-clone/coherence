#!/bin/bash

set -u

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <run_id> <prompt_file> [target_tokens] [model] [max_tokens] [temp_low] [temp_high] [top_p] [n_attempts] [restart_every]" >&2
    exit 2
fi

RUN_ID="$1"
PROMPT_FILE="$2"
TARGET_TOKENS="${3:-1000000}"
MODEL_NAME="${4:-mistral}"
MAX_TOKENS="${5:-16384}"
TEMP_LOW="${6:-1.0}"
TEMP_HIGH="${7:-1.4}"
TOP_P="${8:-0.95}"
N_ATTEMPTS="${9:-5}"
RESTART_EVERY="${10:-200}"
EXTRA_ARGS=("${@:11}")

source /home/sarin/pienicolol/bin/activate
cd /home/sarin/pienicolol/src/coherence || exit 1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

stop_requested=0
child_pid=""

handle_stop() {
    stop_requested=1
    if [ -n "${child_pid}" ] && kill -0 "${child_pid}" 2>/dev/null; then
        kill -TERM "${child_pid}" 2>/dev/null || true
    fi
}

trap handle_stop INT TERM

while :; do
    python -u generated_texts_og/run_longbook.py \
        --model "${MODEL_NAME}" \
        --prompt_file "${PROMPT_FILE}" \
        --target_tokens "${TARGET_TOKENS}" \
        --max_tokens "${MAX_TOKENS}" \
        --temp_low "${TEMP_LOW}" \
        --temp_high "${TEMP_HIGH}" \
        --top_p "${TOP_P}" \
        --n_attempts "${N_ATTEMPTS}" \
        --restart_every "${RESTART_EVERY}" \
        --run_id "${RUN_ID}" \
        "${EXTRA_ARGS[@]}" &

    child_pid=$!
    wait "${child_pid}"
    code=$?
    child_pid=""

    if [ "${stop_requested}" -eq 1 ]; then
        echo "$(date -Is) stopped_by_signal"
        exit 0
    fi

    if [ "${code}" -eq 0 ]; then
        echo "$(date -Is) completed"
        break
    fi

    if [ "${code}" -eq 42 ]; then
        echo "$(date -Is) restart_requested"
        sleep 5
        continue
    fi

    echo "$(date -Is) crashed_exit_${code}"
    sleep 10
done
