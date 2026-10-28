#!/bin/bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
cd "${SCRIPT_DIR}/.." || exit 1

mkdir -p generated_texts_og/logs

TARGET_TOKENS="1000000"
MODEL_NAME="mistral"
MAX_TOKENS="16384"
TEMP_LOW="1.0"
TEMP_HIGH="1.4"
TOP_P="0.95"
N_ATTEMPTS="5"
RESTART_EVERY="2000"
METRIC_GATING_PATIENCE="20"
PROMPT_FILE="generated_texts_og/prompts/lighthouse_keeper.md"
CONDITIONING_MODE="extended_no_fixed_conditioning"

RUN_IDS=(
  "run_ext_no_fixed_lighthouse_keeper_rep1_1m"
  "run_ext_no_fixed_lighthouse_keeper_rep2_1m"
  "run_ext_no_fixed_lighthouse_keeper_rep3_1m"
)

stop_requested=0
child_pid=""

handle_stop() {
    stop_requested=1
    if [ -n "${child_pid}" ] && kill -0 "${child_pid}" 2>/dev/null; then
        kill -TERM "${child_pid}" 2>/dev/null || true
        wait "${child_pid}" 2>/dev/null || true
        child_pid=""
    fi
}

trap handle_stop INT TERM

for run_id in "${RUN_IDS[@]}"; do
    run_log="generated_texts_og/logs/${run_id}.log"

    if [ "${stop_requested}" -eq 1 ]; then
        echo "$(date -Is) batch_stopped_before_${run_id}"
        exit 0
    fi

    if [ -f "${run_log}" ] && grep -q "completed$" "${run_log}"; then
        echo "$(date -Is) skipping_completed_${run_id}"
        continue
    fi

    echo "$(date -Is) starting ${run_id} with ${PROMPT_FILE} in ${CONDITIONING_MODE}"
    /bin/bash generated_texts_og/run_longbook_resume_wrapper.sh \
        "${run_id}" \
        "${PROMPT_FILE}" \
        "${TARGET_TOKENS}" \
        "${MODEL_NAME}" \
        "${MAX_TOKENS}" \
        "${TEMP_LOW}" \
        "${TEMP_HIGH}" \
        "${TOP_P}" \
        "${N_ATTEMPTS}" \
        "${RESTART_EVERY}" \
        --conditioning_mode "${CONDITIONING_MODE}" \
        --metric_gating_patience "${METRIC_GATING_PATIENCE}" >> "${run_log}" 2>&1 &

    child_pid=$!
    wait "${child_pid}"
    code=$?
    child_pid=""

    if [ "${stop_requested}" -eq 1 ]; then
        echo "$(date -Is) batch_stopped_during_${run_id}"
        exit 0
    fi

    if [ "${code}" -ne 0 ]; then
        echo "$(date -Is) batch_aborting_after_${run_id}_exit_${code}"
        exit "${code}"
    fi

    echo "$(date -Is) finished ${run_id}"
done

echo "$(date -Is) batch_completed"
