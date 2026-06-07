#!/bin/bash

set -u

source /home/sarin/pienicolol/bin/activate
cd /home/sarin/pienicolol/src/coherence || exit 1

mkdir -p generated_texts_og/logs

RUN_IDS=(
  "run_markov_roll_space_opera_last_star_1m"
  "run_min_fixed_technical_reference_1m"
)

PROMPTS=(
  "generated_texts_og/prompts/space_opera_last_star.md"
  "generated_texts_og/prompts/technical_reference.md"
)

MODES=(
  "markov_rolling_conditioning"
  "minimal_fixed_conditioning"
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

trap handle_stop INT TERM HUP

for idx in "${!RUN_IDS[@]}"; do
    run_id="${RUN_IDS[$idx]}"
    prompt_file="${PROMPTS[$idx]}"
    mode="${MODES[$idx]}"
    run_log="generated_texts_og/logs/${run_id}.log"

    if [ "${stop_requested}" -eq 1 ]; then
        echo "$(date -Is) batch_stopped_before_${run_id}"
        exit 0
    fi

    echo "$(date -Is) starting ${run_id} with ${prompt_file} in ${mode}"
    /bin/bash generated_texts_og/run_longbook_resume_wrapper.sh \
        "${run_id}" \
        "${prompt_file}" \
        "1000000" \
        "mistral" \
        "16384" \
        "1.0" \
        "1.4" \
        "0.95" \
        "5" \
        "200" \
        --conditioning_mode "${mode}" > "${run_log}" 2>&1 &

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
