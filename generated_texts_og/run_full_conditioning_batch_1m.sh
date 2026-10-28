#!/bin/bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
cd "${SCRIPT_DIR}/.." || exit 1

mkdir -p generated_texts_og/logs

RUN_IDS=(
  "run_fc_biography_1m"
  "run_fc_fantasy_forgotten_god_1m"
  "run_fc_historical_library_1m"
  "run_fc_kingdom_of_kush_1m"
  "run_fc_lighthouse_keeper_1m"
  "run_fc_philosophy_book_1m"
  "run_fc_romance_drama_1m"
  "run_fc_space_opera_last_star_1m"
  "run_fc_technical_reference_1m"
)

PROMPTS=(
  "generated_texts_og/prompts/biography.md"
  "generated_texts_og/prompts/fantasy_forgotten_god.md"
  "generated_texts_og/prompts/historical_library.md"
  "generated_texts_og/prompts/kingdom_of_kush.md"
  "generated_texts_og/prompts/lighthouse_keeper.md"
  "generated_texts_og/prompts/philosophy_book.md"
  "generated_texts_og/prompts/romance_drama.md"
  "generated_texts_og/prompts/space_opera_last_star.md"
  "generated_texts_og/prompts/technical_reference.md"
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
    run_log="generated_texts_og/logs/${run_id}.log"

    if [ "${stop_requested}" -eq 1 ]; then
        echo "$(date -Is) batch_stopped_before_${run_id}"
        exit 0
    fi

    echo "$(date -Is) starting ${run_id} with ${prompt_file}"
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
        "200" > "${run_log}" 2>&1 &

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
