#!/bin/bash

set -u

source /home/sarin/pienicolol/bin/activate
cd /home/sarin/pienicolol/src/coherence || exit 1

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

# Format: run_id|prompt_file|conditioning_mode
#
# Minimal modes: two additional replicates per mode.
# Basic mode: six prompt families x two replicates, high-temperature only.
RUN_SPECS=(
  "run_min_fixed_space_opera_last_star_rep1_1m|generated_texts_og/prompts/space_opera_last_star.md|minimal_fixed_conditioning"
  "run_min_fixed_technical_reference_rep2_1m|generated_texts_og/prompts/technical_reference.md|minimal_fixed_conditioning"
  "run_markov_roll_space_opera_last_star_rep2_1m|generated_texts_og/prompts/space_opera_last_star.md|markov_rolling_conditioning"
  "run_markov_roll_technical_reference_rep1_1m|generated_texts_og/prompts/technical_reference.md|markov_rolling_conditioning"
  "run_basic_biography_rep1_1m|generated_texts_og/prompts/biography.md|basic_conditioning"
  "run_basic_biography_rep2_1m|generated_texts_og/prompts/biography.md|basic_conditioning"
  "run_basic_cyberpunk_grid_rep1_1m|generated_texts_og/prompts/cyberpunk_grid.md|basic_conditioning"
  "run_basic_cyberpunk_grid_rep2_1m|generated_texts_og/prompts/cyberpunk_grid.md|basic_conditioning"
  "run_basic_historical_library_rep1_1m|generated_texts_og/prompts/historical_library.md|basic_conditioning"
  "run_basic_historical_library_rep2_1m|generated_texts_og/prompts/historical_library.md|basic_conditioning"
  "run_basic_lighthouse_keeper_rep1_1m|generated_texts_og/prompts/lighthouse_keeper.md|basic_conditioning"
  "run_basic_lighthouse_keeper_rep2_1m|generated_texts_og/prompts/lighthouse_keeper.md|basic_conditioning"
  "run_basic_romance_drama_rep1_1m|generated_texts_og/prompts/romance_drama.md|basic_conditioning"
  "run_basic_romance_drama_rep2_1m|generated_texts_og/prompts/romance_drama.md|basic_conditioning"
  "run_basic_science_book_rep1_1m|generated_texts_og/prompts/science_book.md|basic_conditioning"
  "run_basic_science_book_rep2_1m|generated_texts_og/prompts/science_book.md|basic_conditioning"
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

for spec in "${RUN_SPECS[@]}"; do
    IFS="|" read -r run_id prompt_file conditioning_mode <<< "${spec}"
    run_log="generated_texts_og/logs/${run_id}.log"

    if [ "${stop_requested}" -eq 1 ]; then
        echo "$(date -Is) batch_stopped_before_${run_id}"
        exit 0
    fi

    if [ -f "${run_log}" ] && grep -q "completed$" "${run_log}"; then
        echo "$(date -Is) skipping_completed_${run_id}"
        continue
    fi

    echo "$(date -Is) starting ${run_id} with ${prompt_file} in ${conditioning_mode}"
    /bin/bash generated_texts_og/run_longbook_resume_wrapper.sh \
        "${run_id}" \
        "${prompt_file}" \
        "${TARGET_TOKENS}" \
        "${MODEL_NAME}" \
        "${MAX_TOKENS}" \
        "${TEMP_LOW}" \
        "${TEMP_HIGH}" \
        "${TOP_P}" \
        "${N_ATTEMPTS}" \
        "${RESTART_EVERY}" \
        --conditioning_mode "${conditioning_mode}" \
        --metric_gating_patience "${METRIC_GATING_PATIENCE}" > >(tee -a "${run_log}") 2>&1 &

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
