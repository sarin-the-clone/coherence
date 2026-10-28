#!/bin/bash

# Activate the desired environment before launch, or set COHERENCE_PYTHON.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
cd "${SCRIPT_DIR}/.." || exit 1
PYTHON="${COHERENCE_PYTHON:-python3}"

while :; do
    "${PYTHON}" -u generated_texts_og/run_longbook.py \
        --model mistral \
        --prompt_file generated_texts_og/prompts/cyberpunk_grid.md \
        --target_tokens 1000000 \
        --max_tokens 16384 \
        --temp_low 1.0 \
        --temp_high 1.4 \
        --top_p 0.95 \
        --n_attempts 5 \
        --restart_every 200 \
        --run_id run_cyberpunk_grid_1m_selfcond

    code=$?
    if [ "$code" -eq 0 ]; then
        echo "$(date -Is) completed"
        break
    fi

    if [ "$code" -eq 42 ]; then
        echo "$(date -Is) restart_requested"
        sleep 5
        continue
    fi

    echo "$(date -Is) crashed_exit_$code"
    sleep 10
done
