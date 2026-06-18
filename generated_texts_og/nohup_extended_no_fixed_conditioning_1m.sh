#!/bin/bash

set -u

cd /home/sarin/pienicolol/src/coherence || exit 1

mkdir -p generated_texts_og/logs

batch_log="generated_texts_og/logs/run_extended_no_fixed_conditioning_1m.nohup.log"

nohup setsid /bin/bash ./generated_texts_og/run_extended_no_fixed_conditioning_1m.sh > "${batch_log}" 2>&1 &
pid="$!"

echo "${pid}" > generated_texts_og/logs/run_extended_no_fixed_conditioning_1m.pid
echo "Started run_extended_no_fixed_conditioning_1m with PID ${pid}"
echo "Batch log: ${batch_log}"
echo "Per-run logs: generated_texts_og/logs/run_ext_no_fixed_lighthouse_keeper_rep<N>_1m.log"
