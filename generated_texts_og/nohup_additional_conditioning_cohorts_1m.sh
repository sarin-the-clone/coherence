#!/bin/bash

set -u

cd /home/sarin/pienicolol/src/coherence || exit 1

mkdir -p generated_texts_og/logs

batch_log="generated_texts_og/logs/run_additional_conditioning_cohorts_1m.nohup.log"

nohup ./generated_texts_og/run_additional_conditioning_cohorts_1m.sh > "${batch_log}" 2>&1 &
pid="$!"

echo "${pid}" > generated_texts_og/logs/run_additional_conditioning_cohorts_1m.pid
echo "Started run_additional_conditioning_cohorts_1m with PID ${pid}"
echo "Batch log: ${batch_log}"
echo "Per-run logs: generated_texts_og/logs/<run_id>.log"
