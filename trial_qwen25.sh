#!/bin/bash
# Trial Experiment for Qwen2.5-3B-Instruct
# Usage: bash trial_qwen25.sh [max_tokens_value]
# Example: bash trial_qwen25.sh 12288

MAX_TOKENS=${1:-12288}

echo ""
echo "================================================"
echo "Running Trial with Qwen2.5-3B-Instruct"
echo "MAX_TOKENS: $MAX_TOKENS"
echo "================================================"
echo ""

# Use the historical fiction prompt for testing
PROMPT_FILE="generated_texts_og/prompts/historical_library.md"

if [ ! -f "$PROMPT_FILE" ]; then
    echo "❌ Error: Prompt file not found: $PROMPT_FILE"
    exit 1
fi

echo "📖 Prompt: $PROMPT_FILE"
echo "🎯 Target: 50,000 tokens (Trial run)"
echo "🔧 Max Tokens per Chunk: $MAX_TOKENS"
echo "🚀 GPU: RTX 5070 Ti (12GB VRAM)"
echo "ℹ️  Model: Qwen2.5-3B (Better creative writing than 1.5B)"
echo ""

echo "Starting trial..."
echo ""

# Run the trial
poetry run python3 generated_texts_og/run_longbook.py \
    --model qwen25_small \
    --prompt_file "$PROMPT_FILE" \
    --target_tokens 50000 \
    --max_tokens "$MAX_TOKENS" \
    --temp_low 1.0 --temp_high 1.3 \
    --top_p 0.95 \
    --run_id "trial_qwen25_${MAX_TOKENS}" \
    --restart_every 200

EXIT_CODE=$?

echo ""
echo "================================================"
echo "Trial Complete"
echo "Exit Code: $EXIT_CODE"
echo "================================================"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Trial successful!"
    echo "Output: generated_texts_og/prompts/trial_qwen25_${MAX_TOKENS}/"
else
    echo "❌ Trial failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
