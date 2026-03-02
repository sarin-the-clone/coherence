#!/bin/bash

# Experiment Sequence
# 1. 4096 tokens, High Entropy (8.7 - Default in adaptive_thresholds.json)
# echo "Starting Run 1: 4096 tokens, High Entropy..."
# python3 generated_texts_og/run_longbook.py --prompt "$(cat generated_texts_og/prompts/scifi_cyberpunk_detective.md)" --chunks 300 --model mistral --max_tokens 4096
# echo "Run 1 Complete."

# # 2. 4096 tokens, Lower Entropy (8.35)
# echo "Starting Run 2: 4096 tokens, Low Entropy (8.35)..."
# python3 generated_texts_og/run_longbook.py --prompt "$(cat generated_texts_og/prompts/scifi_cyberpunk_detective.md)" --chunks 300 --model mistral --max_tokens 4096 --entropy_min 8.35
# echo "Run 2 Complete."

# # 3. 8192 tokens, Lower Entropy (8.35)
# echo "Starting Run 3: 8192 tokens, Low Entropy (8.35)..."
# python3 generated_texts_og/run_longbook.py --prompt "$(cat generated_texts_og/prompts/scifi_cyberpunk_detective.md)" --chunks 300 --model mistral --max_tokens 8192 --entropy_min 8.35 --ppl_min 330

# 4. The 3-Day Epic: 1000 Chunks, 4096 tokens (Adaptive Gating)
# echo "Starting Run 4: 2000 Chunks, 4096 tokens (The 3-Day Run, Pure Adaptive)..."
# python3 generated_texts_og/run_longbook.py --prompt "$(cat generated_texts_og/prompts/historical_library.md)" --chunks 2000 --model mistral --max_tokens 4096

# Run 4: Historical Fiction
# python3 generated_texts_og/run_longbook.py \
#     --model mistral \
#     --prompt "$(cat generated_texts_og/prompts/historical_library.md)" \
#     --chunks 2000 \
#     --temp_low 1.0 --temp_high 1.4 \
#     --top_p 0.95
#########################################################################################
# Run 5: Historical Fiction 
# python3 generated_texts_og/run_longbook.py \
#     --model mistral \
#     --prompt "$(cat generated_texts_og/prompts/historical_library.md)" \
#     --chunks 2000 \
#     --temp_low 1.0 --temp_high 1.4 \
#     --top_p 0.95 \
#     --max_tokens 2048

# Run 6 
# Run 6 
# python3 generated_texts_og/run_longbook.py \
#     --model mistral \
#     --prompt "$(cat generated_texts_og/prompts/historical_library.md)" \
#     --target_tokens 1000000 \
#     --temp_low 1.0 --temp_high 1.4 \
#     --top_p 0.95 \
#     --max_tokens 4096

# Run 7
# Run 7
# python3 generated_texts_og/run_longbook.py \
#     --model mistral \
#     --prompt "$(cat generated_texts_og/prompts/kingdom_of_kush.md)" \
#     --target_tokens 1000000 \
#     --temp_low 1.0 --temp_high 1.4 \
#     --top_p 0.95 \
#     --max_tokens 8192 \
#     --resume generated_texts_og/prompts/20251219_105854

# Run 8 (COMPLETE)
# python3 generated_texts_og/run_longbook.py \
#     --model mistral \
#     --prompt "$(cat generated_texts_og/prompts/kingdom_of_kush.md)" \
#     --target_tokens 1000000 \
#     --temp_low 1.0 --temp_high 1.4 \
#     --top_p 0.95 \
#     --max_tokens 16384 \
#     --resume generated_texts_og/prompts/20251220_041136_vanquisher_of_shadows_amani_s_quest_to_unite_kush_and_egypt_fantasy

# Run 9 (Robustness Test - Clean Start)
# python3 generated_texts_og/run_longbook.py \
#     --model mistral \
#     --prompt "$(cat generated_texts_og/prompts/kingdom_of_kush.md)" \
#     --target_tokens 1000000 \
#     --temp_low 1.0 --temp_high 1.4 \
#     --top_p 0.95 \
#     --max_tokens 16384 \
#     --resume generated_texts_og/prompts/20251221_103143

if [ "$1" == "10" ]; then
    # Run 10 (Fresh Start - 8k Tokens for Stability)
    while :; do
        python3 generated_texts_og/run_longbook.py \
            --model mistral \
            --prompt "$(cat generated_texts_og/prompts/kingdom_of_kush.md)" \
            --target_tokens 1000000 \
            --temp_low 1.0 --temp_high 1.4 \
            --top_p 0.95 \
            --max_tokens 8192 \
            --run_id run_10_kush \
            --resume generated_texts_og/prompts/20251222_065034/ \
            --restart_every 200

        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 42 ]; then
            echo "🔄 Exit code 42 detected. Cleaning VRAM and restarting..."
            sleep 5
            continue
        elif [ $EXIT_CODE -eq 0 ]; then
            echo "✅ Run Complete."
            break
        else
            echo "❌ Crashed with exit code $EXIT_CODE. Restarting in 10s..."
            sleep 10
            continue
        fi
    done
fi

if [ "$1" == "11" ]; then
    # Run 11 (Space Opera - OOM Monitoring)
    while :; do
        echo "Starting Run 11..."
        # NOTE: Using --run_id allows auto-resume without specifying --resume path manually
        python3 generated_texts_og/run_longbook.py \
            --model mistral \
            --prompt_file "generated_texts_og/prompts/space_opera_last_star.md" \
            --target_tokens 1000000 \
            --max_tokens 16384 \
            --temp_low 1.0 --temp_high 1.4 \
            --top_p 0.95 \
            --run_id run_11_space_opera_high_temp \
            --restart_every 200

        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ Run 11 Complete."
            break
        else
            echo "❌ Run 11 Crashed (Code $EXIT_CODE)." 
            
            echo "$(date): Run 11 Crashed with exit code $EXIT_CODE" >> run_11_oom_events.log
            
            echo "Restarting in 10s..."
            sleep 10
        fi
    done
fi

if [ "$1" == "12" ]; then
    # Run 12 (Space Opera - Low Temp Control 0.7-1.0)
    while :; do
        echo "Starting Run 12..."
        python3 generated_texts_og/run_longbook.py \
            --model mistral \
            --prompt_file "generated_texts_og/prompts/space_opera_last_star.md" \
            --target_tokens 1000000 \
            --max_tokens 16384 \
            --temp_low 0.7 --temp_high 1.0 \
            --top_p 0.95 \
            --run_id run_12_low_temp \
            --restart_every 200

        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ Run 12 Complete."
            break
        else
            echo "❌ Run 12 Crashed (Code $EXIT_CODE)." 
            echo "$(date): Run 12 Crashed with exit code $EXIT_CODE" >> run_12_oom_events.log
            echo "Restarting in 10s..."
            sleep 10
        fi
    done
fi

echo "All Runs Complete."

if [ "$1" == "13" ]; then
    # Run 13 (High Temp Chaos 1.4-1.8)
    while :; do
        echo "Starting Run 13 (Chaos Mode)..."
        python3 generated_texts_og/run_longbook.py \
            --model mistral \
            --prompt_file "generated_texts_og/prompts/space_opera_last_star.md" \
            --target_tokens 1000000 \
            --max_tokens 16384 \
            --temp_low 1.4 --temp_high 1.8 \
            --top_p 0.95 \
            --run_id run_13_chaos \
            --restart_every 200 \
            --disable_gating

        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ Run 13 Complete."
            break
        else
            echo "❌ Run 13 Crashed (Code $EXIT_CODE)." 
            echo "$(date): Run 13 Crashed with exit code $EXIT_CODE" >> run_13_oom_events.log
            echo "Restarting in 10s..."
            sleep 10
        fi
    done
fi

if [ "$1" == "14" ]; then
    # Run 14 (Chaos Mode, 4M Tokens - NO GATING)
    while :; do
        echo "Starting Run 14 (Chaos Mode, 4M Tokens, No Gating)..."
        python3 -u generated_texts_og/run_longbook.py \
            --model mistral \
            --prompt_file "generated_texts_og/prompts/space_opera_last_star.md" \
            --target_tokens 4000000 \
            --max_tokens 16384 \
            --temp_low 1.4 --temp_high 1.8 \
            --top_p 0.95 \
            --run_id run_14_chaos_4m \
            --restart_every 200 \
            --disable_gating

        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ Run 14 Complete."
            break
        else
            echo "❌ Run 14 Crashed (Code $EXIT_CODE)." 
            echo "$(date): Run 14 Crashed with exit code $EXIT_CODE" >> run_14_oom_events.log
            echo "Restarting in 10s..."
            sleep 10
        fi
    done
fi

if [[ "$1" -ge 15 && "$1" -le 32 ]]; then
    # Experiment Batch 15-32 (1M Tokens, 16k Context, Temp 1.0-1.4, Gating Active)
    # 6 Prompt Types * 3 Replicates
    
    EXPERIMENT_ID=$1
    
    # Define Prompt Mapping
    declare -A PROMPTS
    declare -A GENRE
    declare -A REP
    
    PROMPTS[15]="generated_texts_og/prompts/historical_library.md"; GENRE[15]="Historical Fiction"; REP[15]=1
    PROMPTS[16]="generated_texts_og/prompts/historical_library.md"; GENRE[16]="Historical Fiction"; REP[16]=2
    PROMPTS[17]="generated_texts_og/prompts/historical_library.md"; GENRE[17]="Historical Fiction"; REP[17]=3
    
    PROMPTS[18]="generated_texts_og/prompts/romance_drama.md"; GENRE[18]="Romance Drama"; REP[18]=1
    PROMPTS[19]="generated_texts_og/prompts/romance_drama.md"; GENRE[19]="Romance Drama"; REP[19]=2
    PROMPTS[20]="generated_texts_og/prompts/romance_drama.md"; GENRE[20]="Romance Drama"; REP[20]=3
    
    PROMPTS[21]="generated_texts_og/prompts/cyberpunk_grid.md"; GENRE[21]="Cyberpunk"; REP[21]=1
    PROMPTS[22]="generated_texts_og/prompts/cyberpunk_grid.md"; GENRE[22]="Cyberpunk"; REP[22]=2
    PROMPTS[23]="generated_texts_og/prompts/cyberpunk_grid.md"; GENRE[23]="Cyberpunk"; REP[23]=3
    
    PROMPTS[24]="generated_texts_og/prompts/science_book.md"; GENRE[24]="Science Book"; REP[24]=1
    PROMPTS[25]="generated_texts_og/prompts/science_book.md"; GENRE[25]="Science Book"; REP[25]=2
    PROMPTS[26]="generated_texts_og/prompts/science_book.md"; GENRE[26]="Science Book"; REP[26]=3
    
    PROMPTS[27]="generated_texts_og/prompts/biography.md"; GENRE[27]="Biography"; REP[27]=1
    PROMPTS[28]="generated_texts_og/prompts/biography.md"; GENRE[28]="Biography"; REP[28]=2
    PROMPTS[29]="generated_texts_og/prompts/biography.md"; GENRE[29]="Biography"; REP[29]=3
    
    PROMPTS[30]="generated_texts_og/prompts/lighthouse_keeper.md"; GENRE[30]="General Fiction"; REP[30]=1
    PROMPTS[31]="generated_texts_og/prompts/lighthouse_keeper.md"; GENRE[31]="General Fiction"; REP[31]=2
    PROMPTS[32]="generated_texts_og/prompts/lighthouse_keeper.md"; GENRE[32]="General Fiction"; REP[32]=3
    
    PROMPT_FILE=${PROMPTS[$EXPERIMENT_ID]}
    GENRE_NAME=${GENRE[$EXPERIMENT_ID]}
    REPLICATE=${REP[$EXPERIMENT_ID]}
    RUN_ID="run_${EXPERIMENT_ID}_${GENRE_NAME// /_}_rep${REPLICATE}"
    RUN_ID=$(echo "$RUN_ID" | tr '[:upper:]' '[:lower:]') # Lowercase
    
    while :; do
        echo "Starting Run $EXPERIMENT_ID: $GENRE_NAME (Replicate $REPLICATE)..."
        echo "  Prompt: $PROMPT_FILE"
        echo "  Run ID: $RUN_ID"
        
        # Verify Prompt File Exists
        if [ ! -f "$PROMPT_FILE" ]; then
            echo "❌ Error: Prompt file not found: $PROMPT_FILE"
            exit 1
        fi
        
        python3 generated_texts_og/run_longbook.py \
            --model mistral \
            --prompt_file "$PROMPT_FILE" \
            --target_tokens 1000000 \
            --max_tokens 16384 \
            --temp_low 1.0 --temp_high 1.4 \
            --top_p 0.95 \
            --run_id "$RUN_ID" \
            --restart_every 200
            # Gating is active by default (no --disable_gating)

        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ Run $EXPERIMENT_ID Complete."
            break
        else
            echo "❌ Run $EXPERIMENT_ID Crashed (Code $EXIT_CODE)." 
            echo "$(date): Run $EXPERIMENT_ID Crashed with exit code $EXIT_CODE" >> "run_${EXPERIMENT_ID}_oom_events.log"
            echo "Restarting in 10s..."
            sleep 10
        fi
    done
fi



if [[ "$1" -ge 33 && "$1" -le 50 ]]; then
    # Experiment Batch 33-50 (1M Tokens, 16k Context, Temp 1.0-1.4, Gating Active)
    # 6 Prompt Types * 3 Replicates
    
    EXPERIMENT_ID=$1
    
    # Define Prompt Mapping
    declare -A PROMPTS
    declare -A GENRE
    declare -A REP
    
    PROMPTS[33]="generated_texts_og/prompts/historical_library.md"; GENRE[33]="Historical Fiction"; REP[33]=1
    PROMPTS[34]="generated_texts_og/prompts/historical_library.md"; GENRE[34]="Historical Fiction"; REP[34]=2
    PROMPTS[35]="generated_texts_og/prompts/historical_library.md"; GENRE[35]="Historical Fiction"; REP[35]=3
    
    PROMPTS[36]="generated_texts_og/prompts/romance_drama.md"; GENRE[36]="Romance Drama"; REP[36]=1
    PROMPTS[37]="generated_texts_og/prompts/romance_drama.md"; GENRE[37]="Romance Drama"; REP[37]=2
    PROMPTS[38]="generated_texts_og/prompts/romance_drama.md"; GENRE[38]="Romance Drama"; REP[38]=3
    
    PROMPTS[39]="generated_texts_og/prompts/cyberpunk_grid.md"; GENRE[39]="Cyberpunk"; REP[39]=1
    PROMPTS[40]="generated_texts_og/prompts/cyberpunk_grid.md"; GENRE[40]="Cyberpunk"; REP[40]=2
    PROMPTS[41]="generated_texts_og/prompts/cyberpunk_grid.md"; GENRE[41]="Cyberpunk"; REP[41]=3
    
    PROMPTS[42]="generated_texts_og/prompts/science_book.md"; GENRE[42]="Science Book"; REP[42]=1
    PROMPTS[43]="generated_texts_og/prompts/science_book.md"; GENRE[43]="Science Book"; REP[43]=2
    PROMPTS[44]="generated_texts_og/prompts/science_book.md"; GENRE[44]="Science Book"; REP[44]=3
    
    PROMPTS[45]="generated_texts_og/prompts/biography.md"; GENRE[45]="Biography"; REP[45]=1
    PROMPTS[46]="generated_texts_og/prompts/biography.md"; GENRE[46]="Biography"; REP[46]=2
    PROMPTS[47]="generated_texts_og/prompts/biography.md"; GENRE[47]="Biography"; REP[47]=3
    
    PROMPTS[48]="generated_texts_og/prompts/lighthouse_keeper.md"; GENRE[48]="General Fiction"; REP[48]=1
    PROMPTS[49]="generated_texts_og/prompts/lighthouse_keeper.md"; GENRE[49]="General Fiction"; REP[49]=2
    PROMPTS[50]="generated_texts_og/prompts/lighthouse_keeper.md"; GENRE[50]="General Fiction"; REP[50]=3
    
    PROMPT_FILE=${PROMPTS[$EXPERIMENT_ID]}
    GENRE_NAME=${GENRE[$EXPERIMENT_ID]}
    REPLICATE=${REP[$EXPERIMENT_ID]}
    RUN_ID="run_${EXPERIMENT_ID}_${GENRE_NAME// /_}_rep${REPLICATE}"
    RUN_ID=$(echo "$RUN_ID" | tr '[:upper:]' '[:lower:]') # Lowercase
    
    while :; do
        echo "Starting Run $EXPERIMENT_ID: $GENRE_NAME (Replicate $REPLICATE)..."
        echo "  Prompt: $PROMPT_FILE"
        echo "  Run ID: $RUN_ID"
        
        # Verify Prompt File Exists
        if [ ! -f "$PROMPT_FILE" ]; then
            echo "❌ Error: Prompt file not found: $PROMPT_FILE"
            exit 1
        fi
        
        python3 -u generated_texts_og/run_longbook.py \
            --model mistral \
            --prompt_file "$PROMPT_FILE" \
            --target_tokens 1000000 \
            --max_tokens 16384 \
            --temp_low 0.6 --temp_high 0.9 \
            --top_p 0.95 \
            --run_id "$RUN_ID" \
            --restart_every 200
            # Gating is active by default (no --disable_gating)

        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ Run $EXPERIMENT_ID Complete."
            break
        else
            echo "❌ Run $EXPERIMENT_ID Crashed (Code $EXIT_CODE)." 
            echo "$(date): Run $EXPERIMENT_ID Crashed with exit code $EXIT_CODE" >> "run_${EXPERIMENT_ID}_oom_events.log"
            echo "Restarting in 10s..."
            sleep 10
        fi
    done
fi

if [ "$1" == "51" ]; then
    # Run 51 (Qwen 72B Test - 50k Tokens)
    while :; do
        echo "Starting Run 51 (Qwen 72B Test)..."
        python3 -u generated_texts_og/run_longbook.py \
            --model qwen25_72b \
            --prompt_file "generated_texts_og/prompts/space_opera_last_star.md" \
            --target_tokens 150000 \
            --max_tokens 16384 \
            --temp_low 0.8 --temp_high 1.0 \
            --top_p 0.95 \
            --run_id run_51_qwen72b_test \
            --restart_every 200 \
            --repetition_penalty 1.1 \
            --n_attempts 3

        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ Run 51 Complete."
            break
        else
            echo "❌ Run 51 Crashed (Code $EXIT_CODE)." 
            echo "$(date): Run 51 Crashed with exit code $EXIT_CODE" >> run_51_oom_events.log
            echo "Restarting in 10s..."
            sleep 10
        fi
    done
fi

