# Reproduction guide

This guide separates a software installation check, regeneration of analyses from existing measurements, and new production calculations. Run commands from the repository root using the Poetry environment described in the [README](../README.md).

## Lightweight installation check

```bash
poetry run python -m utils.validate_installation
poetry run python -m unittest discover -s tests -v
```

The installation check uses UTF-8 bytes, small descriptor examples, and synthetic PLS-DA data on CPU. It requires neither the book corpus nor model weights and makes no network requests. A successful check establishes that those software paths work in the current environment; it does not validate the scientific results against the original corpus. The unit tests check descriptor selection, wavelet behavior, and PLS-DA utility functions.

## Inputs needed for the combined analysis

PLS-DA reads persistent descriptor exports using `--input-dir`, which defaults to `natural_texts/Output/` in this checkout. It joins the natural raw semantic measurements with the natural scaling/token caches by book ID and reads the generated raw descriptor table directly. It does not read the original temporary semantic JSON files or generated chunk records. Complete 44-feature rows are selected before feature-set ablations, so `--feature-set sc` and `--feature-set token_stat` require the same input set.

All paths below are relative to `--input-dir`:

| Input | Relative path | Source / role |
| --- | --- | --- |
| Natural semantic export | `pca/sc_raw/sc_pca_features_robust_z.csv` | Raw values despite the filename: the accompanying summary records `standardization_mode: "none"`. Retains natural-book IDs, token counts, and the nine semantic descriptors. Only `source=natural` rows are loaded. |
| Natural scaling/information measurements | `batch_results/batch_*.json` | Existing reference-book descriptor records. |
| Natural distribution/recurrence measurements | `gutenberg_benchmark_cache/token_naturalness_metrics.jsonl` | Book-level token-statistics cache. |
| Natural predictability measurements | `gutenberg_benchmark_cache/token_predictability_metrics.jsonl` | Book-level conditional entropy descriptors. |
| Generated descriptor export | `pca/sc_combined/sc_combined_pca_generated_scores.csv` | Contains all 44 raw canonical descriptors plus run ID, conditioning mode, bin, and token count. Additional diagnostics and PCA columns are ignored. |
| Natural size-bin definitions/statistics | `analysis/size_binned_metrics.json` | Optional, or override with `--size-stats`. If absent, the loader uses built-in bins. Preserve the original definitions for exact reproduction. PLS-DA refits median/IQR normalization inside each training fold. |
| Optional landmark semantic measurements | `pca/generated_prefixes/landmark_sc.json` | Optional, or override with `--landmark-sc`. Controls whether landmarks appear in the biplot. |
| Landmark endpoint descriptors | `pca/generated_prefixes/landmark_prefix_projection_scores.csv` | Required when landmark semantic measurements are supplied; override with `--landmark-csv`. |

The supplied natural semantic export has 1,064 natural books; joining the natural measurement tables and applying complete-case/size-bin selection yields 1,047 books. The generated export contains 245 complete prefixes from 49 runs, with five prefixes per run. Those book/run ID sets match the saved combined PLS-DA cross-validation table. The natural semantic values also agree with the newer bin-standardized semantic export after reversing its scaling. Keep the recorded IDs and token counts rather than estimating lengths from text or plot coordinates.

The CSV files above preserve measurements, not just latent scores, and are sufficient for this PLS-DA route. Original generated chunk records are still needed to audit accepted text or regenerate its measurements, but are not needed to refit the model from these exports. Older PCA/measurement scripts retain their original temporary-file interfaces; those are described separately in the production section below.

Run the PLS-DA input check before launching:

```bash
python -m utils.check_reproduction_inputs --input-dir natural_texts/Output
```

It uses only the Python standard library and the dependency-free descriptor schema, reads input paths without changing them, and prints a JSON report. Exit status 1 means required files or export headers are missing. A passing check does not establish complete feature values, matched corpus membership, model provenance, or numerical agreement with the published results.

For another persistent input folder, use the same `--input-dir /path/to/data` option in the preflight and PLS-DA commands, preserving the relative layout in the table. The optional `--size-stats`, `--landmark-sc`, and `--landmark-csv` flags select individual files outside that layout. Input location does not change the estimator, descriptor definitions, bin normalization, group-aware folds, or run-level scoring.

## Analysis entry points and outputs

| Task | Entry point | Main output |
| --- | --- | --- |
| One run's descriptor trajectories | `analysis/plot_chunk_metrics.py --folder <run>` | `<run>/_metrics_evolution.jpg` by default. |
| Grouped generated descriptor trajectories, including the Fig. 1 plot | `analysis/plot_aggregate_metrics.py` | Images and aggregate cache under `generated_texts_og/Output/`. |
| Natural descriptors versus book length | `analysis/plot_metrics_vs_size.py` | `natural_texts/Output/analysis/metrics_vs_size_all.png`. |
| Combined descriptor PCA | `analysis/pca_sc_combined_size_bins.py` | Natural-fit PCA plots and loadings under `natural_texts/Output/pca/sc_combined/size_bins/`. |
| PLS-DA, VIP, and cross-validation | `analysis/plsda_discriminant.py` | Per-feature-set reports, prediction CSVs, VIP tables, and figures under `natural_texts/Output/pca/discriminant/`. |

The aggregate plotting script reads run folders directly. `analysis/extract_aggregate_metrics.py` is a historical aggregation utility and is not a prerequisite for the current grouped plots.

Once the complete inputs are present, rerun the three PLS-DA feature sets with:

```bash
poetry run python analysis/plsda_discriminant.py \
  --input-dir natural_texts/Output \
  --feature-set combined --n-components 3 \
  --output-root reproduction_outputs/discriminant
poetry run python analysis/plsda_discriminant.py \
  --input-dir natural_texts/Output \
  --feature-set sc --n-components 3 \
  --output-root reproduction_outputs/discriminant
poetry run python analysis/plsda_discriminant.py \
  --input-dir natural_texts/Output \
  --feature-set token_stat --n-components 3 \
  --output-root reproduction_outputs/discriminant
```

Each output directory contains `report.txt`, `cv_run_book_predictions.csv`, `vip.csv`, `vip_barplot.png`, `plsda_scores.png`, `plsda_biplot.png`, `plsda_per_bin.png`, and `cva_scores.png`. Group-aware cross-validation keeps each generated run's prefixes together and averages its held-out prediction scores for the primary run-level evaluation. VIP and biplot outputs use a final model fitted to the full descriptor matrix.

The older combined PCA command below still uses the original loaders in `pca_sc_natural_combined.py`: natural semantic JSON files under `/tmp`, generated run folders, the semantic prefix JSON, and the generated prefix-token cache. Its prerequisites differ from the export-based PLS-DA command. Once those measurement products are present, run PCA with size-bin median/IQR normalization:

```bash
poetry run python analysis/pca_sc_combined_size_bins.py \
  --per-bin-normalization \
  --output-dir reproduction_outputs/pca \
  --poster-dir reproduction_outputs/poster
```

These commands produce new outputs from the current inputs and installed package versions. Check input coverage and reported sample counts before comparing them with saved results. Do not interpret a run that drops incomplete rows as an exact reproduction of the original analysis.

## Full production workflow

The following stages describe the measurement-generation scripts. They are substantial model/data calculations, separate from both the lightweight validation command and refitting PLS-DA from saved exports. Exact historical regeneration additionally requires the original inputs, accepted outputs, configurations, and software/model versions. The existing PLS-DA exports do not require these stages to be rerun.

### 1. Generate books and retain run records

Use `generated_texts_og/run_longbook.py` with a story contract, a model key from `generated_texts_og/config/model_config.yaml`, sampling parameters, and a conditioning mode. For example:

```bash
poetry run python generated_texts_og/run_longbook.py \
  --model mistral \
  --prompt_file generated_texts_og/prompts/historical_library.md \
  --target_tokens 1000000 \
  --max_tokens 16384 \
  --temp_low 1.0 --temp_high 1.4 \
  --top_p 0.95 \
  --conditioning_mode extended_conditioning \
  --n_attempts 5 \
  --run_id new_history_extended
```

This is a new experiment using explicit settings. The cohort shell scripts retain other prompt/conditioning combinations; inspect them before launching multiple book-scale runs. Use `--help` for supported modes and resume options. The resume wrapper also handles the generator's restart exit code (`42`) when process restarts are requested.

The shell scripts locate the checkout from their script paths and default to `python3` from the active environment. Set `COHERENCE_PYTHON=/path/to/environment/bin/python` when selecting another interpreter. They inherit Hugging Face settings from the caller: set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` only when the required model/tokenizer files are already cached.

Keep the full run folder, including metadata, all attempt files, acceptance markers, and memory dumps. Summary tables and a final book alone do not preserve retry/acceptance history.

For retrospective token-statistics trajectories on one existing run, replace `PATH_TO_RUN` with its directory:

```bash
poetry run python analysis/add_token_naturalness_to_generated.py \
  --folder PATH_TO_RUN --by-chunk-only \
  --metric-set all --device cpu --chunk-stride 25
```

This writes `token_naturalness_by_chunk.jsonl` for cumulative accepted-text prefixes. Use `--chunk-stride 1` for every accepted chunk; smaller strides increase computation. This sidecar is used by the additional-descriptor plot presets and is separate from the matched-size prefix cache computed below.

### 2. Prepare reference texts and token descriptors

`natural_texts/Output/batch_results/` records the reference book IDs and existing scaling/information measurements. `natural_texts/cache_gutenberg_benchmark.py` uses these records to build a compressed text cache and compute further token descriptors:

```bash
poetry run python natural_texts/cache_gutenberg_benchmark.py status
poetry run python natural_texts/cache_gutenberg_benchmark.py build-cache
poetry run python natural_texts/cache_gutenberg_benchmark.py compute-metrics \
  --metric-set fast --device cpu \
  --output natural_texts/Output/gutenberg_benchmark_cache/token_naturalness_metrics.jsonl
poetry run python natural_texts/cache_gutenberg_benchmark.py compute-metrics \
  --metric-set token_predictability --device cpu \
  --output natural_texts/Output/gutenberg_benchmark_cache/token_predictability_metrics.jsonl
```

The cache defaults to `natural_texts/Output/gutenberg_benchmark_cache/books.zip`, with its manifest alongside it. Cache construction downloads texts; availability and text versions can differ from the original inputs. CPU is explicit in the example, but high-order predictability calculations on long books can be expensive. Use `--help` for limits, worker count, devices, and path overrides. Existing completed book IDs are skipped unless recomputation is explicitly requested.

The initial reference scaling/information records were produced by `natural_texts/batch_processor.py` from `natural_texts/Output/long_books_with_metadata.csv`; that script also maintains `processed_books.json`. It has positional arguments for batch number, batch size, and device, downloads reference text, and writes batch results. Rebuilding an existing historical corpus requires the original selection/metadata and progress state; do not run it over the supplied records as an installation check.

### 3. Compute semantic descriptors

Semantic calculations use the separate embedding model defined as `DEFAULT_MODEL` in `semantic_coherence.py`. The existing scripts load that model in float16 and choose CUDA when available. Their configured batch size is 64, which may need adjustment for different hardware; this configuration is not a measured minimum-memory guarantee.

For the natural semantic corpus, first provide the benchmark text archive, then run:

```bash
poetry run python compute_sc_natural_median.py
poetry run python compute_sc_natural_extra.py --group a
```

Repeat `compute_sc_natural_extra.py` for groups `b` through `i` as required by the intended corpus. The lists are encoded in those scripts. Despite its historical name, `compute_sc_natural_median.py` currently contains multiple books per size bin. These scripts write `/tmp/natural_books_sc_median.json` and `/tmp/natural_sc_extra_*.json`; the older PCA loader combines matching tables. These temporary calculation files are not PLS-DA prerequisites when using the persistent exports.

For accepted generated text and the matched-prefix token descriptors:

```bash
poetry run python compute_sc_prefix_bins.py
poetry run python analysis/compute_generated_prefix_token_metrics.py
```

The generated semantic script uses character truncations corresponding to its configured representative lengths for bins 0–3 and the full endpoint for the open-ended top bin. The token-metric script uses token prefixes for bins 0–3 and the full endpoint for the top bin, writing `/tmp/sc_generated_prefix_token_metrics.json`. Preserve these definitions when comparing results. `compute_sc_prefix_bin4.py` is an older separate top-bin computation; its fixed-length truncation should not replace the current endpoint definition. Newly measured datasets must be exported with the columns and relative paths in the PLS-DA input table before using that loader; retain raw descriptor values, IDs, bins, and token counts in those exports.

For the optional full-book landmark projection:

```bash
poetry run python compute_sc_landmarks.py
```

This requires the selected landmark text files and endpoint CSV from the input table. The measurement script writes `/tmp/landmark_sc.json`; preserve that result in a persistent location and pass its path with `--landmark-sc` when refitting PLS-DA, or place it at the default persistent path in the input table. Landmark measurements are projected after fitting and are not training/reference rows of the combined discriminant.

### 4. Regenerate plots and reports

For PLS-DA, run the preflight and the export-based commands above. For the older PCA command or grouped trajectories, retain the original measurement products and generated run records described in those sections. For grouped trajectories:

```bash
poetry run python analysis/plot_aggregate_metrics.py
```

This command writes the grouped legacy/additional-descriptor plots and Fig. 1 plot under `generated_texts_og/Output/`. Its output/cache paths are configurable; use `--help`. Save fresh outputs separately when comparing with the existing figures, and use `--refresh-cache` after changing source measurements.

## Tokenization and model provenance

The research token backend is `Tokenizer(kind="gpt4")`: tiktoken's encoding for `gpt-4` (the encoding name and tiktoken version are recorded in tokenizer metadata). The wrapper applies NFKC Unicode normalization, whitespace normalization, and, by default, spaces around punctuation; lowercase and digit normalization are disabled by default. These preprocessing choices are part of the measurement definition, not just display cleanup. The offline installation check deliberately uses the separate byte backend.

The generation configuration currently names models including `mistralai/Mistral-7B-Instruct-v0.3`; the semantic model is `Alibaba-NLP/gte-Qwen2-7B-instruct`. The model loaders in this checkout do not pin immutable model/tokenizer revisions. Consequently, a download made today is not evidence of which revision was used in an earlier run.

Available run metadata, caches, and saved outputs should be retained as evidence of what was executed. Without the original machine or a saved environment/model snapshot, the exact historical revision hashes, complete package environment, CUDA/driver versions, hardware, and wall-clock timings remain unverified. The lockfile and installation instructions describe the current code environment; they do not retroactively establish those historical details. For new calculations, record resolved model/tokenizer commits, package versions, hardware, inputs, and complete run parameters alongside the outputs.
