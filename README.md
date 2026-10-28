# Coherence Analysis of Long-Form Generated Text

Code for generating book-length text with local language models and comparing its statistical and semantic organization with natural books. The pipeline combines token-based scaling and information measures, token-usage statistics, and similarities between passage embeddings. Partial least-squares discriminant analysis (PLS-DA) evaluates how these descriptors distinguish the natural and generated corpora; PCA provides complementary visualizations.

Start with the installation check below. The [reproduction guide](docs/reproduction.md) maps the analysis scripts to their required inputs and outputs, describes the full workflow, and identifies inputs that a fresh clone does not supply.

## Installation and validation

Run commands from the repository root. Python 3.10–3.12 and Poetry are supported; dependency versions are recorded in [pyproject.toml](pyproject.toml) and [poetry.lock](poetry.lock). The existing PyTorch source targets Linux with CUDA 12.8. CPU-only and other-platform installations need a compatible PyTorch build rather than that CUDA-specific source.

```bash
poetry install
poetry run python -m utils.validate_installation
poetry run python -m unittest discover -s tests -v
```

The validation command runs a small CPU example using UTF-8 byte tokenization, descriptor calculations, and synthetic PLS-DA data. It does not download model weights or require the research corpora. Its purpose is to check the installed software; it does not reproduce the manuscript's measurements or performance scores. The research tokenization and model-based steps are described separately below.

Optional tooling is installed with `poetry install --with ebooks` for ebook conversion/audio utilities or `poetry install --with nlp` for spaCy tokenization. The spaCy language model is a separate download.

Generation and semantic analysis load local Hugging Face models. They require model downloads, access approval where applicable, and sufficient memory for the selected model, its context, and its batch size. The supplied generation configurations select CUDA. No measured runtime or minimum GPU-memory specification from the original calculation machine is available in this checkout.

## Repository map

| Location | Purpose |
| --- | --- |
| [`generated_texts_og/run_longbook.py`](generated_texts_og/run_longbook.py) | Generate or resume one book; CLI for conditioning, sampling, gating, and retries. |
| [`generated_texts_og/longbook_engine.py`](generated_texts_og/longbook_engine.py) | Generation loop, context and memory management, sanitization, and acceptance logic. |
| [`generated_texts_og/config/model_config.yaml`](generated_texts_og/config/model_config.yaml) | Model identifiers, devices, context limits, and sampling defaults. |
| [`generated_texts_og/prompts/`](generated_texts_og/prompts/) | Story-contract templates; generated run folders are also written here. |
| [`metrics.py`](metrics.py) and [`tokenizer.py`](tokenizer.py) | Token-based descriptor estimators and text normalization/tokenization. |
| [`semantic_coherence.py`](semantic_coherence.py) | Passage embeddings, hierarchical window pooling, and cosine-similarity descriptors. |
| [`analysis/descriptor_schema.py`](analysis/descriptor_schema.py) | Canonical multivariate descriptor selection and excluded aliases. |
| [`analysis/plsda_discriminant.py`](analysis/plsda_discriminant.py) | PLS-DA, group-aware cross-validation, VIP, score plots, and biplots. |
| [`analysis/plsda_inputs.py`](analysis/plsda_inputs.py) | Load the persistent natural/generated descriptor exports used by PLS-DA. |
| [`analysis/plot_aggregate_metrics.py`](analysis/plot_aggregate_metrics.py) | Descriptor trajectories aggregated by conditioning mode. |
| [`analysis/pca_sc_combined_size_bins.py`](analysis/pca_sc_combined_size_bins.py) | PCA combining semantic and token descriptors on the semantic subcorpus. |
| [`natural_texts/Output/`](natural_texts/Output/) | Reference-corpus metadata, descriptor tables, and analysis outputs. |
| [`natural_texts/Selected_Books/`](natural_texts/Selected_Books/) | Selected natural-book texts and associated trajectory measurements. |
| [`docs/reproduction.md`](docs/reproduction.md) | Workflow, input requirements, artifact map, and reproducibility limits. |

## Descriptors

The canonical combined analysis has 44 descriptors, selected in `analysis/descriptor_schema.py`:

- Eight scaling/informational descriptors: `beta` (autocorrelation decay), `alpha` (spectral scaling), `H_DFA1` and `H_WT_db4` (Hurst estimates), `entropy_bits`, `perplexity`, `lz_complexity`, and `mi_decay_gamma`.
- Twenty-seven token-statistics descriptors covering token distributions (`td_*`), recurrence gaps (`tr_*`), and finite-context predictability (`tp_*`). These include Zipf and Heaps exponents.
- Nine semantic descriptors: global similarity, novelty, and adjacent-window similarity, each at W2k, W8k, and W40k. The labels refer to character windows of 2,048, 8,000, and 40,000 characters. Larger windows use hierarchical pooling of subwindow embeddings.

`perplexity` in this code is `2 ** entropy_bits`, calculated from empirical unigram token frequencies. It is not language-model likelihood perplexity. Raw metric files include further diagnostics and aliases that are excluded from the canonical multivariate feature set.

The research tokenizer is the `gpt4` backend in `tokenizer.py`, implemented with tiktoken. Its normalization options matter for comparisons; see the implementation and the [reproduction guide](docs/reproduction.md#tokenization-and-model-provenance). Semantic analysis uses a separate embedding model, `Alibaba-NLP/gte-Qwen2-7B-instruct` by default.

## Generate a book

Use an existing story contract to avoid a separate prompt-generation step. This example requests one chunk; it still loads the full configured model and runs the generation pipeline.

```bash
poetry run python generated_texts_og/run_longbook.py \
  --model mistral \
  --prompt_file generated_texts_og/prompts/historical_library.md \
  --chunks 1 \
  --max_tokens 2048 \
  --conditioning_mode extended_conditioning \
  --run_id example_history
```

For a book-scale run, use `--target_tokens 1000000` instead of `--chunks 1` and select the desired sampling and conditioning settings. Use a new `--run_id` for a new experiment: the engine may resume a matching existing run. The printed output identifies the run and text paths.

Run folders under `generated_texts_og/prompts/` contain `metadata.json`, chunk-attempt payloads (`chunk_NNN_attempt_A.txt`, containing JSON), acceptance markers, and memory/debug artifacts. `accumulated_metrics` is a field inside chunk-attempt payloads. Completed book text is saved under `generated_texts_og/Texts/`, with EPUB export under `generated_texts_og/Epub/`.

For settings and resume options:

```bash
poetry run python generated_texts_og/run_longbook.py --help
```

The batch `.sh` files record experiment cohorts and settings. Review those settings before running them: they schedule large calculations. `run_experiments.sh` contains historical experiment commands rather than a general paper-reproduction entry point.

The batch scripts resolve the checkout from their own location. They use `python3` from the active environment by default; set `COHERENCE_PYTHON=/path/to/environment/bin/python` to select another interpreter. Download/offline behavior follows the caller's Hugging Face settings, including `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE`.

## Plot a generated run

Replace `PATH_TO_RUN` with the actual run directory printed by generation:

```bash
poetry run python analysis/plot_chunk_metrics.py \
  --folder PATH_TO_RUN
```

The default image is written inside that run directory. Additional token-statistics presets require a `token_naturalness_by_chunk.jsonl` sidecar; these descriptors can be computed retrospectively with the scripts listed in the reproduction guide.

## Inspect or rerun the analyses

Existing PLS-DA reports, VIP tables, out-of-fold predictions, and figures are under [`natural_texts/Output/pca/discriminant/`](natural_texts/Output/pca/discriminant/), in `combined`, `sc`, and `token_stat` subdirectories.

PLS-DA reads persistent descriptor exports already in the repository. Its input root defaults to `natural_texts/Output/`; it does not need the original temporary JSON files, generated chunk records, or model weights. Check the inputs before starting:

```bash
python -m utils.check_reproduction_inputs --input-dir natural_texts/Output
```

This read-only, standard-library command checks file presence and export CSV headers, prints a JSON report, and exits with status 1 if required inputs are missing or malformed. It does not validate descriptor values, complete corpus membership, historical provenance, or agreement with the manuscript.

The combined analysis can be rerun into a separate output directory:

```bash
poetry run python analysis/plsda_discriminant.py \
  --input-dir natural_texts/Output \
  --feature-set combined \
  --n-components 3 \
  --output-root reproduction_outputs/discriminant
```

The natural raw semantic export is `pca/sc_raw/sc_pca_features_robust_z.csv` (raw values despite the filename); the natural scaling/token descriptors are joined from `batch_results/` and `gutenberg_benchmark_cache/`. The generated export, `pca/sc_combined/sc_combined_pca_generated_scores.csv`, contains all 44 descriptors, not just PCA scores. These paths are relative to `--input-dir`. The supplied inputs yield 1,047 complete natural books and 245 prefixes from 49 generated runs.

Use `--input-dir /path/to/persistent/data` for another folder containing the same relative layout. `--size-stats`, `--landmark-sc`, and `--landmark-csv` override individual optional files; their defaults also resolve under the input root. The [reproduction guide](docs/reproduction.md#inputs-needed-for-the-combined-analysis) lists the exact layout and separates this PLS-DA route from the older PCA/measurement scripts that still use temporary calculation files.

## Data and provenance

The reference corpus is based on Project Gutenberg; the repository contains selected texts, benchmark metadata, and computed descriptors. The complete downloaded book cache and the original generated run records are not supplied by a fresh clone. Their expected paths and the scripts that produce them are documented in the reproduction guide. Do not assume that a downloaded text or a newly generated book exactly recreates a historical input.

The available files do not establish the original machine's complete software environment, model commit hashes, or GPU timings. Current installation metadata describes how to run the current code. It must not be interpreted as a recovered record of the original calculation environment.

## Citing the code

Use the repository version or commit used for your analysis; software citation metadata is in [CITATION.cff](CITATION.cff). A release DOI can be added there once a tagged version has been archived.
