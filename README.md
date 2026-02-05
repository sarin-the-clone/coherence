# Coherence Analysis of Long-Form Generated Text

This project provides a framework for generating long-form texts (novel-length) using LLMs and analyzing their "coherence" against a corpus of natural texts (Gutenberg and modern books). It employs a suite of structural and informational metrics to quantify textual properties and uses PCA to visualize the trajectory of generated books relative to natural genres.

## Project Structure

```
coherence/
├── analysis/                 # Scripts for plotting and PCA analysis
│   ├── plot_chunk_metrics.py     # Plots metric evolution for individual generated books
│   ├── plot_aggregate_metrics.py # Aggregates and plots metrics across runs
│   └── pca_on_natural_texts.py   # Main PCA analysis script comparing all books
├── generated_texts_og/       # Generated book artifacts and generation logic
│   ├── prompts/                  # Prompt templates (e.g., initial_prompt.md)
│   ├── run_longbook.py           # Core script for generating a single book
│   └── longbook_engine.py        # Engine handling context management and generation loop
├── natural_texts/            # Corpus of reference books (Natural Texts)
│   ├── books/                    # Raw text files (Gutenberg etc.)
│   └── Output/                   # Processed batches and metadata
├── metrics.py                # Core library of coherence metrics (Entropy, LZ, Alpha/Beta)
├── tokenizer.py              # Tokenization utilities
├── run_experiments.sh        # Main shell script to orchestrate batched experiments
└── pyproject.toml            # Poetry dependency configuration
```

## Key Metrics

The analysis relies on several key metrics defined in `metrics.py`:

*   **Structural Metrics**:
    *   `alpha`, `beta`: Power-law exponents related to word frequency and correlations.
    *   `Hurst` (H_DFA1, H_DFA3, etc): Detrended Fluctuation Analysis exponents measuring long-range correlations.
    *   `D`: Fractal dimension.
*   **Informational Metrics**:
    *   `entropy_bits`: Shannon entropy.
    *   `perplexity`: Model perplexity (using the generating or reference model).
    *   `lz_complexity`: Lempel-Ziv complexity (normalized).
    *   `mi_decay_gamma`: Rate of mutual information decay over distance.

## Usage

> **Note**: The commands below are rudimentary examples to get you started. Most scripts have many additional parameters (model config, batch sizes, plot settings) not described here.

### 1. Environment Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
# Install dependencies
poetry install

# Activate shell
poetry shell
```

### 2. Generating Prompts

To generate a new "story contract" prompt file for a specific genre/theme:

```bash
python3 generated_texts_og/quick_generate_prompt.py \
    --genre "Fiction" \
    --theme "A detailed theme description..." \
    --output "generated_texts_og/prompts/my_prompt.md"
```

### 3. Generating Books

Use `run_longbook.py` to generate a novel-length text. This script iteratively generates chunks, maintaining context and computing metrics.

```bash
python3 generated_texts_og/run_longbook.py \
    --model mistral \
    --prompt_file "generated_texts_og/prompts/my_prompt.md" \
    --target_tokens 1000000 \
    --max_tokens 16384 \
    --temp_low 1.0 --temp_high 1.4 \
    --top_p 0.95 \
    --run_id "my_run_001" \
    --restart_every 200
```

Alternatively, `run_experiments.sh` can be used to manage batch experiments.

### 3. Metric Analysis & Plotting

After generation, metrics are stored in `accumulated_metrics.json` within each run folder.

**Plot individual book trajectory:**
```bash
python3 analysis/plot_chunk_metrics.py \
  --run_dir generated_texts_og/prompts/YOUR_RUN_FOLDER \
  --out generated_texts_og/Output/my_plot.jpg
```

**Run PCA Comparison:**
To project generated books onto the space defined by natural literature:
```bash
python3 analysis/project_on_natural_books_PCs.py
```
This generates density plots showing where your generated text falls compared to genres like Fiction, Sci-Fi, and Non-Fiction.

**Aggregate Analysis:**
```bash
python3 analysis/plot_aggregate_metrics.py
```
Produces an overlay of multiple runs to visualize general trends.

## Data
*   **Natural Texts**: Sourced primarily from Project Gutenberg. (Note: Some modern books exist in the repo for illustration purposes but are minimal).
*   **Generated Texts**: Output to `generated_texts_og/prompts/`. Each run creates a timestamped folder containing the full text, chunks, and metric logs.

## Configuration

*   **Model Config**: `generated_texts_og/config/model_config.yaml` (Defines LLM endpoints/paths).
*   **Prompt Templates**: `generated_texts_og/prompts/` (Markdown files defining story contracts).

