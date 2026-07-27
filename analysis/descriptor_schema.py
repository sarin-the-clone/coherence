"""Canonical descriptor selection for multivariate analysis.

Raw metric files retain all diagnostic and operational fields. This module
preserves scaling/informational descriptors and all token descriptors with a
distinct interpretation. PCA and PLS-DA exclude only exact token aliases.
"""

SC_WINDOWS = ("W2k", "W8k", "W40k")
SC_METRICS = ("off_diag_mean", "novelty_mean", "sim_lag1_mean")
SC_FEATURES = tuple(f"{window}_{metric}" for window in SC_WINDOWS for metric in SC_METRICS)

# entropy_bits is the canonical unigram entropy descriptor. tp_entropy_0 has
# the same intended definition but remains only in raw outputs for auditing the
# known tokenization/source-provenance mismatch.
STAT_INFO_METRICS = (
    "beta",
    "alpha",
    "H_DFA1",
    "H_WT_db4",
    "entropy_bits",
    "perplexity",
    "lz_complexity",
    "mi_decay_gamma",
)

TOKEN_DISTRIBUTION_METRICS = (
    "td_vocab_size",
    "td_type_token_ratio",
    "td_hapax_type_fraction",
    "td_hapax_token_fraction",
    "td_top1_token_fraction",
    "td_top10_token_fraction",
    "td_zipf_exponent",
    "td_heaps_exponent",
)

TOKEN_RECURRENCE_METRICS = (
    "tr_repeat_gap_mean",
    "tr_repeat_gap_mean_log2",
    "tr_repeat_gap_median_log2",
    "tr_repeat_gap_cv",
    "tr_repeat_gap_burstiness",
    "tr_repeat_gap_logbin_entropy",
)

# The five empirical entropies are the source measurements. Their derived
# measures remain in the matrix because they describe distinct, useful views of
# contextual predictability: absolute gain, relative redundancy, and the rate
# of entropy reduction. They must be interpreted as dependent descriptors.
TOKEN_PREDICTABILITY_METRICS = (
    "tp_cond_entropy_1",
    "tp_cond_entropy_2",
    "tp_cond_entropy_3",
    "tp_cond_entropy_4",
    "tp_redundancy_1",
    "tp_entropy_gain_1",
    "tp_redundancy_2",
    "tp_entropy_gain_2",
    "tp_redundancy_3",
    "tp_entropy_gain_3",
    "tp_redundancy_4",
    "tp_entropy_gain_4",
    "tp_entropy_slope",
)

TOKEN_METRICS = (
    *TOKEN_DISTRIBUTION_METRICS,
    *TOKEN_RECURRENCE_METRICS,
    *TOKEN_PREDICTABILITY_METRICS,
)

CANONICAL_FEATURES = (*SC_FEATURES, *STAT_INFO_METRICS, *TOKEN_METRICS)
CANONICAL_NON_SC_FEATURES = (*STAT_INFO_METRICS, *TOKEN_METRICS)

# Excluded token features remain in the raw metric outputs. Each is an exact
# duplicate of a retained column, so retaining both would double-weight an
# identical measurement rather than preserve a distinct interpretation.
EXCLUDED_EXACT_ALIASES = {
    "td_zipf_slope": "-td_zipf_exponent",
    "tr_repeat_rate": "1 - td_type_token_ratio",
    "tp_entropy_drop_4": "tp_entropy_0 - tp_cond_entropy_4",
    "tp_context_efficiency_4": "1 - tp_cond_entropy_4 / tp_entropy_0",
}

EXCLUDED_PROVENANCE_DUPLICATES = {
    "tp_entropy_0": "same intended unigram entropy as entropy_bits; raw values differ "
                    "because the stored pipelines are not aligned",
}

EXCLUDED_MODEL_FEATURES = {
    **EXCLUDED_EXACT_ALIASES,
    **EXCLUDED_PROVENANCE_DUPLICATES,
}


def available_features(columns, include_sc=True):
    """Return canonical descriptors present in a data frame, in stable order."""
    candidates = CANONICAL_FEATURES if include_sc else CANONICAL_NON_SC_FEATURES
    available = set(columns)
    return [feature for feature in candidates if feature in available]
