"""Compact causal ad-targeting experiment package."""

from .data import (
    CRITEO_V21_SHA256,
    CRITEO_V21_URL,
    FEATURE_COLUMNS,
    OUTCOME_COLUMNS,
    TREATMENT_COLUMN,
    DataSplits,
    DevelopmentSplits,
    DisjointSampleMetadata,
    LoadedDisjointSamples,
    download_criteo,
    load_criteo,
    load_disjoint_criteo_samples_with_metadata,
    split_development_data,
    split_data,
    summarize_data,
)

__all__ = [
    "CRITEO_V21_SHA256",
    "CRITEO_V21_URL",
    "FEATURE_COLUMNS",
    "OUTCOME_COLUMNS",
    "TREATMENT_COLUMN",
    "DataSplits",
    "DevelopmentSplits",
    "DisjointSampleMetadata",
    "LoadedDisjointSamples",
    "download_criteo",
    "load_criteo",
    "load_disjoint_criteo_samples_with_metadata",
    "split_development_data",
    "split_data",
    "summarize_data",
]
