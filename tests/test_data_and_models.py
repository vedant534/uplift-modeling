from __future__ import annotations

import csv
import gzip
import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src.data import (
    COMPLEMENT_SAMPLING_METHOD,
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    ReservedSampleSpec,
    SAMPLING_METHOD,
    UNION_COMPLEMENT_SAMPLING_METHOD,
    _sample_complement_row_indices,
    _sample_row_indices,
    load_criteo_with_metadata,
    load_disjoint_criteo_samples_with_metadata,
    load_pinned_criteo_sample_with_metadata,
    plan_staged_criteo_samples,
    split_development_data,
)
from src.models import fit_treatment_model, predict_treatment


def _write_fixture(
    path: Path,
    rows: int = 100,
    *,
    feature_offset: float = 0.0,
    columns: tuple[str, ...] = EXPECTED_COLUMNS,
) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in range(rows):
            writer.writerow(
                [
                    *[
                        float(row * 100 + column) + feature_offset
                        for column in range(12)
                    ],
                    row % 2,
                    row % 3 == 0,
                    row % 4 == 0,
                    row % 2,
                ]
            )


def _index_digest(indices: np.ndarray) -> str:
    canonical = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _explicit_complement_sample(
    total_rows: int,
    excluded_rows: np.ndarray,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    complement = np.setdiff1d(
        np.arange(total_rows, dtype=np.int64),
        np.asarray(excluded_rows, dtype=np.int64),
        assume_unique=True,
    )
    ranks = _sample_row_indices(len(complement), sample_size, seed)
    return complement[ranks]


class DataAndModelContractTests(unittest.TestCase):
    def test_seeded_sample_spans_full_source_and_records_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv.gz"
            _write_fixture(path)
            loaded = load_criteo_with_metadata(
                path,
                sample_size=20,
                seed=42,
                verify_official_hash=False,
            )
            self.assertEqual(len(loaded.frame), 20)
            self.assertEqual(loaded.metadata.source_row_count, 100)
            self.assertEqual(
                loaded.metadata.sampling_method,
                "uniform_without_replacement_over_source_row_indices",
            )
            self.assertGreater(loaded.metadata.selected_max_source_row, 20)
            self.assertIsNotNone(loaded.metadata.selected_row_index_sha256)
            self.assertEqual(
                loaded.metadata.source_sha256_verification,
                "schema_only_nonofficial_hash",
            )

    def test_disjoint_development_and_final_samples_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv.gz"
            _write_fixture(path)
            development_seed = 11
            final_evaluation_seed = 29
            development_size = 23
            final_evaluation_size = 17

            expected_development = np.sort(
                np.random.default_rng(development_seed).choice(
                    100,
                    size=development_size,
                    replace=False,
                    shuffle=False,
                )
            )
            complement = np.setdiff1d(
                np.arange(100), expected_development, assume_unique=True
            )
            expected_final_ranks = np.sort(
                np.random.default_rng(final_evaluation_seed).choice(
                    len(complement),
                    size=final_evaluation_size,
                    replace=False,
                    shuffle=False,
                )
            )
            expected_final = complement[expected_final_ranks]

            with mock.patch("src.data.pd.read_csv", wraps=pd.read_csv) as read_csv:
                loaded = load_disjoint_criteo_samples_with_metadata(
                    path,
                    development_sample_size=development_size,
                    final_evaluation_sample_size=final_evaluation_size,
                    development_seed=development_seed,
                    final_evaluation_seed=final_evaluation_seed,
                    verify_official_hash=False,
                )
            self.assertEqual(read_csv.call_count, 1)

            repeated = load_disjoint_criteo_samples_with_metadata(
                path,
                development_sample_size=development_size,
                final_evaluation_sample_size=final_evaluation_size,
                development_seed=development_seed,
                final_evaluation_seed=final_evaluation_seed,
                verify_official_hash=False,
            )
            pd.testing.assert_frame_equal(loaded.development, repeated.development)
            pd.testing.assert_frame_equal(
                loaded.final_evaluation, repeated.final_evaluation
            )
            self.assertEqual(loaded.metadata.to_dict(), repeated.metadata.to_dict())

            development_rows = (
                loaded.development["f0"].to_numpy(dtype=np.int64) // 100
            )
            final_evaluation_rows = (
                loaded.final_evaluation["f0"].to_numpy(dtype=np.int64) // 100
            )
            np.testing.assert_array_equal(development_rows, expected_development)
            np.testing.assert_array_equal(final_evaluation_rows, expected_final)
            self.assertEqual(
                np.intersect1d(development_rows, final_evaluation_rows).size,
                0,
            )

            metadata = loaded.metadata
            self.assertEqual(metadata.source_row_overlap_count, 0)
            self.assertTrue(metadata.source_row_overlap_verified)
            self.assertEqual(metadata.development.sampling_method, SAMPLING_METHOD)
            self.assertEqual(
                metadata.final_evaluation.sampling_method,
                COMPLEMENT_SAMPLING_METHOD,
            )
            self.assertEqual(metadata.development.sampling_seed, development_seed)
            self.assertEqual(
                metadata.final_evaluation.sampling_seed, final_evaluation_seed
            )
            self.assertEqual(
                metadata.development.requested_sample_size, development_size
            )
            self.assertEqual(
                metadata.final_evaluation.requested_sample_size,
                final_evaluation_size,
            )
            self.assertEqual(
                metadata.development.selected_row_index_sha256,
                _index_digest(expected_development),
            )
            self.assertEqual(
                metadata.final_evaluation.selected_row_index_sha256,
                _index_digest(expected_final),
            )
            self.assertEqual(
                metadata.development.source_sha256,
                metadata.final_evaluation.source_sha256,
            )
            self.assertTrue(metadata.to_dict()["source_row_overlap_verified"])

    def test_complement_sampler_matches_explicit_complement_exhaustively(self) -> None:
        for total_rows in range(1, 8):
            for exclusion_mask in range(1 << total_rows):
                excluded = np.array(
                    [
                        row
                        for row in range(total_rows)
                        if exclusion_mask & (1 << row)
                    ],
                    dtype=np.int64,
                )
                available_rows = total_rows - len(excluded)
                for sample_size in range(1, available_rows + 1):
                    for seed in (0, 17):
                        with self.subTest(
                            total_rows=total_rows,
                            exclusion_mask=exclusion_mask,
                            sample_size=sample_size,
                            seed=seed,
                        ):
                            actual = _sample_complement_row_indices(
                                total_rows,
                                excluded,
                                sample_size,
                                seed,
                            )
                            expected = _explicit_complement_sample(
                                total_rows,
                                excluded,
                                sample_size,
                                seed,
                            )
                            np.testing.assert_array_equal(actual, expected)
                            self.assertEqual(len(actual), len(np.unique(actual)))
                            self.assertEqual(
                                np.intersect1d(
                                    actual,
                                    excluded,
                                    assume_unique=True,
                                ).size,
                                0,
                            )

    def test_staged_plan_reserves_prior_samples_and_excludes_their_union(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv.gz"
            total_rows = 80
            _write_fixture(path, rows=total_rows)
            development_size = 10
            development_seed = 11
            prior_one_size = 20
            prior_one_seed = 29
            prior_two_size = 18
            prior_two_seed = 31
            final_size = 12
            final_seed = 37

            development_rows = _sample_row_indices(
                total_rows, development_size, development_seed
            )
            prior_one_rows = _explicit_complement_sample(
                total_rows,
                development_rows,
                prior_one_size,
                prior_one_seed,
            )
            prior_two_rows = _explicit_complement_sample(
                total_rows,
                development_rows,
                prior_two_size,
                prior_two_seed,
            )
            pre_final_union = np.unique(
                np.concatenate(
                    (development_rows, prior_one_rows, prior_two_rows)
                )
            )
            final_rows = _explicit_complement_sample(
                total_rows,
                pre_final_union,
                final_size,
                final_seed,
            )
            reserved = (
                ReservedSampleSpec(
                    sample_role="prior_final_1",
                    sample_size=prior_one_size,
                    sampling_seed=prior_one_seed,
                    expected_selected_row_index_sha256=_index_digest(
                        prior_one_rows
                    ),
                ),
                ReservedSampleSpec(
                    sample_role="prior_final_2",
                    sample_size=prior_two_size,
                    sampling_seed=prior_two_seed,
                    expected_selected_row_index_sha256=_index_digest(
                        prior_two_rows
                    ),
                ),
            )

            plan = plan_staged_criteo_samples(
                path,
                development_sample_size=development_size,
                final_evaluation_sample_size=final_size,
                development_seed=development_seed,
                final_evaluation_seed=final_seed,
                reserved_accessed_samples=reserved,
                expected_development_row_index_sha256=_index_digest(
                    development_rows
                ),
                verify_official_hash=False,
            )
            repeated = plan_staged_criteo_samples(
                path,
                development_sample_size=development_size,
                final_evaluation_sample_size=final_size,
                development_seed=development_seed,
                final_evaluation_seed=final_seed,
                reserved_accessed_samples=reserved,
                expected_development_row_index_sha256=_index_digest(
                    development_rows
                ),
                verify_official_hash=False,
            )

            self.assertEqual(plan.to_dict(), repeated.to_dict())
            np.testing.assert_array_equal(
                plan.development.selected_row_indices, development_rows
            )
            np.testing.assert_array_equal(
                plan.pin_for("prior_final_1").selected_row_indices,
                prior_one_rows,
            )
            np.testing.assert_array_equal(
                plan.pin_for("prior_final_2").selected_row_indices,
                prior_two_rows,
            )
            np.testing.assert_array_equal(
                plan.final_evaluation.selected_row_indices, final_rows
            )
            self.assertEqual(
                plan.development.selected_row_index_sha256,
                _index_digest(development_rows),
            )
            self.assertEqual(
                plan.final_evaluation.selected_row_index_sha256,
                _index_digest(final_rows),
            )
            self.assertEqual(
                plan.pre_final_excluded_sample_roles,
                ("development", "prior_final_1", "prior_final_2"),
            )
            self.assertEqual(
                plan.pre_final_excluded_union_row_count,
                len(pre_final_union),
            )
            self.assertEqual(
                plan.pre_final_excluded_union_row_index_sha256,
                _index_digest(pre_final_union),
            )
            self.assertEqual(plan.final_overlap_with_excluded_union_count, 0)
            self.assertEqual(
                plan.final_evaluation.excluded_union_row_index_sha256,
                _index_digest(pre_final_union),
            )
            self.assertEqual(
                plan.final_evaluation.sampling_method,
                UNION_COMPLEMENT_SAMPLING_METHOD,
            )
            self.assertEqual(
                np.intersect1d(final_rows, development_rows).size,
                0,
            )
            self.assertEqual(
                np.intersect1d(final_rows, prior_one_rows).size,
                0,
            )
            self.assertEqual(
                np.intersect1d(final_rows, prior_two_rows).size,
                0,
            )

            overlap_by_pair = {
                frozenset(
                    (overlap.left_sample_role, overlap.right_sample_role)
                ): overlap.overlap_count
                for overlap in plan.pairwise_overlaps
            }
            self.assertEqual(
                overlap_by_pair[
                    frozenset(("prior_final_1", "prior_final_2"))
                ],
                np.intersect1d(prior_one_rows, prior_two_rows).size,
            )
            self.assertGreater(
                overlap_by_pair[
                    frozenset(("prior_final_1", "prior_final_2"))
                ],
                0,
            )
            for pair, overlap_count in overlap_by_pair.items():
                if "final_evaluation" in pair:
                    self.assertEqual(overlap_count, 0)
            all_union = np.unique(np.concatenate((pre_final_union, final_rows)))
            self.assertEqual(plan.all_pinned_union_row_count, len(all_union))
            self.assertEqual(
                plan.all_pinned_union_row_index_sha256,
                _index_digest(all_union),
            )

            invalid_reserved = (
                replace(
                    reserved[0],
                    expected_selected_row_index_sha256="0" * 64,
                ),
            )
            with self.assertRaisesRegex(
                ValueError, "not recorded digest"
            ):
                plan_staged_criteo_samples(
                    path,
                    development_sample_size=development_size,
                    final_evaluation_sample_size=final_size,
                    development_seed=development_seed,
                    final_evaluation_seed=final_seed,
                    reserved_accessed_samples=invalid_reserved,
                    verify_official_hash=False,
                )

    def test_pinned_samples_load_in_separate_verified_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv.gz"
            _write_fixture(path, rows=80)

            with mock.patch("src.data.pd.read_csv", wraps=pd.read_csv) as read_csv:
                plan = plan_staged_criteo_samples(
                    path,
                    development_sample_size=20,
                    final_evaluation_sample_size=15,
                    development_seed=11,
                    final_evaluation_seed=29,
                    verify_official_hash=False,
                )
                self.assertEqual(read_csv.call_count, 0)

                development = load_pinned_criteo_sample_with_metadata(
                    path,
                    plan.development,
                    verify_official_hash=False,
                )
                self.assertEqual(read_csv.call_count, 1)
                development_rows = (
                    development.frame["f0"].to_numpy(dtype=np.int64) // 100
                )
                np.testing.assert_array_equal(
                    development_rows,
                    plan.development.selected_row_indices,
                )
                self.assertEqual(
                    development.metadata.selected_row_index_sha256,
                    plan.development.selected_row_index_sha256,
                )

                final_evaluation = load_pinned_criteo_sample_with_metadata(
                    path,
                    plan.final_evaluation,
                    verify_official_hash=False,
                )
                self.assertEqual(read_csv.call_count, 2)
                final_rows = (
                    final_evaluation.frame["f0"].to_numpy(dtype=np.int64) // 100
                )
                np.testing.assert_array_equal(
                    final_rows,
                    plan.final_evaluation.selected_row_indices,
                )
                self.assertEqual(
                    final_evaluation.metadata.selected_row_index_sha256,
                    plan.final_evaluation.selected_row_index_sha256,
                )
                self.assertEqual(
                    np.intersect1d(development_rows, final_rows).size,
                    0,
                )

            tampered_source = Path(directory) / "tampered.csv.gz"
            _write_fixture(tampered_source, rows=80, feature_offset=0.5)
            with self.assertRaisesRegex(ValueError, "source SHA-256"):
                load_pinned_criteo_sample_with_metadata(
                    tampered_source,
                    plan.development,
                    verify_official_hash=False,
                )

            bad_schema_source = Path(directory) / "bad-schema.csv.gz"
            bad_columns = (*EXPECTED_COLUMNS[:-1], "post_treatment_leak")
            _write_fixture(bad_schema_source, rows=80, columns=bad_columns)
            with self.assertRaisesRegex(ValueError, "Unexpected dataset header"):
                load_pinned_criteo_sample_with_metadata(
                    bad_schema_source,
                    plan.development,
                    verify_official_hash=False,
                )

            wrong_count_pin = replace(
                plan.development,
                source_row_count=plan.development.source_row_count + 1,
            )
            with self.assertRaisesRegex(ValueError, "source row count"):
                load_pinned_criteo_sample_with_metadata(
                    path,
                    wrong_count_pin,
                    verify_official_hash=False,
                )
            with self.assertRaisesRegex(ValueError, "do not match their SHA-256"):
                replace(
                    plan.development,
                    selected_row_index_sha256="0" * 64,
                )

    def test_models_require_exact_pretreatment_feature_columns(self) -> None:
        rng = np.random.default_rng(7)
        rows = 400
        frame = pd.DataFrame(
            {
                column: (
                    rng.normal(size=rows)
                    if column in {"f0", "f2", "f7", "f10"}
                    else rng.integers(0, 5, size=rows).astype(float)
                )
                for column in FEATURE_COLUMNS
            }
        )
        treatment = np.tile([0, 1], rows // 2)
        model = fit_treatment_model(frame, treatment, max_iter=100)
        probability = predict_treatment(model, frame.iloc[:10])
        self.assertEqual(probability.shape, (10,))
        self.assertGreater(model.encoded_feature_count, len(FEATURE_COLUMNS))
        with self.assertRaisesRegex(ValueError, "exactly f0 through f11"):
            fit_treatment_model(frame.assign(treatment=treatment), treatment)
        with self.assertRaisesRegex(ValueError, "exactly f0 through f11"):
            fit_treatment_model(frame.loc[:, list(reversed(FEATURE_COLUMNS))], treatment)

    def test_development_split_is_reproducible_and_leaves_no_test_partition(self) -> None:
        rows = 400
        frame = pd.DataFrame(
            {
                **{column: np.arange(rows, dtype=float) for column in FEATURE_COLUMNS},
                "treatment": np.tile([0, 1], rows // 2),
                "conversion": np.tile([0, 0, 0, 1], rows // 4),
            }
        )
        first = split_development_data(frame, seed=17, validation_fraction=0.25)
        second = split_development_data(frame, seed=17, validation_fraction=0.25)
        self.assertEqual(len(first.train), 300)
        self.assertEqual(len(first.validation), 100)
        self.assertFalse(hasattr(first, "test"))
        pd.testing.assert_frame_equal(first.train, second.train)
        pd.testing.assert_frame_equal(first.validation, second.validation)
        with self.assertRaisesRegex(ValueError, "strictly between"):
            split_development_data(frame, validation_fraction=0.0)


if __name__ == "__main__":
    unittest.main()
