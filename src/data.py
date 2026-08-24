"""Loading and splitting utilities for the corrected Criteo uplift dataset."""

from __future__ import annotations

import csv
import gzip
import hashlib
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from numbers import Integral
from pathlib import Path
from typing import IO, Sequence
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = tuple(f"f{i}" for i in range(12))
CONTINUOUS_FEATURE_COLUMNS = ("f0", "f2", "f7", "f10")
CATEGORICAL_FEATURE_COLUMNS = tuple(
    column for column in FEATURE_COLUMNS if column not in CONTINUOUS_FEATURE_COLUMNS
)
TREATMENT_COLUMN = "treatment"
OUTCOME_COLUMNS = ("conversion", "visit")
EXPOSURE_COLUMN = "exposure"

EXPECTED_COLUMNS = (
    *FEATURE_COLUMNS,
    TREATMENT_COLUMN,
    "conversion",
    "visit",
    EXPOSURE_COLUMN,
)

CRITEO_V21_URL = "https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
# Published by Criteo's official Hugging Face dataset repository.
CRITEO_V21_SHA256 = (
    "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
)
CRITEO_DATASET_VERSION = "v2.1 corrected/unbiased"
SAMPLING_METHOD = "uniform_without_replacement_over_source_row_indices"
COMPLEMENT_SAMPLING_METHOD = (
    "uniform_without_replacement_over_complement_of_development_source_row_indices"
)
UNION_COMPLEMENT_SAMPLING_METHOD = (
    "uniform_without_replacement_over_complement_of_all_pinned_source_row_indices"
)

_DOWNLOAD_BLOCK_BYTES = 1024 * 1024
_COUNT_BLOCK_BYTES = 8 * 1024 * 1024
_CSV_CHUNK_ROWS = 250_000


def _nonempty_sample_role(value: object, name: str = "sample_role") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _nonnegative_seed(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    seed = int(value)
    if seed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return seed


def _normalized_sha256(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return digest


@dataclass(frozen=True)
class DataSplits:
    """Train, validation, and untouched final-test partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class DevelopmentSplits:
    """Model-fitting and decision-selection partitions of development data."""

    train: pd.DataFrame
    validation: pd.DataFrame


@dataclass(frozen=True)
class DataLoadMetadata:
    """Reproducible provenance for one loaded source and sample."""

    dataset_version: str
    source_path: str
    source_sha256: str
    source_sha256_verification: str
    source_row_count: int
    requested_sample_size: int
    loaded_sample_size: int
    sampling_method: str
    sampling_seed: int | None
    selected_row_index_sha256: str | None
    selected_min_source_row: int | None
    selected_max_source_row: int | None

    def to_dict(self) -> dict[str, str | int | None]:
        return asdict(self)


@dataclass(frozen=True)
class LoadedData:
    """A validated modeling frame and its source/sample provenance."""

    frame: pd.DataFrame
    metadata: DataLoadMetadata


@dataclass(frozen=True)
class DisjointSampleMetadata:
    """Provenance for development and fresh final-evaluation samples."""

    development: DataLoadMetadata
    final_evaluation: DataLoadMetadata
    source_row_overlap_count: int

    @property
    def source_row_overlap_verified(self) -> bool:
        return self.source_row_overlap_count == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "development": self.development.to_dict(),
            "final_evaluation": self.final_evaluation.to_dict(),
            "source_row_overlap_count": self.source_row_overlap_count,
            "source_row_overlap_verified": self.source_row_overlap_verified,
        }


@dataclass(frozen=True)
class LoadedDisjointSamples:
    """Disjoint development/final frames and their joint provenance."""

    development: pd.DataFrame
    final_evaluation: pd.DataFrame
    metadata: DisjointSampleMetadata


@dataclass(frozen=True)
class ReservedSampleSpec:
    """Recipe and receipt used to reconstruct a previously accessed sample.

    ``excluded_sample_roles`` names earlier pins whose union formed the sampling
    exclusion set. The expected digest makes reconstruction fail closed if the
    recorded seed, size, ordering, or exclusion history is wrong.
    """

    sample_role: str
    sample_size: int
    sampling_seed: int
    expected_selected_row_index_sha256: str
    excluded_sample_roles: tuple[str, ...] = ("development",)

    def __post_init__(self) -> None:
        role = _nonempty_sample_role(self.sample_role)
        size = _positive_integer(self.sample_size, "sample_size")
        seed = _nonnegative_seed(self.sampling_seed, "sampling_seed")
        digest = _normalized_sha256(
            self.expected_selected_row_index_sha256,
            "expected_selected_row_index_sha256",
        )
        if isinstance(self.excluded_sample_roles, str):
            raise ValueError("excluded_sample_roles must be a sequence of roles")
        try:
            excluded = tuple(
                _nonempty_sample_role(value, "excluded sample role")
                for value in self.excluded_sample_roles
            )
        except TypeError as exc:
            raise ValueError(
                "excluded_sample_roles must be a sequence of roles"
            ) from exc
        if len(excluded) != len(set(excluded)):
            raise ValueError("excluded_sample_roles must not contain duplicates")
        if role in excluded:
            raise ValueError("a reserved sample cannot exclude its own sample role")
        object.__setattr__(self, "sample_role", role)
        object.__setattr__(self, "sample_size", size)
        object.__setattr__(self, "sampling_seed", seed)
        object.__setattr__(self, "expected_selected_row_index_sha256", digest)
        object.__setattr__(self, "excluded_sample_roles", excluded)


@dataclass(frozen=True)
class PinnedSourceSample:
    """Immutable source-index authorization for one staged sample read."""

    sample_role: str
    dataset_version: str
    source_path: str
    source_sha256: str
    source_sha256_verification: str
    source_row_count: int
    sample_size: int
    sampling_method: str
    sampling_seed: int
    selected_row_index_sha256: str
    selected_min_source_row: int
    selected_max_source_row: int
    excluded_sample_roles: tuple[str, ...]
    excluded_union_row_count: int
    excluded_union_row_index_sha256: str
    _selected_row_indices_le_i8: bytes = field(repr=False, compare=False)
    _excluded_union_row_indices_le_i8: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        role = _nonempty_sample_role(self.sample_role)
        source_count = _positive_integer(self.source_row_count, "source_row_count")
        size = _positive_integer(self.sample_size, "sample_size")
        if size > source_count:
            raise ValueError("sample_size cannot exceed source_row_count")
        seed = _nonnegative_seed(self.sampling_seed, "sampling_seed")
        source_digest = _normalized_sha256(self.source_sha256, "source_sha256")
        index_digest = _normalized_sha256(
            self.selected_row_index_sha256, "selected_row_index_sha256"
        )
        excluded_digest = _normalized_sha256(
            self.excluded_union_row_index_sha256,
            "excluded_union_row_index_sha256",
        )
        if not isinstance(self._selected_row_indices_le_i8, bytes):
            raise ValueError("pinned source-row indices must be immutable bytes")
        expected_bytes = size * np.dtype("<i8").itemsize
        if len(self._selected_row_indices_le_i8) != expected_bytes:
            raise ValueError("pinned source-row byte length does not match sample_size")
        indices = np.frombuffer(self._selected_row_indices_le_i8, dtype="<i8")
        if (
            indices[0] < 0
            or indices[-1] >= source_count
            or np.any(np.diff(indices) <= 0)
        ):
            raise ValueError(
                "pinned source-row indices must be sorted, unique, and within bounds"
            )
        if hashlib.sha256(self._selected_row_indices_le_i8).hexdigest() != index_digest:
            raise ValueError(
                "pinned source-row bytes do not match their SHA-256 digest"
            )
        if (
            isinstance(self.selected_min_source_row, (bool, np.bool_))
            or not isinstance(self.selected_min_source_row, Integral)
            or int(indices[0]) != int(self.selected_min_source_row)
        ):
            raise ValueError("selected_min_source_row does not match pinned indices")
        if (
            isinstance(self.selected_max_source_row, (bool, np.bool_))
            or not isinstance(self.selected_max_source_row, Integral)
            or int(indices[-1]) != int(self.selected_max_source_row)
        ):
            raise ValueError("selected_max_source_row does not match pinned indices")
        if (
            isinstance(self.excluded_union_row_count, (bool, np.bool_))
            or not isinstance(self.excluded_union_row_count, Integral)
            or not 0 <= int(self.excluded_union_row_count) <= source_count
        ):
            raise ValueError("excluded_union_row_count must be within source bounds")
        excluded_count = int(self.excluded_union_row_count)
        if not isinstance(self._excluded_union_row_indices_le_i8, bytes):
            raise ValueError("excluded source-row indices must be immutable bytes")
        expected_excluded_bytes = excluded_count * np.dtype("<i8").itemsize
        if len(self._excluded_union_row_indices_le_i8) != expected_excluded_bytes:
            raise ValueError(
                "excluded source-row byte length does not match "
                "excluded_union_row_count"
            )
        excluded_indices = np.frombuffer(
            self._excluded_union_row_indices_le_i8, dtype="<i8"
        )
        if excluded_indices.size and (
            excluded_indices[0] < 0
            or excluded_indices[-1] >= source_count
            or np.any(np.diff(excluded_indices) <= 0)
        ):
            raise ValueError(
                "excluded source-row indices must be sorted, unique, and within bounds"
            )
        if (
            hashlib.sha256(self._excluded_union_row_indices_le_i8).hexdigest()
            != excluded_digest
        ):
            raise ValueError(
                "excluded source-row bytes do not match their SHA-256 digest"
            )
        if np.intersect1d(indices, excluded_indices, assume_unique=True).size:
            raise ValueError("pinned source-row indices overlap their exclusion union")
        excluded_roles = tuple(
            _nonempty_sample_role(value, "excluded sample role")
            for value in self.excluded_sample_roles
        )
        if len(excluded_roles) != len(set(excluded_roles)):
            raise ValueError("excluded_sample_roles must not contain duplicates")
        for value, name in (
            (self.dataset_version, "dataset_version"),
            (self.source_path, "source_path"),
            (self.source_sha256_verification, "source_sha256_verification"),
            (self.sampling_method, "sampling_method"),
        ):
            _nonempty_sample_role(value, name)
        object.__setattr__(self, "sample_role", role)
        object.__setattr__(self, "source_sha256", source_digest)
        object.__setattr__(self, "source_row_count", source_count)
        object.__setattr__(self, "sample_size", size)
        object.__setattr__(self, "sampling_seed", seed)
        object.__setattr__(self, "selected_row_index_sha256", index_digest)
        object.__setattr__(
            self, "selected_min_source_row", int(self.selected_min_source_row)
        )
        object.__setattr__(
            self, "selected_max_source_row", int(self.selected_max_source_row)
        )
        object.__setattr__(self, "excluded_sample_roles", excluded_roles)
        object.__setattr__(self, "excluded_union_row_count", excluded_count)
        object.__setattr__(
            self, "excluded_union_row_index_sha256", excluded_digest
        )

    @property
    def selected_row_indices(self) -> np.ndarray:
        """Return a read-only little-endian int64 view of the pinned indices."""

        return np.frombuffer(self._selected_row_indices_le_i8, dtype="<i8")

    @property
    def excluded_union_row_indices(self) -> np.ndarray:
        """Return a read-only view of the rows this pin was constrained to avoid."""

        return np.frombuffer(self._excluded_union_row_indices_le_i8, dtype="<i8")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe pin metadata without embedding the large index array."""

        return {
            "sample_role": self.sample_role,
            "dataset_version": self.dataset_version,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_sha256_verification": self.source_sha256_verification,
            "source_row_count": self.source_row_count,
            "sample_size": self.sample_size,
            "sampling_method": self.sampling_method,
            "sampling_seed": self.sampling_seed,
            "selected_row_index_sha256": self.selected_row_index_sha256,
            "selected_min_source_row": self.selected_min_source_row,
            "selected_max_source_row": self.selected_max_source_row,
            "excluded_sample_roles": list(self.excluded_sample_roles),
            "excluded_union_row_count": self.excluded_union_row_count,
            "excluded_union_row_index_sha256": (
                self.excluded_union_row_index_sha256
            ),
        }


@dataclass(frozen=True)
class SampleOverlap:
    """Pairwise source-row overlap recorded by a staged sampling plan."""

    left_sample_role: str
    right_sample_role: str
    overlap_count: int

    def __post_init__(self) -> None:
        left = _nonempty_sample_role(self.left_sample_role, "left_sample_role")
        right = _nonempty_sample_role(self.right_sample_role, "right_sample_role")
        if left == right:
            raise ValueError("pairwise overlap roles must be distinct")
        if (
            isinstance(self.overlap_count, (bool, np.bool_))
            or not isinstance(self.overlap_count, Integral)
            or int(self.overlap_count) < 0
        ):
            raise ValueError("overlap_count must be a non-negative integer")
        object.__setattr__(self, "left_sample_role", left)
        object.__setattr__(self, "right_sample_role", right)
        object.__setattr__(self, "overlap_count", int(self.overlap_count))

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


@dataclass(frozen=True)
class StagedSamplePlan:
    """Index-only plan permitting development and final data to be read apart."""

    development: PinnedSourceSample
    reserved_accessed_samples: tuple[PinnedSourceSample, ...]
    final_evaluation: PinnedSourceSample
    pairwise_overlaps: tuple[SampleOverlap, ...]
    pre_final_excluded_sample_roles: tuple[str, ...]
    pre_final_excluded_union_row_count: int
    pre_final_excluded_union_row_index_sha256: str
    final_overlap_with_excluded_union_count: int
    all_pinned_union_row_count: int
    all_pinned_union_row_index_sha256: str

    def __post_init__(self) -> None:
        samples = self.all_samples
        for sample in samples:
            PinnedSourceSample.__post_init__(sample)
        roles = tuple(sample.sample_role for sample in samples)
        if roles[0] != "development" or roles[-1] != "final_evaluation":
            raise ValueError(
                "staged plans require development and final_evaluation boundary roles"
            )
        if len(roles) != len(set(roles)):
            raise ValueError("sample roles in a staged plan must be unique")
        source_identity = {
            (
                sample.dataset_version,
                sample.source_sha256,
                sample.source_row_count,
            )
            for sample in samples
        }
        if len(source_identity) != 1:
            raise ValueError("all staged sample pins must identify the same source")

        if samples[0].excluded_sample_roles or len(
            samples[0].excluded_union_row_indices
        ):
            raise ValueError("development pin must not carry an exclusion union")
        earlier_pins: dict[str, PinnedSourceSample] = {
            samples[0].sample_role: samples[0]
        }
        for sample in samples[1:]:
            unknown_exclusions = [
                role
                for role in sample.excluded_sample_roles
                if role not in earlier_pins
            ]
            if unknown_exclusions:
                raise ValueError(
                    f"sample {sample.sample_role!r} excludes roles that are not "
                    f"earlier pins: {unknown_exclusions}"
                )
            expected_exclusion_union = _union_row_indices(
                earlier_pins[role].selected_row_indices
                for role in sample.excluded_sample_roles
            )
            if not np.array_equal(
                sample.excluded_union_row_indices,
                expected_exclusion_union,
            ):
                raise ValueError(
                    f"sample {sample.sample_role!r} exclusion union does not match "
                    "its named earlier pins"
                )
            earlier_pins[sample.sample_role] = sample

        expected_pairs = len(samples) * (len(samples) - 1) // 2
        if len(self.pairwise_overlaps) != expected_pairs:
            raise ValueError("pairwise_overlaps does not cover every sample pair")
        observed_pairs: set[frozenset[str]] = set()
        pins_by_role = {sample.sample_role: sample for sample in samples}
        for overlap in self.pairwise_overlaps:
            pair = frozenset(
                (overlap.left_sample_role, overlap.right_sample_role)
            )
            if len(pair) != 2 or not pair <= pins_by_role.keys():
                raise ValueError("pairwise_overlaps contains an unknown sample pair")
            if pair in observed_pairs:
                raise ValueError("pairwise_overlaps contains a duplicate sample pair")
            observed_pairs.add(pair)
            left = pins_by_role[overlap.left_sample_role].selected_row_indices
            right = pins_by_role[overlap.right_sample_role].selected_row_indices
            actual_overlap = int(
                np.intersect1d(left, right, assume_unique=True).size
            )
            if overlap.overlap_count != actual_overlap:
                raise ValueError("recorded pairwise overlap does not match pinned rows")

        expected_pre_final_roles = roles[:-1]
        if tuple(self.pre_final_excluded_sample_roles) != expected_pre_final_roles:
            raise ValueError(
                "pre_final_excluded_sample_roles must list all pre-final pins in order"
            )
        pre_final_union = _union_row_indices(
            sample.selected_row_indices for sample in samples[:-1]
        )
        if int(self.pre_final_excluded_union_row_count) != len(pre_final_union):
            raise ValueError("pre-final exclusion-union count does not match pins")
        pre_final_digest = _row_index_sha256(pre_final_union)
        if (
            _normalized_sha256(
                self.pre_final_excluded_union_row_index_sha256,
                "pre_final_excluded_union_row_index_sha256",
            )
            != pre_final_digest
        ):
            raise ValueError("pre-final exclusion-union digest does not match pins")
        if samples[-1].excluded_sample_roles != expected_pre_final_roles:
            raise ValueError("final pin must exclude every pre-final sample role")
        if not np.array_equal(
            samples[-1].excluded_union_row_indices,
            pre_final_union,
        ):
            raise ValueError("final pin exclusion union does not match pre-final pins")
        final_overlap = int(
            np.intersect1d(
                samples[-1].selected_row_indices,
                pre_final_union,
                assume_unique=True,
            ).size
        )
        if int(self.final_overlap_with_excluded_union_count) != final_overlap:
            raise ValueError("final/exclusion-union overlap count does not match pins")
        if final_overlap:
            raise ValueError(
                "final-evaluation pin overlaps the pre-final exclusion union"
            )

        all_union = _union_row_indices(
            sample.selected_row_indices for sample in samples
        )
        if int(self.all_pinned_union_row_count) != len(all_union):
            raise ValueError("all-pinned union count does not match pins")
        all_union_digest = _row_index_sha256(all_union)
        if (
            _normalized_sha256(
                self.all_pinned_union_row_index_sha256,
                "all_pinned_union_row_index_sha256",
            )
            != all_union_digest
        ):
            raise ValueError("all-pinned union digest does not match pins")

    @property
    def all_samples(self) -> tuple[PinnedSourceSample, ...]:
        return (
            self.development,
            *self.reserved_accessed_samples,
            self.final_evaluation,
        )

    def pin_for(self, sample_role: str) -> PinnedSourceSample:
        role = _nonempty_sample_role(sample_role)
        matches = [pin for pin in self.all_samples if pin.sample_role == role]
        if len(matches) != 1:
            raise KeyError(f"sampling plan has no unique pin for role {role!r}")
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "development": self.development.to_dict(),
            "reserved_accessed_samples": [
                sample.to_dict() for sample in self.reserved_accessed_samples
            ],
            "final_evaluation": self.final_evaluation.to_dict(),
            "pairwise_overlaps": [
                overlap.to_dict() for overlap in self.pairwise_overlaps
            ],
            "pre_final_excluded_sample_roles": list(
                self.pre_final_excluded_sample_roles
            ),
            "pre_final_excluded_union_row_count": (
                self.pre_final_excluded_union_row_count
            ),
            "pre_final_excluded_union_row_index_sha256": (
                self.pre_final_excluded_union_row_index_sha256
            ),
            "final_overlap_with_excluded_union_count": (
                self.final_overlap_with_excluded_union_count
            ),
            "all_pinned_union_row_count": self.all_pinned_union_row_count,
            "all_pinned_union_row_index_sha256": (
                self.all_pinned_union_row_index_sha256
            ),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_DOWNLOAD_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def download_criteo(destination: str | Path) -> Path:
    """Download the corrected Criteo v2.1 gzip and verify its published hash.

    Bytes are streamed into a sibling ``.part`` file. The completed file only
    becomes visible at ``destination`` after its SHA256 has been verified.
    """

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        actual_hash = _sha256(destination)
        if actual_hash != CRITEO_V21_SHA256:
            raise ValueError(
                f"Existing dataset failed SHA256 verification: {destination} "
                f"(expected {CRITEO_V21_SHA256}, got {actual_hash})"
            )
        return destination

    part_path = destination.with_name(f"{destination.name}.part")
    request = Request(CRITEO_V21_URL, headers={"User-Agent": "causal-ad-targeting/1.0"})
    digest = hashlib.sha256()

    try:
        with urlopen(request, timeout=60) as response, part_path.open("wb") as output:
            for block in iter(lambda: response.read(_DOWNLOAD_BLOCK_BYTES), b""):
                output.write(block)
                digest.update(block)

        actual_hash = digest.hexdigest()
        if actual_hash != CRITEO_V21_SHA256:
            raise ValueError(
                "Downloaded dataset failed SHA256 verification "
                f"(expected {CRITEO_V21_SHA256}, got {actual_hash})"
            )

        os.replace(part_path, destination)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise

    return destination


def _open_text(path: Path) -> IO[str]:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return path.open(mode="rt", encoding="utf-8", newline="")


def _open_binary(path: Path) -> IO[bytes]:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode="rb")
    return path.open(mode="rb")


def _validate_header(path: Path) -> None:
    with _open_text(path) as stream:
        header = next(csv.reader(stream), None)

    if header is None:
        raise ValueError(f"Dataset is empty: {path}")
    if len(header) != len(set(header)):
        raise ValueError(f"Dataset header contains duplicate columns: {header}")
    if tuple(header) != EXPECTED_COLUMNS:
        missing = sorted(set(EXPECTED_COLUMNS) - set(header))
        unexpected = sorted(set(header) - set(EXPECTED_COLUMNS))
        raise ValueError(
            "Unexpected dataset header. "
            f"Expected {list(EXPECTED_COLUMNS)}; missing={missing}; "
            f"unexpected={unexpected}; actual={header}"
        )


def _count_rows(path: Path) -> int:
    newline_count = 0
    last_byte = b""
    with _open_binary(path) as stream:
        for block in iter(lambda: stream.read(_COUNT_BLOCK_BYTES), b""):
            newline_count += block.count(b"\n")
            last_byte = block[-1:]

    line_count = newline_count + int(bool(last_byte) and last_byte != b"\n")
    return max(0, line_count - 1)  # exclude the header


def _validated_source_provenance(
    path: Path, verify_official_hash: bool
) -> tuple[str, str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    _validate_header(path)
    actual_hash = _sha256(path)
    if path.suffix.lower() == ".gz" and verify_official_hash:
        if actual_hash != CRITEO_V21_SHA256:
            raise ValueError(
                "Dataset failed corrected Criteo v2.1 SHA256 verification: "
                f"{path} (expected {CRITEO_V21_SHA256}, got {actual_hash})"
            )
        hash_verification = "official_v2.1_sha256_match"
    elif actual_hash == CRITEO_V21_SHA256:
        hash_verification = "official_v2.1_sha256_match"
    else:
        hash_verification = "schema_only_nonofficial_hash"

    total_rows = _count_rows(path)
    if total_rows == 0:
        raise ValueError(f"Dataset contains no data rows: {path}")
    return actual_hash, hash_verification, total_rows


def _validate_outcome(outcome: str) -> None:
    if outcome not in OUTCOME_COLUMNS:
        raise ValueError(f"outcome must be one of {OUTCOME_COLUMNS}, got {outcome!r}")


def _sample_row_indices(total_rows: int, sample_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(total_rows, size=sample_size, replace=False, shuffle=False)
    )


def _sample_complement_row_indices(
    total_rows: int,
    excluded_rows: np.ndarray,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    """Sample source indices uniformly from the complement of excluded rows."""

    excluded = np.asarray(excluded_rows, dtype=np.int64)
    if excluded.ndim != 1:
        raise ValueError("excluded_rows must be one-dimensional")
    if excluded.size and (
        excluded[0] < 0
        or excluded[-1] >= total_rows
        or np.any(np.diff(excluded) <= 0)
    ):
        raise ValueError(
            "excluded_rows must be sorted, unique, and within source bounds"
        )

    available_rows = total_rows - len(excluded)
    if sample_size <= 0 or sample_size > available_rows:
        raise ValueError(
            f"sample_size must be in [1, {available_rows:,}] for the source complement"
        )

    # Sample ranks in the compact complement, then expand those ranks back to
    # source indices. For excluded source index e[j], e[j] - j is its insertion
    # position in complement-rank space. This avoids materializing an O(n) array
    # containing every source index.
    complement_ranks = _sample_row_indices(available_rows, sample_size, seed)
    excluded_rank_positions = excluded - np.arange(len(excluded), dtype=np.int64)
    selected = complement_ranks + np.searchsorted(
        excluded_rank_positions, complement_ranks, side="right"
    )
    selected = selected.astype(np.int64, copy=False)
    if np.intersect1d(selected, excluded, assume_unique=True).size:
        raise AssertionError("complement sampler selected an excluded source row")
    return selected


def _row_index_sha256(selected_rows: np.ndarray) -> str:
    canonical = np.asarray(selected_rows, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _canonical_row_indices(
    rows: np.ndarray,
    *,
    total_rows: int,
    name: str,
    allow_empty: bool,
) -> np.ndarray:
    indices = np.asarray(rows, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not allow_empty and indices.size == 0:
        raise ValueError(f"{name} must not be empty")
    if indices.size and (
        indices[0] < 0
        or indices[-1] >= total_rows
        or np.any(np.diff(indices) <= 0)
    ):
        raise ValueError(f"{name} must be sorted, unique, and within source bounds")
    return indices


def _row_index_bytes(rows: np.ndarray) -> bytes:
    return np.asarray(rows, dtype="<i8").tobytes(order="C")


def _union_row_indices(row_sets: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(rows, dtype=np.int64) for rows in row_sets]
    if not arrays:
        return np.empty(0, dtype=np.int64)
    if len(arrays) == 1:
        return arrays[0].copy()
    return np.unique(np.concatenate(arrays))


def _sampling_method_for_exclusions(excluded_roles: tuple[str, ...]) -> str:
    if not excluded_roles:
        return SAMPLING_METHOD
    if excluded_roles == ("development",):
        return COMPLEMENT_SAMPLING_METHOD
    return UNION_COMPLEMENT_SAMPLING_METHOD


def _build_pinned_source_sample(
    *,
    sample_role: str,
    path: Path,
    actual_hash: str,
    hash_verification: str,
    total_rows: int,
    selected_rows: np.ndarray,
    sampling_method: str,
    sampling_seed: int,
    excluded_sample_roles: tuple[str, ...],
    excluded_union_rows: np.ndarray,
) -> PinnedSourceSample:
    selected = _canonical_row_indices(
        selected_rows,
        total_rows=total_rows,
        name="selected_rows",
        allow_empty=False,
    )
    excluded = _canonical_row_indices(
        excluded_union_rows,
        total_rows=total_rows,
        name="excluded_union_rows",
        allow_empty=True,
    )
    overlap = np.intersect1d(selected, excluded, assume_unique=True)
    if overlap.size:
        raise AssertionError(
            f"sample role {sample_role!r} overlaps its exclusion union by "
            f"{overlap.size} rows"
        )
    return PinnedSourceSample(
        sample_role=sample_role,
        dataset_version=CRITEO_DATASET_VERSION,
        source_path=str(path),
        source_sha256=actual_hash,
        source_sha256_verification=hash_verification,
        source_row_count=total_rows,
        sample_size=len(selected),
        sampling_method=sampling_method,
        sampling_seed=sampling_seed,
        selected_row_index_sha256=_row_index_sha256(selected),
        selected_min_source_row=int(selected[0]),
        selected_max_source_row=int(selected[-1]),
        excluded_sample_roles=excluded_sample_roles,
        excluded_union_row_count=len(excluded),
        excluded_union_row_index_sha256=_row_index_sha256(excluded),
        _selected_row_indices_le_i8=_row_index_bytes(selected),
        _excluded_union_row_indices_le_i8=_row_index_bytes(excluded),
    )


def plan_staged_criteo_samples(
    path: str | Path,
    *,
    development_sample_size: int,
    final_evaluation_sample_size: int,
    development_seed: int = 42,
    final_evaluation_seed: int = 43,
    reserved_accessed_samples: Sequence[ReservedSampleSpec] = (),
    expected_development_row_index_sha256: str | None = None,
    verify_official_hash: bool = True,
) -> StagedSamplePlan:
    """Pin index-only samples for a strict development/final access boundary.

    Planning validates only source identity/schema/count and performs deterministic
    index arithmetic; it does not call ``pandas.read_csv`` or return row contents.
    Each reserved specification reconstructs a previously accessed selection and
    verifies its recorded digest. The fresh final sample is drawn uniformly from
    the complement of the union of development and all accepted reserved pins.
    Call :func:`load_pinned_criteo_sample_with_metadata` separately for each stage.
    """

    path = Path(path)
    actual_hash, hash_verification, total_rows = _validated_source_provenance(
        path, verify_official_hash
    )
    development_size = _positive_integer(
        development_sample_size, "development_sample_size"
    )
    final_size = _positive_integer(
        final_evaluation_sample_size, "final_evaluation_sample_size"
    )
    development_sampling_seed = _nonnegative_seed(
        development_seed, "development_seed"
    )
    final_sampling_seed = _nonnegative_seed(
        final_evaluation_seed, "final_evaluation_seed"
    )
    if development_size > total_rows:
        raise ValueError(
            f"development_sample_size={development_size:,} exceeds dataset size "
            f"{total_rows:,}"
        )

    development_rows = _sample_row_indices(
        total_rows, development_size, development_sampling_seed
    )
    development_digest = _row_index_sha256(development_rows)
    if expected_development_row_index_sha256 is not None:
        expected_development_digest = _normalized_sha256(
            expected_development_row_index_sha256,
            "expected_development_row_index_sha256",
        )
        if development_digest != expected_development_digest:
            raise ValueError(
                "reconstructed development source-row digest does not match the "
                "recorded digest"
            )

    empty_exclusion = np.empty(0, dtype=np.int64)
    development_pin = _build_pinned_source_sample(
        sample_role="development",
        path=path,
        actual_hash=actual_hash,
        hash_verification=hash_verification,
        total_rows=total_rows,
        selected_rows=development_rows,
        sampling_method=SAMPLING_METHOD,
        sampling_seed=development_sampling_seed,
        excluded_sample_roles=(),
        excluded_union_rows=empty_exclusion,
    )
    pins_by_role: dict[str, PinnedSourceSample] = {
        development_pin.sample_role: development_pin
    }

    try:
        reserved_specs = tuple(reserved_accessed_samples)
    except TypeError as exc:
        raise ValueError(
            "reserved_accessed_samples must be a sequence of ReservedSampleSpec"
        ) from exc
    reserved_pins: list[PinnedSourceSample] = []
    for position, spec in enumerate(reserved_specs):
        if not isinstance(spec, ReservedSampleSpec):
            raise ValueError(
                "reserved_accessed_samples must contain only ReservedSampleSpec "
                f"instances; item {position} is {type(spec).__name__}"
            )
        role = spec.sample_role
        if role in {"development", "final_evaluation"} or role in pins_by_role:
            raise ValueError(f"sample role {role!r} is duplicated or reserved")
        unknown_exclusions = [
            excluded_role
            for excluded_role in spec.excluded_sample_roles
            if excluded_role not in pins_by_role
        ]
        if unknown_exclusions:
            raise ValueError(
                f"reserved sample {role!r} refers to unpinned earlier roles: "
                f"{unknown_exclusions}"
            )
        excluded_union = _union_row_indices(
            pins_by_role[excluded_role].selected_row_indices
            for excluded_role in spec.excluded_sample_roles
        )
        available_rows = total_rows - len(excluded_union)
        if spec.sample_size > available_rows:
            raise ValueError(
                f"reserved sample {role!r} requests {spec.sample_size:,} rows but "
                f"only {available_rows:,} remain outside its exclusion union"
            )
        if excluded_union.size:
            reserved_rows = _sample_complement_row_indices(
                total_rows,
                excluded_union,
                spec.sample_size,
                spec.sampling_seed,
            )
        else:
            reserved_rows = _sample_row_indices(
                total_rows, spec.sample_size, spec.sampling_seed
            )
        reconstructed_digest = _row_index_sha256(reserved_rows)
        if reconstructed_digest != spec.expected_selected_row_index_sha256:
            raise ValueError(
                f"reconstructed reserved sample {role!r} has source-row digest "
                f"{reconstructed_digest}, not recorded digest "
                f"{spec.expected_selected_row_index_sha256}"
            )
        reserved_pin = _build_pinned_source_sample(
            sample_role=role,
            path=path,
            actual_hash=actual_hash,
            hash_verification=hash_verification,
            total_rows=total_rows,
            selected_rows=reserved_rows,
            sampling_method=_sampling_method_for_exclusions(
                spec.excluded_sample_roles
            ),
            sampling_seed=spec.sampling_seed,
            excluded_sample_roles=spec.excluded_sample_roles,
            excluded_union_rows=excluded_union,
        )
        pins_by_role[role] = reserved_pin
        reserved_pins.append(reserved_pin)

    pre_final_roles = tuple(pins_by_role)
    pre_final_union = _union_row_indices(
        pin.selected_row_indices for pin in pins_by_role.values()
    )
    final_available_rows = total_rows - len(pre_final_union)
    if final_size > final_available_rows:
        raise ValueError(
            f"final_evaluation_sample_size={final_size:,} exceeds the "
            f"{final_available_rows:,} rows outside all pre-final pins"
        )
    final_rows = _sample_complement_row_indices(
        total_rows,
        pre_final_union,
        final_size,
        final_sampling_seed,
    )
    final_pin = _build_pinned_source_sample(
        sample_role="final_evaluation",
        path=path,
        actual_hash=actual_hash,
        hash_verification=hash_verification,
        total_rows=total_rows,
        selected_rows=final_rows,
        sampling_method=_sampling_method_for_exclusions(pre_final_roles),
        sampling_seed=final_sampling_seed,
        excluded_sample_roles=pre_final_roles,
        excluded_union_rows=pre_final_union,
    )

    all_samples = (development_pin, *reserved_pins, final_pin)
    pairwise_overlaps: list[SampleOverlap] = []
    for left_position, left in enumerate(all_samples):
        for right in all_samples[left_position + 1 :]:
            pairwise_overlaps.append(
                SampleOverlap(
                    left_sample_role=left.sample_role,
                    right_sample_role=right.sample_role,
                    overlap_count=int(
                        np.intersect1d(
                            left.selected_row_indices,
                            right.selected_row_indices,
                            assume_unique=True,
                        ).size
                    ),
                )
            )
    final_union_overlap = int(
        np.intersect1d(
            final_rows,
            pre_final_union,
            assume_unique=True,
        ).size
    )
    if final_union_overlap:
        raise AssertionError(
            "final-evaluation selection overlaps the union of pre-final pins"
        )
    all_pinned_union = _union_row_indices(
        sample.selected_row_indices for sample in all_samples
    )
    return StagedSamplePlan(
        development=development_pin,
        reserved_accessed_samples=tuple(reserved_pins),
        final_evaluation=final_pin,
        pairwise_overlaps=tuple(pairwise_overlaps),
        pre_final_excluded_sample_roles=pre_final_roles,
        pre_final_excluded_union_row_count=len(pre_final_union),
        pre_final_excluded_union_row_index_sha256=_row_index_sha256(
            pre_final_union
        ),
        final_overlap_with_excluded_union_count=final_union_overlap,
        all_pinned_union_row_count=len(all_pinned_union),
        all_pinned_union_row_index_sha256=_row_index_sha256(all_pinned_union),
    )


def _read_sample(
    path: Path,
    columns: list[str],
    dtypes: dict[str, str],
    selected_rows: np.ndarray,
) -> pd.DataFrame:
    selected_chunks: list[pd.DataFrame] = []
    row_offset = 0
    for chunk in pd.read_csv(
        path,
        usecols=columns,
        dtype=dtypes,
        chunksize=_CSV_CHUNK_ROWS,
    ):
        chunk_end = row_offset + len(chunk)
        left = np.searchsorted(selected_rows, row_offset, side="left")
        right = np.searchsorted(selected_rows, chunk_end, side="left")
        if right > left:
            local_rows = selected_rows[left:right] - row_offset
            selected_chunks.append(chunk.iloc[local_rows])
        row_offset = chunk_end

    if not selected_chunks:
        return pd.DataFrame(columns=columns).astype(dtypes)
    return pd.concat(selected_chunks, ignore_index=True)


def _validate_loaded_data(frame: pd.DataFrame, outcome: str) -> None:
    feature_values = frame.loc[:, FEATURE_COLUMNS].to_numpy(copy=False)
    if not np.isfinite(feature_values).all():
        raise ValueError("Feature columns contain missing or non-finite values")

    for column in (TREATMENT_COLUMN, outcome):
        values = frame[column].unique()
        if not np.isin(values, (0, 1)).all():
            raise ValueError(f"{column!r} must be binary 0/1; observed {values.tolist()}")


def _model_columns_and_dtypes(outcome: str) -> tuple[list[str], dict[str, str]]:
    columns = [*FEATURE_COLUMNS, TREATMENT_COLUMN, outcome]
    dtypes = {
        **{column: "float32" for column in CONTINUOUS_FEATURE_COLUMNS},
        # The categorical modalities are serialized as floating-point values.
        # Keep float64 precision so distinct anonymized category IDs cannot be
        # merged before sparse one-hot encoding.
        **{column: "float64" for column in CATEGORICAL_FEATURE_COLUMNS},
        TREATMENT_COLUMN: "uint8",
        outcome: "uint8",
    }
    return columns, dtypes


def _load_metadata(
    *,
    path: Path,
    actual_hash: str,
    hash_verification: str,
    total_rows: int,
    loaded_rows: int,
    sampling_method: str,
    sampling_seed: int | None,
    selected_rows: np.ndarray | None,
) -> DataLoadMetadata:
    return DataLoadMetadata(
        dataset_version=CRITEO_DATASET_VERSION,
        source_path=str(path),
        source_sha256=actual_hash,
        source_sha256_verification=hash_verification,
        source_row_count=total_rows,
        requested_sample_size=loaded_rows,
        loaded_sample_size=loaded_rows,
        sampling_method=sampling_method,
        sampling_seed=sampling_seed,
        selected_row_index_sha256=(
            _row_index_sha256(selected_rows) if selected_rows is not None else None
        ),
        selected_min_source_row=(
            int(selected_rows[0]) if selected_rows is not None else 0
        ),
        selected_max_source_row=(
            int(selected_rows[-1]) if selected_rows is not None else total_rows - 1
        ),
    )


def _reverify_pinned_source_sample(pin: PinnedSourceSample) -> np.ndarray:
    if not isinstance(pin, PinnedSourceSample):
        raise TypeError("pin must be a PinnedSourceSample")

    # A frozen dataclass prevents ordinary mutation, but re-run every receipt and
    # constraint check at the access boundary as a defense against deserialization
    # bugs or callers that bypassed normal attribute assignment.
    PinnedSourceSample.__post_init__(pin)
    selected_rows = pin.selected_row_indices
    if len(selected_rows) != pin.sample_size:
        raise ValueError("pinned index count does not match sample_size")
    if _row_index_sha256(selected_rows) != pin.selected_row_index_sha256:
        raise ValueError("pinned source-row index digest verification failed")
    excluded_rows = pin.excluded_union_row_indices
    if len(excluded_rows) != pin.excluded_union_row_count:
        raise ValueError("pinned exclusion-union count verification failed")
    if _row_index_sha256(excluded_rows) != pin.excluded_union_row_index_sha256:
        raise ValueError("pinned exclusion-union digest verification failed")
    if np.intersect1d(selected_rows, excluded_rows, assume_unique=True).size:
        raise ValueError("pinned rows violate their exclusion-union constraint")
    return selected_rows


def load_pinned_criteo_sample_with_metadata(
    path: str | Path,
    pin: PinnedSourceSample,
    outcome: str = "conversion",
    *,
    verify_official_hash: bool = True,
) -> LoadedData:
    """Load exactly one verified pin, leaving every other staged pin unopened.

    The source header, content hash, and row count are revalidated at each stage.
    The selected and exclusion-union index bytes are independently checked for
    digest, count, ordering, bounds, uniqueness, and disjointness before one
    chunked source scan materializes only the requested sample in the result.
    """

    _validate_outcome(outcome)
    selected_rows = _reverify_pinned_source_sample(pin)
    path = Path(path)
    actual_hash, hash_verification, total_rows = _validated_source_provenance(
        path, verify_official_hash
    )
    if pin.dataset_version != CRITEO_DATASET_VERSION:
        raise ValueError(
            f"pin dataset version {pin.dataset_version!r} does not match "
            f"{CRITEO_DATASET_VERSION!r}"
        )
    if actual_hash != pin.source_sha256:
        raise ValueError(
            "source SHA-256 does not match the pinned source: "
            f"expected {pin.source_sha256}, got {actual_hash}"
        )
    if total_rows != pin.source_row_count:
        raise ValueError(
            "source row count does not match the pinned source: "
            f"expected {pin.source_row_count:,}, got {total_rows:,}"
        )
    if hash_verification != pin.source_sha256_verification:
        raise ValueError(
            "source verification status does not match the pinned source: "
            f"expected {pin.source_sha256_verification!r}, got "
            f"{hash_verification!r}"
        )

    columns, dtypes = _model_columns_and_dtypes(outcome)
    frame = _read_sample(
        path=path,
        columns=columns,
        dtypes=dtypes,
        selected_rows=selected_rows,
    ).loc[:, columns]
    if len(frame) != pin.sample_size:
        raise RuntimeError(
            f"Expected {pin.sample_size:,} pinned rows, loaded {len(frame):,}"
        )
    _validate_loaded_data(frame, outcome)
    metadata = _load_metadata(
        path=path,
        actual_hash=actual_hash,
        hash_verification=hash_verification,
        total_rows=total_rows,
        loaded_rows=len(frame),
        sampling_method=pin.sampling_method,
        sampling_seed=pin.sampling_seed,
        selected_rows=selected_rows,
    )
    if metadata.selected_row_index_sha256 != pin.selected_row_index_sha256:
        raise AssertionError("loaded sample metadata does not match the pinned digest")
    if metadata.selected_min_source_row != pin.selected_min_source_row:
        raise AssertionError("loaded sample minimum source row does not match its pin")
    if metadata.selected_max_source_row != pin.selected_max_source_row:
        raise AssertionError("loaded sample maximum source row does not match its pin")
    return LoadedData(frame=frame, metadata=metadata)


def load_criteo_with_metadata(
    path: str | Path,
    outcome: str = "conversion",
    sample_size: int | None = None,
    seed: int = 42,
    *,
    verify_official_hash: bool = True,
) -> LoadedData:
    """Load an exact seeded sample using compact dtypes.

    The returned frame contains only ``f0`` through ``f11``, ``treatment``,
    and the selected outcome. ``exposure`` is validated as part of the source
    schema but is intentionally never returned as a candidate model feature.
    Sampling is uniform without replacement and reads the source in chunks,
    which avoids loading all 14 million rows before selecting the sample.
    """

    _validate_outcome(outcome)
    path = Path(path)
    actual_hash, hash_verification, total_rows = _validated_source_provenance(
        path, verify_official_hash
    )

    if sample_size is None:
        requested_rows = total_rows
    else:
        if sample_size <= 0:
            raise ValueError("sample_size must be a positive integer or None")
        if sample_size > total_rows:
            raise ValueError(
                f"sample_size={sample_size:,} exceeds dataset size {total_rows:,}"
            )
        requested_rows = sample_size

    columns, dtypes = _model_columns_and_dtypes(outcome)

    selected_rows: np.ndarray | None = None
    if requested_rows == total_rows:
        frame = pd.read_csv(path, usecols=columns, dtype=dtypes)
    else:
        selected_rows = _sample_row_indices(total_rows, requested_rows, seed)
        frame = _read_sample(
            path=path,
            columns=columns,
            dtypes=dtypes,
            selected_rows=selected_rows,
        )

    frame = frame.loc[:, columns]
    if len(frame) != requested_rows:
        raise RuntimeError(f"Expected {requested_rows:,} rows, loaded {len(frame):,}")
    _validate_loaded_data(frame, outcome)
    metadata = _load_metadata(
        path=path,
        actual_hash=actual_hash,
        hash_verification=hash_verification,
        total_rows=total_rows,
        loaded_rows=len(frame),
        sampling_method=(SAMPLING_METHOD if selected_rows is not None else "all_source_rows"),
        sampling_seed=(seed if selected_rows is not None else None),
        selected_rows=selected_rows,
    )
    return LoadedData(frame=frame, metadata=metadata)


def load_disjoint_criteo_samples_with_metadata(
    path: str | Path,
    outcome: str = "conversion",
    *,
    development_sample_size: int,
    final_evaluation_sample_size: int,
    development_seed: int = 42,
    final_evaluation_seed: int = 43,
    verify_official_hash: bool = True,
) -> LoadedDisjointSamples:
    """Load deterministic development and fresh final-evaluation samples.

    The development rows are sampled uniformly without replacement from the
    full source. Final-evaluation rows are then sampled uniformly without
    replacement from the complement of those development source-row indices.
    Both samples are retrieved in one chunked ``read_csv`` pass and carry
    separate sizes, seeds, methods, and source-index digests in their metadata.
    """

    _validate_outcome(outcome)
    path = Path(path)
    actual_hash, hash_verification, total_rows = _validated_source_provenance(
        path, verify_official_hash
    )

    for name, value in (
        ("development_sample_size", development_sample_size),
        ("final_evaluation_sample_size", final_evaluation_sample_size),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    development_rows_requested = int(development_sample_size)
    final_rows_requested = int(final_evaluation_sample_size)
    if development_rows_requested + final_rows_requested > total_rows:
        raise ValueError(
            "development_sample_size + final_evaluation_sample_size exceeds "
            f"dataset size {total_rows:,}"
        )

    development_rows = _sample_row_indices(
        total_rows, development_rows_requested, development_seed
    )
    final_evaluation_rows = _sample_complement_row_indices(
        total_rows,
        development_rows,
        final_rows_requested,
        final_evaluation_seed,
    )
    overlap = np.intersect1d(
        development_rows, final_evaluation_rows, assume_unique=True
    )
    if overlap.size:
        raise AssertionError(
            f"development/final source selections overlap by {overlap.size} rows"
        )

    combined_rows = np.sort(
        np.concatenate((development_rows, final_evaluation_rows))
    )
    if len(combined_rows) != development_rows_requested + final_rows_requested:
        raise AssertionError("combined source selections contain duplicate rows")

    columns, dtypes = _model_columns_and_dtypes(outcome)
    combined_frame = _read_sample(
        path=path,
        columns=columns,
        dtypes=dtypes,
        selected_rows=combined_rows,
    ).loc[:, columns]
    if len(combined_frame) != len(combined_rows):
        raise RuntimeError(
            f"Expected {len(combined_rows):,} rows, loaded {len(combined_frame):,}"
        )

    development_mask = np.isin(
        combined_rows, development_rows, assume_unique=True
    )
    development = combined_frame.iloc[np.flatnonzero(development_mask)].reset_index(
        drop=True
    )
    final_evaluation = combined_frame.iloc[
        np.flatnonzero(~development_mask)
    ].reset_index(drop=True)
    if len(development) != development_rows_requested:
        raise RuntimeError(
            f"Expected {development_rows_requested:,} development rows, "
            f"loaded {len(development):,}"
        )
    if len(final_evaluation) != final_rows_requested:
        raise RuntimeError(
            f"Expected {final_rows_requested:,} final-evaluation rows, "
            f"loaded {len(final_evaluation):,}"
        )
    _validate_loaded_data(development, outcome)
    _validate_loaded_data(final_evaluation, outcome)

    development_metadata = _load_metadata(
        path=path,
        actual_hash=actual_hash,
        hash_verification=hash_verification,
        total_rows=total_rows,
        loaded_rows=len(development),
        sampling_method=SAMPLING_METHOD,
        sampling_seed=development_seed,
        selected_rows=development_rows,
    )
    final_evaluation_metadata = _load_metadata(
        path=path,
        actual_hash=actual_hash,
        hash_verification=hash_verification,
        total_rows=total_rows,
        loaded_rows=len(final_evaluation),
        sampling_method=COMPLEMENT_SAMPLING_METHOD,
        sampling_seed=final_evaluation_seed,
        selected_rows=final_evaluation_rows,
    )
    metadata = DisjointSampleMetadata(
        development=development_metadata,
        final_evaluation=final_evaluation_metadata,
        source_row_overlap_count=int(overlap.size),
    )
    if not metadata.source_row_overlap_verified:
        raise AssertionError("development/final source-row overlap was not eliminated")
    return LoadedDisjointSamples(
        development=development,
        final_evaluation=final_evaluation,
        metadata=metadata,
    )


def load_criteo(
    path: str | Path,
    outcome: str = "conversion",
    sample_size: int | None = None,
    seed: int = 42,
    *,
    verify_official_hash: bool = True,
) -> pd.DataFrame:
    """Backward-compatible frame-only wrapper around the provenance loader."""

    return load_criteo_with_metadata(
        path,
        outcome=outcome,
        sample_size=sample_size,
        seed=seed,
        verify_official_hash=verify_official_hash,
    ).frame


def split_data(
    frame: pd.DataFrame,
    outcome: str = "conversion",
    seed: int = 42,
) -> DataSplits:
    """Create 60/20/20 splits stratified by joint treatment/outcome status."""

    _validate_outcome(outcome)
    required = {*FEATURE_COLUMNS, TREATMENT_COLUMN, outcome}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Cannot split data; missing columns: {missing}")

    joint_strata = (
        frame[TREATMENT_COLUMN].to_numpy(dtype=np.uint8) * 2
        + frame[outcome].to_numpy(dtype=np.uint8)
    )
    train, holdout = train_test_split(
        frame,
        test_size=0.40,
        random_state=seed,
        shuffle=True,
        stratify=joint_strata,
    )

    holdout_strata = (
        holdout[TREATMENT_COLUMN].to_numpy(dtype=np.uint8) * 2
        + holdout[outcome].to_numpy(dtype=np.uint8)
    )
    validation, test = train_test_split(
        holdout,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
        stratify=holdout_strata,
    )

    return DataSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def split_development_data(
    frame: pd.DataFrame,
    outcome: str = "conversion",
    seed: int = 42,
    validation_fraction: float = 0.20,
) -> DevelopmentSplits:
    """Create train/validation development splits for pre-test decisions.

    The fresh final-evaluation sample is loaded separately and must never be
    passed to this function. Stratifying jointly by treatment and outcome keeps
    both rare-outcome treatment cells represented in the validation data used
    to freeze model, policy, and budget decisions.
    """

    _validate_outcome(outcome)
    required = {*FEATURE_COLUMNS, TREATMENT_COLUMN, outcome}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Cannot split development data; missing columns: {missing}")
    fraction = float(validation_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")

    joint_strata = (
        frame[TREATMENT_COLUMN].to_numpy(dtype=np.uint8) * 2
        + frame[outcome].to_numpy(dtype=np.uint8)
    )
    train, validation = train_test_split(
        frame,
        test_size=fraction,
        random_state=seed,
        shuffle=True,
        stratify=joint_strata,
    )
    return DevelopmentSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
    )


def summarize_data(
    frame: pd.DataFrame,
    outcome: str = "conversion",
) -> dict[str, int | float]:
    """Return the minimal randomized-experiment descriptive statistics."""

    _validate_outcome(outcome)
    missing = {TREATMENT_COLUMN, outcome} - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot summarize data; missing columns: {sorted(missing)}")

    treated = frame[TREATMENT_COLUMN] == 1
    treatment_count = int(treated.sum())
    control_count = int((~treated).sum())
    if treatment_count == 0 or control_count == 0:
        raise ValueError("Both treatment and control rows are required")

    treated_rate = float(frame.loc[treated, outcome].mean())
    control_rate = float(frame.loc[~treated, outcome].mean())
    return {
        "sample_size": int(len(frame)),
        "treatment_count": treatment_count,
        "control_count": control_count,
        "treatment_ratio": float(treatment_count / len(frame)),
        "outcome_rate_treated": treated_rate,
        "outcome_rate_control": control_rate,
        "naive_ate": treated_rate - control_rate,
        "positive_outcomes": int(frame[outcome].sum()),
    }
