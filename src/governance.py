"""Fail-closed governance for one-shot final evaluations.

The state transition is deliberately one way::

    preflight -> decision freeze -> final-access claim -> completion receipt

Each record is canonical JSON created with exclusive ``x`` mode and fsynced.
Existing, partial, malformed, or tampered records are never overwritten or
cleaned up automatically.  A failed claim therefore remains a replay blocker
until a human handles the incident outside this module.

No timestamp is generated.  A caller may place one in ``caller_metadata`` when
external time provenance exists; if supplied, it becomes part of the hashed
record semantics.

Threat model: roots and parent directories are expected to live in a trusted,
quiescent workspace while a governance transition runs.  Leaf symlinks and
non-regular files fail closed, but this module does not attempt to defeat a
concurrent local writer that can rename or replace parent directories.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
MAX_GOVERNANCE_RECORD_BYTES = 8 * 1024 * 1024
MAX_HASHED_FILE_BYTES = 8 * 1024 * 1024 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024
_EVALUATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_ARTIFACT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class GovernanceError(RuntimeError):
    """Base class for governance failures."""


class GovernanceConflictError(GovernanceError):
    """A one-shot path already exists and must not be overwritten."""


class GovernanceIntegrityError(GovernanceError):
    """A hash, canonical record, manifest, or chain binding is invalid."""


@dataclass(frozen=True)
class EvaluationPaths:
    """Unique immutable-record paths for one evaluation identifier."""

    metrics_dir: Path
    evaluation_id: str
    freeze_path: Path
    claim_path: Path
    receipt_path: Path

    @property
    def immutable_paths(self) -> tuple[Path, Path, Path]:
        return (self.freeze_path, self.claim_path, self.receipt_path)


@dataclass(frozen=True)
class ImmutableRecord:
    """Path and exact persisted-byte digest returned after exclusive creation."""

    path: Path
    sha256: str


@dataclass(frozen=True)
class CodeManifest:
    """Deterministically ordered hashes for an explicit set of code files."""

    files: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.files:
            raise ValueError("code manifest must contain at least one file")
        normalized: list[tuple[str, str]] = []
        seen: set[str] = set()
        for path, digest in self.files:
            safe_path = _validate_relative_posix_path(path, "code manifest path")
            safe_digest = validate_sha256(digest, "code manifest sha256")
            if safe_path in seen:
                raise ValueError(f"duplicate code manifest path: {safe_path}")
            seen.add(safe_path)
            normalized.append((safe_path, safe_digest))
        ordered = tuple(sorted(normalized))
        if tuple(normalized) != ordered:
            raise ValueError("code manifest files must be sorted by relative path")
        object.__setattr__(self, "files", ordered)

    def to_record(self) -> dict[str, Any]:
        return {
            "algorithm": "sha256",
            "files": [
                {"path": path, "sha256": digest} for path, digest in self.files
            ],
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.to_record())

    @classmethod
    def from_record(cls, value: object) -> CodeManifest:
        if not isinstance(value, dict) or set(value) != {"algorithm", "files"}:
            raise GovernanceIntegrityError("code manifest has an invalid schema")
        if value["algorithm"] != "sha256" or not isinstance(value["files"], list):
            raise GovernanceIntegrityError("code manifest has an invalid algorithm")
        files: list[tuple[str, str]] = []
        for entry in value["files"]:
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                raise GovernanceIntegrityError("code manifest entry has an invalid schema")
            if not isinstance(entry["path"], str) or not isinstance(
                entry["sha256"], str
            ):
                raise GovernanceIntegrityError(
                    "code manifest path and sha256 must be strings"
                )
            files.append((entry["path"], entry["sha256"]))
        try:
            return cls(tuple(files))
        except ValueError as exc:
            raise GovernanceIntegrityError(str(exc)) from exc


def sanitize_evaluation_id(value: object) -> str:
    """Validate an identifier without lossy normalization or path traversal."""

    if not isinstance(value, str) or _EVALUATION_ID.fullmatch(value) is None:
        raise ValueError(
            "evaluation_id must be 1-128 ASCII letters, digits, '_' or '-', "
            "start with a letter or digit, and contain no path separators"
        )
    return value


def derive_evaluation_paths(
    metrics_dir: str | os.PathLike[str], evaluation_id: object
) -> EvaluationPaths:
    """Derive the only three governance paths for an evaluation identifier."""

    identifier = sanitize_evaluation_id(evaluation_id)
    directory = Path(metrics_dir)
    prefix = f"final-evaluation-{identifier}"
    return EvaluationPaths(
        metrics_dir=directory,
        evaluation_id=identifier,
        freeze_path=directory / f"{prefix}.freeze.json",
        claim_path=directory / f"{prefix}.claim.json",
        receipt_path=directory / f"{prefix}.receipt.json",
    )


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def preflight_evaluation(paths: EvaluationPaths) -> None:
    """Refuse a new run when any freeze, claim, or receipt path already exists."""

    _validate_evaluation_paths(paths)
    if _lexists(paths.metrics_dir) and not paths.metrics_dir.is_dir():
        raise GovernanceConflictError(
            f"metrics path exists but is not a directory: {paths.metrics_dir}"
        )
    existing = [path for path in paths.immutable_paths if _lexists(path)]
    if existing:
        raise GovernanceConflictError(
            "evaluation is already claimed or partially materialized; refusing replay: "
            + ", ".join(str(path) for path in existing)
        )


def _validate_evaluation_paths(paths: EvaluationPaths) -> None:
    if not isinstance(paths, EvaluationPaths):
        raise TypeError("paths must be EvaluationPaths")
    expected = derive_evaluation_paths(paths.metrics_dir, paths.evaluation_id)
    if paths != expected:
        raise ValueError("EvaluationPaths does not match its metrics_dir/evaluation_id")


def validate_sha256(value: object, name: str = "sha256") -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 hex digest")
    return value


def sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(bytes(value)).hexdigest()


def _positive_size_limit(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer byte count")
    return value


def _open_regular_file(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        # A FIFO must fail the regular-file check instead of blocking in open().
        flags |= os.O_NONBLOCK
    try:
        descriptor = os.open(os.fspath(path), flags)
    except (FileNotFoundError, OSError) as exc:
        raise GovernanceIntegrityError(f"cannot open regular file: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise GovernanceIntegrityError(f"path is not a regular file: {path}")
        return descriptor, metadata
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_file_bytes(
    path: Path, *, max_bytes: int | None = None
) -> bytes:
    limit = _positive_size_limit(
        MAX_GOVERNANCE_RECORD_BYTES if max_bytes is None else max_bytes,
        "max_bytes",
    )
    descriptor, metadata = _open_regular_file(path)
    try:
        if metadata.st_size > limit:
            raise GovernanceIntegrityError(
                f"file exceeds {limit}-byte read limit: {path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise GovernanceIntegrityError(
                f"file exceeds {limit}-byte read limit: {path}"
            )
        return data
    finally:
        os.close(descriptor)


def sha256_file(
    path: str | os.PathLike[str], *, max_bytes: int | None = None
) -> str:
    """Stream-hash a bounded regular, non-symlink file's exact bytes."""

    limit = _positive_size_limit(
        MAX_HASHED_FILE_BYTES if max_bytes is None else max_bytes,
        "max_bytes",
    )
    source = Path(path)
    descriptor, metadata = _open_regular_file(source)
    try:
        if metadata.st_size > limit:
            raise GovernanceIntegrityError(
                f"file exceeds {limit}-byte hash limit: {source}"
            )
        digest = hashlib.sha256()
        total = 0
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while True:
                chunk = stream.read(min(_HASH_CHUNK_BYTES, limit - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise GovernanceIntegrityError(
                        f"file exceeds {limit}-byte hash limit: {source}"
                    )
                digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _normalize_json(value: object, location: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON number at {location}")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"JSON object key at {location} must be a string")
            normalized[key] = _normalize_json(item, f"{location}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json(item, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"unsupported JSON value at {location}: {type(value).__name__}")


def canonical_json_bytes(payload: object) -> bytes:
    """Return stable UTF-8 JSON bytes with sorted keys and one trailing newline."""

    normalized = _normalize_json(payload)
    text = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def canonical_json_sha256(payload: object) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(os.fspath(directory), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_canonical_json_exclusive(
    path: str | os.PathLike[str], payload: object
) -> ImmutableRecord:
    """Create canonical JSON via ``x`` mode, fsync it, and never overwrite.

    If writing or syncing fails after creation, the partial path is intentionally
    retained.  Subsequent preflight/creation calls will treat it as a conflict.
    """

    destination = Path(path)
    data = canonical_json_bytes(payload)
    if len(data) > MAX_GOVERNANCE_RECORD_BYTES:
        raise ValueError(
            "canonical governance record exceeds "
            f"{MAX_GOVERNANCE_RECORD_BYTES}-byte limit"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise GovernanceConflictError(
            f"immutable record already exists; refusing overwrite: {destination}"
        ) from exc
    os.chmod(destination, 0o444)
    _fsync_directory(destination.parent)
    digest = sha256_file(destination)
    if digest != sha256_bytes(data):
        raise GovernanceIntegrityError(
            f"persisted bytes differ from canonical payload: {destination}"
        )
    return ImmutableRecord(destination, digest)


def _decode_canonical_json(raw: bytes, source: Path) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GovernanceIntegrityError(f"invalid JSON record: {source}") from exc
    if not isinstance(value, dict):
        raise GovernanceIntegrityError(f"governance record is not an object: {source}")
    try:
        canonical = canonical_json_bytes(value)
    except ValueError as exc:
        raise GovernanceIntegrityError(f"non-canonical JSON record: {source}") from exc
    if raw != canonical:
        raise GovernanceIntegrityError(f"record bytes are not canonical: {source}")
    return value


def _read_canonical_json_with_hash(path: Path) -> tuple[dict[str, Any], str]:
    """Parse and hash the exact same read, avoiding a pathname re-read race."""

    raw = _read_regular_file_bytes(path)
    return _decode_canonical_json(raw, path), sha256_bytes(raw)


def read_canonical_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and prove a governance record is canonical JSON object bytes."""

    value, _ = _read_canonical_json_with_hash(Path(path))
    return value


def _validate_relative_posix_path(value: object, name: str) -> str:
    if not isinstance(value, str) or value in {"", ".", ".."} or "\\" in value:
        raise ValueError(f"{name} must be a normalized relative POSIX path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != value
    ):
        raise ValueError(f"{name} must be a normalized relative POSIX path")
    return value


def _rooted_regular_file(
    root: str | os.PathLike[str], path: str | os.PathLike[str], role: str
) -> tuple[Path, str]:
    root_path = Path(root)
    try:
        resolved_root = root_path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"{role} root does not exist: {root_path}") from exc
    if not resolved_root.is_dir():
        raise ValueError(f"{role} root is not a directory: {root_path}")
    supplied = Path(path)
    candidate = supplied if supplied.is_absolute() else resolved_root / supplied
    if candidate.is_symlink():
        raise ValueError(f"{role} file must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(resolved_root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ValueError(f"{role} file escapes its root or does not exist: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"{role} path is not a regular file: {path}")
    relative_text = _validate_relative_posix_path(relative.as_posix(), f"{role} path")
    return resolved, relative_text


def build_code_manifest(
    code_root: str | os.PathLike[str],
    files: Sequence[str | os.PathLike[str]],
) -> CodeManifest:
    """Hash an explicit file list; caller ordering does not affect the manifest."""

    if isinstance(files, (str, bytes, os.PathLike)):
        raise ValueError("files must be an explicit sequence of paths")
    entries: dict[str, str] = {}
    for supplied in files:
        resolved, relative = _rooted_regular_file(code_root, supplied, "code")
        if relative in entries:
            raise ValueError(f"duplicate code file after normalization: {relative}")
        entries[relative] = sha256_file(resolved)
    if not entries:
        raise ValueError("files must contain at least one code path")
    return CodeManifest(tuple(sorted(entries.items())))


def verify_code_manifest(
    manifest: CodeManifest, code_root: str | os.PathLike[str]
) -> None:
    """Raise if any explicit code-manifest path or byte hash changed."""

    if not isinstance(manifest, CodeManifest):
        raise TypeError("manifest must be CodeManifest")
    for relative, expected in manifest.files:
        try:
            resolved, normalized = _rooted_regular_file(code_root, relative, "code")
        except ValueError as exc:
            raise GovernanceIntegrityError(str(exc)) from exc
        if normalized != relative:
            raise GovernanceIntegrityError(f"code path changed: {relative}")
        actual = sha256_file(resolved)
        if actual != expected:
            raise GovernanceIntegrityError(f"code manifest hash changed: {relative}")


def _record(
    record_type: str,
    evaluation_id: str,
    fields: Mapping[str, object],
    caller_metadata: Mapping[str, object] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "record_type": record_type,
        "schema_version": SCHEMA_VERSION,
        "evaluation_id": evaluation_id,
        **dict(fields),
    }
    if caller_metadata is not None:
        if not isinstance(caller_metadata, Mapping):
            raise TypeError("caller_metadata must be a mapping")
        payload["caller_metadata"] = dict(caller_metadata)
    _normalize_json(payload)
    return payload


def create_decision_freeze(
    paths: EvaluationPaths,
    decision_payload: Mapping[str, object],
    *,
    caller_metadata: Mapping[str, object] | None = None,
) -> ImmutableRecord:
    """Preflight and exclusively freeze validation-selected decision semantics."""

    if not isinstance(decision_payload, Mapping) or not decision_payload:
        raise ValueError("decision_payload must be a non-empty mapping")
    preflight_evaluation(paths)
    payload = _record(
        "final_evaluation_decision_freeze",
        paths.evaluation_id,
        {"decision": dict(decision_payload)},
        caller_metadata,
    )
    return write_canonical_json_exclusive(paths.freeze_path, payload)


def _require_absent(path: Path, role: str) -> None:
    if _lexists(path):
        raise GovernanceConflictError(
            f"{role} path already exists; refusing replay/overwrite: {path}"
        )


def _require_record_with_hash(
    path: Path, record_type: str, evaluation_id: str
) -> tuple[dict[str, Any], str]:
    value, digest = _read_canonical_json_with_hash(path)
    if value.get("record_type") != record_type:
        raise GovernanceIntegrityError(f"unexpected record_type in {path}")
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != SCHEMA_VERSION
    ):
        raise GovernanceIntegrityError(f"unsupported schema_version in {path}")
    if value.get("evaluation_id") != evaluation_id:
        raise GovernanceIntegrityError(f"evaluation_id mismatch in {path}")
    return value, digest


def _require_exact_record_keys(
    value: Mapping[str, object], required: set[str], role: str
) -> None:
    optional = {"caller_metadata"}
    keys = set(value)
    missing = sorted(required - keys)
    extra = sorted(keys - required - optional)
    if missing or extra:
        raise GovernanceIntegrityError(
            f"{role} schema mismatch; missing={missing}, extra={extra}"
        )
    if "caller_metadata" in value and not isinstance(value["caller_metadata"], dict):
        raise GovernanceIntegrityError(f"{role} caller_metadata must be an object")


def _validate_freeze_record(value: Mapping[str, object]) -> None:
    _require_exact_record_keys(
        value,
        {
            "record_type",
            "schema_version",
            "evaluation_id",
            "decision",
        },
        "decision freeze",
    )
    if not isinstance(value["decision"], dict) or not value["decision"]:
        raise GovernanceIntegrityError("decision freeze decision must be a non-empty object")


def _validate_claim_record(value: Mapping[str, object]) -> CodeManifest:
    _require_exact_record_keys(
        value,
        {
            "record_type",
            "schema_version",
            "evaluation_id",
            "protocol_sha256",
            "freeze_sha256",
            "outcome",
            "source_sha256",
            "development_index_sha256",
            "reserved_index_sha256",
            "final_index_sha256",
            "code_manifest",
            "code_manifest_sha256",
        },
        "final-access claim",
    )
    try:
        for field in (
            "protocol_sha256",
            "freeze_sha256",
            "source_sha256",
            "development_index_sha256",
            "reserved_index_sha256",
            "final_index_sha256",
            "code_manifest_sha256",
        ):
            validate_sha256(value[field], field)
        _nonempty_text(value["outcome"], "outcome")
    except ValueError as exc:
        raise GovernanceIntegrityError(str(exc)) from exc
    manifest = CodeManifest.from_record(value["code_manifest"])
    if value["code_manifest_sha256"] != canonical_json_sha256(
        value["code_manifest"]
    ):
        raise GovernanceIntegrityError(
            "claim code-manifest digest does not bind the embedded manifest"
        )
    return manifest


def _validate_receipt_record(value: Mapping[str, object]) -> None:
    _require_exact_record_keys(
        value,
        {
            "record_type",
            "schema_version",
            "evaluation_id",
            "protocol_sha256",
            "freeze_sha256",
            "claim_sha256",
            "code_manifest_sha256",
            "output_artifacts",
        },
        "completion receipt",
    )
    try:
        for field in (
            "protocol_sha256",
            "freeze_sha256",
            "claim_sha256",
            "code_manifest_sha256",
        ):
            validate_sha256(value[field], field)
    except ValueError as exc:
        raise GovernanceIntegrityError(str(exc)) from exc


def create_final_access_claim(
    paths: EvaluationPaths,
    *,
    protocol_path: str | os.PathLike[str],
    protocol_sha256: str,
    freeze_sha256: str,
    outcome: str,
    source_sha256: str,
    development_index_sha256: str,
    reserved_index_sha256: str,
    final_index_sha256: str,
    code_manifest: CodeManifest,
    code_root: str | os.PathLike[str],
    caller_metadata: Mapping[str, object] | None = None,
) -> ImmutableRecord:
    """Create the immutable claim that must precede access to final outcomes."""

    _validate_evaluation_paths(paths)
    _require_absent(paths.claim_path, "claim")
    _require_absent(paths.receipt_path, "receipt")
    expected_freeze = validate_sha256(freeze_sha256, "freeze_sha256")
    expected_protocol = validate_sha256(protocol_sha256, "protocol_sha256")
    freeze_record, actual_freeze = _require_record_with_hash(
        paths.freeze_path,
        "final_evaluation_decision_freeze",
        paths.evaluation_id,
    )
    _validate_freeze_record(freeze_record)
    if actual_freeze != expected_freeze:
        raise GovernanceIntegrityError("decision freeze hash changed before claim")
    if sha256_file(protocol_path) != expected_protocol:
        raise GovernanceIntegrityError("protocol hash does not match protocol file")
    verify_code_manifest(code_manifest, code_root)
    clean_outcome = _nonempty_text(outcome, "outcome")
    fields = {
        "protocol_sha256": expected_protocol,
        "freeze_sha256": expected_freeze,
        "outcome": clean_outcome,
        "source_sha256": validate_sha256(source_sha256, "source_sha256"),
        "development_index_sha256": validate_sha256(
            development_index_sha256, "development_index_sha256"
        ),
        "reserved_index_sha256": validate_sha256(
            reserved_index_sha256, "reserved_index_sha256"
        ),
        "final_index_sha256": validate_sha256(
            final_index_sha256, "final_index_sha256"
        ),
        "code_manifest": code_manifest.to_record(),
        "code_manifest_sha256": code_manifest.sha256,
    }
    payload = _record(
        "final_evaluation_access_claim",
        paths.evaluation_id,
        fields,
        caller_metadata,
    )
    return write_canonical_json_exclusive(paths.claim_path, payload)


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _verify_claim_payload(
    paths: EvaluationPaths,
    *,
    protocol_path: str | os.PathLike[str],
    code_root: str | os.PathLike[str],
    expected_freeze_sha256: str,
    expected_claim_sha256: str,
) -> tuple[dict[str, Any], str, str, str]:
    freeze, freeze_hash = _require_record_with_hash(
        paths.freeze_path,
        "final_evaluation_decision_freeze",
        paths.evaluation_id,
    )
    _validate_freeze_record(freeze)
    claim, claim_hash = _require_record_with_hash(
        paths.claim_path,
        "final_evaluation_access_claim",
        paths.evaluation_id,
    )
    manifest = _validate_claim_record(claim)
    if freeze_hash != validate_sha256(
        expected_freeze_sha256, "expected_freeze_sha256"
    ):
        raise GovernanceIntegrityError("decision freeze hash changed")
    if claim_hash != validate_sha256(expected_claim_sha256, "expected_claim_sha256"):
        raise GovernanceIntegrityError("final-access claim hash changed")
    if claim.get("freeze_sha256") != freeze_hash:
        raise GovernanceIntegrityError("claim does not bind the current freeze hash")
    protocol_hash = sha256_file(protocol_path)
    if claim.get("protocol_sha256") != protocol_hash:
        raise GovernanceIntegrityError("claim does not bind the current protocol hash")
    if claim.get("code_manifest_sha256") != manifest.sha256:
        raise GovernanceIntegrityError("claim code-manifest digest is invalid")
    verify_code_manifest(manifest, code_root)
    return claim, freeze_hash, claim_hash, protocol_hash


def verify_final_access_claim(
    paths: EvaluationPaths,
    *,
    protocol_path: str | os.PathLike[str],
    code_root: str | os.PathLike[str],
    expected_freeze_sha256: str,
    expected_claim_sha256: str,
    require_receipt_absent: bool = True,
) -> dict[str, Any]:
    """Verify freeze, claim, protocol, and code before final-outcome access."""

    _validate_evaluation_paths(paths)
    if require_receipt_absent:
        _require_absent(paths.receipt_path, "receipt")
    claim, _, _, _ = _verify_claim_payload(
        paths,
        protocol_path=protocol_path,
        code_root=code_root,
        expected_freeze_sha256=expected_freeze_sha256,
        expected_claim_sha256=expected_claim_sha256,
    )
    return claim


def _validate_artifact_name(value: object) -> str:
    if not isinstance(value, str) or _ARTIFACT_NAME.fullmatch(value) is None:
        raise ValueError(
            "artifact name must be 1-128 ASCII letters, digits, '_', '-', or '.'"
        )
    return value


def _output_manifest(
    output_artifacts: Mapping[str, str | os.PathLike[str]],
    output_root: str | os.PathLike[str],
) -> list[dict[str, str]]:
    if not isinstance(output_artifacts, Mapping) or not output_artifacts:
        raise ValueError("output_artifacts must be a non-empty mapping")
    entries: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for raw_name, supplied in output_artifacts.items():
        name = _validate_artifact_name(raw_name)
        resolved, relative = _rooted_regular_file(output_root, supplied, "output")
        if relative in seen_paths:
            raise ValueError(f"duplicate output artifact path: {relative}")
        seen_paths.add(relative)
        entries.append(
            {"name": name, "path": relative, "sha256": sha256_file(resolved)}
        )
    entries.sort(key=lambda item: (item["name"], item["path"]))
    return entries


def create_completion_receipt(
    paths: EvaluationPaths,
    *,
    protocol_path: str | os.PathLike[str],
    protocol_sha256: str,
    freeze_sha256: str,
    claim_sha256: str,
    code_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    output_artifacts: Mapping[str, str | os.PathLike[str]],
    caller_metadata: Mapping[str, object] | None = None,
) -> ImmutableRecord:
    """Exclusively bind a verified claim to exact completed output bytes."""

    _validate_evaluation_paths(paths)
    _require_absent(paths.receipt_path, "receipt")
    expected_protocol = validate_sha256(protocol_sha256, "protocol_sha256")
    expected_freeze = validate_sha256(freeze_sha256, "freeze_sha256")
    expected_claim = validate_sha256(claim_sha256, "claim_sha256")
    claim, _, _, _ = _verify_claim_payload(
        paths,
        protocol_path=protocol_path,
        code_root=code_root,
        expected_freeze_sha256=expected_freeze,
        expected_claim_sha256=expected_claim,
    )
    if claim.get("protocol_sha256") != expected_protocol:
        raise GovernanceIntegrityError("receipt protocol hash disagrees with claim")
    artifacts = _output_manifest(output_artifacts, output_root)
    # Close the verification-to-write window as far as a file-based API can.
    _verify_claim_payload(
        paths,
        protocol_path=protocol_path,
        code_root=code_root,
        expected_freeze_sha256=expected_freeze,
        expected_claim_sha256=expected_claim,
    )
    if _output_manifest(output_artifacts, output_root) != artifacts:
        raise GovernanceIntegrityError(
            "output artifacts changed while completion receipt was being prepared"
        )
    payload = _record(
        "final_evaluation_completion_receipt",
        paths.evaluation_id,
        {
            "protocol_sha256": expected_protocol,
            "freeze_sha256": expected_freeze,
            "claim_sha256": expected_claim,
            "code_manifest_sha256": claim["code_manifest_sha256"],
            "output_artifacts": artifacts,
        },
        caller_metadata,
    )
    return write_canonical_json_exclusive(paths.receipt_path, payload)


def verify_completion_receipt(
    paths: EvaluationPaths,
    *,
    protocol_path: str | os.PathLike[str],
    code_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    expected_freeze_sha256: str,
    expected_claim_sha256: str,
    expected_receipt_sha256: str,
) -> dict[str, str]:
    """Verify the complete immutable chain, code, protocol, and output hashes."""

    _validate_evaluation_paths(paths)
    receipt, receipt_hash = _require_record_with_hash(
        paths.receipt_path,
        "final_evaluation_completion_receipt",
        paths.evaluation_id,
    )
    _validate_receipt_record(receipt)
    if receipt_hash != validate_sha256(
        expected_receipt_sha256, "expected_receipt_sha256"
    ):
        raise GovernanceIntegrityError("completion receipt hash changed")
    claim, actual_freeze, actual_claim, actual_protocol = _verify_claim_payload(
        paths,
        protocol_path=protocol_path,
        code_root=code_root,
        expected_freeze_sha256=expected_freeze_sha256,
        expected_claim_sha256=expected_claim_sha256,
    )
    bindings = {
        "freeze_sha256": actual_freeze,
        "claim_sha256": actual_claim,
        "protocol_sha256": actual_protocol,
        "code_manifest_sha256": claim["code_manifest_sha256"],
    }
    for field, expected in bindings.items():
        if receipt.get(field) != expected:
            raise GovernanceIntegrityError(f"receipt {field} binding is invalid")
    artifacts = receipt.get("output_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise GovernanceIntegrityError("receipt output_artifacts is invalid")
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    for entry in artifacts:
        if not isinstance(entry, dict) or set(entry) != {"name", "path", "sha256"}:
            raise GovernanceIntegrityError("receipt output artifact schema is invalid")
        try:
            name = _validate_artifact_name(entry["name"])
            relative = _validate_relative_posix_path(
                entry["path"], "receipt output path"
            )
            expected = validate_sha256(entry["sha256"], "output sha256")
        except ValueError as exc:
            raise GovernanceIntegrityError(str(exc)) from exc
        if name in seen_names or relative in seen_paths:
            raise GovernanceIntegrityError("receipt has duplicate output entries")
        seen_names.add(name)
        seen_paths.add(relative)
        try:
            resolved, normalized = _rooted_regular_file(
                output_root, relative, "output"
            )
        except ValueError as exc:
            raise GovernanceIntegrityError(str(exc)) from exc
        if normalized != relative or sha256_file(resolved) != expected:
            raise GovernanceIntegrityError(f"output artifact changed: {relative}")
    return {
        "freeze_sha256": actual_freeze,
        "claim_sha256": actual_claim,
        "receipt_sha256": receipt_hash,
        "protocol_sha256": actual_protocol,
        "code_manifest_sha256": str(claim["code_manifest_sha256"]),
    }


__all__ = [
    "SCHEMA_VERSION",
    "MAX_GOVERNANCE_RECORD_BYTES",
    "MAX_HASHED_FILE_BYTES",
    "CodeManifest",
    "EvaluationPaths",
    "GovernanceConflictError",
    "GovernanceError",
    "GovernanceIntegrityError",
    "ImmutableRecord",
    "build_code_manifest",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "create_completion_receipt",
    "create_decision_freeze",
    "create_final_access_claim",
    "derive_evaluation_paths",
    "preflight_evaluation",
    "read_canonical_json",
    "sanitize_evaluation_id",
    "sha256_bytes",
    "sha256_file",
    "validate_sha256",
    "verify_code_manifest",
    "verify_completion_receipt",
    "verify_final_access_claim",
    "write_canonical_json_exclusive",
]
