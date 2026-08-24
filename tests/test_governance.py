from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from src import governance as governance_module
from src.governance import (
    GovernanceConflictError,
    GovernanceIntegrityError,
    MAX_GOVERNANCE_RECORD_BYTES,
    build_code_manifest,
    canonical_json_bytes,
    canonical_json_sha256,
    create_completion_receipt,
    create_decision_freeze,
    create_final_access_claim,
    derive_evaluation_paths,
    preflight_evaluation,
    read_canonical_json,
    sanitize_evaluation_id,
    sha256_bytes,
    sha256_file,
    verify_code_manifest,
    verify_completion_receipt,
    verify_final_access_claim,
    write_canonical_json_exclusive,
)


def digest(label: str) -> str:
    return sha256_bytes(label.encode("utf-8"))


class GovernanceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.metrics = self.root / "metrics"
        self.code_root = self.root / "checkout"
        self.output_root = self.root / "outputs"
        self.metrics.mkdir()
        self.code_root.mkdir()
        self.output_root.mkdir()
        (self.code_root / "src").mkdir()
        self.code_a = self.code_root / "src" / "a.py"
        self.code_b = self.code_root / "src" / "b.py"
        self.code_a.write_text("A = 1\n", encoding="utf-8")
        self.code_b.write_text("B = 2\n", encoding="utf-8")
        self.protocol = self.root / "protocol.json"
        self.protocol.write_bytes(canonical_json_bytes({"protocol": 1}))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _manifest(self):
        return build_code_manifest(
            self.code_root, [Path("src/a.py"), Path("src/b.py")]
        )

    def _claim(self, evaluation_id: str = "eval_001"):
        paths = derive_evaluation_paths(self.metrics, evaluation_id)
        freeze = create_decision_freeze(
            paths,
            {
                "selected_policy": "response",
                "selected_budget": 0.2,
                "selection_sample": "validation",
            },
        )
        manifest = self._manifest()
        claim = create_final_access_claim(
            paths,
            protocol_path=self.protocol,
            protocol_sha256=sha256_file(self.protocol),
            freeze_sha256=freeze.sha256,
            outcome="conversion",
            source_sha256=digest("source"),
            development_index_sha256=digest("development"),
            reserved_index_sha256=digest("reserved"),
            final_index_sha256=digest("final"),
            code_manifest=manifest,
            code_root=self.code_root,
        )
        return paths, freeze, claim, manifest

    def _complete(self, evaluation_id: str = "eval_001"):
        paths, freeze, claim, manifest = self._claim(evaluation_id)
        metric = self.output_root / "metric.csv"
        figure = self.output_root / "figure.png"
        metric.write_text("metric,value\nnet,12\n", encoding="utf-8")
        figure.write_bytes(b"not-a-real-png-but-exact-output-bytes")
        outputs = {"figure": figure, "metrics": metric}
        receipt = create_completion_receipt(
            paths,
            protocol_path=self.protocol,
            protocol_sha256=sha256_file(self.protocol),
            freeze_sha256=freeze.sha256,
            claim_sha256=claim.sha256,
            code_root=self.code_root,
            output_root=self.output_root,
            output_artifacts=outputs,
        )
        return paths, freeze, claim, receipt, manifest, outputs


class IdentifierAndPathTests(GovernanceTestCase):
    def test_identifier_validation_is_non_lossy_and_blocks_traversal(self) -> None:
        for valid in ("a", "Eval_2026-08-17", "ABC123"):
            self.assertEqual(sanitize_evaluation_id(valid), valid)
        for invalid in (
            "",
            ".",
            "..",
            "../escape",
            "a/b",
            "a\\b",
            " leading",
            "trailing ",
            "évaluation",
            "a" * 129,
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    sanitize_evaluation_id(invalid)

    def test_derived_paths_are_unique_and_under_metrics_directory(self) -> None:
        paths = derive_evaluation_paths(self.metrics, "run-7")
        self.assertEqual(len(set(paths.immutable_paths)), 3)
        for path in paths.immutable_paths:
            self.assertEqual(path.parent, self.metrics)
            self.assertIn("run-7", path.name)

    def test_preflight_refuses_any_existing_or_broken_symlink_path(self) -> None:
        for index, role in enumerate(("freeze_path", "claim_path", "receipt_path")):
            paths = derive_evaluation_paths(self.metrics, f"existing-{index}")
            getattr(paths, role).write_text("partial", encoding="utf-8")
            with self.subTest(role=role):
                with self.assertRaises(GovernanceConflictError):
                    preflight_evaluation(paths)

        paths = derive_evaluation_paths(self.metrics, "broken-link")
        os.symlink("missing-target", paths.claim_path)
        self.assertFalse(paths.claim_path.exists())
        with self.assertRaises(GovernanceConflictError):
            preflight_evaluation(paths)


class CanonicalRecordTests(GovernanceTestCase):
    def test_canonical_json_is_order_independent_and_exclusively_created(self) -> None:
        first = {"z": 2, "a": {"y": 1, "x": [3, 4]}}
        second = {"a": {"x": [3, 4], "y": 1}, "z": 2}
        expected = b'{"a":{"x":[3,4],"y":1},"z":2}\n'
        self.assertEqual(canonical_json_bytes(first), expected)
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertEqual(canonical_json_sha256(first), canonical_json_sha256(second))

        path = self.metrics / "canonical.json"
        original_fsync = os.fsync
        with mock.patch("src.governance.os.fsync", wraps=original_fsync) as fsync:
            record = write_canonical_json_exclusive(path, first)
        self.assertGreaterEqual(fsync.call_count, 2)
        self.assertEqual(path.read_bytes(), expected)
        self.assertEqual(record.sha256, sha256_bytes(expected))
        with self.assertRaises(GovernanceConflictError):
            write_canonical_json_exclusive(path, second)

    def test_canonical_json_rejects_nonfinite_and_noncanonical_records(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            canonical_json_bytes({"bad": float("nan")})
        path = self.metrics / "pretty.json"
        path.write_text(json.dumps({"b": 2, "a": 1}, indent=2), encoding="utf-8")
        with self.assertRaisesRegex(GovernanceIntegrityError, "not canonical"):
            read_canonical_json(path)

    def test_failed_write_leaves_replay_blocking_partial_path(self) -> None:
        path = self.metrics / "partial.json"
        with mock.patch("src.governance.os.fsync", side_effect=OSError("disk failed")):
            with self.assertRaisesRegex(OSError, "disk failed"):
                write_canonical_json_exclusive(path, {"record": "partial"})
        self.assertTrue(path.exists())
        with self.assertRaises(GovernanceConflictError):
            write_canonical_json_exclusive(path, {"record": "retry"})

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO support is required")
    def test_regular_file_reader_rejects_fifo_without_blocking(self) -> None:
        fifo = self.root / "untrusted-fifo"
        os.mkfifo(fifo)
        project_root = Path(__file__).resolve().parents[1]
        script = (
            "from src.governance import sha256_file; "
            f"sha256_file({os.fspath(fifo)!r})"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("not a regular file", completed.stderr)

    def test_record_and_stream_hash_size_limits_fail_closed(self) -> None:
        oversized_record = self.metrics / "oversized-record.json"
        oversized_record.write_bytes(b"x" * (MAX_GOVERNANCE_RECORD_BYTES + 1))
        with self.assertRaisesRegex(GovernanceIntegrityError, "read limit"):
            read_canonical_json(oversized_record)

        destination = self.metrics / "bounded-write.json"
        with mock.patch("src.governance.MAX_GOVERNANCE_RECORD_BYTES", 32):
            with self.assertRaisesRegex(ValueError, "record exceeds"):
                write_canonical_json_exclusive(
                    destination, {"payload": "x" * 64}
                )
        self.assertFalse(destination.exists())

        artifact = self.output_root / "bounded.bin"
        artifact.write_bytes(b"abcdef")
        self.assertEqual(
            sha256_file(artifact, max_bytes=6), sha256_bytes(b"abcdef")
        )
        with self.assertRaisesRegex(GovernanceIntegrityError, "hash limit"):
            sha256_file(artifact, max_bytes=5)
        with self.assertRaises(ValueError):
            sha256_file(artifact, max_bytes=True)


class CodeManifestTests(GovernanceTestCase):
    def test_manifest_is_deterministic_and_detects_code_tampering(self) -> None:
        forward = build_code_manifest(
            self.code_root, ["src/a.py", self.code_b]
        )
        reverse = build_code_manifest(
            self.code_root, [self.code_b, "src/a.py"]
        )
        self.assertEqual(forward, reverse)
        self.assertEqual(forward.sha256, reverse.sha256)
        self.assertEqual(
            [entry[0] for entry in forward.files], ["src/a.py", "src/b.py"]
        )
        verify_code_manifest(forward, self.code_root)
        self.code_a.write_text("A = 999\n", encoding="utf-8")
        with self.assertRaisesRegex(GovernanceIntegrityError, "hash changed"):
            verify_code_manifest(forward, self.code_root)

    def test_manifest_rejects_outside_paths_duplicates_and_symlinks(self) -> None:
        outside = self.root / "outside.py"
        outside.write_text("outside = True\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "escapes"):
            build_code_manifest(self.code_root, [outside])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_code_manifest(self.code_root, ["src/a.py", self.code_a])
        link = self.code_root / "src" / "link.py"
        os.symlink(self.code_a, link)
        with self.assertRaisesRegex(ValueError, "symlink"):
            build_code_manifest(self.code_root, [link])

    def test_manifest_hashing_enforces_the_shared_file_size_limit(self) -> None:
        with mock.patch("src.governance.MAX_HASHED_FILE_BYTES", 4):
            with self.assertRaisesRegex(GovernanceIntegrityError, "hash limit"):
                build_code_manifest(self.code_root, ["src/a.py"])


class GovernanceChainTests(GovernanceTestCase):
    def test_complete_chain_binds_every_required_digest_without_auto_timestamp(self) -> None:
        paths, freeze, claim, receipt, manifest, outputs = self._complete()
        claim_payload = read_canonical_json(paths.claim_path)
        self.assertEqual(claim_payload["evaluation_id"], "eval_001")
        self.assertEqual(claim_payload["protocol_sha256"], sha256_file(self.protocol))
        self.assertEqual(claim_payload["freeze_sha256"], freeze.sha256)
        self.assertEqual(claim_payload["outcome"], "conversion")
        self.assertEqual(claim_payload["source_sha256"], digest("source"))
        self.assertEqual(
            claim_payload["development_index_sha256"], digest("development")
        )
        self.assertEqual(claim_payload["reserved_index_sha256"], digest("reserved"))
        self.assertEqual(claim_payload["final_index_sha256"], digest("final"))
        self.assertEqual(claim_payload["code_manifest_sha256"], manifest.sha256)
        self.assertNotIn("timestamp", json.dumps(claim_payload).lower())

        receipt_payload = read_canonical_json(paths.receipt_path)
        self.assertEqual(receipt_payload["claim_sha256"], claim.sha256)
        self.assertEqual(receipt_payload["freeze_sha256"], freeze.sha256)
        self.assertEqual(
            receipt_payload["protocol_sha256"], sha256_file(self.protocol)
        )
        self.assertEqual(
            [item["name"] for item in receipt_payload["output_artifacts"]],
            ["figure", "metrics"],
        )
        self.assertNotIn("timestamp", json.dumps(receipt_payload).lower())

        verified = verify_completion_receipt(
            paths,
            protocol_path=self.protocol,
            code_root=self.code_root,
            output_root=self.output_root,
            expected_freeze_sha256=freeze.sha256,
            expected_claim_sha256=claim.sha256,
            expected_receipt_sha256=receipt.sha256,
        )
        self.assertEqual(verified["receipt_sha256"], receipt.sha256)
        for name, path in outputs.items():
            entry = next(
                item
                for item in receipt_payload["output_artifacts"]
                if item["name"] == name
            )
            self.assertEqual(entry["sha256"], sha256_file(path))

    def test_caller_supplied_timestamp_is_hashed_as_metadata(self) -> None:
        paths = derive_evaluation_paths(self.metrics, "caller-time")
        record = create_decision_freeze(
            paths,
            {"decision": "response"},
            caller_metadata={"timestamp": "2026-08-17T12:00:00Z"},
        )
        payload = read_canonical_json(record.path)
        self.assertEqual(
            payload["caller_metadata"]["timestamp"], "2026-08-17T12:00:00Z"
        )

    def test_claim_verifies_before_final_access_and_refuses_replay(self) -> None:
        paths, freeze, claim, _ = self._claim()
        verified = verify_final_access_claim(
            paths,
            protocol_path=self.protocol,
            code_root=self.code_root,
            expected_freeze_sha256=freeze.sha256,
            expected_claim_sha256=claim.sha256,
        )
        self.assertEqual(verified["record_type"], "final_evaluation_access_claim")
        with self.assertRaises(GovernanceConflictError):
            preflight_evaluation(paths)
        with self.assertRaises(GovernanceConflictError):
            create_final_access_claim(
                paths,
                protocol_path=self.protocol,
                protocol_sha256=sha256_file(self.protocol),
                freeze_sha256=freeze.sha256,
                outcome="conversion",
                source_sha256=digest("source"),
                development_index_sha256=digest("development"),
                reserved_index_sha256=digest("reserved"),
                final_index_sha256=digest("final"),
                code_manifest=self._manifest(),
                code_root=self.code_root,
            )

    def test_claim_verification_hashes_and_parses_each_record_from_one_read(self) -> None:
        paths, freeze, claim, _ = self._claim("single-read")
        original_reader = governance_module._read_regular_file_bytes
        with mock.patch(
            "src.governance._read_regular_file_bytes", wraps=original_reader
        ) as reader:
            verified = verify_final_access_claim(
                paths,
                protocol_path=self.protocol,
                code_root=self.code_root,
                expected_freeze_sha256=freeze.sha256,
                expected_claim_sha256=claim.sha256,
            )
        self.assertEqual(verified["outcome"], "conversion")
        read_paths = [call.args[0] for call in reader.call_args_list]
        self.assertEqual(read_paths.count(paths.freeze_path), 1)
        self.assertEqual(read_paths.count(paths.claim_path), 1)

    def test_claim_verification_rejects_reanchored_schema_variants(self) -> None:
        variants = ("boolean-version", "missing-outcome", "extra-field")
        for variant in variants:
            with self.subTest(variant=variant):
                paths, freeze, _, _ = self._claim(f"schema-{variant}")
                payload = read_canonical_json(paths.claim_path)
                if variant == "boolean-version":
                    payload["schema_version"] = True
                elif variant == "missing-outcome":
                    payload.pop("outcome")
                else:
                    payload["unexpected"] = "must fail closed"
                os.chmod(paths.claim_path, 0o644)
                paths.claim_path.write_bytes(canonical_json_bytes(payload))
                forged_anchor = sha256_file(paths.claim_path)
                with self.assertRaises(GovernanceIntegrityError):
                    verify_final_access_claim(
                        paths,
                        protocol_path=self.protocol,
                        code_root=self.code_root,
                        expected_freeze_sha256=freeze.sha256,
                        expected_claim_sha256=forged_anchor,
                    )

    def test_claim_rejects_manifest_type_coercion_and_normalized_digest(self) -> None:
        paths, freeze, _, _ = self._claim("manifest-type-coercion")
        payload = read_canonical_json(paths.claim_path)
        numeric_code_path = self.code_root / "1"
        numeric_code_path.write_bytes(self.code_a.read_bytes())
        embedded_manifest = payload["code_manifest"]
        embedded_manifest["files"][0]["path"] = 1
        normalized_manifest = json.loads(json.dumps(embedded_manifest))
        normalized_manifest["files"][0]["path"] = "1"
        payload["code_manifest_sha256"] = canonical_json_sha256(normalized_manifest)
        self.assertNotEqual(
            payload["code_manifest_sha256"],
            canonical_json_sha256(embedded_manifest),
        )
        os.chmod(paths.claim_path, 0o644)
        paths.claim_path.write_bytes(canonical_json_bytes(payload))
        forged_anchor = sha256_file(paths.claim_path)
        with self.assertRaisesRegex(GovernanceIntegrityError, "must be strings"):
            verify_final_access_claim(
                paths,
                protocol_path=self.protocol,
                code_root=self.code_root,
                expected_freeze_sha256=freeze.sha256,
                expected_claim_sha256=forged_anchor,
            )

    def test_partial_claim_blocks_recreation_and_preflight(self) -> None:
        paths = derive_evaluation_paths(self.metrics, "partial-claim")
        freeze = create_decision_freeze(paths, {"decision": "response"})
        os.chmod(paths.claim_path.parent, 0o755)
        paths.claim_path.write_bytes(b'{"record_type":"partial"')
        with self.assertRaises(GovernanceConflictError):
            preflight_evaluation(paths)
        with self.assertRaises(GovernanceConflictError):
            create_final_access_claim(
                paths,
                protocol_path=self.protocol,
                protocol_sha256=sha256_file(self.protocol),
                freeze_sha256=freeze.sha256,
                outcome="conversion",
                source_sha256=digest("source"),
                development_index_sha256=digest("development"),
                reserved_index_sha256=digest("reserved"),
                final_index_sha256=digest("final"),
                code_manifest=self._manifest(),
                code_root=self.code_root,
            )

    def test_claim_rejects_wrong_protocol_or_freeze_hash(self) -> None:
        paths = derive_evaluation_paths(self.metrics, "wrong-hash")
        freeze = create_decision_freeze(paths, {"decision": "response"})
        common = dict(
            paths=paths,
            protocol_path=self.protocol,
            outcome="conversion",
            source_sha256=digest("source"),
            development_index_sha256=digest("development"),
            reserved_index_sha256=digest("reserved"),
            final_index_sha256=digest("final"),
            code_manifest=self._manifest(),
            code_root=self.code_root,
        )
        with self.assertRaisesRegex(GovernanceIntegrityError, "freeze hash"):
            create_final_access_claim(
                **common,
                protocol_sha256=sha256_file(self.protocol),
                freeze_sha256=digest("wrong-freeze"),
            )
        self.assertFalse(paths.claim_path.exists())
        with self.assertRaisesRegex(GovernanceIntegrityError, "protocol hash"):
            create_final_access_claim(
                **common,
                protocol_sha256=digest("wrong-protocol"),
                freeze_sha256=freeze.sha256,
            )

    def test_completion_receipt_is_one_shot(self) -> None:
        paths, freeze, claim, receipt, _, outputs = self._complete()
        with self.assertRaises(GovernanceConflictError):
            create_completion_receipt(
                paths,
                protocol_path=self.protocol,
                protocol_sha256=sha256_file(self.protocol),
                freeze_sha256=freeze.sha256,
                claim_sha256=claim.sha256,
                code_root=self.code_root,
                output_root=self.output_root,
                output_artifacts=outputs,
            )
        self.assertTrue(receipt.path.exists())

    def test_code_change_after_claim_blocks_completion(self) -> None:
        paths, freeze, claim, _ = self._claim("code-change")
        output = self.output_root / "metric.csv"
        output.write_text("ok\n", encoding="utf-8")
        self.code_a.write_text("A = 999\n", encoding="utf-8")
        with self.assertRaisesRegex(GovernanceIntegrityError, "hash changed"):
            create_completion_receipt(
                paths,
                protocol_path=self.protocol,
                protocol_sha256=sha256_file(self.protocol),
                freeze_sha256=freeze.sha256,
                claim_sha256=claim.sha256,
                code_root=self.code_root,
                output_root=self.output_root,
                output_artifacts={"metric": output},
            )
        self.assertFalse(paths.receipt_path.exists())

    def test_oversized_output_blocks_completion_receipt(self) -> None:
        paths, freeze, claim, _ = self._claim("oversized-output")
        output = self.output_root / "oversized.bin"
        output.write_bytes(b"x" * 64)
        with mock.patch("src.governance.MAX_HASHED_FILE_BYTES", 32):
            with self.assertRaisesRegex(GovernanceIntegrityError, "hash limit"):
                create_completion_receipt(
                    paths,
                    protocol_path=self.protocol,
                    protocol_sha256=sha256_file(self.protocol),
                    freeze_sha256=freeze.sha256,
                    claim_sha256=claim.sha256,
                    code_root=self.code_root,
                    output_root=self.output_root,
                    output_artifacts={"oversized": output},
                )
        self.assertFalse(paths.receipt_path.exists())

    def test_output_path_traversal_and_symlink_are_rejected(self) -> None:
        paths, freeze, claim, _ = self._claim("bad-output")
        outside = self.root / "outside.csv"
        outside.write_text("outside\n", encoding="utf-8")
        common = dict(
            paths=paths,
            protocol_path=self.protocol,
            protocol_sha256=sha256_file(self.protocol),
            freeze_sha256=freeze.sha256,
            claim_sha256=claim.sha256,
            code_root=self.code_root,
            output_root=self.output_root,
        )
        with self.assertRaisesRegex(ValueError, "escapes"):
            create_completion_receipt(
                **common, output_artifacts={"outside": outside}
            )
        link = self.output_root / "link.csv"
        os.symlink(outside, link)
        with self.assertRaisesRegex(ValueError, "symlink"):
            create_completion_receipt(**common, output_artifacts={"link": link})

    def test_tampering_with_each_chain_layer_is_detected(self) -> None:
        # Decision-freeze tampering.
        paths, freeze, claim, _, _, _ = self._complete("tamper-freeze")
        os.chmod(paths.freeze_path, 0o644)
        paths.freeze_path.write_bytes(canonical_json_bytes({"tampered": True}))
        with self.assertRaises(GovernanceIntegrityError):
            verify_final_access_claim(
                paths,
                protocol_path=self.protocol,
                code_root=self.code_root,
                expected_freeze_sha256=freeze.sha256,
                expected_claim_sha256=claim.sha256,
                require_receipt_absent=False,
            )

        # Protocol tampering.
        paths, freeze, claim, _, _, _ = self._complete("tamper-protocol")
        self.protocol.write_bytes(canonical_json_bytes({"protocol": 2}))
        with self.assertRaisesRegex(GovernanceIntegrityError, "protocol"):
            verify_final_access_claim(
                paths,
                protocol_path=self.protocol,
                code_root=self.code_root,
                expected_freeze_sha256=freeze.sha256,
                expected_claim_sha256=claim.sha256,
                require_receipt_absent=False,
            )

        # Restore protocol and create another completed, independently anchored chain.
        self.protocol.write_bytes(canonical_json_bytes({"protocol": 1}))
        paths, freeze, claim, receipt, _, outputs = self._complete("tamper-output")
        next(iter(outputs.values())).write_bytes(b"changed output")
        with self.assertRaisesRegex(GovernanceIntegrityError, "output artifact changed"):
            verify_completion_receipt(
                paths,
                protocol_path=self.protocol,
                code_root=self.code_root,
                output_root=self.output_root,
                expected_freeze_sha256=freeze.sha256,
                expected_claim_sha256=claim.sha256,
                expected_receipt_sha256=receipt.sha256,
            )

        # Tamper with read-only claim bytes after explicitly changing permissions.
        paths, freeze, claim, _, _, _ = self._complete("tamper-claim")
        os.chmod(paths.claim_path, 0o644)
        paths.claim_path.write_bytes(canonical_json_bytes({"tampered": True}))
        with self.assertRaises(GovernanceIntegrityError):
            verify_final_access_claim(
                paths,
                protocol_path=self.protocol,
                code_root=self.code_root,
                expected_freeze_sha256=freeze.sha256,
                expected_claim_sha256=claim.sha256,
                require_receipt_absent=False,
            )

        paths, freeze, claim, receipt, _, _ = self._complete("tamper-receipt")
        os.chmod(paths.receipt_path, 0o644)
        paths.receipt_path.write_bytes(canonical_json_bytes({"tampered": True}))
        with self.assertRaises(GovernanceIntegrityError):
            verify_completion_receipt(
                paths,
                protocol_path=self.protocol,
                code_root=self.code_root,
                output_root=self.output_root,
                expected_freeze_sha256=freeze.sha256,
                expected_claim_sha256=claim.sha256,
                expected_receipt_sha256=receipt.sha256,
            )


if __name__ == "__main__":
    unittest.main()
