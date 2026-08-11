from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from daxalgo_numeric_transformer.chmt_q.data.dataset import (
    DatasetCapabilities,
    DatasetManifest,
    DatasetShard,
)
from daxalgo_numeric_transformer.chmt_q.data.research_authorization import (
    COMPILER_REPORT_NAME,
    LocalResearchAuthorization,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
AUTHORIZED_AT = datetime(2026, 7, 25, 10, 30, tzinfo=UTC)


def _manifest(*, instrument_id: str = "BTC-USDT") -> DatasetManifest:
    capabilities = DatasetCapabilities(
        training_ready=False,
        rights_confirmed=False,
        provenance_complete=False,
        readiness_evidence_sha256=HASH_A,
        rights_evidence_sha256=HASH_B,
        provenance_evidence_sha256=HASH_C,
    )
    return DatasetManifest(
        instrument_id=instrument_id,
        source_manifest_sha256=HASH_D,
        builder_sha256=HASH_A,
        capabilities=capabilities,
        row_count=1,
        shards=(
            DatasetShard(
                path="shards/part-00000.arrow",
                byte_size=1,
                row_count=1,
                sha256=HASH_B,
                schema_sha256=HASH_C,
                first_example_id="example-001",
                last_example_id="example-001",
                first_decision_ns=1,
                last_decision_ns=1,
            ),
        ),
        arrow_schema_sha256=HASH_C,
    )


def _write_report(root: Path, manifest: DatasetManifest) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "status": "compiled_non_promotable",
        "dataset_id": manifest.dataset_id,
        "dataset_rows": manifest.row_count,
        "instrument_id": manifest.instrument_id,
        "source_manifest_sha256": manifest.source_manifest_sha256,
        "capabilities": manifest.capabilities.to_payload(),
    }
    path = root / COMPILER_REPORT_NAME
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _create(root: Path, manifest: DatasetManifest) -> LocalResearchAuthorization:
    return LocalResearchAuthorization.create(
        manifest,
        root,
        authorizer="Dhruv",
        authorized_at_utc=AUTHORIZED_AT,
        reason="Explicit local CHMT-Q development research.",
        rights_confirmed=True,
        incomplete_readiness_acknowledged=True,
        incomplete_provenance_acknowledged=True,
    )


def test_create_binds_dataset_capabilities_report_and_fixed_scope(tmp_path: Path) -> None:
    manifest = _manifest()
    report_path = _write_report(tmp_path, manifest)

    authorization = _create(tmp_path, manifest)

    assert authorization.dataset_id == manifest.dataset_id
    assert authorization.instrument_id == manifest.instrument_id
    assert authorization.source_manifest_sha256 == manifest.source_manifest_sha256
    assert (
        authorization.compiler_report_sha256 == hashlib.sha256(report_path.read_bytes()).hexdigest()
    )
    assert authorization.local_development_training_allowed is True
    assert authorization.local_development_validation_allowed is True
    assert authorization.promotion_allowed is False
    assert authorization.export_allowed is False
    assert authorization.serving_allowed is False
    assert authorization.outer_holdout_allowed is False
    assert len(authorization.authorization_sha256) == 64
    assert len(authorization.research_dataset_id) == 64
    assert authorization.research_dataset_id != authorization.dataset_id
    authorization.validate_against(manifest, tmp_path)


def test_canonical_round_trip_and_content_addressing(tmp_path: Path) -> None:
    manifest = _manifest()
    _write_report(tmp_path, manifest)
    authorization = _create(tmp_path, manifest)

    serialized = authorization.to_json()
    restored = LocalResearchAuthorization.from_json(serialized)

    assert restored == authorization
    assert restored.to_json() == serialized
    changed = replace(authorization, reason="A different local research reason.")
    assert changed.authorization_sha256 != authorization.authorization_sha256
    assert changed.research_dataset_id != authorization.research_dataset_id

    with pytest.raises(ValueError, match="canonical"):
        LocalResearchAuthorization.from_json(serialized + "\n")
    with pytest.raises(ValueError, match="authorization SHA-256"):
        LocalResearchAuthorization.from_json(serialized.replace("Explicit local", "Tampered local"))
    duplicate = serialized.replace(
        '"authorization_sha256":',
        '"authorization_sha256":"' + authorization.authorization_sha256 + '",'
        '"authorization_sha256":',
        1,
    )
    with pytest.raises(ValueError, match="duplicate JSON field"):
        LocalResearchAuthorization.from_json(duplicate)


@pytest.mark.parametrize(
    "field",
    [
        "rights_confirmed",
        "incomplete_readiness_acknowledged",
        "incomplete_provenance_acknowledged",
    ],
)
def test_create_requires_explicit_confirmations(tmp_path: Path, field: str) -> None:
    manifest = _manifest()
    _write_report(tmp_path, manifest)
    values = {
        "rights_confirmed": True,
        "incomplete_readiness_acknowledged": True,
        "incomplete_provenance_acknowledged": True,
    }
    values[field] = False

    with pytest.raises(ValueError, match=field):
        LocalResearchAuthorization.create(
            manifest,
            tmp_path,
            authorizer="Dhruv",
            authorized_at_utc=AUTHORIZED_AT,
            reason="Explicit local CHMT-Q development research.",
            **values,
        )


def test_timestamp_must_be_canonical_utc(tmp_path: Path) -> None:
    manifest = _manifest()
    _write_report(tmp_path, manifest)
    non_utc = datetime(2026, 7, 25, 11, 30, tzinfo=timezone(timedelta(hours=1)))

    with pytest.raises(ValueError, match="must use UTC"):
        LocalResearchAuthorization.create(
            manifest,
            tmp_path,
            authorizer="Dhruv",
            authorized_at_utc=non_utc,
            reason="Explicit local CHMT-Q development research.",
            rights_confirmed=True,
            incomplete_readiness_acknowledged=True,
            incomplete_provenance_acknowledged=True,
        )


def test_validation_rejects_manifest_and_report_drift(tmp_path: Path) -> None:
    manifest = _manifest()
    report_path = _write_report(tmp_path, manifest)
    authorization = _create(tmp_path, manifest)

    changed_manifest = _manifest(instrument_id="ETH-USDT")
    with pytest.raises(ValueError, match="dataset_id"):
        authorization.validate_against(changed_manifest, tmp_path)

    report_path.write_text(
        report_path.read_text(encoding="utf-8").replace(
            '"status": "compiled_non_promotable"',
            '"status": "changed"',
        ),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="compiler-report SHA-256"):
        authorization.validate_against(manifest, tmp_path)


def test_create_rejects_report_not_bound_to_manifest(tmp_path: Path) -> None:
    manifest = _manifest()
    report_path = _write_report(tmp_path, manifest)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["instrument_id"] = "ETH-USDT"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="compiler report instrument_id"):
        _create(tmp_path, manifest)


def test_atomic_write_new_and_regular_non_symlink_read(tmp_path: Path) -> None:
    manifest = _manifest()
    _write_report(tmp_path / "dataset", manifest)
    authorization = _create(tmp_path / "dataset", manifest)
    path = tmp_path / "authorization.json"

    assert authorization.write_new(path) == path.absolute()
    assert path.read_text(encoding="ascii") == authorization.to_json()
    assert LocalResearchAuthorization.read(path) == authorization
    with pytest.raises(FileExistsError, match="already exists"):
        authorization.write_new(path)
    with pytest.raises(ValueError, match="regular file"):
        LocalResearchAuthorization.read(tmp_path)


def test_read_rejects_symlink(tmp_path: Path) -> None:
    manifest = _manifest()
    _write_report(tmp_path / "dataset", manifest)
    authorization = _create(tmp_path / "dataset", manifest)
    target = authorization.write_new(tmp_path / "authorization.json")
    link = tmp_path / "authorization-link.json"
    try:
        os.symlink(target, link)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(ValueError, match="symbolic link|reparse point"):
        LocalResearchAuthorization.read(link)
