from __future__ import annotations

import json
from dataclasses import replace

import pytest

from daxalgo_numeric_transformer.chmt_q.config import ChmtQConfig
from daxalgo_numeric_transformer.chmt_q.experiment import (
    ExperimentConfig,
    source_tree_sha256,
)
from daxalgo_numeric_transformer.chmt_q.training import TrainingConfig

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        dataset_id=HASH_A,
        instrument_id="BTC-USD",
        split_manifest_sha256=HASH_B,
        scaler_artifact_sha256=HASH_E,
        source_tree_sha256=HASH_C,
        dependency_lock_sha256=HASH_D,
        fold_id="fold-01",
        seed=17,
        model=ChmtQConfig.tiny(),
        training=TrainingConfig(seed=17, micro_batch_size=2, accumulation_steps=1),
        max_epochs=3,
        early_stop_patience=2,
    )


def test_experiment_config_round_trip_identity_and_no_overwrite(tmp_path) -> None:
    config = _config()
    restored = ExperimentConfig.from_json(config.to_json())
    assert restored == config
    assert restored.run_id == restored.config_sha256
    assert restored.checkpoint_identity()["dataset_id"] == HASH_A
    assert restored.checkpoint_identity()["instrument_id"] == "BTC-USD"
    assert restored.checkpoint_identity()["scaler_artifact_sha256"] == HASH_E
    assert not restored.local_research_only
    assert restored.device == "cpu"
    assert replace(config, instrument_id="ETH-USD").run_id != config.run_id
    assert replace(config, scaler_artifact_sha256="f" * 64).run_id != config.run_id
    research = replace(config, local_research_authorization_sha256="f" * 64)
    assert research.local_research_only
    assert research.run_id != config.run_id
    assert research.checkpoint_identity()["local_research_authorization_sha256"] == "f" * 64

    path = config.write_new(tmp_path / "config.json")
    assert ExperimentConfig.from_json(path.read_text(encoding="utf-8")) == config
    with pytest.raises(FileExistsError):
        config.write_new(path)


def test_experiment_config_rejects_tampering_unknowns_and_seed_mismatch() -> None:
    config = _config()
    with pytest.raises(ValueError, match="fixed at 1"):
        replace(config, checkpoint_every_epochs=2)

    document = json.loads(config.to_json())
    document["payload"]["max_epochs"] = 4
    with pytest.raises(ValueError, match="SHA-256"):
        ExperimentConfig.from_json(json.dumps(document))

    document = json.loads(config.to_json())
    document["payload"]["training"]["typo"] = True
    with pytest.raises(ValueError, match="unknown training"):
        ExperimentConfig.from_json(json.dumps(document))

    with pytest.raises(ValueError, match="seed"):
        ExperimentConfig(
            dataset_id=HASH_A,
            instrument_id="BTC-USD",
            split_manifest_sha256=HASH_B,
            scaler_artifact_sha256=HASH_E,
            source_tree_sha256=HASH_C,
            dependency_lock_sha256=HASH_D,
            fold_id="fold-01",
            seed=17,
            model=ChmtQConfig.tiny(),
            training=TrainingConfig(seed=29),
        )

    document = json.loads(config.to_json())
    document["payload"]["max_epochs"] = True
    document["config_sha256"] = "0" * 64
    with pytest.raises(TypeError, match="max_epochs"):
        ExperimentConfig.from_json(json.dumps(document))

    with pytest.raises(ValueError, match="canonical"):
        ExperimentConfig.from_json(json.dumps(json.loads(config.to_json()), indent=2))


def test_source_tree_hash_is_content_and_relative_path_stable(tmp_path) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    for root in (first, second):
        package = root / "daxalgo_numeric_transformer"
        package.mkdir(parents=True)
        (package / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
        (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    assert source_tree_sha256(first) == source_tree_sha256(second)
    (second / "daxalgo_numeric_transformer" / "a.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert source_tree_sha256(first) != source_tree_sha256(second)


def test_source_tree_hash_excludes_virtual_environments_and_requires_package(
    tmp_path,
) -> None:
    root = tmp_path / "source"
    package = root / "daxalgo_numeric_transformer"
    package.mkdir(parents=True)
    (package / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    original = source_tree_sha256(root)

    site_packages = root / ".venv" / "Lib" / "site-packages" / "third_party"
    site_packages.mkdir(parents=True)
    (site_packages / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert source_tree_sha256(root) == original

    missing_package = tmp_path / "missing-package"
    missing_package.mkdir()
    (missing_package / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="daxalgo_numeric_transformer"):
        source_tree_sha256(missing_package)
