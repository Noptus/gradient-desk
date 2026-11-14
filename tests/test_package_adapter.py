import json
from pathlib import Path
import struct
import tarfile

import pytest

from tools.package_adapter import package


def make_files(path):
    (path / "LICENSE").write_text("Apache-2.0\n")
    (path / "NOTICE").write_text("Reference adapter\n")
    (path / "adapter_config.json").write_text(json.dumps(
        {"fine_tune_type": "lora", "num_layers": 8, "data": "/private/data",
         "lora_parameters": {"rank": 8, "scale": 16.0, "dropout": 0.0}}))
    (path / "provenance.json").write_text(json.dumps(
        {"machine": {"label": "private hostname"},
         "base_repo": "Qwen/Qwen3-8B-MLX-4bit", "base_revision": "a" * 40,
         "weights_sha256": "b" * 64, "config_sha256": "c" * 64,
         "dataset_sha256": {"train": "d" * 64, "valid": "e" * 64}}))
    header = json.dumps({"layers.0.lora_b": {"dtype": "F32", "shape": [1],
                                             "data_offsets": [0, 4]}}).encode()
    weight = path / "adapters.safetensors"
    weight.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.5))
    return weight


def test_package_is_deterministic_and_public_metadata_is_limited(tmp_path):
    weights = make_files(tmp_path)
    first, second = (tmp_path / name for name in ("one.tar.gz", "two.tar.gz"))
    args = {"name": "m4-pilot", "step": 180, "license_dir": tmp_path}
    assert package(tmp_path, weights, first, **args) == package(tmp_path, weights, second, **args)
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first) as archive:
        assert len(archive.getnames()) == 6
        assert all(member.isfile() and member.mtime == 0 for member in archive)
        metadata = json.load(archive.extractfile("m4-pilot/provenance.json"))
        assert metadata["checkpoint_step"] == 180
        assert "machine" not in metadata
        config = json.load(archive.extractfile("m4-pilot/adapter_config.json"))
        assert "data" not in config
    with pytest.raises(FileExistsError):
        package(tmp_path, weights, first, **args)


def test_package_rejects_truncated_weights(tmp_path):
    weights = make_files(tmp_path)
    weights.write_bytes(b"bad")
    with pytest.raises(ValueError, match="Truncated"):
        package(tmp_path, weights, tmp_path / "broken.tar.gz",
                name="m4-pilot", step=180, license_dir=tmp_path)
