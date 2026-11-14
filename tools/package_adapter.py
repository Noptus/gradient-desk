"""Create a reproducible release archive for one verified MLX LoRA checkpoint."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct
import tarfile
import tempfile


SHA256 = re.compile(r"[a-f0-9]{64}")
REVISION = re.compile(r"[a-f0-9]{40}")
NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
MAX_WEIGHTS = 128 * 1024 * 1024


def read_regular(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError(f"Expected a regular file of at most {limit} bytes: {path.name}")
    with path.open("rb") as stream:
        blob = stream.read(limit + 1)
    if not 0 < len(blob) <= limit:
        raise ValueError(f"Unexpected file size: {path.name}")
    return blob


def json_bytes(data: dict) -> bytes:
    return (json.dumps(data, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def json_file(path: Path) -> dict:
    data = json.loads(read_regular(path, 100_000))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return data


def validated_weights(path: Path) -> bytes:
    contents = read_regular(path, MAX_WEIGHTS)
    if len(contents) < 10:
        raise ValueError("Truncated safetensors header")
    header_size = struct.unpack_from("<Q", contents)[0]
    if not 2 <= header_size <= min(100_000, len(contents) - 9):
        raise ValueError("Invalid safetensors header size")
    header = json.loads(contents[8:8 + header_size])
    if not isinstance(header, dict):
        raise ValueError("Expected a safetensors header object")
    tensors = [key for key in header if key != "__metadata__"]
    if not tensors or any(not key.endswith((".lora_a", ".lora_b")) for key in tensors):
        raise ValueError("Weights must contain only LoRA tensors")
    return contents


def public_provenance(metadata: dict, step: int) -> dict:
    fields = {"base_repo": REPOSITORY, "base_revision": REVISION,
              "weights_sha256": SHA256, "config_sha256": SHA256}
    out = {"checkpoint_step": step}
    for key, pattern in fields.items():
        value = metadata.get(key)
        if not isinstance(value, str) or not pattern.fullmatch(value):
            raise ValueError(f"Invalid provenance field: {key}")
        out[key] = value
    datasets = metadata.get("dataset_sha256")
    if not isinstance(datasets, dict) or set(datasets) != {"train", "valid"}:
        raise ValueError("Expected train and validation digests")
    if any(not isinstance(d, str) or not SHA256.fullmatch(d) for d in datasets.values()):
        raise ValueError("Invalid dataset digest")
    out["dataset_sha256"] = datasets
    return out


def safe_config(config: dict) -> dict:
    if config.get("fine_tune_type") != "lora" or type(config.get("num_layers")) is not int:
        raise ValueError("Expected an MLX LoRA configuration")
    parameters = config.get("lora_parameters")
    parameter_keys = {"rank", "scale", "dropout"}
    if not isinstance(parameters, dict) or not parameter_keys <= parameters.keys():
        raise ValueError("Missing LoRA parameters")
    for key in parameter_keys:
        if type(parameters[key]) not in (int, float):
            raise ValueError("LoRA parameters must be numeric")
    return {
        "fine_tune_type": "lora",
        "num_layers": config["num_layers"],
        "lora_parameters": {key: parameters[key] for key in parameter_keys},
    }


def package(
    adapter: Path,
    weights: Path,
    output: Path,
    *,
    name: str,
    step: int,
    license_dir: Path,
) -> dict:
    if not NAME.fullmatch(name) or step <= 0:
        raise ValueError("Choose a release-safe folder name and a positive checkpoint step")
    config = safe_config(json_file(adapter / "adapter_config.json"))
    info = public_provenance(json_file(adapter / "provenance.json"), step)
    files = {
        "LICENSE": read_regular(license_dir / "LICENSE", 100_000),
        "NOTICE": read_regular(license_dir / "NOTICE", 100_000),
        "adapter_config.json": json_bytes(config),
        "adapters.safetensors": validated_weights(weights),
        "provenance.json": json_bytes(info),
    }
    info = {"files": {key: hashlib.sha256(data).hexdigest()
                      for key, data in sorted(files.items())}}
    files["SHA256SUMS"] = "".join(
        f"{value}  {key}\n" for key, value in info["files"].items()
    ).encode()
    with tempfile.NamedTemporaryFile(dir=output.parent) as temporary:
        with gzip.GzipFile(fileobj=temporary, filename="", mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                for key, data in sorted(files.items()):
                    member = tarfile.TarInfo(f"{name}/{key}")
                    member.mode = 0o644
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
        temporary.flush()
        os.link(temporary.name, output)
    with output.open("rb") as stream:
        info["archive_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    return info


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("adapter", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--step", type=int, required=True)
    cli.add_argument("--weights", type=Path)
    cli.add_argument("--name", default="m4-pilot")
    args = cli.parse_args()
    try:
        result = package(args.adapter, args.weights or args.adapter / "adapters.safetensors",
                         args.output, name=args.name, step=args.step,
                         license_dir=Path(__file__).resolve().parents[1])
    except (OSError, ValueError) as error:
        cli.exit(2, f"package_adapter: {error}\n")
    print(json_bytes(result).decode(), end="")


if __name__ == "__main__":
    main()
