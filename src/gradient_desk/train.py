"""Reproducible Apple Silicon QLoRA from a verified local checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace

from .dataset import MODEL_REPO, MODEL_REVISION, read_jsonl

EXPECTED_WEIGHTS_SHA256 = "fcc83d7537bee76cc143be3e25f8954b816798f93528103db1cddb02977c5617"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--data", type=Path)
    cli = parser.parse_args(argv)

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.lora import CONFIG_DEFAULTS, train_model
    from mlx_lm.tuner.datasets import CompletionsDataset
    import numpy as np
    import yaml

    class ActionDataset(CompletionsDataset):
        """Keep the supplied no-thinking response prefix out of the loss."""

        def process(self, row):
            messages = [{"role": "user", "content": row["prompt"]}]
            prefix = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=False,
            )
            tokens = self.tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": row["completion"]}],
                add_generation_prompt=False, enable_thinking=False,
            )
            if tokens[:len(prefix)] != prefix:
                raise ValueError("Training and inference chat templates yield different prompt prefixes")
            if len(tokens) > args.max_seq_length:
                raise ValueError(
                    f"Case exceeds max_seq_length: {len(tokens)} > {args.max_seq_length}. "
                    "Increase the limit or shorten the telemetry."
                )
            return tokens, len(prefix)

    config = yaml.safe_load(cli.config.read_text(encoding="utf-8"))
    overrides = {k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()
                 if k != "config" and v is not None}
    args = SimpleNamespace(**(CONFIG_DEFAULTS | config | overrides))
    model_path = Path(args.model)
    if args.fine_tune_type != "lora" or not args.mask_prompt or not args.train:
        parser.error("The pilot requires completion-masked LoRA training")
    if args.iters <= 0 or args.learning_rate <= 0:
        parser.error("Iterations and the learning rate must be positive")
    output = Path(args.adapter_path)
    if output.exists():
        parser.error(f"Adapter output already exists: {output}")
    if sha256(model_path / "model.safetensors") != EXPECTED_WEIGHTS_SHA256:
        parser.error("Base weight digest differs from the pinned Qwen checkpoint")
    if not mx.metal.is_available():
        parser.error("This trainer needs an Apple Silicon GPU with Metal")

    np.random.seed(args.seed)
    print(f"Loading {MODEL_REPO}@{MODEL_REVISION}", flush=True)
    model, tokenizer = load(str(model_path.resolve()), tokenizer_config={"trust_remote_code": False})

    files = {key: Path(args.data) / f"{key}.jsonl" for key in ("train", "valid")}
    datasets = {}
    for key, path in files.items():
        rows = read_jsonl(path)
        data = ActionDataset(rows, tokenizer, "prompt", "completion", True)
        sizes = [len(data.process(row)[0]) for row in rows]
        if not sizes:
            raise ValueError(f"No training examples in {path}")
        datasets[key] = data
        print(f"{key}: {len(sizes)} examples, max sequence {max(sizes)} tokens", flush=True)
    output.mkdir(parents=True)
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], check=True,
                          capture_output=True, text=True).stdout.strip()
    provenance = {
        "base_repo": MODEL_REPO,
        "base_revision": MODEL_REVISION,
        "weights_sha256": EXPECTED_WEIGHTS_SHA256,
        "machine": {"chip": chip, "platform": platform.platform()},
        "dataset_sha256": {key: sha256(path) for key, path in files.items()},
        "config_sha256": sha256(cli.config),
        "overrides": overrides,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    train_model(args, model, datasets["train"], datasets["valid"])


if __name__ == "__main__":
    main()
