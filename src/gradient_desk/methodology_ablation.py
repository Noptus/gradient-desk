"""Replay exactly the methodology examples seen by mixed QLoRA step 120."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .dataset import read_jsonl, write_jsonl
from .evaluate import _load_model, _predictions, evaluate


SEED = 425
MIXED_STEP = 120
EXPECTED_EXPOSURES = 49


def _batch_order(rows):
    """Read the actual order yielded by the pinned MLX trainer at batch size 1."""
    import mlx.core as mx
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import iterate_batches
    import numpy as np

    class Indexed:
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            return {"prompt": str(index + 1), "completion": "token"}

        def process(self, row):
            return [int(row["prompt"]), 0], 0

    random = np.random.get_state()
    device = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        np.random.seed(SEED)
        batches = iterate_batches(CacheDataset(Indexed()), batch_size=1, max_seq_length=32)
        return [int(tokens[0, 0].item()) - 1 for tokens, _ in batches]
    finally:
        mx.set_default_device(device)
        np.random.set_state(random)


def prepare(mixed: Path, output: Path) -> dict:
    rows = read_jsonl(mixed / "train.jsonl")
    train_cases = read_jsonl(mixed / "train-cases.jsonl")
    valid = [r for r in read_jsonl(mixed / "valid-cases.jsonl") if r["track"] == "methodology"]
    if len(rows) != 240 or len(train_cases) != 176 or len(valid) != 16:
        raise ValueError("Expected the exact reference mixed training/validation corpus")
    if any(r["split"] != "train" for r in train_cases) or any(r["split"] != "valid" for r in valid):
        raise ValueError("Methodology ablation inputs have conflicting split membership")
    cases = {case["prompt"]: case for case in train_cases if case["track"] == "methodology"}
    if len(cases) != 32:
        raise ValueError("Expected 32 distinct methodology training cases")
    if any(set(row) != {"prompt", "completion"} for row in rows):
        raise ValueError("Unexpected training record format")
    mixed_order = _batch_order(rows)[:MIXED_STEP]
    matches = [(position, rows[index]) for position, index in enumerate(mixed_order, 1)
               if rows[index]["prompt"] in cases]
    if len(matches) != EXPECTED_EXPOSURES:
        raise ValueError(f"Expected {EXPECTED_EXPOSURES} methodology exposures, saw {len(matches)}")
    reference = [row for _, row in matches]
    method_order = _batch_order(reference)
    ablated = [None] * len(reference)
    for index, row in zip(method_order, reference, strict=True):
        ablated[index] = row
    if [ablated[i] for i in _batch_order(ablated)] != reference:
        raise RuntimeError("The ablation loader changed the methodology sample sequence")
    order_receipt = [{"mixed_step": step, "case_id": cases[row["prompt"]]["case_id"]}
                     for step, row in matches]
    receipt = {
        "seed": SEED,
        "mixed_step": MIXED_STEP,
        "methodology_updates": len(reference),
        "removed_neural_updates": MIXED_STEP - len(reference),
        "methodology_order": order_receipt,
        "order_sha256": hashlib.sha256(
            json.dumps(order_receipt, sort_keys=True, separators=(",", ":")).encode(),
        ).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train.jsonl", ablated)
    write_jsonl(output / "valid.jsonl", [
        {"prompt": row["prompt"], "completion": row["label"]} for row in valid
    ])
    (output / "exposure-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def _method_panel(source: Path, target: Path) -> Path:
    rows = [row for row in read_jsonl(source) if row["track"] == "methodology"]
    if not rows or any(row["split"] != "test" for row in rows):
        raise ValueError(f"Expected held-out methodology cases in {source}")
    write_jsonl(target, rows)
    return target


def score(model_path: Path, mixed: Path, ablation: Path, main_report: Path, panels: list[Path], output: Path) -> None:
    from mlx_lm.tuner.utils import load_adapters

    from .compare import write_comparison
    from .evaluate import _adapter_swaps, _swap_modules

    if not panels:
        raise ValueError("Provide the primary synthetic panel followed by external panels")
    if not (main_report / "checkpoint-selection.json").is_file():
        raise ValueError("Complete the reference run before evaluating the secondary ablation")
    batch = json.loads((main_report / "batch-parity.json").read_text(encoding="utf-8"))["used"]
    selected = json.loads((main_report / "checkpoint-selection.json").read_text(encoding="utf-8"))["selected_step"]
    model, tokenizer = _load_model(model_path, None)
    topology = dict(model.named_modules())

    def predict(_model, _adapter, rows):
        return _predictions(model, tokenizer, rows, progress=True, batch_size=batch)

    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, panel in enumerate(panels):
        panel_dir = output / panel.stem
        panel_dir.mkdir(parents=True, exist_ok=True)
        reference_dir = main_report if index == 0 else main_report / "external" / panel.stem
        reports.append((panel_dir, reference_dir,
                        _method_panel(panel, panel_dir / "case-methodology.jsonl")))
    model = load_adapters(model, str(mixed.resolve()))
    model.load_weights(str((mixed / f"{MIXED_STEP:07d}_adapters.safetensors").resolve()), strict=False)
    model.eval()
    for folder, _, cases in reports:
        evaluate(cases, folder / "mixed-step-120.jsonl", model_path, mixed, None, predict, batch_size=batch)
    original, _ = _adapter_swaps(model, topology)
    _swap_modules(model, original)
    model = load_adapters(model, str(ablation.resolve()))
    model.eval()
    for folder, reference, cases in reports:
        evaluate(cases, folder / "methodology-step-49.jsonl", model_path, ablation, None, predict, batch_size=batch)
        for name in ("base", "adapter"):
            saved = _method_panel(reference / f"{name}.jsonl", folder / f"primary-{name}.jsonl")
            write_comparison(saved, folder / "methodology-step-49.jsonl", folder / f"ablation-vs-{name}.json")
        write_comparison(folder / "mixed-step-120.jsonl", folder / "methodology-step-49.jsonl",
                         folder / "ablation-vs-mixed-step-120.json")
    (output / "reference-settings.json").write_text(
        json.dumps({"batch_size": batch, "primary_selected_step": selected, "matched_mixed_step": MIXED_STEP,
                    "methodology_step": EXPECTED_EXPOSURES}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--mixed-data", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)

    run = commands.add_parser("score")
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--mixed-adapter", type=Path, required=True)
    run.add_argument("--ablation-adapter", type=Path, required=True)
    run.add_argument("--main-report", type=Path, required=True)
    run.add_argument("--panel", type=Path, action="append", required=True)
    run.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.mixed_data, args.output), indent=2))
    else:
        score(args.model, args.mixed_adapter, args.ablation_adapter, args.main_report, args.panel, args.output)


if __name__ == "__main__":
    main()
