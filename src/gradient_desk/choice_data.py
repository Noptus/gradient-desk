"""Stable answer menus and split-safe preparation for conditional QLoRA."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import median
from typing import Iterable

from .dataset import METHODS, read_jsonl, write_jsonl
from .traces import ACTIONS, oracle_label

CODES = "ABCDEFGH"
INTERVENTIONS = {
    "resume": "Continue training with the current optimizer settings.",
    "lower_lr": "Reduce the learning rate to one quarter of its current value.",
    "clip_grad": "Enable global gradient-norm clipping at 0.5.",
    "accumulate": "Average consecutive pairs of microbatches, halving optimizer update count.",
}


def _choice_ids(row: dict) -> list[str]:
    choices = row.get("choices")
    if not isinstance(choices, dict) or not 2 <= len(choices) <= len(CODES):
        raise ValueError("Expected between two and eight answer choices")
    if not all(
        isinstance(key, str) and key.strip() == key and key
        and isinstance(text, str) and text.strip()
        for key, text in choices.items()
    ):
        raise ValueError("Choice IDs and text must be strings")
    if len({text.strip() for text in choices.values()}) != len(choices):
        raise ValueError("Answer choices must have distinct text")
    return sorted(choices)


def layout(row: dict, variant: int = 0) -> tuple[str, list[str]]:
    """Return a prompt and the canonical IDs in positional answer order.

    The case ID seeds a shuffle. Subsequent variants rotate that exact order,
    so no class repeats its position until all positions are visited. The
    ground truth and probability target are never read here.
    """
    order = _choice_ids(row)
    if type(variant) is not int or variant < 0:
        raise ValueError("Variant must be a nonnegative integer")
    key = row.get("case_id")
    prompt = row.get("prompt")
    if not isinstance(key, str) or not key or not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Expected a nonempty case ID and prompt")
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest(), "big")
    random.Random(seed).shuffle(order)
    offset = variant % len(order)
    order = order[offset:] + order[:offset]
    menu = "\n".join(f"{code}. {row['choices'][name]}" for code, name in zip(CODES, order))
    return (
        f"{prompt.strip()}\n\nChoose the best answer. Reply with its letter only.\n\n{menu}",
        order,
    )


def target_distribution(row: dict, order: list[str]) -> list[float]:
    """Supervise the model with either a hard reference or a soft action target."""
    label = row.get("label")
    allowed = set(_choice_ids(row))
    if label not in allowed or len(order) != len(allowed) or set(order) != allowed:
        raise ValueError("Label and order must match the declared choices")
    values = row.get("target_dist")
    if values is None:
        return [float(name == label) for name in order]
    if not isinstance(values, dict) or set(values) != allowed:
        raise ValueError("Soft target must cover every choice")
    if any(type(value) not in (int, float) or value < 0 or not math.isfinite(value)
           for value in values.values()):
        raise ValueError("Soft target values must be nonnegative and finite")
    norm = math.fsum(values.values())
    if norm <= 0:
        raise ValueError("Soft targets must have positive total mass")
    return [float(values[name]) / norm for name in order]


def normalized_action_losses(row: dict) -> dict[str, float]:
    losses = {action: float(row["actions"][action]["validation_loss"]) for action in ACTIONS}
    if not all(math.isfinite(loss) for loss in losses.values()):
        raise ValueError("Counterfactual losses must be finite")
    low, high = min(losses.values()), max(losses.values())
    if low == high:
        return dict.fromkeys(ACTIONS, 0.0)
    return {action: (loss - low) / (high - low) for action, loss in losses.items()}


def _ratio(value: float, reference: float) -> float:
    return value / max(abs(reference), 1e-8)


def _numbers(values: Iterable[float]) -> str:
    return ", ".join(f"{float(value):.3g}" for value in values)


def compact_intervention(telemetry: dict) -> str:
    """Base measurements on observed checkpoint history, including units."""
    gradients = list(telemetry["gradient_norm_history"])
    training = list(telemetry["train_loss_history"])
    validation = [point["loss"] for point in telemetry["validation_loss_history"]]
    if not gradients or not training or not validation:
        raise ValueError("Telemetry trails cannot be empty")
    for group in (gradients, training, validation):
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in group):
            raise ValueError("Telemetry trails must be finite numbers")
    optimizer = telemetry.get("optimizer", "unknown")
    momentum = telemetry.get("momentum")
    batch = telemetry.get("microbatch_size", telemetry.get("batch_size"))
    return (
        "Select a training intervention for this checkpoint. Minimize loss on later evaluation data. "
        "Each action uses the same next eight microbatches.\n"
        f"Loss: {telemetry['objective']}; optimizer: {optimizer}; momentum: {momentum}; "
        f"learning rate: {telemetry['lr']:.3g}; microbatch: {batch}.\n"
        f"Monitoring loss, early to late: {_numbers(validation)}.\n"
        f"Last four training losses: {_numbers(training[-4:])}; "
        f"recent median / earlier median: {_ratio(median(training[-4:]), median(training[:4])):.3g}.\n"
        f"Last four gradient norms: {_numbers(gradients[-4:])}; "
        f"recent maximum / earlier median: {_ratio(max(gradients[-4:]), median(gradients[:4])):.3g}."
    )


def adapt_legacy(row: dict) -> dict:
    """Translate the original action/review schema without changing any labels."""
    if "choices" in row:
        return row
    original = row["prompt"]
    if row["track"] == "training":
        info = row.get("telemetry") or json.loads(original.split("\nTelemetry: ", 1)[1])
        normal = {
            **row,
            "prompt": compact_intervention(info),
            "legacy_prompt": original,
            "telemetry": info,
            "choices": INTERVENTIONS,
        }
        if row.get("split") == "train":
            regret = normalized_action_losses(row)
            mass = {action: math.exp(-value / 0.18) for action, value in regret.items()}
            total = math.fsum(mass.values())
            normal["target_dist"] = {
                action: 0.5 * float(action == row["label"]) + 0.5 * value / total
                for action, value in mass.items()
            }
        return normal
    if row["track"] != "methodology":
        raise ValueError(f"No answer menu for legacy track {row['track']!r}")
    delimiter = "Case: "
    if delimiter not in original:
        raise ValueError("Legacy methodology prompt has no case")
    return {
        **row,
        "prompt": "Identify the highest-priority next step for investment research.\n\n"
        + original.split(delimiter, 1)[1],
        "legacy_prompt": original,
        "choices": METHODS,
    }


def validate(rows: list[dict]) -> dict:
    """Validate identity, exact prompt and declared source-group boundaries."""
    case_ids = set()
    prompt_split = {}
    group_split = {}
    counts: defaultdict = defaultdict(Counter)
    for row in rows:
        split = row.get("split")
        track = row.get("track")
        family = row.get("family")
        if (split not in ("train", "valid", "test") or not isinstance(track, str)
                or not track or not isinstance(family, str) or not family):
            raise ValueError("Cases need a train/valid/test split, track and family")
        _, order = layout(row)
        target_distribution(row, order)
        key = row["case_id"]
        if key in case_ids:
            raise ValueError(f"Duplicate case ID: {key}")
        case_ids.add(key)
        prompt_key = " ".join(row["prompt"].split())
        if prompt_split.setdefault(prompt_key, split) != split:
            raise ValueError(f"Prompt occurs across splits: {key}")
        if row.get("group_id") is not None:
            group = (track, str(row["group_id"]))
            if group_split.setdefault(group, split) != split:
                raise ValueError(f"Source group occurs across splits: {group}")
        if "actions" in row:
            normalized_action_losses(row)
            if row["label"] != oracle_label({action: result["validation_loss"]
                                              for action, result in row["actions"].items()}):
                raise ValueError(f"Measured action label differs from the near-tie oracle: {key}")
        counts[split][track] += 1
    return {split: dict(sorted(totals.items())) for split, totals in sorted(counts.items())}


def merge(paths: list[Path], output: Path) -> dict:
    """Assemble hashed, normalized splits without implicit test sampling."""
    raw = [(path, path.read_bytes()) for path in paths]
    rows = [adapt_legacy(json.loads(line)) for _, payload in raw
            for line in payload.splitlines() if line.strip()]
    counts = validate(rows)
    if not {"train", "valid"} <= set(counts):
        raise ValueError("Preparation needs both training and validation cases")
    output.mkdir(parents=True, exist_ok=False)
    for split in counts:
        write_jsonl(output / f"{split}.jsonl", [row for row in rows if row["split"] == split])
    receipt = {
        "schema": 1,
        "counts": counts,
        "inputs": [{"name": path.name, "sha256": hashlib.sha256(payload).hexdigest()}
                   for path, payload in raw],
        "outputs": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(output.glob("*.jsonl"))},
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def freeze_test(paths: list[Path], campaign: Path, output: Path) -> dict:
    """Freeze test prompts and comparison endpoints against the campaign splits."""
    train = read_jsonl(campaign / "train.jsonl")
    validation = read_jsonl(campaign / "valid.jsonl")
    if {row["split"] for row in train} != {"train"} or {row["split"] for row in validation} != {"valid"}:
        raise ValueError("Campaign splits are absent or mislabeled")
    tests = [adapt_legacy(row) for path in paths for row in read_jsonl(path)]
    if {row["split"] for row in tests} != {"test"}:
        raise ValueError("Freeze requires exclusively test cases")
    validate(train + validation + tests)
    fields = ("case_id", "track", "family", "split", "prompt", "label", "choices",
              "benchmark", "evaluation_block", "group_id", "legacy_prompt")
    clean = []
    for row in sorted(tests, key=lambda row: row["case_id"]):
        projected = {name: row[name] for name in fields if name in row}
        if "actions" in row:
            projected["actions"] = {name: {"validation_loss": result["validation_loss"]}
                                    for name, result in row["actions"].items()}
        clean.append(projected)
    counts = Counter(row.get("benchmark", row["track"]) for row in clean)
    blocks = defaultdict(set)
    for row in clean:
        blocks[row.get("benchmark", row["track"])].add(
            row.get("evaluation_block", row.get("group_id", row["case_id"])))
    output.mkdir(parents=True, exist_ok=False)
    test_path = output / "test.jsonl"
    write_jsonl(test_path, clean)
    receipt = {
        "schema": 1,
        "cases_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "benchmarks": {name: {"cases": counts[name], "blocks": len(blocks[name])}
                       for name in sorted(counts)},
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-test-against", type=Path, metavar="CAMPAIGN")
    cli = parser.parse_args()
    receipt = (freeze_test(cli.input, cli.freeze_test_against, cli.output)
               if cli.freeze_test_against else merge(cli.input, cli.output))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
