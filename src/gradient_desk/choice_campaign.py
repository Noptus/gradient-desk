"""Select a checkpoint on development cases and score a frozen held-out suite."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

from .choice_data import normalized_action_losses, validate
from .choice_model import EXPECTED_WEIGHTS_SHA256, _file_sha256, _load, _score_rows
from .dataset import read_jsonl

# The pre-v2 "methodology" research-advice track is excluded from automatic
# selection panels unless the caller names benchmarks explicitly; it is kept for
# provenance only and is not part of the v2 choice benchmark suite.
LEGACY_TRACKS = frozenset({"methodology"})
MAX_PER_BENCHMARK = 32
SELECTION_SCHEMA = 2
EVALUATION_SCHEMA = 2

SCORE_FIELDS = ("case_id", "track", "family", "split", "prompt", "choices")


def _install_lora(model, config_path: Path) -> None:
    from mlx_lm.tuner.utils import linear_to_lora_layers

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["fine_tune_type"] != "lora":
        raise ValueError("Checkpoint campaign only supports LoRA adapters")
    model.freeze()
    linear_to_lora_layers(model, config["num_layers"], config["lora_parameters"])


def _swap_lora_weights(model, checkpoint: Path) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    expected = dict(tree_flatten(model.trainable_parameters()))
    weights = mx.load(str(checkpoint))
    if not expected or set(weights) != set(expected):
        raise ValueError(f"Adapter parameters differ from expected LoRA tensors: {checkpoint}")
    for key, parameter in expected.items():
        value = weights[key]
        if value.shape != parameter.shape or value.dtype != parameter.dtype:
            raise ValueError(f"Invalid adapter shape or dtype for {key}: {checkpoint}")
    model.load_weights(list(weights.items()), strict=False)
    model.eval()


# --------------------------------------------------------------------------- #
# Pure helpers (no MLX): selection set construction and panel utility.
# --------------------------------------------------------------------------- #
def _benchmark(row: dict) -> str:
    """Panel key, matching :func:`choice_report._panel`."""
    return row.get("benchmark", row["track"])


def _block(row: dict) -> str:
    """Independent dependence block, matching the report's grouping key."""
    return str(row.get("evaluation_block", row.get("group_id", row["case_id"])))


def _group_rank(case_id: str) -> int:
    """Stable, label-independent ordering key derived from the case ID only."""
    return int.from_bytes(hashlib.sha256(case_id.encode()).digest()[:8], "big")


def select_cases(rows: list[dict], *, limit: int = MAX_PER_BENCHMARK) -> list[dict]:
    """Pick up to ``limit`` validation cases per benchmark, diversifying blocks.

    Order blocks and cases by stable hashes of their IDs, then round-robin
    across blocks before taking a second case from any block.
    """
    if limit < 1:
        raise ValueError("limit-per-benchmark must be positive")
    chosen: list[dict] = []
    by_benchmark: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_benchmark[_benchmark(row)][_block(row)].append(row)
    for benchmark in sorted(by_benchmark):
        blocks = by_benchmark[benchmark]
        for cases in blocks.values():
            cases.sort(key=lambda row: (_group_rank(row["case_id"]), row["case_id"]))
        ordered_blocks = sorted(blocks, key=lambda block: (_group_rank(block), block))
        picked: list[dict] = []
        depth = 0
        while len(picked) < limit and depth < max(len(blocks[b]) for b in ordered_blocks):
            for block in ordered_blocks:
                if depth < len(blocks[block]) and len(picked) < limit:
                    picked.append(blocks[block][depth])
            depth += 1
        chosen.extend(picked)
    return chosen


def _blind(row: dict) -> dict:
    return {key: row[key] for key in SCORE_FIELDS}


def case_utility(row: dict, prediction: str) -> float:
    """Per-case utility in ``[0, 1]`` for the already-chosen ``prediction``.

    Label benchmarks score 1.0 for an exact match.  Measured-intervention
    benchmarks (rows carrying ``actions``) score ``1 - normalized_regret`` so
    that the best action scores 1.0 and the worst scores 0.0; ties across the
    loss range collapse to 1.0.  ``label`` is read here only to *measure* a
    prediction that was produced without it.
    """
    if "actions" in row:
        return 1.0 - normalized_action_losses(row)[prediction]
    return float(prediction == row["label"])


def panel_utilities(rows: list[dict], predictions: dict[str, str]) -> dict[str, float]:
    """Mean utility per benchmark panel for a mapping of case ID -> prediction."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        panel = _benchmark(row)
        sums[panel] += case_utility(row, predictions[row["case_id"]])
        counts[panel] += 1
    return {panel: sums[panel] / counts[panel] for panel in sums}


def _equal_weight_mean(panels: dict[str, float]) -> float:
    """Equal-weight mean over panels (not over cases)."""
    return sum(panels.values()) / len(panels)


def _require_split(rows: list[dict], split: str, what: str) -> None:
    present = {row["split"] for row in rows}
    if present != {split}:
        raise ValueError(f"{what} must contain only the {split!r} split; saw {sorted(present)}")


# --------------------------------------------------------------------------- #
# select: score listed checkpoints on validation, record the best step.
# --------------------------------------------------------------------------- #
def select(
    validation: Path,
    model_path: Path,
    adapter: Path,
    output: Path,
    *,
    steps: list[int],
    variants: int,
    batch_size: int,
    limit_per_benchmark: int = MAX_PER_BENCHMARK,
    benchmarks: list[str] | None = None,
) -> dict:
    """Score each checkpoint on validation and write a selection receipt.

    ``steps`` are evaluated with the model resident; the adapter's LoRA layers
    are attached once and each step overwrites only those tensors.  The chosen
    step maximises the equal-weighted mean of per-benchmark utility, breaking an
    exact tie toward the earliest step.
    """
    if not steps:
        raise ValueError("select needs one or more checkpoint steps")
    if len(set(steps)) != len(steps):
        raise ValueError("Duplicate checkpoint steps requested")
    if variants < 1 or batch_size < 1:
        raise ValueError("variants and batch-size must be positive")
    if output.exists():
        raise FileExistsError(f"Selection output already exists: {output}")

    rows = read_jsonl(validation)
    if not rows:
        raise ValueError("No validation cases supplied")
    _require_split(rows, "valid", "Validation cases")
    validate(rows)

    if benchmarks:
        wanted = set(benchmarks)
        rows = [row for row in rows if _benchmark(row) in wanted]
        missing = wanted - {_benchmark(row) for row in rows}
        if missing:
            raise ValueError(f"Requested benchmarks absent from validation: {sorted(missing)}")
    else:
        # Default suite excludes the legacy methodology track entirely.
        rows = [row for row in rows if row["track"] not in LEGACY_TRACKS]
        if not rows:
            raise ValueError("No non-legacy benchmarks in validation; pass --benchmark to opt in")

    cases = select_cases(rows, limit=limit_per_benchmark)
    cases.sort(key=lambda row: row["case_id"])
    case_ids = [row["case_id"] for row in cases]
    blind_cases = [_blind(row) for row in cases]
    by_id = {row["case_id"]: row for row in cases}

    checkpoints = {}
    for step in steps:
        path = adapter / f"{step:07d}_adapters.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"No adapter checkpoint at step {step}: {path}")
        checkpoints[step] = path

    validation_sha = _file_sha256(validation)
    config_path = adapter / "adapter_config.json"
    config_sha = _file_sha256(config_path)
    base_sha = _file_sha256(model_path / "model.safetensors")
    if base_sha != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("Base checkpoint digest does not match the pinned Qwen snapshot")
    provenance = json.loads((adapter / "provenance.json").read_text(encoding="utf-8"))
    if (provenance["weights_sha256"] != base_sha or provenance["config_sha256"] != config_sha
            or provenance["dataset_sha256"]["valid"] != validation_sha):
        raise ValueError("Selection inputs do not match the adapter's training receipt")

    started = time.monotonic()
    print("Scoring development cases with the frozen base", flush=True)
    model, tokenizer = _load(model_path, None)
    base_predictions = {rec["case_id"]: rec["prediction"] for rec in
                        _score_rows(model, tokenizer, blind_cases, variants, batch_size)}
    base_utility = panel_utilities(cases, base_predictions)

    _install_lora(model, config_path)

    scored = {}
    for step, path in checkpoints.items():
        _swap_lora_weights(model, path)
        records = list(_score_rows(model, tokenizer, blind_cases, variants, batch_size))
        predictions = {rec["case_id"]: rec["prediction"] for rec in records}
        panels = panel_utilities(cases, predictions)
        scored[step] = {
            "panel_utility": {panel: panels[panel] for panel in sorted(panels)},
            "mean_panel_utility": _equal_weight_mean(panels),
            "weights_sha256": _file_sha256(path),
        }
        print(f"step {step}: mean panel utility {scored[step]['mean_panel_utility']:.4f}", flush=True)

    # Highest equal-weight mean; earliest step wins an exact tie.
    chosen = min(steps, key=lambda step: (-scored[step]["mean_panel_utility"], step))

    receipt = {
        "schema": SELECTION_SCHEMA,
        "selected_step": chosen,
        "criterion": "equal-weighted mean per-benchmark utility; earliest step breaks exact ties",
        "variants": variants,
        "batch_size": batch_size,
        "limit_per_benchmark": limit_per_benchmark,
        "benchmarks": sorted({_benchmark(row) for row in cases}),
        "validation": validation.name,
        "validation_sha256": validation_sha,
        "adapter_config_sha256": config_sha,
        "base_weights_sha256": base_sha,
        "base_panel_utility": dict(sorted(base_utility.items())),
        "selected_weights_sha256": scored[chosen]["weights_sha256"],
        "case_ids": case_ids,
        "case_ids_sha256": hashlib.sha256("\n".join(case_ids).encode()).hexdigest(),
        "labels_per_benchmark": _labels_per_benchmark(by_id, case_ids),
        "steps": {str(step): scored[step] for step in steps},
        "elapsed_seconds": time.monotonic() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, allow_nan=False, sort_keys=True, indent=2) + "\n")
    print(f"Selected step {chosen}; receipt at {output}", flush=True)
    return receipt


def _labels_per_benchmark(by_id: dict[str, dict], case_ids: list[str]) -> dict[str, dict[str, int]]:
    """Report (not predict) the label mix per panel, for later auditing only."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for case_id in case_ids:
        row = by_id[case_id]
        counts[_benchmark(row)][row["label"]] += 1
    return {panel: dict(sorted(labels.items())) for panel, labels in sorted(counts.items())}


# --------------------------------------------------------------------------- #
# evaluate: verify the receipt, then paired base/tuned scoring on frozen test.
# --------------------------------------------------------------------------- #
def _verify_receipt(
    selection: dict, cases: Path, adapter: Path, model_path: Path,
    expected_cases_sha256: str,
) -> tuple[int, Path]:
    """Fail loudly unless the frozen bytes and checkpoint match the receipt.

    Returns the selected step and its checkpoint path.  All checks happen before
    any model weights are loaded so a mismatch never wastes a model load.
    """
    frozen_sha = _file_sha256(cases)
    if frozen_sha != expected_cases_sha256:
        raise ValueError(
            f"Frozen test bytes {frozen_sha} do not match --expected-cases-sha256 "
            f"{expected_cases_sha256}"
        )

    step = selection["selected_step"]
    checkpoint = adapter / f"{step:07d}_adapters.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint missing: {checkpoint}")
    # Exact checkpoint file hash must match the receipt's recorded selection.
    weights_sha = _file_sha256(checkpoint)
    if weights_sha != selection["selected_weights_sha256"]:
        raise ValueError("Selected checkpoint bytes differ from the selection receipt")
    config_sha = _file_sha256(adapter / "adapter_config.json")
    if config_sha != selection["adapter_config_sha256"]:
        raise ValueError("Adapter config digest differs from the selection receipt")
    # Base model: stat the pinned weights file and compare to the receipt's base
    # digest if the earlier receipt recorded one; avoid rehashing when not needed.
    base_weights = model_path / "model.safetensors"
    if not base_weights.is_file():
        raise FileNotFoundError(f"Base weights not found: {base_weights}")
    expected_base = selection["base_weights_sha256"]
    if _file_sha256(base_weights) != expected_base:
        raise ValueError("Base weights digest differs from the selection receipt")
    return step, checkpoint


def evaluate(
    cases: Path,
    model_path: Path,
    adapter: Path,
    selection: Path,
    output: Path,
    *,
    expected_cases_sha256: str,
    variants: int,
    batch_size: int,
) -> dict:
    """Score the unadapted base then the selected tuned model, write a report.

    Verification (frozen bytes, checkpoint, config, base) runs first; only then
    is the base loaded and scored, followed by a freshly-adapter-loaded tuned
    model.  Both runs use identical prompts, deterministic permutation variants
    and batch size.  Output goes into a brand-new directory only.
    """
    if variants < 1 or batch_size < 1:
        raise ValueError("variants and batch-size must be positive")

    receipt = json.loads(selection.read_text(encoding="utf-8"))
    if receipt.get("schema") != SELECTION_SCHEMA or receipt.get("variants") != variants:
        raise ValueError("Test option variants must match the selection receipt")
    step, checkpoint = _verify_receipt(
        receipt, cases, adapter, model_path, expected_cases_sha256
    )

    rows = read_jsonl(cases)
    if not rows:
        raise ValueError("No evaluation cases supplied")
    _require_split(rows, "test", "Evaluation cases")
    validate(rows)
    shared = set(receipt.get("case_ids", ())) & {row["case_id"] for row in rows}
    if shared:
        raise ValueError(f"Test cases reuse validation IDs: {sorted(shared)}")

    output.mkdir(parents=True, exist_ok=False)  # new directory exclusively
    blind_rows = [_blind(row) for row in rows]

    # Base first: no adapter installed, so the base graph is never perturbed.
    print("Scoring frozen unadapted base", flush=True)
    base_started = time.perf_counter()
    base_model, tokenizer = _load(model_path, None)
    base = list(_score_rows(base_model, tokenizer, blind_rows, variants, batch_size))
    base_seconds = time.perf_counter() - base_started
    _write_predictions(output / "base.jsonl", base)

    # Install the selected checkpoint after all base predictions are fixed.
    print(f"Scoring selected tuned model at step {step}", flush=True)
    tuned_started = time.perf_counter()
    _install_lora(base_model, adapter / "adapter_config.json")
    _swap_lora_weights(base_model, checkpoint)
    tuned = list(_score_rows(base_model, tokenizer, blind_rows, variants, batch_size))
    tuned_seconds = time.perf_counter() - tuned_started
    _write_predictions(output / "tuned.jsonl", tuned)

    from .choice_report import compare

    report = compare(rows, base, tuned)
    report["selection"] = {
        "selected_step": step,
        "selected_weights_sha256": receipt["selected_weights_sha256"],
        "adapter_config_sha256": receipt["adapter_config_sha256"],
    }
    report["checksums"] = {
        "cases_sha256": expected_cases_sha256,
        "base_predictions_sha256": _file_sha256(output / "base.jsonl"),
        "tuned_predictions_sha256": _file_sha256(output / "tuned.jsonl"),
        "selection_sha256": _file_sha256(selection),
    }
    report["timing_seconds"] = {
        "base": base_seconds, "tuned": tuned_seconds, "variants": variants,
        "batch_size": batch_size,
    }
    report["schema_evaluation"] = EVALUATION_SCHEMA
    with (output / "paired-report.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, allow_nan=False, sort_keys=True, indent=2) + "\n")
    print(f"Paired report written to {output / 'paired-report.json'}", flush=True)
    return report


def _write_predictions(path: Path, records: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    sel = commands.add_parser("select", help="Score checkpoints on validation and pick a step")
    sel.add_argument("--validation", type=Path, required=True)
    sel.add_argument("--model", type=Path, required=True)
    sel.add_argument("--adapter", type=Path, required=True)
    sel.add_argument("--steps", type=_positive_int, nargs="+", default=[60, 120, 180, 240])
    sel.add_argument("--output", type=Path, required=True)
    sel.add_argument("--limit-per-benchmark", type=_positive_int, default=MAX_PER_BENCHMARK)
    sel.add_argument("--variants", type=_positive_int, default=2)
    sel.add_argument("--batch-size", type=_positive_int, default=8)
    sel.add_argument("--benchmark", dest="benchmarks", action="append",
                     help="Opt in explicit benchmark names (overrides the legacy-track exclusion)")

    ev = commands.add_parser("evaluate", help="Paired base/tuned evaluation on frozen test cases")
    ev.add_argument("--cases", type=Path, required=True)
    ev.add_argument("--model", type=Path, required=True)
    ev.add_argument("--adapter", type=Path, required=True)
    ev.add_argument("--selection", type=Path, required=True)
    ev.add_argument("--expected-cases-sha256", required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--variants", type=_positive_int, default=2)
    ev.add_argument("--batch-size", type=_positive_int, default=8)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "select":
        receipt = select(
            args.validation, args.model, args.adapter, args.output,
            steps=args.steps, variants=args.variants, batch_size=args.batch_size,
            limit_per_benchmark=args.limit_per_benchmark, benchmarks=args.benchmarks,
        )
        print(json.dumps({"selected_step": receipt["selected_step"],
                          "output": str(args.output)}, indent=2))
    else:
        report = evaluate(
            args.cases, args.model, args.adapter, args.selection, args.output,
            expected_cases_sha256=args.expected_cases_sha256,
            variants=args.variants, batch_size=args.batch_size,
        )
        print(json.dumps({"confirmed_improvements": report["confirmed_improvements"],
                          "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
