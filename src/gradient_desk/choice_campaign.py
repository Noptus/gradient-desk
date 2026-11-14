"""Select a checkpoint on development cases and score a frozen held-out suite."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import time

from .choice_data import normalized_action_losses, validate
from .choice_model import (
    EXPECTED_WEIGHTS_SHA256, _candidate_head, _encode_prompt, _file_sha256, _load, _score_rows,
    candidate_logits,
)
from .dataset import read_jsonl

# The pre-v2 "methodology" research-advice track is excluded from automatic
# selection panels unless the caller names benchmarks explicitly; it is kept for
# provenance only and is not part of the v2 choice benchmark suite.
LEGACY_TRACKS = frozenset({"methodology"})
MAX_PER_BENCHMARK = 32
SELECTION_SCHEMA = 2
EVALUATION_SCHEMA = 2
SHARED_SOURCES = {
    "treasury_curve_audit": "us_treasury_daily_par_curve",
    "treasury_intervention": "us_treasury_daily_par_curve",
}
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


class _FrozenPrefix:
    """Reuse unchanged Qwen activations across scores for identical token batches."""

    def __init__(self, model, config: dict):
        num_layers = config["num_layers"]
        if (model.model_type != "qwen3" or config["fine_tune_type"] != "lora"
                or config["lora_parameters"].get("keys") is not None
                or not 0 < num_layers < len(model.layers)):
            raise ValueError("Prefix caching requires Qwen3 and LoRA on its final layers only")
        self.boundary = len(model.layers) - num_layers
        self.cache = {}

    def __call__(self, model, batch_ids, lengths, candidate_matrix):
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        key = tuple(tuple(row) for row in batch_ids.tolist())
        hidden = self.cache.get(key)
        if hidden is None:
            hidden = model.model.embed_tokens(batch_ids)
            mask = create_attention_mask(hidden, None)
            for layer in model.layers[:self.boundary]:
                hidden = layer(hidden, mask, None)
            mx.eval(hidden)
            self.cache[key] = hidden
        mask = create_attention_mask(hidden, None)
        for layer in model.layers[self.boundary:]:
            hidden = layer(hidden, mask, None)
        return _candidate_head(model, model.model.norm(hidden), lengths, candidate_matrix)


def _verify_prefix_parity(model, tokenizer, rows, variants, batch_size, cached):
    """Compare both layouts on the shortest and longest development prompts."""
    ordered = sorted(rows, key=lambda row: len(_encode_prompt(tokenizer, row, 0)[0]))
    sample = [ordered[0]]
    if len(ordered) > 1:
        sample.append(ordered[-1])
    reference = _score_rows(model, tokenizer, sample, variants, batch_size, progress=False)
    reused = _score_rows(
        model, tokenizer, sample, variants, batch_size, progress=False, logits_fn=cached,
    )
    max_difference = 0.0
    for original, optimized in zip(reference, reused, strict=True):
        if original["prediction"] != optimized["prediction"]:
            raise RuntimeError("Frozen-prefix prediction failed parity against full Qwen inference")
        for label, probability in original["probabilities"].items():
            other = optimized["probabilities"][label]
            if not math.isclose(probability, other, rel_tol=1e-6, abs_tol=1e-7):
                raise RuntimeError("Frozen-prefix probabilities failed parity against full Qwen inference")
            max_difference = max(max_difference, abs(probability - other))
    return max_difference


# --------------------------------------------------------------------------- #
# Pure helpers (no MLX): selection set construction and panel utility.
# --------------------------------------------------------------------------- #
def _benchmark(row: dict) -> str:
    """Panel key, matching :func:`choice_report._panel`."""
    return row.get("benchmark", row["track"])


def _block(row: dict) -> str:
    """Use the declared dependence block, matching the report."""
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
    reuse_frozen_prefix: bool = False,
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
    logits_fn = candidate_logits
    prefix_parity = {}
    if reuse_frozen_prefix:
        logits_fn = _FrozenPrefix(model, json.loads(config_path.read_text(encoding="utf-8")))
        prefix_parity["base_max_probability_error"] = _verify_prefix_parity(
            model, tokenizer, blind_cases, variants, batch_size, logits_fn,
        )
        print("Verified frozen-prefix parity on the base model", flush=True)
    base_predictions = {rec["case_id"]: rec["prediction"] for rec in
                        _score_rows(model, tokenizer, blind_cases, variants, batch_size, logits_fn=logits_fn)}
    base_utility = panel_utilities(cases, base_predictions)

    _install_lora(model, config_path)

    scored = {}
    for step, path in checkpoints.items():
        _swap_lora_weights(model, path)
        if reuse_frozen_prefix and not scored:
            prefix_parity["adapted_max_probability_error"] = _verify_prefix_parity(
                model, tokenizer, blind_cases, variants, batch_size, logits_fn,
            )
            print("Verified frozen-prefix parity on the first trained checkpoint", flush=True)
        records = _score_rows(model, tokenizer, blind_cases, variants, batch_size, logits_fn=logits_fn)
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
        "frozen_prefix_parity": prefix_parity or None,
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
    base_weights = model_path / "model.safetensors"
    if not base_weights.is_file():
        raise FileNotFoundError(f"Base weights not found: {base_weights}")
    expected_base = selection["base_weights_sha256"]
    if _file_sha256(base_weights) != expected_base:
        raise ValueError("Base weights digest differs from the selection receipt")
    return step, checkpoint


def _prior_weights(path: Path, expected_sha: str, current_config: dict) -> tuple[Path, dict]:
    weights = path / "adapters.safetensors"
    config_path = path / "adapter_config.json"
    provenance_path = path / "provenance.json"
    if _file_sha256(weights) != expected_sha:
        raise ValueError("Prior weights differ from the declared release")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if any(config[key] != current_config[key] for key in ("fine_tune_type", "num_layers", "lora_parameters")):
        raise ValueError("Prior adapter layout differs from the selected checkpoint layout")
    if provenance.get("weights_sha256") != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("Prior adapter was fitted against a different base model")
    return weights, {
        "checkpoint_step": provenance["checkpoint_step"],
        "weights_sha256": expected_sha,
        "config_sha256": _file_sha256(config_path),
        "provenance_sha256": _file_sha256(provenance_path),
    }


def _infer_pair(model_path, config_path, checkpoint, rows, variants, batch_size,
                *, reuse_frozen_prefix, prior_adapter, expected_prior_sha256):
    import mlx.core as mx

    started = time.perf_counter()
    model, tokenizer = _load(model_path, None)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    scorer = _FrozenPrefix(model, config) if reuse_frozen_prefix else candidate_logits
    parity = {}
    timings = {"model_load": time.perf_counter() - started}

    def score(role):
        print(f"Scoring the {role} model", flush=True)
        then = time.perf_counter()
        if reuse_frozen_prefix:
            parity[role] = _verify_prefix_parity(model, tokenizer, rows, variants, batch_size, scorer)
            print(f"Verified frozen-prefix parity for {role}", flush=True)
        values = list(_score_rows(model, tokenizer, rows, variants, batch_size, logits_fn=scorer))
        timings[role] = time.perf_counter() - then
        return values

    base = score("base")
    _install_lora(model, config_path)
    _swap_lora_weights(model, checkpoint)
    tuned = score("tuned")
    prior, prior_meta = None, None
    if prior_adapter is not None:
        try:
            weights, prior_meta = _prior_weights(prior_adapter, expected_prior_sha256, config)
            _swap_lora_weights(model, weights)
            prior = score("prior")
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            prior_meta = {"status": "unavailable", "reason": str(exc)}
            print(f"Skipping prior diagnostic: {exc}", flush=True)
    return base, tuned, prior, prior_meta, timings, parity, mx.get_peak_memory()


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
    reuse_frozen_prefix: bool = False,
    prior_adapter: Path | None = None,
    expected_prior_sha256: str | None = None,
) -> dict:
    """Write complete paired results after verifying every primary input."""
    if variants < 1 or batch_size < 1:
        raise ValueError("variants and batch-size must be positive")
    if (prior_adapter is None) != (expected_prior_sha256 is None):
        raise ValueError("The prior adapter and its expected weight hash must be supplied together")

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

    output.mkdir(parents=True, exist_ok=False)
    blind_rows = [_blind(row) for row in rows]
    scorer_args = (model_path, adapter / "adapter_config.json", checkpoint, blind_rows, variants, batch_size)
    prior_args = {"prior_adapter": prior_adapter, "expected_prior_sha256": expected_prior_sha256}
    fallback = None
    try:
        scores = _infer_pair(*scorer_args, reuse_frozen_prefix=reuse_frozen_prefix, **prior_args)
    except (RuntimeError, ValueError) as exc:
        if not reuse_frozen_prefix:
            raise
        fallback = str(exc)
        print(f"Repeating both primary models using full forwards: {fallback}", flush=True)
    if fallback is not None:
        import mlx.core as mx

        gc.collect()
        mx.clear_cache()
        scores = _infer_pair(*scorer_args, reuse_frozen_prefix=False, **prior_args)
    base, tuned, prior, prior_meta, timings, parity, peak_memory = scores

    from .choice_report import compare

    report = compare(rows, base, tuned)
    report["improvement_sources"] = {
        status: sorted({SHARED_SOURCES.get(name, name) for name, panel in report["panels"].items()
                        if panel[f"{status}_improvement"]})
        for status in ("observed", "confirmed")
    }
    _write_predictions(output / "base.jsonl", base)
    _write_predictions(output / "tuned.jsonl", tuned)
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
    report["scoring"] = {
        "timing_seconds": timings,
        "variants": variants,
        "batch_size": batch_size,
        "frozen_prefix_max_probability_error": parity or None,
        "frozen_prefix_fallback": fallback,
        "peak_mlx_bytes": peak_memory,
    }
    if prior is not None:
        _write_predictions(output / "prior.jsonl", prior)
        previous = compare(rows, base, prior)["panels"]
        prior_meta.update({
            "status": "secondary_diagnostic",
            "predictions_sha256": _file_sha256(output / "prior.jsonl"),
            "panels": {name: {
                "primary": row["candidate"]["primary"],
                "accuracy": row["candidate"]["accuracy"],
                "gain_vs_base": row["gain"],
                "selected_gain_vs_prior": report["panels"][name]["gain"] - row["gain"],
            } for name, row in previous.items()},
        })
    if prior_meta is not None:
        report["prior_diagnostic"] = prior_meta
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
    sel.add_argument("--reuse-frozen-prefix", action="store_true",
                     help="Cache verified frozen Qwen-layer outputs across development checkpoint scores")

    ev = commands.add_parser("evaluate", help="Paired base/tuned evaluation on frozen test cases")
    ev.add_argument("--cases", type=Path, required=True)
    ev.add_argument("--model", type=Path, required=True)
    ev.add_argument("--adapter", type=Path, required=True)
    ev.add_argument("--selection", type=Path, required=True)
    ev.add_argument("--expected-cases-sha256", required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--variants", type=_positive_int, default=2)
    ev.add_argument("--batch-size", type=_positive_int, default=8)
    ev.add_argument("--reuse-frozen-prefix", action="store_true",
                    help="Reuse frozen Qwen3 layers after checking probability parity")
    ev.add_argument("--prior-adapter", type=Path, help="Optional previous release for a secondary diagnostic")
    ev.add_argument("--expected-prior-sha256", help="Published sha256 of the prior adapter tensors")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "select":
        receipt = select(
            args.validation, args.model, args.adapter, args.output,
            steps=args.steps, variants=args.variants, batch_size=args.batch_size,
            limit_per_benchmark=args.limit_per_benchmark, benchmarks=args.benchmarks,
            reuse_frozen_prefix=args.reuse_frozen_prefix,
        )
        print(json.dumps({"selected_step": receipt["selected_step"],
                          "output": str(args.output)}, indent=2))
    else:
        report = evaluate(
            args.cases, args.model, args.adapter, args.selection, args.output,
            expected_cases_sha256=args.expected_cases_sha256,
            variants=args.variants, batch_size=args.batch_size,
            reuse_frozen_prefix=args.reuse_frozen_prefix,
            prior_adapter=args.prior_adapter, expected_prior_sha256=args.expected_prior_sha256,
        )
        print(json.dumps({"improvement_sources": report["improvement_sources"],
                          "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
