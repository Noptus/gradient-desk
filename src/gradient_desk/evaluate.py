"""Paired base/adapter evaluation with explicit failures and intervention regret."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
import importlib.metadata
import json
import math
from pathlib import Path
import re
import time

from .dataset import METHODS, read_jsonl, write_jsonl
from .traces import ACTIONS


@dataclass(frozen=True)
class Prediction:
    label: str | None
    raw: str
    seconds: float


def parse_prediction(text: str, choices) -> str | None:
    candidate = text.strip()
    if len(candidate) > 128:
        return None
    if candidate.startswith("`") and candidate.endswith("`"):
        candidate = candidate[1:-1]
    return candidate if re.fullmatch("[a-z_]+", candidate) and candidate in choices else None


def _load_model(model_path: Path, adapter: Path | None):
    try:
        from mlx_lm import load
    except ImportError as exc:
        raise RuntimeError("Install the 'mlx' extra for Apple Silicon inference") from exc
    return load(
        str(model_path.resolve()),
        adapter_path=str(adapter.resolve()) if adapter else None,
        tokenizer_config={"trust_remote_code": False},
    )


def generate_predictions(
    model_path: Path, adapter: Path | None, rows: list[dict], *,
    progress: bool = True, batch_size: int = 1,
):
    model, tokenizer = _load_model(model_path, adapter)
    yield from _predictions(model, tokenizer, rows, progress=progress, batch_size=batch_size)


def _predictions(model, tokenizer, rows: list[dict], *, progress: bool, batch_size: int):
    import mlx.core as mx
    from mlx_lm import generate
    from mlx_lm.generate import batch_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)
    pending = iter(rows)
    done = 0

    while batch := list(islice(pending, batch_size)):
        started = time.perf_counter()
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                add_generation_prompt=True, enable_thinking=False,
            )
            for row in batch
        ]
        if len(batch) == 1:
            texts = [generate(model, tokenizer, prompts[0], max_tokens=16, sampler=sampler)]
        else:
            texts = batch_generate(
                model, tokenizer, prompts, max_tokens=16, sampler=sampler,
                prefill_batch_size=batch_size, completion_batch_size=batch_size,
                prefill_step_size=384,
            ).texts
        seconds = (time.perf_counter() - started) / len(batch)
        for row, raw in zip(batch, texts, strict=True):
            choices = ACTIONS if row["track"] == "training" else METHODS
            yield Prediction(parse_prediction(raw, choices), raw, seconds)
        done += len(batch)
        if progress and (done == len(batch) or done // 20 > (done - len(batch)) // 20 or done == len(rows)):
            print(f"Scored {done}/{len(rows)} cases, peak MLX memory {mx.get_peak_memory()/1e9:.2f} GB", flush=True)


def intervention_regret(row: dict, selected: str | None) -> dict:
    if row["track"] != "training":
        return {"relative_regret": None, "normalized_regret": None}
    losses = {action: result["validation_loss"] for action, result in row["actions"].items()}
    best, worst = min(losses.values()), max(losses.values())
    # For regret scoring, an invalid response keeps the existing configuration.
    regret = losses[selected or "resume"] - best
    return {
        "relative_regret": regret / max(abs(best), 1e-12),
        "normalized_regret": regret / (worst - best) if worst > best else 0.0,
    }


def policy_rules(row: dict) -> str | None:
    if row["track"] != "training":
        return None
    raw = row["prompt"].split("\nTelemetry: ", maxsplit=1)
    if len(raw) != 2:
        return "resume"
    info = row.get("telemetry") or json.loads(raw[1])
    norms = info["gradient_norm_history"]
    increasing = info["validation_loss_history"][-1]["loss"] > 1.05 * info["validation_loss_history"][-2]["loss"]
    if max(norms[-4:]) > 3.0:
        return "clip_grad"
    if info["lr"] >= 0.15 and increasing:
        return "lower_lr"
    if info.get("microbatch_size", 32) <= 8 and min(norms[-4:]) > 0 and max(norms[-4:]) > 2 * min(norms[-4:]):
        return "accumulate"
    return "resume"


def _metrics(results: list[dict]) -> dict:
    predictions = Counter(r["prediction"] or "<invalid>" for r in results)
    matches = sum(r["prediction"] == r["label"] for r in results)
    valid = len(results) - predictions["<invalid>"]
    out = {
        "n": len(results),
        "accuracy": matches / len(results),
        "valid_rate": valid / len(results),
        "predictions": dict(predictions),
        "gold": dict(Counter(r["label"] for r in results)),
        "mean_seconds": math.fsum(r["seconds"] for r in results) / len(results),
    }
    labels = sorted({r["label"] for r in results})
    out["macro_accuracy"] = math.fsum(
        sum(r["prediction"] == label for r in results if r["label"] == label)
        / sum(r["label"] == label for r in results)
        for label in labels
    ) / len(labels)
    regrets = [r["relative_regret"] for r in results if r["relative_regret"] is not None]
    if regrets:
        from statistics import median

        out["mean_relative_regret"] = math.fsum(regrets) / len(regrets)
        out["median_relative_regret"] = median(regrets)
        out["mean_log_relative_regret"] = math.fsum(math.log1p(r) for r in regrets) / len(regrets)
        out["mean_normalized_regret"] = math.fsum(r["normalized_regret"] for r in results) / len(results)
    return out


def evaluate(
    cases: Path,
    output: Path,
    model_path: Path | None,
    adapter: Path | None,
    baseline: str | None,
    predictor: Callable | None = None,
    *,
    batch_size: int = 1,
    custom_engine: str | None = None,
) -> dict:
    rows = read_jsonl(cases)
    if not rows:
        raise ValueError("Expected at least one evaluation case")
    if baseline:
        rows = [r for r in rows if r["track"] == "training"]
        if not rows:
            raise ValueError("The requested baseline requires training-coach cases")
        policy = policy_rules if baseline == "rules" else lambda _row: "resume"
        predictions = (Prediction(policy(row), "", 0.0) for row in rows)
    else:
        predictions = (predictor(model_path, adapter, rows) if predictor else
                       generate_predictions(model_path, adapter, rows, batch_size=batch_size))
    results = [
        {
            "case_id": row["case_id"],
            "track": row["track"],
            "family": row["family"],
            "split": row["split"],
            "block": str(row.get("evaluation_block", row.get("seed_index", row["case_id"]))),
            "label": row["label"],
            "prediction": prediction.label,
            "raw": prediction.raw,
            "seconds": prediction.seconds,
            **intervention_regret(row, prediction.label),
        }
        for row, prediction in zip(rows, predictions, strict=True)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, results)
    report = {
        "engine": custom_engine or baseline or ("adapter" if adapter else "base"),
        "model_dir": model_path.name if model_path else None,
        "adapter_dir": adapter.name if adapter else None,
        "inference_batch_size": batch_size if not baseline else None,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mlx", "mlx-lm", "transformers")
        } if not (baseline or custom_engine or (predictor and not model_path)) else {},
        "by_track": {key: _metrics([r for r in results if r["track"] == key])
                     for key in sorted({r["track"] for r in results})},
        "by_family": {
            f"{track}/{family}": _metrics(
                [r for r in results if r["track"] == track and r["family"] == family]
            )
            for track, family in sorted({(r["track"], r["family"]) for r in results})
        },
    }
    output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _checkpoint_selection(model, predict, model_path, adapter, output, validation, steps, batch_size):
    if not steps:
        raise ValueError("Checkpoint selection requires one or more explicit steps")
    validation_rows = read_jsonl(validation)
    if (not validation_rows or {row["split"] for row in validation_rows} != {"valid"}
            or {row["track"] for row in validation_rows} != {"training", "methodology"}):
        raise ValueError("Checkpoint selection needs validation cases in both tracks")

    scores = {}
    for step in steps:
        path = adapter / f"{step:07d}_adapters.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"No adapter checkpoint at step {step}")
        model.load_weights(str(path.resolve()), strict=False)
        model.eval()
        report = evaluate(
            validation, output / "validation" / f"step-{step}.jsonl",
            model_path, adapter, None, predict, batch_size=batch_size,
        )["by_track"]
        training = report["training"]["mean_normalized_regret"]
        methods = report["methodology"]["macro_accuracy"]
        scores[step] = {
            "training_mean_normalized_regret": training,
            "methodology_macro_accuracy": methods,
            "selection_score": ((1 - training) + methods) / 2,
        }
    step = max(scores, key=lambda item: (scores[item]["selection_score"], -item))
    model.load_weights(str((adapter / f"{step:07d}_adapters.safetensors").resolve()), strict=False)
    model.eval()
    selected = {"selected_step": step, "validation": validation.name, "checkpoint_scores": scores,
                "criterion": "mean of (one minus training normalized regret) and methodology macro accuracy"}
    (output / "checkpoint-selection.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    print(f"Selected checkpoint step {step} using validation cases", flush=True)


def _swap_modules(model, replacement):
    from mlx.utils import tree_unflatten

    model.update_modules(tree_unflatten(list(replacement.items())))
    model.eval()
    active = dict(model.named_modules())
    if any(active.get(name) is not layer for name, layer in replacement.items()):
        raise RuntimeError("Model layer swap did not take effect at every expected path")


def _adapter_swaps(model, base_modules):
    adapted = dict(model.named_modules())
    changed = {name: layer for name, layer in base_modules.items()
               if name and name in adapted and adapted[name] is not layer}
    if not changed:
        raise RuntimeError("No LoRA modules were installed")
    return changed, {name: adapted[name] for name in changed}


def paired_eval(
    cases: Path, output: Path, model_path: Path, adapter: Path, batch_size: int,
    extra_cases: tuple[Path, ...] = (),
    validation: Path | None = None,
    checkpoints: tuple[int, ...] = (),
) -> dict:
    """Evaluate base and adapter in one process without duplicating the base weights."""
    from mlx_lm.tuner.utils import load_adapters

    panels = [(cases, output), *((path, output / "external" / path.stem) for path in extra_cases)]
    if len({target for _, target in panels}) != len(panels):
        raise ValueError("Evaluation panels must have distinct base names")
    ids = set()
    families = set()
    for source, _ in panels:
        panel_rows = read_jsonl(source)
        if not panel_rows or {row["split"] for row in panel_rows} != {"test"}:
            raise ValueError("Paired test reports require only designated test cases")
        for row in panel_rows:
            if row["case_id"] in ids:
                raise ValueError(f"Repeated evaluation case: {row['case_id']}")
            ids.add(row["case_id"])
            families.add((row["track"], row["family"]))
    rows = read_jsonl(cases)
    parity_rows = read_jsonl(validation) if validation else rows
    if validation and any(row["case_id"] in ids or (row["track"], row["family"]) in families
                          for row in parity_rows):
        raise ValueError("Validation overlaps the test case IDs or families")
    if bool(validation) != bool(checkpoints):
        raise ValueError("Provide both validation cases and checkpoint steps")
    if not (adapter / "adapters.safetensors").is_file():
        raise FileNotFoundError(f"Missing adapter weights under {adapter}")

    print("Loading pinned base for paired evaluation", flush=True)
    model, tokenizer = _load_model(model_path, None)
    parity = {"requested": batch_size, "used": batch_size, "checks": []}
    if batch_size > 1:
        sample = []
        for track in ("training", "methodology"):
            sample.extend([r for r in parity_rows if r["track"] == track][:max(1, batch_size // 2)])
        one = list(_predictions(model, tokenizer, sample, progress=False, batch_size=1))
        many = list(_predictions(model, tokenizer, sample, progress=False, batch_size=batch_size))
        parity["checks"] = [
            {"case_id": row["case_id"], "matches": a.raw == b.raw}
            for row, a, b in zip(sample, one, many, strict=True)
        ]
        if len(sample) < 2 or not all(check["matches"] for check in parity["checks"]):
            batch_size = 1
            parity["used"] = 1

    def predict(_path, _adapter, case_rows):
        return _predictions(model, tokenizer, case_rows, progress=True, batch_size=batch_size)

    output.mkdir(parents=True, exist_ok=True)
    (output / "batch-parity.json").write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")
    selected_layers = None
    if validation:
        base_modules = dict(model.named_modules())
        model = load_adapters(model, str(adapter.resolve()))
        model.eval()
        _checkpoint_selection(model, predict, model_path, adapter, output, validation, checkpoints, batch_size)
        base_layers, selected_layers = _adapter_swaps(model, base_modules)
        _swap_modules(model, base_layers)
        current = dict(model.named_modules())
        if current.keys() != base_modules.keys() or any(
            current[name] is not original for name, original in base_modules.items()
        ):
            raise RuntimeError("Exact base module graph was not restored")
        if batch_size > 1:
            restored = list(_predictions(model, tokenizer, sample, progress=False, batch_size=1))
            if [p.raw for p in restored] != [p.raw for p in one]:
                raise RuntimeError("Base response changed after detaching the selected adapter")

    print(f"Evaluating base with batch size {batch_size}", flush=True)
    for source, target in panels:
        evaluate(source, target / "base.jsonl", model_path, None, None, predict, batch_size=batch_size)
    print("Loading adapter into the same frozen base", flush=True)
    if selected_layers:
        _swap_modules(model, selected_layers)
    else:
        model = load_adapters(model, str(adapter.resolve()))
        model.eval()
    from .compare import write_comparison

    comparisons = {}
    for source, target in panels:
        evaluate(source, target / "adapter.jsonl", model_path, adapter, None, predict, batch_size=batch_size)
        comparisons[source.stem] = write_comparison(
            target / "base.jsonl", target / "adapter.jsonl", target / "adapter-vs-base.json",
        )
    return comparisons
