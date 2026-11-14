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
    match = re.fullmatch(r"`?([a-z_]+)`?", candidate)
    return match[1] if match and match[1] in choices else None


def generate_predictions(
    model_path: Path, adapter: Path | None, rows: list[dict], *,
    progress: bool = True, batch_size: int = 1,
):
    try:
        import mlx.core as mx
        from mlx_lm import generate, load
        from mlx_lm.generate import batch_generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise RuntimeError("Install the 'mlx' extra for Apple Silicon inference") from exc
    model, tokenizer = load(
        str(model_path.resolve()),
        adapter_path=str(adapter.resolve()) if adapter else None,
        tokenizer_config={"trust_remote_code": False},
    )
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
        "engine": baseline or ("adapter" if adapter else "base"),
        "model_dir": model_path.name if model_path else None,
        "adapter_dir": adapter.name if adapter else None,
        "inference_batch_size": batch_size if not baseline else None,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mlx", "mlx-lm", "transformers")
        } if not (baseline or predictor) else {},
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
