"""Measure CPU training interventions on a held-out historical industry panel."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
import platform
from pathlib import Path
import time

from gradient_desk.dataset import write_jsonl
from gradient_desk.traces import (
    ACTIONS,
    FUTURE_MICROBATCHES,
    NEAR_TIE_ABSOLUTE,
    NEAR_TIE_RELATIVE,
    WARMUP_STEPS,
    format_telemetry,
    oracle_label,
)

from panel import ARCHIVE_SHA256, ARCHIVE_URL, FEATURES, Day, build_panel, parse_archive

FAMILY = "public_french_industry_mlp"
INDUSTRY_COUNT = 12
SCALER_DAYS = 60
MONITOR_DAYS = 12
FINAL_DAYS = 16
ANCHORS = tuple(map(date.fromisoformat, (
    "2011-08-01", "2013-06-03", "2018-02-01",
    "2020-02-18", "2022-06-01", "2023-03-01",
)))
LEARNING_RATES = (0.01, 0.04, 0.12, 0.24)
MOMENTUM = 0.8


@dataclass(frozen=True)
class Window:
    scaler: list[Day]
    monitor: list[Day]
    warmup: list[Day]
    future: list[Day]
    final: list[Day]


def window(days: list[Day], anchor: date) -> Window:
    """Group full industry cross-sections with a one-date label embargo."""
    try:
        start = next(i for i, day in enumerate(days) if day.date >= anchor)
    except StopIteration as exc:
        raise ValueError(f"Anchor {anchor} is later than the panel") from exc
    monitor_end = start - 1
    monitor_start = monitor_end - MONITOR_DAYS
    scaler_end = monitor_start - 1
    scaler_start = scaler_end - SCALER_DAYS
    warmup_end = start + WARMUP_STEPS
    future_start = warmup_end + 1
    future_end = future_start + FUTURE_MICROBATCHES
    final_start = future_end + 1
    final_end = final_start + FINAL_DAYS
    if scaler_start < 0 or final_end > len(days):
        raise ValueError(f"Insufficient complete records around anchor {anchor}")
    phases = Window(
        days[scaler_start:scaler_end], days[monitor_start:monitor_end],
        days[start:warmup_end], days[future_start:future_end], days[final_start:final_end],
    )
    portions = (phases.scaler, phases.monitor, phases.warmup, phases.future, phases.final)
    if not all(left[-1].label_date < right[0].date for left, right in zip(portions, portions[1:])):
        raise ValueError("A label straddles a phase boundary")
    if any(left.label_date != right.date for part in portions for left, right in zip(part, part[1:])):
        raise ValueError("An incomplete return row breaks a requested phase")
    return phases


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def _rng_hash(state) -> str:
    return hashlib.sha256(bytes(state.tolist())).hexdigest()


def _tensors(torch, days):
    x = torch.tensor([day.features for day in days], dtype=torch.float32, device="cpu")
    y = torch.tensor([day.targets for day in days], dtype=torch.float32, device="cpu")
    return x, y


def _phases(torch, windows: Window):
    scaler, _ = _tensors(torch, windows.scaler)
    mean = scaler.mean((0, 1))
    scale = scaler.std((0, 1), unbiased=False).clamp_min(1e-6)
    matrices = []
    for days in (windows.monitor, windows.warmup, windows.future, windows.final):
        matrix, targets = _tensors(torch, days)
        matrices.append(((matrix - mean) / scale, targets))
    return matrices


def _validation(torch, model, dataset):
    x, y = dataset
    with torch.no_grad():
        return torch.nn.functional.mse_loss(model(x).squeeze(-1), y).item()


def _train_microbatch(torch, model, x, y, denominator=1):
    loss = torch.nn.functional.mse_loss(model(x).squeeze(-1), y)
    (loss / denominator).backward()
    return loss.item()


def _norm(torch, model):
    squares = [parameter.grad.detach().double().square().sum() for parameter in model.parameters()]
    return torch.sqrt(torch.stack(squares).sum()).item()


def _phase_record(windows: Window) -> dict:
    return {
        name: {
            "first_feature_date": days[0].date.isoformat(),
            "last_feature_date": days[-1].date.isoformat(),
            "last_label_date": days[-1].label_date.isoformat(),
            "n_sessions": len(days),
        }
        for name, days in vars(windows).items()
    }


def _branch(torch, model, optimizer, checkpoint, future, final, sample_ids, action):
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    torch.set_rng_state(checkpoint["rng"])
    accumulation = 2 if action == "accumulate" else 1
    if action == "lower_lr":
        optimizer.param_groups[0]["lr"] *= 0.25
    clipping = 0.5 if action == "clip_grad" else None
    results = []
    consumed = []
    started = time.perf_counter()
    for start in range(0, FUTURE_MICROBATCHES, accumulation):
        optimizer.zero_grad(set_to_none=True)
        for offset in range(start, start + accumulation):
            _train_microbatch(torch, model, future[0][offset], future[1][offset], accumulation)
            consumed.extend(sample_ids[offset])
        results.append(_norm(torch, model))
        if clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
        optimizer.step()
    loss = _validation(torch, model, final)
    if not math.isfinite(loss):
        raise RuntimeError(f"Nonfinite measured loss for {action}")
    return {
        "validation_loss": loss,
        "elapsed_seconds": time.perf_counter() - started,
        "examples_seen": len(consumed),
        "future_sample_ids": consumed,
        "future_sample_digest": _hash(consumed),
        "rng_start_digest": _rng_hash(checkpoint["rng"]),
        "rng_end_digest": _rng_hash(torch.get_rng_state()),
        "optimizer_steps": FUTURE_MICROBATCHES // accumulation,
        "forward_backward_calls": FUTURE_MICROBATCHES,
        "validation_forward_calls": 1,
        "accumulation_steps": accumulation,
        "effective_batch_size": INDUSTRY_COUNT * accumulation,
        "lr": optimizer.param_groups[0]["lr"],
        "clip_grad_norm": clipping,
        "gradient_norm_history": results,
    }


def _case(torch, industry_names, windows: Window, anchor, block, profile):
    seed = 640000 + 1000 * block
    torch.random.default_generator.manual_seed(seed)
    monitor, warmup, future, final = _phases(torch, windows)
    model = torch.nn.Sequential(
        torch.nn.Linear(len(FEATURES), 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)
    ).cpu()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATES[profile], momentum=MOMENTUM)
    history = [{"step": 0, "loss": _validation(torch, model, monitor)}]
    train_losses, norms = [], []
    for step, (x, y) in enumerate(zip(*warmup, strict=True), 1):
        optimizer.zero_grad(set_to_none=True)
        train_losses.append(_train_microbatch(torch, model, x, y))
        norms.append(_norm(torch, model))
        optimizer.step()
        if step % 4 == 0:
            history.append({"step": step, "loss": _validation(torch, model, monitor)})
    features, targets = warmup
    observed_features = features.reshape(-1, features.shape[-1])
    telemetry = {
        "device": "cpu", "dtype": "float32",
        "model": "6-16-1 tanh MLP", "optimizer": "SGD", "momentum": MOMENTUM,
        "objective": "mean_squared_error", "training_steps": WARMUP_STEPS,
        "training_examples": targets.numel(), "validation_evaluations": len(history),
        "validation_examples_per_evaluation": MONITOR_DAYS * INDUSTRY_COUNT,
        "train_loss_history": train_losses, "validation_loss_history": history,
        "gradient_norm_history": norms, "gradient_norm": norms[-1],
        "lr": LEARNING_RATES[profile], "clip_grad_norm": None,
        "microbatch_size": INDUSTRY_COUNT, "batch_size": INDUSTRY_COUNT, "accumulation_steps": 1,
        "data": {
            "observed_examples": targets.numel(), "features": len(FEATURES),
            "feature_mean": observed_features.mean(0).tolist(),
            "feature_std": observed_features.std(0, unbiased=False).tolist(),
            "feature_abs_max": observed_features.abs().max().item(), "missing_fraction": 0.0,
        },
        "target": {
            "mean": targets.mean().item(), "std": targets.std(unbiased=False).item(),
            "min": targets.min().item(), "max": targets.max().item(),
            "positive_fraction": None,
        },
    }
    prompt = format_telemetry(telemetry)
    checkpoint = {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "rng": torch.get_rng_state().clone(),
    }
    sample_ids = [
        [f"{day.date.isoformat()}/{industry}" for industry in industry_names] for day in windows.future
    ]
    order = (block + profile) % len(ACTIONS)
    actions = {
        action: _branch(torch, model, optimizer, checkpoint, future, final, sample_ids, action)
        for action in ACTIONS[order:] + ACTIONS[:order]
    }
    sample_digest = _hash([sample for group in sample_ids for sample in group])
    if {row["future_sample_digest"] for row in actions.values()} != {sample_digest}:
        raise RuntimeError("Action branches used different training inputs")
    losses = {action: actions[action]["validation_loss"] for action in ACTIONS}
    label = oracle_label(losses)
    minimum = min(losses.values())
    for result in actions.values():
        result["regret"] = result["validation_loss"] - minimum
    return {
        "schema_version": 1,
        "case_id": f"{FAMILY}-p{profile}-s{block}", "track": "training",
        "family": FAMILY, "split": "test", "evaluation_block": anchor.isoformat(),
        "seed": seed, "seed_index": block, "profile_index": profile,
        "prompt": prompt, "label": label, "actions": actions,
        "oracle": {"absolute_tolerance": NEAR_TIE_ABSOLUTE,
                   "relative_tolerance": NEAR_TIE_RELATIVE, "tie_order": list(ACTIONS)},
        "label_regret": losses[label] - minimum,
        "budget": {"matching": "ordered industry-date examples and forward/backward calls",
                   "equivalent_future_data": True, "future_examples": sum(map(len, sample_ids)),
                   "future_sample_digest": sample_digest, "microbatch_size": INDUSTRY_COUNT},
        "source": {"archive_sha256": ARCHIVE_SHA256,
                   "phase_dates": _phase_record(windows)},
    }


def _measurement_hash(rows: list[dict]) -> str:
    without_wall_time = [
        {
            **row,
            "actions": {
                action: {key: value for key, value in result.items() if key != "elapsed_seconds"}
                for action, result in row["actions"].items()
            },
        }
        for row in rows
    ]
    payload = json.dumps(without_wall_time, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def build(archive: Path, output: Path) -> dict:
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    vintage, industries, data = parse_archive(archive)
    if len(industries) != INDUSTRY_COUNT:
        raise ValueError(f"Expected {INDUSTRY_COUNT} industries")
    days = build_panel(data)
    windows = [window(days, anchor) for anchor in ANCHORS]
    if not all(a.final[-1].label_date < b.scaler[0].date for a, b in zip(windows, windows[1:])):
        raise ValueError("Benchmark windows overlap")
    rows = [
        _case(torch, industries, phases, anchor, block, profile)
        for block, (anchor, phases) in enumerate(zip(ANCHORS, windows, strict=True))
        for profile in range(len(LEARNING_RATES))
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    write_jsonl(temporary, rows)
    temporary.replace(output)
    metadata = {
        "schema": 1, "source_url": ARCHIVE_URL, "source_sha256": ARCHIVE_SHA256,
        "database_vintage": vintage, "device": "cpu", "torch_version": torch.__version__,
        "python_version": platform.python_version(), "os": platform.system(),
        "machine": platform.machine(), "cases": len(rows),
        "case_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "measurement_sha256": _measurement_hash(rows),
        "industries": industries, "features": list(FEATURES), "label": "next trading-session return in percent",
        "learning_rates": LEARNING_RATES, "anchors": [anchor.isoformat() for anchor in ANCHORS],
        "outcome_counts": dict(Counter(row["label"] for row in rows)),
        "validation": "terminal MSE in squared percentage points",
        "limits": "Six predeclared market windows, one small MLP, archived values, no investment performance claim.",
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="Pinned Dartmouth ZIP")
    parser.add_argument("--output", required=True, type=Path, help="Local eval-ready cases JSONL")
    options = parser.parse_args()
    print(json.dumps(build(options.archive, options.output), indent=2))
