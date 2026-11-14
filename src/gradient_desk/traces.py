"""Cheap, measured CPU counterfactuals for a training-action coach.

Export: python -m gradient_desk.traces --output traces.jsonl --seeds-per-case 8
No dependency beyond optional torch and the standard library. Importing this
module does not import torch. Splits are family-held-out, not random row splits;
these synthetic tasks are not evidence of efficacy on real financial data.

Budget contract: every trial sees the same ordered future microbatches, once.
resume/lower_lr/clip_grad take one optimizer step per microbatch; accumulate
averages two microbatch objectives before a step. Thus accumulation changes
noise and cadence, NOT total examples or the number of forward/backward calls.
Wall times (including validation, excluding checkpoint restoration) are noisy
measurements, deliberately excluded from prompts and determinism guarantees.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Iterator

ACTIONS = ("resume", "lower_lr", "clip_grad", "accumulate")
# External callers may call the last intervention increase_accumulation; the
# wire label and model response are always accumulate.
ACTION_ALIASES = {"increase_accumulation": "accumulate"}
FAMILY_SPLITS = {
    "nonlinear_regression": "train",
    "imbalanced_classification": "train",
    "noisy_classification": "valid",
    "heteroscedastic_regression": "valid",
    "tabular_regime_shift": "test",
    "rank_cross_section": "test",
}
# Independent, predeclared seed namespaces. A local seed index is serialized too.
SPLIT_SEED_BASES = {"train": 110000, "valid": 220000, "test": 330000}
PROFILES = (
    {"lr": 0.003, "batch_size": 16},
    {"lr": 0.03, "batch_size": 16},
    {"lr": 0.15, "batch_size": 16},
    {"lr": 0.45, "batch_size": 16},
    {"lr": 0.15, "batch_size": 8},
    {"lr": 0.45, "batch_size": 32},
)
WARMUP_STEPS = 12
FUTURE_MICROBATCHES = 8
VALIDATION_SIZE = 96
NEAR_TIE_ABSOLUTE = 1e-5
NEAR_TIE_RELATIVE = 1e-3
INSTRUCTION = (
    "You are a neural-network training coach. Choose the next intervention "
    "using only the pre-action telemetry. Reply with exactly one action: "
    "resume, lower_lr, clip_grad, accumulate. "
    "resume keeps the current settings; lower_lr multiplies LR by 0.25; "
    "clip_grad enables global gradient-norm clipping at 0.5; accumulate "
    "averages two consecutive microbatches per optimizer update. All choices "
    "use the same next eight microbatches; accumulate makes four updates "
    "instead of eight. No extra examples are available to any action."
)


def format_telemetry(telemetry: dict) -> str:
    def compact(value):
        if isinstance(value, float):
            return float(f"{value:.4g}")
        if isinstance(value, dict):
            return {key: compact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value

    encoded = json.dumps(compact(telemetry), separators=(",", ":"), allow_nan=False)
    return INSTRUCTION + "\nTelemetry: " + encoded


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Trace generation requires optional PyTorch; install torch in your "
            "telemetry environment. The main package need not depend on it."
        ) from exc
    return torch


def oracle_label(losses: dict[str, float]) -> str:
    """Minimum measured loss, with a fixed tolerance and menu-order resolver.

    Resume wins whenever it is within abs + rel * abs(minimum) of the minimum.
    The same fixed menu order resolves near ties not involving resume.
    """
    if set(losses) != set(ACTIONS) or not all(
        math.isfinite(v) for v in losses.values()
    ):
        raise ValueError("Expected one finite validation loss for every action")
    best = min(losses.values())
    tolerance = NEAR_TIE_ABSOLUTE + NEAR_TIE_RELATIVE * abs(best)
    return next(action for action in ACTIONS if losses[action] <= best + tolerance)


def _data(torch, family, seed, batch_size):
    """Fixed stream and separate validation draw; hidden recipe never prompted."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_size = (WARMUP_STEPS + FUTURE_MICROBATCHES) * batch_size

    def draw(n, validation=False):
        x = torch.randn(n, 6, generator=generator)
        noise = torch.randn(n, generator=generator)
        signal = 1.2 * x[:, 0] - 0.8 * x[:, 1] + 0.5 * x[:, 2]
        if family == "nonlinear_regression":
            y = torch.sin(1.8 * x[:, 0]) + 0.4 * x[:, 1] * x[:, 2] + 0.1 * noise
        elif family == "imbalanced_classification":
            probability = torch.sigmoid(signal - 2.3)
            y = (torch.rand(n, generator=generator) < probability).float()
        elif family == "noisy_classification":
            y = (signal + 0.2 * noise > 0).float()
            flip = torch.rand(n, generator=generator) < 0.25
            y = torch.where(flip, 1.0 - y, y)
        elif family == "heteroscedastic_regression":
            y = torch.sin(x[:, 0]) + 0.5 * x[:, 1]
            y = y + (0.1 + 0.7 * x[:, 2].abs()) * noise
        elif family == "tabular_regime_shift":
            # The observed stream has already begun shifting at the checkpoint.
            # Validation and future data represent the new regime; no future
            # statistics or shift indicator are exposed to the coach.
            start = 0 if validation else (WARMUP_STEPS // 2) * batch_size
            x[start:, 0] += 1.0
            y = 0.8 * x[:, 0] - 0.6 * x[:, 1] + 0.1 * noise
            y[start:] = -0.5 * x[start:, 0] + x[start:, 1] + 0.1 * noise[start:]
        elif family == "rank_cross_section":
            # Within-microbatch relative return ordering, not return MSE.
            y = torch.tanh(signal) + 0.4 * x[:, 3] * x[:, 4] + 0.3 * noise
        else:
            raise ValueError(f"Unknown family: {family}")
        return x, y

    train = draw(train_size)
    validation = draw(VALIDATION_SIZE, validation=True)
    return train, validation


def _objective(torch, family, prediction, target):
    if family in ("imbalanced_classification", "noisy_classification"):
        return torch.nn.functional.binary_cross_entropy_with_logits(prediction, target)
    if family == "rank_cross_section":
        i, j = torch.triu_indices(len(target), len(target), offset=1)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            prediction[i] - prediction[j], (target[i] > target[j]).float()
        )
    return torch.nn.functional.mse_loss(prediction, target)


def _validation(torch, family, model, data):
    # Fixed cross-sections for ranking, independent of training batch size.
    model.eval()
    with torch.no_grad():
        losses = [
            _objective(torch, family, model(x).squeeze(-1), y).item()
            for x, y in zip(data[0].split(16), data[1].split(16))
        ]
    model.train()
    return sum(losses) / len(losses)


def _gradient_norm(torch, model):
    return float(torch.sqrt(sum(
        parameter.grad.detach().square().sum()
        for parameter in model.parameters() if parameter.grad is not None
    )).item())


def _digest(tensor):
    # No numpy dependency. Used only for the tiny CPU RNG state tensors.
    return hashlib.sha256(bytes(tensor.tolist())).hexdigest()


def _case(torch, family, profile_index, seed_index):
    split = FAMILY_SPLITS[family]
    family_index = tuple(FAMILY_SPLITS).index(family)
    # Multiplicative seed index avoids overlaps even with large requested N.
    seed = SPLIT_SEED_BASES[split] + family_index * 1000 + profile_index * 100 + seed_index * 1000000
    profile = PROFILES[profile_index]
    batch_size = profile["batch_size"]
    torch.random.default_generator.manual_seed(seed)
    train, validation = _data(torch, family, seed + 17, batch_size)
    model = torch.nn.Sequential(
        torch.nn.Linear(6, 12), torch.nn.Tanh(), torch.nn.Linear(12, 1)
    ).cpu()
    optimizer = torch.optim.SGD(model.parameters(), lr=profile["lr"], momentum=0.8)
    cursor = 0
    train_losses, grad_norms = [], []
    validation_history = [{"step": 0, "loss": _validation(torch, family, model, validation)}]
    for step in range(WARMUP_STEPS):
        x, y = train[0][cursor:cursor + batch_size], train[1][cursor:cursor + batch_size]
        optimizer.zero_grad(set_to_none=True)
        loss = _objective(torch, family, model(x).squeeze(-1), y)
        loss.backward()
        grad_norms.append(_gradient_norm(torch, model))
        optimizer.step()
        train_losses.append(float(loss.item()))
        cursor += batch_size
        if (step + 1) % 4 == 0:
            validation_history.append({
                "step": step + 1, "loss": _validation(torch, family, model, validation)
            })

    seen_x, seen_y = train[0][:cursor], train[1][:cursor]
    classification = family in ("imbalanced_classification", "noisy_classification")
    objective = "binary_cross_entropy" if classification else (
        "pairwise_rank_cross_entropy" if family == "rank_cross_section" else "mean_squared_error"
    )
    telemetry = {
        "device": "cpu", "dtype": "float32",
        "model": "6-12-1 tanh MLP", "optimizer": "SGD", "momentum": 0.8,
        "objective": objective, "training_steps": WARMUP_STEPS,
        "training_examples": cursor, "validation_evaluations": len(validation_history),
        "validation_examples_per_evaluation": VALIDATION_SIZE,
        "train_loss_history": train_losses,
        "validation_loss_history": validation_history,
        "gradient_norm_history": grad_norms,
        "gradient_norm": grad_norms[-1], "lr": profile["lr"],
        "clip_grad_norm": None, "microbatch_size": batch_size,
        "batch_size": batch_size, "accumulation_steps": 1,
        "data": {
            "observed_examples": cursor, "features": 6,
            "feature_mean": seen_x.mean(0).tolist(),
            "feature_std": seen_x.std(0, unbiased=False).tolist(),
            "feature_abs_max": float(seen_x.abs().max().item()),
            "missing_fraction": 0.0,
        },
        "target": {
            "mean": float(seen_y.mean().item()),
            "std": float(seen_y.std(unbiased=False).item()),
            "min": float(seen_y.min().item()), "max": float(seen_y.max().item()),
            "positive_fraction": float(seen_y.mean().item()) if classification else None,
        },
    }
    # Restrict prompt precision to observations useful for a small model. The
    # intervention outcomes keep full precision outside the prompt.
    prompt = format_telemetry(telemetry)
    checkpoint = {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "cursor": cursor, "rng": torch.get_rng_state().clone(),
    }
    future_ids = list(range(cursor, cursor + FUTURE_MICROBATCHES * batch_size))
    future_digest = hashlib.sha256(json.dumps(future_ids).encode()).hexdigest()
    results = {}
    # Rotate evaluation order to reduce systematic wall-clock ordering bias.
    order = ACTIONS[seed_index % 4:] + ACTIONS[:seed_index % 4]
    for action in order:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
        optimizer.zero_grad(set_to_none=True)
        cursor = checkpoint["cursor"]
        torch.set_rng_state(checkpoint["rng"])
        if action == "lower_lr":
            for group in optimizer.param_groups:
                group["lr"] *= 0.25
        accumulation = 2 if action == "accumulate" else 1
        start = time.perf_counter()
        norm_history = []
        consumed = []
        for _ in range(FUTURE_MICROBATCHES // accumulation):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accumulation):
                x = train[0][cursor:cursor + batch_size]
                y = train[1][cursor:cursor + batch_size]
                loss = _objective(torch, family, model(x).squeeze(-1), y) / accumulation
                loss.backward()
                consumed.extend(range(cursor, cursor + batch_size))
                cursor += batch_size
            norm_history.append(_gradient_norm(torch, model))
            if action == "clip_grad":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        validation_loss = _validation(torch, family, model, validation)
        elapsed = time.perf_counter() - start
        if not math.isfinite(validation_loss):
            raise RuntimeError(f"Nonfinite measured loss: {family}, {seed}, {action}")
        results[action] = {
            "validation_loss": validation_loss,
            "elapsed_seconds": elapsed,
            "examples_seen": len(consumed),
            "future_sample_ids": consumed,
            "future_sample_digest": hashlib.sha256(json.dumps(consumed).encode()).hexdigest(),
            "cursor_start": checkpoint["cursor"], "cursor_end": cursor,
            "rng_start_digest": _digest(checkpoint["rng"]),
            "rng_end_digest": _digest(torch.get_rng_state()),
            "optimizer_steps": FUTURE_MICROBATCHES // accumulation,
            "forward_backward_calls": FUTURE_MICROBATCHES,
            "validation_forward_calls": VALIDATION_SIZE // 16,
            "accumulation_steps": accumulation,
            "effective_batch_size": batch_size * accumulation,
            "lr": optimizer.param_groups[0]["lr"],
            "clip_grad_norm": 0.5 if action == "clip_grad" else None,
            "gradient_norm_history": norm_history,
            "clipping_calls": FUTURE_MICROBATCHES if action == "clip_grad" else 0,
        }
    losses = {action: results[action]["validation_loss"] for action in ACTIONS}
    label = oracle_label(losses)
    minimum = min(losses.values())
    for action in ACTIONS:
        results[action]["regret"] = losses[action] - minimum
        results[action]["validation_loss_change"] = losses[action] - validation_history[-1]["loss"]
        results[action]["elapsed_ratio_to_resume"] = (
            results[action]["elapsed_seconds"] / max(results["resume"]["elapsed_seconds"], 1e-12)
        )
    return {
        "schema_version": 1,
        "case_id": f"{family}-p{profile_index}-s{seed_index}",
        "track": "training", "family": family, "split": split,
        "seed": seed, "seed_index": seed_index, "profile_index": profile_index,
        "prompt": prompt, "label": label,
        "actions": {action: results[action] for action in ACTIONS},
        "label_regret": losses[label] - minimum,
        "oracle": {"absolute_tolerance": NEAR_TIE_ABSOLUTE,
                   "relative_tolerance": NEAR_TIE_RELATIVE,
                   "tie_order": list(ACTIONS)},
        "budget": {
            "matching": "ordered future examples; not optimizer steps",
            "equivalent_future_data": True,
            "future_examples": len(future_ids),
            "future_sample_digest": future_digest,
            "microbatch_size": batch_size,
            "microbatches": FUTURE_MICROBATCHES,
            "checkpoint_cursor": checkpoint["cursor"],
            "checkpoint_rng_digest": _digest(checkpoint["rng"]),
            "compute_note": "All trials use 8 forward/backward microbatches and 6 validation forwards. "
                            "accumulate uses 4 rather than 8 SGD updates; clip_grad adds 8 norm/clipping "
                            "passes. Times include validation, exclude restoration, and are not stable benchmarks.",
        },
        "provenance": {"generator": "pytorch_cpu_synthetic_v1", "torch_version": str(torch.__version__),
                       "financial_evidence": False},
    }


def generate_records(seeds_per_case: int = 8) -> Iterator[dict]:
    """Yield 36 * N records, default 288; no label balancing or outcome filtering.

    Numeric results are reproducible in the same torch/platform environment;
    elapsed times are not. Single CPU thread, fixed data order, no dropout or
    augmentation. Only Torch RNG is used. Caller RNG and thread count are
    restored before every yield, including when iteration is stopped early.
    This function changes process-global torch settings briefly: do not call
    concurrently with other torch workloads in the same process.
    """
    if not isinstance(seeds_per_case, int) or isinstance(seeds_per_case, bool) or seeds_per_case < 1:
        raise ValueError("seeds_per_case must be a positive integer")
    torch = _torch()
    for family in FAMILY_SPLITS:
        for profile_index in range(len(PROFILES)):
            for seed_index in range(seeds_per_case):
                threads = torch.get_num_threads()
                deterministic = torch.are_deterministic_algorithms_enabled()
                warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
                try:
                    torch.set_num_threads(1)
                    torch.use_deterministic_algorithms(True)
                    with torch.random.fork_rng(devices=[]):
                        record = _case(torch, family, profile_index, seed_index)
                finally:
                    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
                    torch.set_num_threads(threads)
                yield record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSONL file (existing files are not overwritten)")
    parser.add_argument("--seeds-per-case", type=int, default=8, help="Seeds per family/profile; total rows = 36 * N")
    args = parser.parse_args(argv)
    if args.seeds_per_case < 1:
        parser.error("--seeds-per-case must be positive")
    _torch()  # Fail before creating an output file if the optional dependency is absent.
    counts = {action: 0 for action in ACTIONS}
    with args.output.open("x", encoding="utf-8") as stream:
        for record in generate_records(args.seeds_per_case):
            stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            counts[record["label"]] += 1
    print(json.dumps({"output": str(args.output), "rows": sum(counts.values()), "labels": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
