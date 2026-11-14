"""Independent Treasury public-data counterfactual intervention generator.

Export: python -m gradient_desk.treasury_traces --csv PATH --output DIR

This mirrors the budget/oracle/telemetry contract of ``gradient_desk.traces``
and ``examples/public_industry/benchmark.py`` on an *independent* public panel:
the U.S. Treasury daily par yield curve. We predict each tenor's **next-session
absolute daily yield change in basis points** from backward-looking features
computed with information available through the decision session's close.

Source (never embedded in code; only its locator and digest are pinned)
    U.S. Department of the Treasury par yield curve rates, 1990-2024.
        https://home.treasury.gov/system/files/276/yield-curve-rates-1990-2024.csv
    SHA-256 pinned in ``SOURCE_SHA256``. The caller stages the verified file
    (default ``.work/v2/treasury-yield-curve-1990-2024.csv``). The raw table is
    read only after its hash is verified; it is never redistributed here.

Timing / leakage
    The CSV header is ``Date,1 Mo,2 Mo,3 Mo,4 Mo,6 Mo,1 Yr,2 Yr,3 Yr,5 Yr,7 Yr,
    10 Yr,20 Yr,30 Yr`` with ``Date`` formatted ``%m/%d/%y`` and rows in
    *descending* date order. We model eight tenors (``TENORS``). Every feature
    for decision session ``t`` uses only rows with date ``<= t``; the label of a
    session is the realized next-observed-session yield change, in basis points.

Splits are by whole chronology, not random rows: train 1990-2013, valid
2014-2018, test 2019-2024. No raw session date appears in more than one split.
This is an archived retrospective snapshot; nothing here is an investment claim
and no price-level conclusion is drawn (we model *changes*, in basis points).
A known Treasury par-yield methodology change (2021-12-06, monotone-convex
interpolation) is recorded in provenance so downstream readers can account for
the break; it is not exposed to the coach.

Budget contract (identical in spirit to ``gradient_desk.traces``): every action
sees the same ordered eight future microbatches, once; each performs exactly
eight forward/backward calls. resume/lower_lr/clip_grad take one optimizer step
per microbatch; accumulate averages two microbatches per step (four updates).
Model, optimizer and Torch RNG are restored to the exact checkpoint before each
branch. Wall times are noisy and excluded from prompts, determinism guarantees,
and the permanent measurement hash.
"""

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
from typing import Sequence

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

# --------------------------------------------------------------------------- #
# Pinned source (data never embedded; only the locator and digest are)
# --------------------------------------------------------------------------- #
SOURCE_URL = "https://home.treasury.gov/system/files/276/yield-curve-rates-1990-2024.csv"
SOURCE_SHA256 = "258a594eb64e9d21d9e8574b0b8580b0a5230dce56bb284ec6aa49a9a260897f"
STAGED_PATH = Path(".work/v2/treasury-yield-curve-1990-2024.csv")

#: Expected CSV header, in file order.
CSV_COLUMNS: tuple[str, ...] = (
    "Date", "1 Mo", "2 Mo", "3 Mo", "4 Mo", "6 Mo", "1 Yr", "2 Yr",
    "3 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr",
)
#: Modelled tenors (a stable, liquid subset), in curve order.
TENORS: tuple[str, ...] = ("6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "30 Yr")
TENOR_COUNT = len(TENORS)
DATE_FORMAT = "%m/%d/%y"  # documented; parsing is explicit (no strptime pivot surprises)

#: Recorded (not exposed) Treasury par-yield interpolation methodology break.
METHOD_BREAK = date(2021, 12, 6)

# Whole-chronology splits (inclusive calendar years); disjoint by construction.
SPLIT_YEARS = {
    "train": (1990, 2013),
    "valid": (2014, 2018),
    "test": (2019, 2024),
}
# Independent, predeclared seed namespaces per split (distinct, non-overlapping).
SPLIT_SEED_BASES = {"train": 710000, "valid": 720000, "test": 730000}

# --------------------------------------------------------------------------- #
# Phase layout within each isolated block (feature sessions), with embargoes.
# A block is a single consecutive run of complete sessions. Phases never share a
# session, and a 1-2 session label embargo separates consecutive phases so no
# phase's next-session label straddles into the following phase's features.
# --------------------------------------------------------------------------- #
LOOKBACK_DAYS = 12   # history used only to compute lagged/backward features
SCALER_DAYS = 16     # fit-only standardization statistics (features and y scale)
MONITOR_DAYS = 10    # pre-action validation (branch-independent)
# WARMUP_STEPS (=12) warmup sessions drive the pre-action optimizer steps.
# FUTURE_MICROBATCHES (=8) branch sessions are the shared future microbatches.
FINAL_DAYS = 10      # terminal scoring sessions (independent of monitor)
EMBARGO = 1          # sessions dropped between phases (1-2 permitted; we use 1)

# Six distinct lagged/curve backward features per tenor (all through t).
FEATURES: tuple[str, ...] = (
    "tenor_index",        # normalized tenor position (static identity feature)
    "change_t_bp",        # own yield change on t vs t-1, in basis points
    "mean_change_5_bp",   # own mean daily change over last 5 sessions, bp
    "std_change_5_bp",    # own daily change std over last 5 sessions, bp (ddof=0)
    "level_minus_mean_5", # own level t minus its 5-session mean level, percent
    "curve_slope_t",      # 10 Yr minus 6 Mo level on t, percent (shared term)
)
MOMENTUM = 0.8

# Four-to-six predefined LR/optimizer profiles per block. The range is chosen to
# stay finite on this well-scaled target yet to support different winners.
PROFILES: tuple[dict, ...] = (
    {"lr": 0.01, "optimizer": "sgd", "momentum": MOMENTUM},
    {"lr": 0.04, "optimizer": "sgd", "momentum": MOMENTUM},
    {"lr": 0.12, "optimizer": "sgd", "momentum": MOMENTUM},
    {"lr": 0.24, "optimizer": "sgd", "momentum": MOMENTUM},
    {"lr": 0.08, "optimizer": "adam", "momentum": MOMENTUM},
)

# Sessions per block = all feature phases + embargoes + the single target row
# that labels the final phase's last feature session.
BLOCK_PHASE_DAYS = (
    LOOKBACK_DAYS + SCALER_DAYS + MONITOR_DAYS + WARMUP_STEPS + FUTURE_MICROBATCHES + FINAL_DAYS
)
BLOCK_EMBARGO_DAYS = 5 * EMBARGO  # between the six phases
BLOCK_SESSIONS = BLOCK_PHASE_DAYS + BLOCK_EMBARGO_DAYS + 1  # +1 label row


# --------------------------------------------------------------------------- #
# Public row dataclass (one supervised decision session, all tenors grouped)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Session:
    """One decision session keyed by its date.

    ``features`` is ``TENOR_COUNT`` per-tenor feature tuples (tenor order matches
    ``TENORS``); each follows ``FEATURES``. ``targets`` holds the ``TENOR_COUNT``
    next-session signed yield changes in basis points. ``label_date`` is the
    next observed session date.
    """

    date: date
    label_date: date
    history_start_date: date
    features: tuple[tuple[float, ...], ...]
    targets: tuple[float, ...]


# --------------------------------------------------------------------------- #
# Archive integrity + decoding (hash verified before any decode)
# --------------------------------------------------------------------------- #
def _verify_hash(payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(
            "Treasury CSV SHA-256 mismatch: refusing to decode untrusted data.\n"
            f"  expected {SOURCE_SHA256}\n  actual   {digest}"
        )
    return digest


def _parse_date(token: str) -> date:
    """Explicit ``%m/%d/%y`` with a fixed 1990-2069 pivot (matches the file)."""
    month, day, year = (part.strip() for part in token.split("/"))
    yy = int(year)
    full = 2000 + yy if yy < 70 else 1900 + yy
    return date(full, int(month), int(day))


def _parse_cell(token: str) -> float:
    token = token.strip()
    if token == "" or token.upper() in ("N/A", "NA", "ND"):
        return math.nan
    try:
        return float(token)
    except ValueError as exc:
        raise ValueError(f"Non-numeric yield cell: {token!r}") from exc


def parse_csv(path: Path) -> tuple[str, list[tuple[date, tuple[float, ...]]]]:
    """Verify the pinned hash, then decode the Treasury CSV.

    Returns ``(source_sha256, rows)`` with ``rows`` ascending by date and
    restricted to the modelled ``TENORS`` (percent). The file is descending by
    date; we sort ascending here. No raw data leaves this process beyond the
    derived rows the caller requests.
    """
    with path.open("rb") as stream:
        payload = stream.read(8_000_001)
    if len(payload) > 8_000_000:
        raise ValueError("Treasury CSV exceeds the 8 MB limit")
    digest = _verify_hash(payload)

    lines = payload.decode("utf-8-sig").splitlines()
    if not lines:
        raise ValueError("Empty Treasury CSV")
    header = [cell.strip() for cell in lines[0].split(",")]
    if tuple(header) != CSV_COLUMNS:
        raise ValueError(f"Unexpected Treasury header: {header!r}")
    tenor_index = [header.index(name) for name in TENORS]

    rows: list[tuple[date, tuple[float, ...]]] = []
    seen: set[date] = set()
    for line in lines[1:]:
        if not line.strip():
            continue
        cells = line.split(",")
        if len(cells) != len(CSV_COLUMNS):
            raise ValueError(f"Expected {len(CSV_COLUMNS)} columns, got {len(cells)}: {line!r}")
        day = _parse_date(cells[0])
        if day in seen:
            raise ValueError(f"Duplicate date in archive: {day.isoformat()}")
        seen.add(day)
        rows.append((day, tuple(_parse_cell(cells[i]) for i in tenor_index)))
    if len(rows) < BLOCK_SESSIONS:
        raise ValueError(f"Too few sessions ({len(rows)}); need at least {BLOCK_SESSIONS}")
    rows.sort(key=lambda item: item[0])
    return digest, rows


# --------------------------------------------------------------------------- #
# Panel construction (leakage-safe backward features; changes in basis points)
# --------------------------------------------------------------------------- #
def _finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(v) for v in values)


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _pstd(values: Sequence[float]) -> float:
    mean = _mean(values)
    return math.sqrt(math.fsum((v - mean) ** 2 for v in values) / len(values))


def build_panel(rows: list[tuple[date, tuple[float, ...]]]) -> list[Session]:
    """Turn ascending daily levels into leakage-safe next-session change rows.

    For each decision index ``t`` (needing a full ``LOOKBACK_DAYS`` history and a
    next row ``t+1``) we build per-tenor backward features from rows ``<= t`` and
    label with the realized ``t+1 - t`` change in basis points. A session is
    skipped entirely if any value in its lookback window, current row, or next
    row is non-finite; windows are never bridged across such gaps (consecutive
    calendar sessions are not required, only consecutive observed rows).
    """
    columns = [[row[1][i] for row in rows] for i in range(TENOR_COUNT)]
    short_i = TENORS.index("6 Mo")
    long_i = TENORS.index("10 Yr")
    panel: list[Session] = []
    for t in range(LOOKBACK_DAYS - 1, len(rows) - 1):
        decision_day, current = rows[t]
        label_day, next_values = rows[t + 1]
        if not (_finite(current) and _finite(next_values)):
            continue
        window_ok = True
        per_tenor: list[tuple[float, ...]] = []
        slope = columns[long_i][t] - columns[short_i][t]
        for i in range(TENOR_COUNT):
            levels = columns[i][t - LOOKBACK_DAYS + 1 : t + 1]
            if not _finite(levels):
                window_ok = False
                break
            changes_bp = [100.0 * (levels[k] - levels[k - 1]) for k in range(1, len(levels))]
            last5 = changes_bp[-5:]
            per_tenor.append((
                (i - (TENOR_COUNT - 1) / 2) / ((TENOR_COUNT - 1) / 2),  # tenor_index in [-1, 1]
                changes_bp[-1],
                _mean(last5),
                _pstd(last5),
                levels[-1] - _mean(levels[-5:]),
                slope,
            ))
        if not window_ok:
            continue
        targets_bp = tuple(100.0 * (next_values[i] - current[i]) for i in range(TENOR_COUNT))
        panel.append(Session(
            date=decision_day, label_date=label_day,
            history_start_date=rows[t - LOOKBACK_DAYS + 1][0],
            features=tuple(per_tenor), targets=targets_bp,
        ))
    return panel


def _in_split(day: date, split: str) -> bool:
    low, high = SPLIT_YEARS[split]
    return low <= day.year <= high


# --------------------------------------------------------------------------- #
# Block windowing with explicit phase embargoes and leakage assertions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Block:
    lookback: list[Session]
    scaler: list[Session]
    monitor: list[Session]
    warmup: list[Session]
    branch: list[Session]
    final: list[Session]


def _phase_names() -> tuple[str, ...]:
    return ("lookback", "scaler", "monitor", "warmup", "branch", "final")


def carve_block(sessions: list[Session], start: int) -> Block:
    """Carve one isolated block of consecutive sessions starting at ``start``.

    Phases are consumed in order with a ``EMBARGO`` session gap between them, so
    that no phase's label (next session) falls inside the following phase's
    feature sessions. Raises if the phases are internally non-consecutive.
    """
    sizes = (LOOKBACK_DAYS, SCALER_DAYS, MONITOR_DAYS, WARMUP_STEPS, FUTURE_MICROBATCHES, FINAL_DAYS)
    cursor = start
    parts: list[list[Session]] = []
    for size in sizes:
        parts.append(sessions[cursor:cursor + size])
        cursor += size + EMBARGO
    block = Block(*parts)
    phases = (block.lookback, block.scaler, block.monitor, block.warmup, block.branch, block.final)
    if any(len(part) != size for part, size in zip(phases, sizes)):
        raise ValueError("Insufficient complete sessions for a block")
    # Internal consecutiveness: each phase's per-session label is the next
    # session's date (no missing row inside a phase).
    for part in phases:
        if any(left.label_date != right.date for left, right in zip(part, part[1:])):
            raise ValueError("A non-consecutive session breaks a requested phase")
    # Strict forward ordering with embargo: every phase's last label precedes
    # the next phase's first feature date (no label straddles a boundary).
    for left, right in zip(phases, phases[1:]):
        if not left[-1].label_date < right[0].date:
            raise ValueError("A label straddles a phase boundary")
    return block


def _blocks_for_split(panel: list[Session], split: str) -> list[tuple[int, Block]]:
    """Choose closed, nonoverlapping source windows including all lag inputs."""
    blocks = []
    previous_tail = date.min
    width = BLOCK_SESSIONS - 1  # final label exists outside the feature-session list
    for base in range(len(panel) - width + 1):
        first, last = panel[base], panel[base + width - 1]
        if (first.history_start_date <= previous_tail
                or not _in_split(first.history_start_date, split)
                or not _in_split(last.label_date, split)):
            continue
        local = panel[base:base + width]
        if not all(left.label_date == right.date for left, right in zip(local, local[1:])):
            continue
        blocks.append((base, carve_block(local, 0)))
        previous_tail = last.label_date
    return blocks


# --------------------------------------------------------------------------- #
# Hash helpers
# --------------------------------------------------------------------------- #
def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _rng_hash(state) -> str:
    return hashlib.sha256(bytes(state.tolist())).hexdigest()


def _phase_record(block: Block) -> dict:
    return {
        name: {
            "first_history_date": sessions[0].history_start_date.isoformat(),
            "first_feature_date": sessions[0].date.isoformat(),
            "last_feature_date": sessions[-1].date.isoformat(),
            "last_label_date": sessions[-1].label_date.isoformat(),
            "n_sessions": len(sessions),
        }
        for name, sessions in zip(_phase_names(), vars(block).values())
    }


# --------------------------------------------------------------------------- #
# Tensor / training helpers (torch is optional and imported lazily)
# --------------------------------------------------------------------------- #
def _tensors(torch, sessions: list[Session]):
    x = torch.tensor([s.features for s in sessions], dtype=torch.float32, device="cpu")
    y = torch.tensor([s.targets for s in sessions], dtype=torch.float32, device="cpu")
    return x, y


def _scaled_phases(torch, block: Block):
    """Fit feature standardization and the y (bp) scale on the scaler phase only.

    Returns scaled (monitor, warmup, branch, final) feature/target pairs where
    targets are divided by the fit-only y scale; the scale is returned too.
    """
    sx, sy = _tensors(torch, block.scaler)
    mean = sx.reshape(-1, sx.shape[-1]).mean(0)
    scale = sx.reshape(-1, sx.shape[-1]).std(0, unbiased=False).clamp_min(1e-6)
    y_scale = float(sy.reshape(-1).std(unbiased=False).clamp_min(1e-6).item())
    phases = {}
    for name in ("monitor", "warmup", "branch", "final"):
        fx, fy = _tensors(torch, getattr(block, name))
        phases[name] = ((fx - mean) / scale, fy / y_scale)
    return phases, y_scale


def _validation(torch, model, dataset):
    x, y = dataset
    with torch.no_grad():
        return torch.nn.functional.mse_loss(model(x).squeeze(-1), y).item()


def _train_microbatch(torch, model, x, y, denominator=1):
    loss = torch.nn.functional.mse_loss(model(x).squeeze(-1), y)
    (loss / denominator).backward()
    return float(loss.item())


def _norm(torch, model):
    squares = [p.grad.detach().double().square().sum() for p in model.parameters() if p.grad is not None]
    return float(torch.sqrt(torch.stack(squares).sum()).item())


def _build_optimizer(torch, model, profile):
    if profile["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=profile["lr"])
    return torch.optim.SGD(model.parameters(), lr=profile["lr"], momentum=profile["momentum"])


def _branch(torch, model, optimizer, checkpoint, branch, final, sample_ids, action):
    """Run one restored action branch; eight forward/backward calls, once."""
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    torch.set_rng_state(checkpoint["rng"])
    accumulation = 2 if action == "accumulate" else 1
    if action == "lower_lr":
        for group in optimizer.param_groups:
            group["lr"] *= 0.25
    clipping = 0.5 if action == "clip_grad" else None
    norms: list[float] = []
    consumed: list[str] = []
    started = time.perf_counter()
    for start in range(0, FUTURE_MICROBATCHES, accumulation):
        optimizer.zero_grad(set_to_none=True)
        for offset in range(start, start + accumulation):
            _train_microbatch(torch, model, branch[0][offset], branch[1][offset], accumulation)
            consumed.extend(sample_ids[offset])
        norms.append(_norm(torch, model))
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
        "effective_batch_size": TENOR_COUNT * accumulation,
        "lr": optimizer.param_groups[0]["lr"],
        "clip_grad_norm": clipping,
        "gradient_norm_history": norms,
        "clipping_calls": FUTURE_MICROBATCHES if action == "clip_grad" else 0,
    }


# --------------------------------------------------------------------------- #
# One measured case (telemetry/prompt frozen before any branch is scored)
# --------------------------------------------------------------------------- #
def _case(torch, split: str, block: Block, block_index: int, profile_index: int) -> dict:
    profile = PROFILES[profile_index]
    seed = SPLIT_SEED_BASES[split] + block_index * 1000 + profile_index
    torch.random.default_generator.manual_seed(seed)
    phases, y_scale = _scaled_phases(torch, block)
    monitor, warmup, branch, final = (phases[name] for name in ("monitor", "warmup", "branch", "final"))

    model = torch.nn.Sequential(
        torch.nn.Linear(len(FEATURES), 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)
    ).cpu()
    optimizer = _build_optimizer(torch, model, profile)

    history = [{"step": 0, "loss": _validation(torch, model, monitor)}]
    train_losses: list[float] = []
    norms: list[float] = []
    for step, (x, y) in enumerate(zip(*warmup, strict=True), 1):
        optimizer.zero_grad(set_to_none=True)
        train_losses.append(_train_microbatch(torch, model, x, y))
        norms.append(_norm(torch, model))
        optimizer.step()
        if step % 4 == 0:
            history.append({"step": step, "loss": _validation(torch, model, monitor)})

    features, targets = warmup
    observed = features.reshape(-1, features.shape[-1])
    # Pre-action telemetry: normalized, branch-independent, no future outcome.
    telemetry = {
        "device": "cpu", "dtype": "float32",
        "model": "6-16-1 tanh MLP", "optimizer": profile["optimizer"],
        "momentum": profile["momentum"] if profile["optimizer"] == "sgd" else None,
        "objective": "mean_squared_error",
        "target_units": "scaled_next_session_yield_change_bp",
        "training_steps": WARMUP_STEPS, "training_examples": targets.numel(),
        "validation_evaluations": len(history),
        "validation_examples_per_evaluation": MONITOR_DAYS * TENOR_COUNT,
        "train_loss_history": train_losses, "validation_loss_history": history,
        "gradient_norm_history": norms, "gradient_norm": norms[-1],
        "lr": profile["lr"], "clip_grad_norm": None,
        "microbatch_size": TENOR_COUNT, "batch_size": TENOR_COUNT, "accumulation_steps": 1,
        "data": {
            "observed_examples": targets.numel(), "features": len(FEATURES),
            "feature_mean": observed.mean(0).tolist(),
            "feature_std": observed.std(0, unbiased=False).tolist(),
            "feature_abs_max": float(observed.abs().max().item()), "missing_fraction": 0.0,
        },
        "target": {
            "mean": float(targets.mean().item()), "std": float(targets.std(unbiased=False).item()),
            "min": float(targets.min().item()), "max": float(targets.max().item()),
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
        [f"{session.date.isoformat()}/{tenor}" for tenor in TENORS] for session in block.branch
    ]
    order = (block_index + profile_index) % len(ACTIONS)
    actions = {
        action: _branch(torch, model, optimizer, checkpoint, branch, final, sample_ids, action)
        for action in ACTIONS[order:] + ACTIONS[:order]
    }
    sample_digest = _hash([sample for group in sample_ids for sample in group])
    if {row["future_sample_digest"] for row in actions.values()} != {sample_digest}:
        raise RuntimeError("Action branches used different training inputs")

    losses = {action: actions[action]["validation_loss"] for action in ACTIONS}
    label = oracle_label(losses)
    minimum = min(losses.values())
    for action in ACTIONS:
        actions[action]["regret"] = losses[action] - minimum
        actions[action]["validation_loss_change"] = losses[action] - history[-1]["loss"]
    phase_dates = _phase_record(block)
    split_low, split_high = SPLIT_YEARS[split]
    return {
        "schema_version": 1,
        "case_id": f"treasury_yield_mlp-{split}-b{block_index}-p{profile_index}",
        "track": "training", "family": "treasury_yield_mlp", "split": split,
        "benchmark": "treasury_intervention",
        "split_id": f"treasury:{split}:{split_low}-{split_high}",
        "evaluation_block": block.branch[0].date.isoformat(),
        "seed": seed, "seed_index": block_index, "profile_index": profile_index,
        "prompt": prompt, "label": label,
        "actions": {action: actions[action] for action in ACTIONS},
        "label_regret": losses[label] - minimum,
        "oracle": {"absolute_tolerance": NEAR_TIE_ABSOLUTE,
                   "relative_tolerance": NEAR_TIE_RELATIVE, "tie_order": list(ACTIONS)},
        "budget": {
            "matching": "ordered tenor-session examples and forward/backward calls; not optimizer steps",
            "equivalent_future_data": True,
            "future_examples": sum(len(group) for group in sample_ids),
            "future_sample_digest": sample_digest,
            "microbatch_size": TENOR_COUNT, "microbatches": FUTURE_MICROBATCHES,
            "forward_backward_calls": FUTURE_MICROBATCHES,
            "compute_note": "All actions use 8 forward/backward microbatches and one terminal validation. "
                            "accumulate makes 4 instead of 8 updates; clip_grad adds 8 clipping passes. "
                            "Times include validation, exclude restoration, and are not stable benchmarks.",
        },
        "source": {
            "source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256,
            "y_scale_bp": y_scale, "phase_dates": phase_dates,
            "tenors": list(TENORS), "features": list(FEATURES),
            "method_break": METHOD_BREAK.isoformat(),
            "retrospective_snapshot": True, "price_level_conclusion": False,
        },
        "provenance": {"generator": "treasury_yield_pytorch_cpu_v1",
                       "torch_version": str(torch.__version__), "financial_evidence": False},
    }


# --------------------------------------------------------------------------- #
# Generation + CLI
# --------------------------------------------------------------------------- #
def _measurement_hash(rows: list[dict]) -> str:
    """Permanent measurement digest, excluding wall-clock fields."""
    without_wall_time = [
        {
            **row,
            "actions": {
                action: {k: v for k, v in result.items() if k != "elapsed_seconds"}
                for action, result in row["actions"].items()
            },
        }
        for row in rows
    ]
    payload = json.dumps(without_wall_time, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def generate_split(torch, panel: list[Session], split: str) -> list[dict]:
    rows: list[dict] = []
    for block_index, (_, block) in enumerate(_blocks_for_split(panel, split)):
        for profile_index in range(len(PROFILES)):
            threads = torch.get_num_threads()
            deterministic = torch.are_deterministic_algorithms_enabled()
            warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
            try:
                torch.set_num_threads(1)
                torch.use_deterministic_algorithms(True)
                with torch.random.fork_rng(devices=[]):
                    rows.append(_case(torch, split, block, block_index, profile_index))
            finally:
                torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
                torch.set_num_threads(threads)
    return rows


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Treasury trace generation requires optional PyTorch; install torch "
            "in your telemetry environment. The main package need not depend on it."
        ) from exc
    return torch


def build(csv: Path, output: Path) -> dict:
    torch = _torch()
    digest, raw = parse_csv(csv)
    panel = build_panel(raw)
    output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict] = {}
    all_rows: list[dict] = []
    for split in SPLIT_YEARS:
        rows = generate_split(torch, panel, split)
        if not rows:
            raise ValueError(f"No complete blocks carved for split {split!r}")
        all_rows.extend(rows)
        write_jsonl(output / f"treasury-{split}.jsonl", rows)
        counts[split] = {"blocks": len({r["seed_index"] for r in rows}), "cases": len(rows),
                         "labels": dict(Counter(r["label"] for r in rows))}
    metadata = {
        "schema": 1, "source_url": SOURCE_URL, "source_sha256": digest,
        "device": "cpu", "torch_version": str(torch.__version__),
        "python_version": platform.python_version(), "os": platform.system(),
        "machine": platform.machine(),
        "split_years": {s: list(v) for s, v in SPLIT_YEARS.items()},
        "split_ids": {s: f"treasury:{s}:{lo}-{hi}" for s, (lo, hi) in SPLIT_YEARS.items()},
        "tenors": list(TENORS), "features": list(FEATURES),
        "profiles": list(PROFILES), "method_break": METHOD_BREAK.isoformat(),
        "label": "next-session signed yield change in basis points (fit-only scaled)",
        "validation": "terminal MSE on fit-only-scaled next-session changes",
        "counts": counts, "cases": len(all_rows),
        "outcome_counts": dict(Counter(r["label"] for r in all_rows)),
        "measurement_sha256": _measurement_hash(all_rows),
        "limits": "Archived retrospective snapshot; small MLP; modelling changes not levels; "
                  "no investment-performance or price-level claim.",
    }
    (output / "treasury.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True,
                        help=f"Staged, hash-verified Treasury CSV (default location {STAGED_PATH})")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for treasury-{train,valid,test}.jsonl")
    args = parser.parse_args(argv)
    metadata = build(args.csv, args.output)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
