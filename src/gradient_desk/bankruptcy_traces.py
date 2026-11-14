"""Independent OOD holdout of PyTorch counterfactual optimizer interventions on
public corporate financial-ratio data (UCI Taiwanese Bankruptcy Prediction).

Export: python -m gradient_desk.bankruptcy_traces --archive PATH --output holdout.jsonl

This mirrors the budget / oracle / telemetry contract of ``gradient_desk.traces``
on an *independent, frozen, test-only* public panel: anonymously aggregated
company financial ratios with published bankruptcy labels (1999-2009). It is a
**training-intervention** benchmark on a small MLP: for each isolated company-row
group we warm up a classifier, then measure the post-action loss of every action
against *published* labels. Nothing here predicts any named business, and no
binary case prediction is emitted; only normalized training telemetry is shown.

Source (never embedded in code; only its locator and digest are pinned)
    UCI Taiwanese Bankruptcy Prediction, DOI 10.24432/C5004D, CC BY 4.0.
        https://archive.ics.uci.edu/static/public/572/taiwanese+bankruptcy+prediction.zip
    SHA-256 pinned in ``SOURCE_SHA256``. The caller stages the verified zip
    (default ``.work/v2/uci-taiwan-bankruptcy.zip``). The sole archive member
    ``data.csv`` (~11.45 MB) is decoded only after the whole archive's hash is
    verified and after explicit zip/decompressed size limits; the raw ratios are
    never redistributed here and are not written outside an ignored location.

Test-only / leakage
    There is no same-domain training protocol: every carved group is emitted on
    the ``test`` split and this generator is never used during validation
    selection. Groups are disjoint source-row partitions (no row reused); the
    dataset carries no issuer identifiers, so we *cannot* guarantee the absence
    of issuer duplication across groups and say so explicitly. Within each group
    the scaler is fit on the scaler phase only, checkpoint selection never looks
    at terminal data, and the prompt/pre-action telemetry is frozen before any
    action is scored.

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
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import time
import zipfile

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
SOURCE_URL = "https://archive.ics.uci.edu/static/public/572/taiwanese+bankruptcy+prediction.zip"
SOURCE_DOI = "10.24432/C5004D"
SOURCE_LICENSE = "CC BY 4.0"
SOURCE_SHA256 = "c346f5ad2618cb198e7ed8306cf2f31fe3bb2ec60acdbbe1736788d50f269aac"
STAGED_PATH = Path(".work/v2/uci-taiwan-bankruptcy.zip")
ARCHIVE_MEMBER = "data.csv"

# Decode limits (bytes). The archive is ~ a few hundred KB compressed with one
# ~11.45 MB member; these caps bound both the stored zip and its inflation.
MAX_ZIP_BYTES = 10 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 18 * 1024 * 1024

#: Explicit binary target column and the number of leading numeric features used.
TARGET_COLUMN = "Bankrupt?"
FEATURE_COUNT = 16

# This is a frozen, test-only OOD holdout: there is no same-domain training or
# validation protocol here.
SPLIT = "test"
#: Predeclared RNG seed namespace for the shuffle of source row indices.
SHUFFLE_SEED = 572024
#: Predeclared per-case Torch seed namespace (distinct from the shuffle seed).
CASE_SEED_BASE = 5720000

# --------------------------------------------------------------------------- #
# Group / phase layout (company rows). A group is a disjoint block of rows.
# --------------------------------------------------------------------------- #
GROUP_MIN = 192
GROUP_MAX = 240
SCALER_ROWS = 60          # fit-only robust scaler statistics (median / IQR)
MONITOR_ROWS = 32         # pre-action monitor (branch-independent)
TERMINAL_ROWS = 24        # terminal scoring rows (>= 24, independent of monitor)
# WARMUP_STEPS (=12) warmup microbatches and FUTURE_MICROBATCHES (=8) branch
# microbatches are sized per profile (4 or 6 rows each), hence "12*4 or 12*6"
# warmup examples and "8*4 or 8*6" branch examples.

#: Fixed positive-class weight for the imbalanced BCE objective (8 or 12).
POS_WEIGHT_CHOICES = (8.0, 12.0)
#: Standardized-feature clamp after robust scaling.
CLAMP = 5.0

# Four-to-five moderate optimizer profiles per group. microbatch is 4 or 6, so
# warmup uses 12*mb and branch 8*mb examples; pos_weight is 8 or 12.
PROFILES: tuple[dict, ...] = (
    {"lr": 0.02, "optimizer": "sgd", "momentum": 0.8, "microbatch": 4, "pos_weight": 8.0},
    {"lr": 0.08, "optimizer": "sgd", "momentum": 0.8, "microbatch": 6, "pos_weight": 12.0},
    {"lr": 0.04, "optimizer": "sgd", "momentum": 0.9, "microbatch": 6, "pos_weight": 8.0},
    {"lr": 0.01, "optimizer": "adam", "momentum": None, "microbatch": 4, "pos_weight": 12.0},
    {"lr": 0.003, "optimizer": "adam", "momentum": None, "microbatch": 6, "pos_weight": 8.0},
)


def _profile_rows(profile: dict) -> int:
    """Rows a profile consumes from a group: scaler + monitor + warmup + branch + terminal."""
    mb = profile["microbatch"]
    return SCALER_ROWS + MONITOR_ROWS + WARMUP_STEPS * mb + FUTURE_MICROBATCHES * mb + TERMINAL_ROWS


#: A group must hold enough rows for the largest profile and stay within bounds.
MAX_PROFILE_ROWS = max(_profile_rows(profile) for profile in PROFILES)
if not GROUP_MIN <= MAX_PROFILE_ROWS <= GROUP_MAX:  # predeclared invariant
    raise AssertionError("Profile phase budget does not fit the declared group size band")


# --------------------------------------------------------------------------- #
# Archive integrity + decoding (hash verified before any decode)
# --------------------------------------------------------------------------- #
def _verify_hash(payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(
            "Bankruptcy archive SHA-256 mismatch: refusing to decode untrusted data.\n"
            f"  expected {SOURCE_SHA256}\n  actual   {digest}"
        )
    return digest


def _read_member(payload: bytes) -> str:
    """Decode the single ``data.csv`` member under explicit size caps.

    The whole-archive hash is verified by the caller first. Here we additionally
    bound the stored archive and the inflated member so a surprising archive
    cannot exhaust memory.
    """
    if len(payload) > MAX_ZIP_BYTES:
        raise ValueError(f"Archive exceeds the {MAX_ZIP_BYTES}-byte zip limit")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if names != [ARCHIVE_MEMBER]:
            raise ValueError(f"Archive must contain only {ARCHIVE_MEMBER!r}; found {names!r}")
        info = archive.getinfo(ARCHIVE_MEMBER)
        if info.file_size > MAX_DECOMPRESSED_BYTES:
            raise ValueError(f"Member exceeds the {MAX_DECOMPRESSED_BYTES}-byte decompressed limit")
        with archive.open(info) as stream:
            raw = stream.read(MAX_DECOMPRESSED_BYTES + 1)
    if len(raw) > MAX_DECOMPRESSED_BYTES:
        raise ValueError("Decompressed member exceeds the declared limit")
    return raw.decode("utf-8-sig")


def _split_row(line: str) -> list[str]:
    # The UCI export is a plain comma table with no quoted fields; keep parsing
    # explicit so a stray quote or embedded comma is rejected, not mis-split.
    if '"' in line:
        raise ValueError("Unexpected quoting in the financial-ratio table")
    return [cell.strip() for cell in line.split(",")]


@dataclass(frozen=True)
class SourceTable:
    feature_names: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]   # canonical de-duplicated feature vectors
    labels: tuple[int, ...]               # aligned published binary labels
    raw_row_count: int
    duplicate_groups: int
    conflicting_groups: int
    conflicting_rows: int


def parse_archive(path: Path) -> tuple[str, SourceTable]:
    """Verify the pinned hash, decode ``data.csv``, and canonicalize features.

    Selects the explicit binary ``Bankrupt?`` target and the first
    ``FEATURE_COUNT`` non-target numeric feature columns. Feature vectors are
    canonicalized (rounded to a fixed precision) and de-duplicated exactly;
    duplicate feature groups with conflicting labels are skipped entirely and
    counted. Returns ``(sha256, SourceTable)``.
    """
    with path.open("rb") as stream:
        payload = stream.read(MAX_ZIP_BYTES + 1)
    digest = _verify_hash(payload)
    text = _read_member(payload)

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty bankruptcy table")
    header = _split_row(lines[0])
    if header.count(TARGET_COLUMN) != 1:
        raise ValueError(f"Expected exactly one {TARGET_COLUMN!r} target column")
    target_index = header.index(TARGET_COLUMN)
    feature_indices = [i for i in range(len(header)) if i != target_index][:FEATURE_COUNT]
    if len(feature_indices) != FEATURE_COUNT:
        raise ValueError(f"Table has fewer than {FEATURE_COUNT} non-target columns")
    feature_names = tuple(header[i] for i in feature_indices)

    # Group canonical feature vectors -> set of labels seen with that vector.
    grouped: dict[tuple[float, ...], set[int]] = {}
    order: list[tuple[float, ...]] = []
    raw_count = 0
    for line in lines[1:]:
        cells = _split_row(line)
        if len(cells) != len(header):
            raise ValueError(f"Expected {len(header)} columns, got {len(cells)}")
        label = _parse_label(cells[target_index])
        vector = tuple(_canonical(cells[i]) for i in feature_indices)
        raw_count += 1
        if vector not in grouped:
            grouped[vector] = set()
            order.append(vector)
        grouped[vector].add(label)

    rows: list[tuple[float, ...]] = []
    labels: list[int] = []
    duplicate_groups = 0
    conflicting_groups = 0
    conflicting_rows = 0
    for vector in order:
        seen = grouped[vector]
        if len(seen) > 1:
            # Conflicting labels on identical features: skip the whole tuple.
            conflicting_groups += 1
            conflicting_rows += 1
            continue
        rows.append(vector)
        labels.append(next(iter(seen)))
    duplicate_groups = raw_count - len(order)  # exact-duplicate feature rows collapsed

    if sum(labels) == 0:
        raise ValueError("No positive (bankrupt) labels survived canonicalization")
    table = SourceTable(
        feature_names=feature_names,
        rows=tuple(rows),
        labels=tuple(labels),
        raw_row_count=raw_count,
        duplicate_groups=duplicate_groups,
        conflicting_groups=conflicting_groups,
        conflicting_rows=conflicting_rows,
    )
    return digest, table


def _parse_label(token: str) -> int:
    token = token.strip()
    if token not in ("0", "1"):
        raise ValueError(f"Non-binary target value: {token!r}")
    return int(token)


def _canonical(token: str) -> float:
    token = token.strip()
    try:
        value = float(token)
    except ValueError as exc:
        raise ValueError(f"Non-numeric feature cell: {token!r}") from exc
    if not math.isfinite(value):
        raise ValueError("Non-finite feature cell in public ratios")
    # Fixed-precision canonicalization so trivially-equal vectors de-duplicate.
    return round(value, 10)


# --------------------------------------------------------------------------- #
# Deterministic shuffle + stratified disjoint grouping
# --------------------------------------------------------------------------- #
def _shuffle_indices(n: int) -> list[int]:
    """Seeded, predeclared Fisher-Yates over source row indices (no numpy/torch)."""
    order = list(range(n))
    for i in range(n - 1, 0, -1):
        digest = hashlib.sha256(f"taiwan-bankruptcy:{SHUFFLE_SEED}:{i}".encode()).digest()
        j = int.from_bytes(digest[:8], "big") % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def carve_groups(table: SourceTable, group_size: int = MAX_PROFILE_ROWS) -> list[list[int]]:
    """Balance rare positives across disjoint source-row groups."""
    if not GROUP_MIN <= group_size <= GROUP_MAX:
        raise ValueError(f"group_size must be within [{GROUP_MIN}, {GROUP_MAX}]")
    shuffled = _shuffle_indices(len(table.rows))
    positives = [i for i in shuffled if table.labels[i] == 1]
    negatives = [i for i in shuffled if table.labels[i] == 0]
    n_groups = len(table.rows) // group_size
    if n_groups < 1:
        raise ValueError("Not enough rows to carve a single group")
    if len(positives) < n_groups:
        n_groups = len(positives)  # guarantee >=1 positive per group
    if n_groups < 1:
        raise ValueError("No positive-bearing group is possible")
    positives = positives[:n_groups * group_size]

    # The branch anchor lies in both profile microbatch schedules. Terminal
    # rows are the same within each group, so all profiles measure the same
    # outcomes while retaining disjoint pre-action datasets.
    shared_branch_start = SCALER_ROWS + MONITOR_ROWS + WARMUP_STEPS * max(
        profile["microbatch"] for profile in PROFILES
    )
    anchors = (SCALER_ROWS, SCALER_ROWS + MONITOR_ROWS,
               shared_branch_start, group_size - TERMINAL_ROWS)
    groups: list[list[int]] = []
    cursor = 0
    for g in range(n_groups):
        positive_rows = positives[g::n_groups]
        need = group_size - len(positive_rows)
        negative_rows = negatives[cursor:cursor + need]
        cursor += need
        if len(negative_rows) != need:
            raise RuntimeError("Insufficient rows for a complete stratified group")
        # Ensure one rare-class example reaches the terminal metric. When the
        # source has enough positives, give the other three phases one apiece.
        reserved = dict(zip((anchors[-1], *anchors[:-1]), positive_rows))
        remaining = positive_rows[len(reserved):] + negative_rows
        fill = iter(remaining[index] for index in _shuffle_indices(len(remaining)))
        groups.append([reserved[index] if index in reserved else next(fill)
                       for index in range(group_size)])
    # Disjointness invariant (explicit leakage guard).
    seen: set[int] = set()
    for g in groups:
        for index in g:
            if index in seen:
                raise RuntimeError("A source row was assigned to two groups")
            seen.add(index)
    return groups


# --------------------------------------------------------------------------- #
# Phase slicing within a group (ordered by the shuffled assignment)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Phases:
    scaler: list[int]
    monitor: list[int]
    warmup: list[int]
    branch: list[int]
    terminal: list[int]


def slice_phases(group: list[int], profile: dict) -> Phases:
    mb = profile["microbatch"]
    sizes = (SCALER_ROWS, MONITOR_ROWS, WARMUP_STEPS * mb, FUTURE_MICROBATCHES * mb)
    cursor = 0
    parts: list[list[int]] = []
    for size in sizes:
        parts.append(group[cursor:cursor + size])
        cursor += size
    if cursor > len(group) - TERMINAL_ROWS:
        raise ValueError("Group too small for the requested phase budget")
    parts.append(group[-TERMINAL_ROWS:])
    phases = Phases(*parts)
    # No phase shares a row (disjoint slices of a disjoint group).
    flat = [i for part in parts for i in part]
    if len(set(flat)) != len(flat):
        raise RuntimeError("Phase slices overlap within a group")
    return phases


# --------------------------------------------------------------------------- #
# Hash helpers
# --------------------------------------------------------------------------- #
def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _rng_hash(state) -> str:
    return hashlib.sha256(bytes(state.tolist())).hexdigest()


# --------------------------------------------------------------------------- #
# Tensor / training helpers (torch is optional and imported lazily)
# --------------------------------------------------------------------------- #
def _features(table: SourceTable, indices: list[int]) -> list[tuple[float, ...]]:
    return [table.rows[i] for i in indices]


def _labels(table: SourceTable, indices: list[int]) -> list[int]:
    return [table.labels[i] for i in indices]


def _robust_stats(torch, rows: list[tuple[float, ...]]):
    """Fit-only robust center/scale: feature median and (IQR/1.349) proxy std."""
    x = torch.tensor(rows, dtype=torch.float32)
    center = x.median(dim=0).values
    q1 = torch.quantile(x, 0.25, dim=0)
    q3 = torch.quantile(x, 0.75, dim=0)
    scale = ((q3 - q1) / 1.349).clamp_min(1e-6)
    return center, scale


def _standardize(torch, rows: list[tuple[float, ...]], center, scale):
    x = torch.tensor(rows, dtype=torch.float32)
    return ((x - center) / scale).clamp(-CLAMP, CLAMP)


def _bce(torch, logits, targets, pos_weight):
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=torch.tensor(pos_weight, dtype=torch.float32)
    )


def _validate(torch, model, x, y, pos_weight):
    model.eval()
    with torch.no_grad():
        loss = _bce(torch, model(x).squeeze(-1), y, pos_weight).item()
    model.train()
    return loss


def _grad_norm(torch, model):
    squares = [p.grad.detach().double().square().sum() for p in model.parameters() if p.grad is not None]
    return float(torch.sqrt(torch.stack(squares).sum()).item())


def _build_optimizer(torch, model, profile):
    if profile["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=profile["lr"])
    return torch.optim.SGD(model.parameters(), lr=profile["lr"], momentum=profile["momentum"])


def _branch(torch, model, optimizer, checkpoint, batches, terminal, sample_ids, action, pos_weight):
    """Run one restored action branch: eight forward/backward calls, once."""
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
            x, y = batches[offset]
            loss = _bce(torch, model(x).squeeze(-1), y, pos_weight) / accumulation
            loss.backward()
            consumed.extend(sample_ids[offset])
        norms.append(_grad_norm(torch, model))
        if clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
        optimizer.step()
    loss = _validate(torch, model, terminal[0], terminal[1], pos_weight)
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
        "lr": optimizer.param_groups[0]["lr"],
        "clip_grad_norm": clipping,
        "gradient_norm_history": norms,
        "clipping_calls": FUTURE_MICROBATCHES if action == "clip_grad" else 0,
    }


# --------------------------------------------------------------------------- #
# One measured case (telemetry/prompt frozen before any branch is scored)
# --------------------------------------------------------------------------- #
def _case(torch, table: SourceTable, group: list[int], group_index: int, profile_index: int) -> dict:
    profile = PROFILES[profile_index]
    mb = profile["microbatch"]
    pos_weight = profile["pos_weight"]
    seed = CASE_SEED_BASE + group_index * 100 + profile_index
    torch.random.default_generator.manual_seed(seed)

    phases = slice_phases(group, profile)
    center, scale = _robust_stats(torch, _features(table, phases.scaler))

    def scaled(indices):
        x = _standardize(torch, _features(table, indices), center, scale)
        y = torch.tensor([float(v) for v in _labels(table, indices)], dtype=torch.float32)
        return x, y

    monitor = scaled(phases.monitor)
    terminal = scaled(phases.terminal)
    warmup_x, warmup_y = scaled(phases.warmup)
    branch_x, branch_y = scaled(phases.branch)

    model = torch.nn.Sequential(
        torch.nn.Linear(FEATURE_COUNT, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)
    ).cpu()
    optimizer = _build_optimizer(torch, model, profile)

    history = [{"step": 0, "loss": _validate(torch, model, monitor[0], monitor[1], pos_weight)}]
    train_losses: list[float] = []
    norms: list[float] = []
    for step in range(WARMUP_STEPS):
        x = warmup_x[step * mb:(step + 1) * mb]
        y = warmup_y[step * mb:(step + 1) * mb]
        optimizer.zero_grad(set_to_none=True)
        loss = _bce(torch, model(x).squeeze(-1), y, pos_weight)
        loss.backward()
        norms.append(_grad_norm(torch, model))
        optimizer.step()
        train_losses.append(float(loss.item()))
        if (step + 1) % 4 == 0:
            history.append({"step": step + 1, "loss": _validate(torch, model, monitor[0], monitor[1], pos_weight)})

    observed = warmup_x
    # Pre-action telemetry: normalized, finite, branch-independent, no outcome and
    # no binary case prediction. positive_fraction is a warmup-only summary.
    positive_fraction = float(warmup_y.mean().item())
    telemetry = {
        "device": "cpu", "dtype": "float32",
        "model": f"{FEATURE_COUNT}-16-1 tanh MLP", "optimizer": profile["optimizer"],
        "momentum": profile["momentum"] if profile["optimizer"] == "sgd" else None,
        "objective": "binary_cross_entropy",
        "pos_weight": pos_weight,
        "training_steps": WARMUP_STEPS, "training_examples": warmup_y.numel(),
        "validation_evaluations": len(history),
        "validation_examples_per_evaluation": MONITOR_ROWS,
        "train_loss_history": train_losses, "validation_loss_history": history,
        "gradient_norm_history": norms, "gradient_norm": norms[-1],
        "lr": profile["lr"], "clip_grad_norm": None,
        "microbatch_size": mb, "batch_size": mb, "accumulation_steps": 1,
        "data": {
            "observed_examples": warmup_y.numel(), "features": FEATURE_COUNT,
            "feature_mean": observed.mean(0).tolist(),
            "feature_std": observed.std(0, unbiased=False).tolist(),
            "feature_abs_max": float(observed.abs().max().item()), "missing_fraction": 0.0,
        },
        "target": {
            "mean": positive_fraction, "std": float(warmup_y.std(unbiased=False).item()),
            "min": float(warmup_y.min().item()), "max": float(warmup_y.max().item()),
            "positive_fraction": positive_fraction,
        },
    }
    prompt = format_telemetry(telemetry)

    checkpoint = {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "rng": torch.get_rng_state().clone(),
    }
    batches = [
        (branch_x[k * mb:(k + 1) * mb], branch_y[k * mb:(k + 1) * mb])
        for k in range(FUTURE_MICROBATCHES)
    ]
    # Stable, non-raw sample identifiers: group-local phase positions only. No raw
    # ratio, label, or source-row index is ever written to a trace.
    sample_ids = [
        [f"g{group_index}/branch/{k * mb + off}" for off in range(mb)]
        for k in range(FUTURE_MICROBATCHES)
    ]
    order = (group_index + profile_index) % len(ACTIONS)
    actions = {
        action: _branch(torch, model, optimizer, checkpoint, batches, terminal, sample_ids, action, pos_weight)
        for action in ACTIONS[order:] + ACTIONS[:order]
    }
    sample_digest = _hash([sample for group_ids in sample_ids for sample in group_ids])
    if {row["future_sample_digest"] for row in actions.values()} != {sample_digest}:
        raise RuntimeError("Action branches used different training inputs")

    losses = {action: actions[action]["validation_loss"] for action in ACTIONS}
    label = oracle_label(losses)
    minimum = min(losses.values())
    for action in ACTIONS:
        actions[action]["regret"] = losses[action] - minimum
        actions[action]["validation_loss_change"] = losses[action] - history[-1]["loss"]

    evaluation_block = {
        "canonical_group": f"taiwan_bankruptcy:test:g{group_index}",
        "group_index": group_index,
        "group_size": len(group),
        "group_positive_count": sum(table.labels[i] for i in group),
        "phase_sizes": {
            "scaler": len(phases.scaler), "monitor": len(phases.monitor),
            "warmup": len(phases.warmup), "branch": len(phases.branch),
            "terminal": len(phases.terminal),
        },
        "issuer_identifiers": False,
        "issuer_duplication_guarantee": False,
        "disjoint_from_other_groups": True,
    }
    return {
        "schema_version": 1,
        "case_id": f"taiwan_bankruptcy_mlp-{SPLIT}-g{group_index}-p{profile_index}",
        "track": "training", "family": "taiwan_bankruptcy_mlp", "split": SPLIT,
        "benchmark": "taiwan_bankruptcy_intervention",
        "evaluation_block": evaluation_block["canonical_group"],
        "group_id": evaluation_block["canonical_group"],
        "group_metadata": evaluation_block,
        "seed": seed, "seed_index": group_index, "profile_index": profile_index,
        "prompt": prompt, "label": label,
        "actions": {action: actions[action] for action in ACTIONS},
        "label_regret": losses[label] - minimum,
        "oracle": {"absolute_tolerance": NEAR_TIE_ABSOLUTE,
                   "relative_tolerance": NEAR_TIE_RELATIVE, "tie_order": list(ACTIONS)},
        "budget": {
            "matching": "ordered company-row microbatches and forward/backward calls; not optimizer steps",
            "equivalent_future_data": True,
            "future_examples": FUTURE_MICROBATCHES * mb,
            "future_sample_digest": sample_digest,
            "microbatch_size": mb, "microbatches": FUTURE_MICROBATCHES,
            "forward_backward_calls": FUTURE_MICROBATCHES,
            "compute_note": "All actions use 8 forward/backward microbatches and one terminal validation. "
                            "accumulate makes 4 instead of 8 updates; clip_grad adds 8 clipping passes. "
                            "Times include validation, exclude restoration, and are not stable benchmarks.",
        },
        "data_source": {
            "source_url": SOURCE_URL, "source_doi": SOURCE_DOI, "source_license": SOURCE_LICENSE,
            "source_sha256": SOURCE_SHA256, "archive_member": ARCHIVE_MEMBER,
            "target_column": TARGET_COLUMN, "feature_names": list(table.feature_names),
            "feature_count": FEATURE_COUNT, "period": "1999-2009",
            "label_basis": "published bankruptcy labels",
            "no_named_business_prediction": True,
        },
        "evaluation_receipts": {
            "pos_weight": pos_weight,
            "robust_scaling": "median center, IQR/1.349 scale, clamp [-5,5], fit on scaler phase only",
            "checkpoint_selection_sees_terminal": False,
            "test_only": True, "same_domain_training": False,
        },
        "provenance": {"generator": "taiwan_bankruptcy_pytorch_cpu_v1",
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


def generate(torch, table: SourceTable) -> list[dict]:
    groups = carve_groups(table)
    if not groups:
        raise ValueError("No complete groups carved from the source")
    rows: list[dict] = []
    for group_index, group in enumerate(groups):
        for profile_index in range(len(PROFILES)):
            threads = torch.get_num_threads()
            deterministic = torch.are_deterministic_algorithms_enabled()
            warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
            try:
                torch.set_num_threads(1)
                torch.use_deterministic_algorithms(True)
                with torch.random.fork_rng(devices=[]):
                    rows.append(_case(torch, table, group, group_index, profile_index))
            finally:
                torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
                torch.set_num_threads(threads)
    return rows


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Bankruptcy trace generation requires optional PyTorch; install torch "
            "in your telemetry environment. The main package need not depend on it."
        ) from exc
    return torch


def build(archive: Path, output: Path) -> dict:
    torch = _torch()
    digest, table = parse_archive(archive)
    rows = generate(torch, table)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")

    n_groups = len({row["seed_index"] for row in rows})
    receipt = {
        "schema": 1,
        "benchmark": "taiwan_bankruptcy_intervention",
        "family": "taiwan_bankruptcy_mlp",
        "split": SPLIT,
        "source": {
            "source_url": SOURCE_URL, "source_doi": SOURCE_DOI,
            "source_license": SOURCE_LICENSE, "source_sha256": digest,
            "archive_member": ARCHIVE_MEMBER, "period": "1999-2009",
        },
        "canonicalization": {
            "raw_rows": table.raw_row_count,
            "unique_feature_groups_kept": len(table.rows),
            "duplicate_feature_rows_collapsed": table.duplicate_groups,
            "conflicting_label_groups_skipped": table.conflicting_groups,
            "conflicting_rows_skipped": table.conflicting_rows,
            "positive_rows_kept": sum(table.labels),
        },
        "grouping": {
            "group_size": MAX_PROFILE_ROWS, "group_min": GROUP_MIN, "group_max": GROUP_MAX,
            "groups": n_groups, "shuffle_seed": SHUFFLE_SEED,
            "stratified_source_wide": True, "disjoint": True,
            "issuer_identifiers": False, "issuer_duplication_guarantee": False,
        },
        "counts": {
            "cases": len(rows), "profiles": len(PROFILES),
            "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
        },
        "environment": {
            "device": "cpu", "torch_version": str(torch.__version__),
            "python_version": platform.python_version(), "os": platform.system(),
            "machine": platform.machine(),
        },
        "limits": "Frozen test-only OOD holdout; small MLP; published labels; "
                  "no named-business prediction and no investment claim.",
        "measurement_sha256": _measurement_hash(rows),
    }
    print(json.dumps(receipt, indent=2))
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True,
                        help=f"Staged, hash-verified UCI zip (default location {STAGED_PATH})")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL file in the legacy trace schema")
    args = parser.parse_args(argv)
    build(args.archive, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
