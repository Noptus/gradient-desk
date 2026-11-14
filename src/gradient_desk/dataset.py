"""Build a split-isolated MLX dataset and its auditable outcome files."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from random import Random

from .traces import ACTIONS

SPLITS = ("train", "valid", "test")
METHODS = {
    "audit_timing": "establish which inputs and universe members were observable at decision time",
    "improve_validation": "repair selection, split, dependence or inference before claiming evidence",
    "model_costs": "measure executable spreads, market impact, fees, or capacity",
    "limit_risk": "add exposure limits, liquidity or downside stress controls before enlargement",
    "clarify_thesis": "state a mechanism, testable prediction and condition that would falsify it",
    "run_pilot": "run a small, capped prospective experiment with stop criteria",
}
MODEL_REPO = "Qwen/Qwen3-8B-MLX-4bit"
MODEL_REVISION = "383413e909f3bc5303ce195ebbdf0339c5a1a2a3"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")


def method_prompt(scenario: str) -> str:
    menu = "; ".join(f"{label}: {meaning}" for label, meaning in METHODS.items())
    return (
        "You review investment research methodology. For this fictional research case, select the "
        "single highest-priority next research action. Reply with one action identifier. "
        f"Available actions: {menu}.\n\nCase: {scenario}"
    )


def methods_rows(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "case_id": case["id"],
            "track": "methodology",
            "family": case["family"],
            "split": case["split"],
            "prompt": method_prompt(case["scenario"]),
            "label": case["label"],
            "rationale": case["rationale"],
        }
        for case in cases
    ]


def validate_cases(rows: list[dict]) -> None:
    case_ids = set()
    family_splits = {}
    prompt_hashes = {}
    coverage = {}
    for row in rows:
        track, family, split = (row[key] for key in ("track", "family", "split"))
        if track not in ("training", "methodology") or split not in SPLITS:
            raise ValueError(f"Unknown track/split: {track!r}/{split!r}")
        actions = ACTIONS if track == "training" else METHODS
        if row["label"] not in actions:
            raise ValueError(f"Invalid label for {row['case_id']}")
        if row["case_id"] in case_ids:
            raise ValueError(f"Duplicate case ID: {row['case_id']}")
        case_ids.add(row["case_id"])
        if family_splits.setdefault((track, family), split) != split:
            raise ValueError(f"Family {track}/{family} crosses splits")
        digest = hashlib.sha256(row["prompt"].encode()).hexdigest()
        if prompt_hashes.setdefault(digest, split) != split:
            raise ValueError(f"Duplicate prompt crosses splits: {row['case_id']}")
        if track == "training":
            if set(row["actions"]) != set(ACTIONS):
                raise ValueError(f"Incomplete interventions: {row['case_id']}")
            if not all(math.isfinite(r["validation_loss"]) for r in row["actions"].values()):
                raise ValueError(f"Invalid loss: {row['case_id']}")
        coverage.setdefault((track, split), set()).add(row["label"])
    expected = {(track, split) for track in ("training", "methodology") for split in SPLITS}
    if set(coverage) != expected:
        raise ValueError("Every split must have cases for both tracks")


def prepare(traces: Path, methods: Path, output: Path, method_weight: int) -> dict:
    rows = read_jsonl(traces) + methods_rows(methods)
    validate_cases(rows)
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLITS:
        cases = [row for row in rows if row["split"] == split]
        examples = [
            {"prompt": row["prompt"], "completion": row["label"]}
            for row in cases
            for _ in range(method_weight if split == "train" and row["track"] == "methodology" else 1)
        ]
        Random(f"gradient-desk-v1:{split}").shuffle(examples)
        write_jsonl(output / f"{split}.jsonl", examples)
        write_jsonl(output / f"{split}-cases.jsonl", cases)
        counts[split] = {
            "examples": len(examples),
            "training": sum(row["track"] == "training" for row in cases),
            "methodology": sum(row["track"] == "methodology" for row in cases),
        }
    return counts
