"""Compute matched deltas, with seed-block intervals for synthetic training traces."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from random import Random
import statistics

from .dataset import read_jsonl


def _percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[int(fraction * (len(values) - 1))]


def _interval(blocks: dict, draws: int = 3000) -> list[float]:
    """Resample matched seed groups while keeping case profiles together."""
    rng = Random(819)
    groups = list(blocks.values())
    values = []
    for allocation in (rng.choices(groups, k=len(groups)) for _ in range(draws)):
        values.append(statistics.mean(x for group in allocation for x in group))
    return [_percentile(values, 0.025), _percentile(values, 0.975)]


def paired_deltas(base_path: Path, candidate_path: Path) -> dict:
    def index(path: Path) -> dict:
        rows = read_jsonl(path)
        by_id = {row["case_id"]: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"Repeated case IDs in {path}")
        return by_id

    base = index(base_path)
    candidate = index(candidate_path)
    if not base or set(base) != set(candidate):
        raise ValueError("Comparison requires identical nonempty case IDs")
    by_family = defaultdict(list)
    for case_id, baseline in base.items():
        trial = candidate[case_id]
        for field in ("family", "track", "split", "label", "block"):
            if baseline[field] != trial[field]:
                raise ValueError(f"{case_id} differs on {field}")
        by_family[(trial["track"], trial["family"])].append((baseline, trial))

    report = {"baseline": base_path.name, "candidate": candidate_path.name, "by_family": {}}
    for (track, family), pairs in sorted(by_family.items()):
        wins = sum(b["prediction"] != b["label"] and t["prediction"] == t["label"]
                   for b, t in pairs)
        losses = sum(b["prediction"] == b["label"] and t["prediction"] != t["label"]
                     for b, t in pairs)
        out = {"n": len(pairs), "newly_correct": wins, "newly_wrong": losses,
               "accuracy_delta": (wins - losses) / len(pairs)}
        if track == "training":
            blocks = defaultdict(list)
            for b, t in pairs:
                group = t["block"]
                blocks[group].append(t["normalized_regret"] - b["normalized_regret"])
            out["mean_normalized_regret_delta"] = statistics.mean(x for group in blocks.values() for x in group)
            if len(blocks) >= 8:
                out["block_bootstrap_95"] = _interval(blocks)
        report["by_family"][f"{track}/{family}"] = out
    return report


def write_comparison(base: Path, candidate: Path, output: Path) -> dict:
    report = paired_deltas(base, candidate)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
