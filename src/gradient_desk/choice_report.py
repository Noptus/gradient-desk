"""Paired, block-aware evaluation for blinded constrained predictions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

from .choice_data import normalized_action_losses
from .dataset import read_jsonl


def _index(rows, name):
    by_id = {row["case_id"]: row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError(f"{name} has duplicate case IDs")
    return by_id


def _panel(row):
    return row.get("benchmark", row["track"])


def _score(row, prediction):
    label = prediction.get("prediction")
    probabilities = prediction.get("probabilities")
    if label not in row["choices"] or (probabilities is not None and
                                       set(probabilities) != set(row["choices"])):
        raise ValueError(f"Wrong prediction menu: {row['case_id']}")
    if probabilities is not None:
        if not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                   for v in probabilities.values()):
            raise ValueError(f"Invalid probabilities: {row['case_id']}")
        if not math.isclose(math.fsum(probabilities.values()), 1.0, abs_tol=1e-5):
            raise ValueError(f"Probabilities must sum to one: {row['case_id']}")
    return label, probabilities


def _bounds(blocks, seed, replications):
    """Percentile interval sampling benchmark dependence blocks with replacement."""
    if len(blocks) < 4:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(replications):
        sample = rng.choices(blocks, k=len(blocks))
        draws.append(math.fsum(delta for delta, _ in sample) / sum(size for _, size in sample))
    draws.sort()
    return [draws[int(0.025 * (replications - 1))], draws[int(0.975 * (replications - 1))]]


def _signflip(deltas, replications, seed):
    """One-sided paired block randomization under exchangeability of policies."""
    gains = [value for value in deltas if value != 0]
    observed = math.fsum(gains)
    if len(gains) <= 15:
        possibilities = itertools.product((-1, 1), repeat=len(gains))
        successes = sum(math.fsum(x * s for x, s in zip(gains, signs)) >= observed - 1e-12
                        for signs in possibilities)
        return successes / (2 ** len(gains))
    rng = random.Random(seed)
    successes = 1
    for _ in range(replications):
        trial = math.fsum(x if rng.getrandbits(1) else -x for x in gains)
        successes += trial >= observed - 1e-12
    return successes / (replications + 1)


def _holm(panels):
    monotone = 0.0
    ordered = sorted(panels.values(), key=lambda row: row["p_block_one_sided"])
    for i, row in enumerate(ordered):
        monotone = max(monotone, min(1.0, (len(ordered) - i) * row["p_block_one_sided"]))
        row["p_holm"] = monotone


def compare(cases, base, candidate, replications=6000, seed=7121):
    if replications < 100:
        raise ValueError("Use at least 100 resamples")
    case_index = _index(cases, "Cases")
    base_index = _index(base, "Base predictions")
    candidate_index = _index(candidate, "Candidate predictions")
    if not cases or case_index.keys() != base_index.keys() or case_index.keys() != candidate_index.keys():
        raise ValueError("Cases and both predictions must cover exactly the same IDs")
    if {row["split"] for row in cases} != {"test"}:
        raise ValueError("This report is reserved for frozen test cases")
    grouped = defaultdict(list)
    for row in cases:
        key = row["case_id"]
        grouped[_panel(row)].append((row, _score(row, base_index[key]), _score(row, candidate_index[key])))

    panels = {}
    for panel, values in sorted(grouped.items()):
        blocks = defaultdict(list)
        policy_scores = defaultdict(list)
        confusion = defaultdict(lambda: defaultdict(Counter))
        by_class = Counter()
        measured = "actions" in values[0][0]
        if any(("actions" in row) != measured for row, _, _ in values):
            raise ValueError(f"Panel mixes different primary endpoints {panel}")
        for row, (b, b_probs), (c, c_probs) in values:
            label = row["label"]
            by_class[label] += 1
            if measured:
                loss = {name: outcome["validation_loss"] for name, outcome in row["actions"].items()}
                oracle = min(loss.values())
                utilities = {name: -value for name, value in normalized_action_losses(row).items()}
            else:
                utilities = {name: float(name == label) for name in row["choices"]}
            for role, action, probs in (("base", b, b_probs), ("candidate", c, c_probs)):
                policy_scores[role].append(utilities[action])
                confusion[role][label][action] += 1
                if measured:
                    policy_scores[f"{role}_raw_regret"].append(loss[action] - oracle)
                if probs is not None:
                    policy_scores[f"{role}_log_loss"].append(-math.log(max(probs[label], 1e-12)))
            block = str(row.get("evaluation_block", row.get("group_id", row["case_id"])))
            blocks[block].append(utilities[c] - utilities[b])
        packed = [(math.fsum(differences), len(differences)) for differences in blocks.values()]
        n = len(values)
        role_stats = {}
        for role in ("base", "candidate"):
            matrix = confusion[role]
            predicted = Counter()
            for label_counts in matrix.values():
                predicted.update(label_counts)
            role_stats[role] = {
                "accuracy": sum(matrix[label][label] for label in by_class) / n,
                "macro_accuracy": sum(matrix[key][key] / count
                                      for key, count in by_class.items()) / len(by_class),
                "action_counts": dict(sorted(predicted.items())),
                "confusion_reference_by_prediction": {
                    label: dict(sorted(matrix[label].items())) for label in sorted(matrix)
                },
                "primary": (-1 if measured else 1) * math.fsum(policy_scores[role]) / n,
            }
            if measured:
                role_stats[role]["mean_raw_regret"] = math.fsum(policy_scores[f"{role}_raw_regret"]) / n
            if len(policy_scores[f"{role}_log_loss"]) == n:
                role_stats[role]["log_loss"] = math.fsum(policy_scores[f"{role}_log_loss"]) / n
        panels[panel] = {
            "n": n,
            "dependence_blocks": len(blocks),
            "labels": dict(sorted(by_class.items())),
            "primary_metric": "mean_normalized_regret" if measured else "accuracy",
            "gain_direction": "positive favors candidate",
            "gain": math.fsum(delta for delta, _ in packed) / n,
            "paired_block_bootstrap95": _bounds(packed, seed, replications),
            "p_block_one_sided": _signflip([delta for delta, _ in packed], replications, seed),
            **role_stats,
        }
    _holm(panels)
    for row in panels.values():
        interval = row["paired_block_bootstrap95"]
        row["observed_improvement"] = row["gain"] >= 0.02
        row["confirmed_improvement"] = bool(
            row["observed_improvement"] and interval and interval[0] > 0 and row["p_holm"] < 0.05
        )
    return {
        "schema": 1,
        "resamples": replications,
        "seed": seed,
        "observed_improvements": sum(r["observed_improvement"] for r in panels.values()),
        "confirmed_improvements": sum(r["confirmed_improvement"] for r in panels.values()),
        "panels": panels,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = {"cases": args.cases, "base": args.base, "candidate": args.candidate}
    report = compare(*(read_jsonl(path) for path in files.values()))
    report["input_sha256"] = {role: hashlib.sha256(path.read_bytes()).hexdigest()
                              for role, path in files.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, allow_nan=False, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
