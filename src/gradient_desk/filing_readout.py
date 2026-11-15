"""Fit and evaluate a supervised SEC section readout on frozen Qwen states."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import file_digest, sha256
import json
import math
from pathlib import Path

from .choice_data import validate
from .choice_model import (
    EXPECTED_WEIGHTS_SHA256, _candidate_table, _encode_prompt, _lm_head, _load,
    _pad_batch, _pad_candidates, _pad_id, candidate_probabilities,
)
from .choice_report import compare
from .dataset import read_jsonl, write_jsonl

TRACK = "sec8k_item_routing"
RIDGE = (0.001, 0.01, 0.1, 1.0, 10.0)
ALPHA = (0.0, 0.25, 0.5, 0.75, 1.0)
VARIANTS = 2
BATCH_SIZE = 8
BLIND_FIELDS = ("case_id", "track", "prompt", "choices")
SEC_SPLITS = (
    "0b6c016bd301220bc2f8ee4ab8722669443faa2dea53be9321c7336989f8723b",
    "893dfbef1567735d474dc86f61c1526e4e4a4b894c1aafa8e576c7b3f2ca1748",
)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _extract(model_path: Path, cases: list[dict]) -> list[tuple]:
    """Return the averaged normalized hidden state and legal probabilities per case."""
    import mlx.core as mx
    import numpy as np

    if _digest(model_path / "model.safetensors") != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("Model does not match the frozen base checkpoint")
    model, tokenizer = _load(model_path, None)
    model.eval()
    table = _candidate_table(tokenizer)
    pad = _pad_id(tokenizer)
    work = [(i, *_encode_prompt(tokenizer, case, variant))
            for i, case in enumerate(cases) for variant in range(VARIANTS)]
    work.sort(key=lambda item: len(item[1]))
    hidden_sums = [None] * len(cases)
    probability_sums = [defaultdict(float) for _ in cases]
    for start in range(0, len(work), BATCH_SIZE):
        batch = work[start:start + BATCH_SIZE]
        ids, lengths = _pad_batch([tokens for _, tokens, _ in batch], pad)
        candidate_ids, mask = _pad_candidates([order for _, _, order in batch], table)
        hidden = model.model(ids)
        last = mx.broadcast_to((lengths - 1).reshape(-1, 1, 1),
                               (hidden.shape[0], 1, hidden.shape[-1]))
        last_hidden = mx.take_along_axis(hidden, last, axis=1)
        full = _lm_head(model, last_hidden).squeeze(1)
        logits = mx.take_along_axis(full, candidate_ids, axis=1).astype(mx.float32)
        probabilities = candidate_probabilities(logits, mask)
        mx.eval(last_hidden, probabilities)
        vectors = np.asarray(last_hidden.squeeze(1).astype(mx.float32), dtype=np.float64)
        masses = np.asarray(probabilities, dtype=np.float64)
        for position, (index, _, order) in enumerate(batch):
            vector = vectors[position]
            vector = vector / (np.linalg.norm(vector) + 1e-12)
            previous = hidden_sums[index]
            hidden_sums[index] = vector if previous is None else previous + vector
            for slot, option in enumerate(order):
                probability_sums[index][option] += float(masses[position, slot])
        if start == 0 or (start // BATCH_SIZE + 1) % 40 == 0 or start + BATCH_SIZE >= len(work):
            print(f"Extracted {min(start + BATCH_SIZE, len(work))}/{len(work)} prompt layouts", flush=True)
    features = []
    for i, case in enumerate(cases):
        base = np.array([probability_sums[i][label] for label in sorted(case["choices"])])
        if not np.isfinite(base).all() or not base.sum() > 0 or not np.isfinite(hidden_sums[i]).all():
            raise ValueError(f"Invalid frozen features for {case['case_id']}")
        features.append((hidden_sums[i] / VARIANTS, base / base.sum()))
    return features


def fit_head(hidden, targets, regularization):
    """Solve ridge in sample space and return a primal matrix with intercept."""
    import numpy as np

    feature_mean = hidden.mean(axis=0)
    target_mean = targets.mean(axis=0)
    centered = hidden - feature_mean
    gram = centered @ centered.T
    gram.flat[::len(gram) + 1] += regularization
    weights = centered.T @ np.linalg.solve(gram, targets - target_mean)
    return np.vstack((weights, target_mean - feature_mean @ weights))


def distribution(head, alpha: float, feature, labels: list[str]) -> dict[str, float]:
    """Score a blinded case from its two-layout representation and base probabilities."""
    import numpy as np

    hidden, base = feature
    scores = np.r_[hidden, 1.0] @ head
    projected = np.clip(scores, 0.0, None)
    if not projected.sum() > 0:
        projected = np.exp(scores - scores.max())
    result = (1 - alpha) * base + alpha * (projected / projected.sum())
    if not np.isfinite(result).all() or not result.sum() > 0:
        raise ValueError("Readout returned an invalid probability distribution")
    return dict(zip(labels, (result / result.sum()).tolist(), strict=True))


def _prediction(probabilities: dict[str, float]) -> str:
    return max(probabilities, key=lambda label: (probabilities[label], label))


def _check_plan(plan: dict) -> None:
    fit = plan["training"]
    if (fit["track"], fit["layouts"], fit["batch_size"]) != (TRACK, VARIANTS, BATCH_SIZE):
        raise ValueError("The fit plan specifies different extraction geometry")
    if tuple(fit["ridge_grid"]) != RIDGE or tuple(fit["head_probability_weights"]) != ALPHA:
        raise ValueError("The approved readout search grid differs from the implementation")


def fit(model: Path, train: Path, validation: Path, selection: Path, plan_path: Path,
        output: Path) -> dict:
    import numpy as np

    if output.exists():
        raise FileExistsError(f"Fit output already exists: {output}")
    plan = _read(plan_path)
    _check_plan(plan)
    train_sha, valid_sha, selection_sha = (_digest(path) for path in (train, validation, selection))
    combined_splits = plan["train_sha256"], plan["validation_sha256"]
    if (train_sha, valid_sha) not in (combined_splits, SEC_SPLITS):
        raise ValueError("Pass the exact original campaign splits or the frozen SEC-only splits")
    if selection_sha != plan["original_development_selection_sha256"]:
        raise ValueError("Selection receipt differs from the registered bytes")
    chosen = _read(selection)
    train_rows = [row for row in read_jsonl(train) if row["track"] == TRACK]
    valid_by_id = {row["case_id"]: row for row in read_jsonl(validation)}
    dev_rows = [valid_by_id[key] for key in chosen["case_ids"]
                if key in valid_by_id and valid_by_id[key]["track"] == TRACK]
    if (len(train_rows), len(dev_rows)) != (plan["training"]["rows"], plan["training"]["development_rows"]):
        raise ValueError("Unexpected development case counts")
    labels = sorted(dev_rows[0]["choices"])
    validate(train_rows + dev_rows)
    if {row["split"] for row in train_rows} != {"train"} or {row["split"] for row in dev_rows} != {"valid"}:
        raise ValueError("Expected distinct train and validation splits")
    if any(sorted(row["choices"]) != labels for row in train_rows + dev_rows):
        raise ValueError("Inconsistent canonical SEC item menu")

    dev = _extract(model, [{key: row[key] for key in BLIND_FIELDS} for row in dev_rows])
    base_accuracy = sum(_prediction(dict(zip(labels, base))) == row["label"]
                        for row, (_, base) in zip(dev_rows, dev, strict=True)) / len(dev_rows)
    if not math.isclose(base_accuracy, chosen["base_panel_utility"][TRACK], abs_tol=1e-12):
        raise ValueError("The frozen base no longer matches the development receipt")
    train_features = _extract(model, [{key: row[key] for key in BLIND_FIELDS} for row in train_rows])
    hidden = np.array([state for state, _ in train_features])
    targets = np.eye(len(labels))[[labels.index(row["label"]) for row in train_rows]]
    heads = {reg: fit_head(hidden, targets, reg) for reg in RIDGE}
    grid = [
        {"regularization": reg, "alpha": alpha,
         "accuracy": sum(_prediction(distribution(heads[reg], alpha, feature, labels)) == row["label"]
                         for row, feature in zip(dev_rows, dev, strict=True)) / len(dev_rows)}
        for reg in RIDGE for alpha in ALPHA
    ]
    winner = min(grid, key=lambda row: (-row["accuracy"], row["alpha"], -row["regularization"]))
    output.mkdir(parents=True)
    weights = output / "weights.npz"
    np.savez(weights, **{TRACK: heads[winner["regularization"]]})
    receipt = {
        "schema": 1, "track": TRACK, "labels": labels, "plan_sha256": _digest(plan_path),
        "base_weights_sha256": EXPECTED_WEIGHTS_SHA256, "weights_sha256": _digest(weights),
        "inputs": {"train_sha256": train_sha, "validation_sha256": valid_sha, "selection_sha256": selection_sha},
        "training_cases": len(train_rows), "development_cases": len(dev_rows),
        "base_development_accuracy": base_accuracy, "selected": winner, "grid": grid,
    }
    _write(output / "fit-receipt.json", receipt)
    return receipt


def score(model: Path, head_dir: Path, cases_path: Path, plan_path: Path, output: Path) -> dict:
    import numpy as np

    if output.exists():
        raise FileExistsError(f"Score output already exists: {output}")
    plan = _read(plan_path)
    _check_plan(plan)
    receipt_path = head_dir / "fit-receipt.json"
    receipt = _read(receipt_path)
    weights_path = head_dir / "weights.npz"
    if receipt.get("track") != TRACK or receipt.get("plan_sha256") != _digest(plan_path):
        raise ValueError("Readout receipt does not match the locked SEC plan")
    if receipt["base_weights_sha256"] != EXPECTED_WEIGHTS_SHA256 or _digest(weights_path) != receipt["weights_sha256"]:
        raise ValueError("Readout weights or model digest differ from the fit receipt")
    with np.load(weights_path, allow_pickle=False) as stored:
        head = stored[TRACK].astype(np.float64, copy=False)
    alpha = receipt["selected"]["alpha"]
    labels = receipt["labels"]
    if alpha not in ALPHA or head.ndim != 2 or head.shape[1] != len(labels) or not np.isfinite(head).all():
        raise ValueError("Readout weights or mixture coefficient are invalid")

    cases_sha = _digest(cases_path)
    if cases_sha != plan["evaluation"]["cases_sha256"]:
        raise ValueError("Fresh test bytes differ from their registered digest")
    rows = read_jsonl(cases_path)
    if len(rows) != plan["evaluation"]["rows"] or {row["split"] for row in rows} != {"test"}:
        raise ValueError("Expected the registered test-only SEC cases")
    if any(row["track"] != TRACK or sorted(row["choices"]) != labels for row in rows):
        raise ValueError("Fresh test cases differ from the readout's SEC item menu")
    validate(rows)
    features = _extract(model, [{key: row[key] for key in BLIND_FIELDS} for row in rows])
    base, readout = [], []
    for row, feature in zip(rows, features, strict=True):
        for records, probabilities in (
            (base, dict(zip(labels, feature[1].tolist(), strict=True))),
            (readout, distribution(head, alpha, feature, labels)),
        ):
            records.append({"case_id": row["case_id"], "prediction": _prediction(probabilities),
                            "probabilities": probabilities})
    report = compare(rows, base, readout)
    output.mkdir(parents=True)
    write_jsonl(output / "base.jsonl", base)
    write_jsonl(output / "readout.jsonl", readout)
    report["checksums"] = {
        "cases_sha256": cases_sha, "plan_sha256": _digest(plan_path),
        "fit_receipt_sha256": _digest(receipt_path), "weights_sha256": receipt["weights_sha256"],
        "base_predictions_sha256": _digest(output / "base.jsonl"),
        "readout_predictions_sha256": _digest(output / "readout.jsonl"),
    }
    report["inference"] = {"model_sha256": EXPECTED_WEIGHTS_SHA256, "batch_size": BATCH_SIZE,
                           "layouts": VARIANTS, "case_count": len(rows), "alpha": alpha}
    _write(output / "paired-report.json", report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("fit", help="Fit on frozen train and development data")
    evaluate = commands.add_parser("score", help="Score the registered issuer-disjoint test")
    for command in (train, evaluate):
        for option in ("model", "plan", "output"):
            command.add_argument(f"--{option}", type=Path, required=True)
    for option in ("train", "validation", "selection"):
        train.add_argument(f"--{option}", type=Path, required=True)
    for option in ("head", "cases"):
        evaluate.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "fit":
            report = fit(args.model, args.train, args.validation, args.selection, args.plan, args.output)
            print(json.dumps(report["selected"], indent=2))
        else:
            report = score(args.model, args.head, args.cases, args.plan, args.output)
            print(json.dumps({name: {"gain": row["gain"], "p": row["p_holm"]}
                              for name, row in report["panels"].items()}, indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"filing_readout: {error}\n")


if __name__ == "__main__":
    main()
