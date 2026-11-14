"""Run a validation-selected telemetry control and a fixed resume control."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .choice_data import validate
from .dataset import read_jsonl, write_jsonl
from .learned import make_baseline


def _interventions(path: Path, split: str) -> list[dict]:
    rows = read_jsonl(path)
    if not rows or {row["split"] for row in rows} != {split}:
        raise ValueError(f"{path} must contain only {split} cases")
    rows = [row for row in rows if "actions" in row]
    if not rows:
        raise ValueError(f"{path} has no measured intervention cases")
    return rows


def _legacy(row: dict) -> dict:
    return {**row, "prompt": row["legacy_prompt"]}


def run(train: Path, validation: Path, test: Path, expected_sha256: str, output: Path) -> dict:
    inputs = {"train": train, "valid": validation, "test": test}
    digests = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()}
    if digests["test"] != expected_sha256:
        raise ValueError("Test file does not match the frozen benchmark digest")
    train_rows, validation_rows, test_rows = [
        _interventions(path, split) for split, path in inputs.items()
    ]
    validate(train_rows + validation_rows + test_rows)
    policy = make_baseline(list(map(_legacy, train_rows)), list(map(_legacy, validation_rows)))
    learned = [{"case_id": row["case_id"], "prediction": policy(_legacy(row))} for row in test_rows]
    fixed = [{"case_id": row["case_id"], "prediction": "resume"} for row in test_rows]
    receipt = {"inputs": digests, "rows": len(test_rows), "learned": asdict(policy.settings)}
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "test-actions.jsonl", test_rows)
    write_jsonl(output / "learned.jsonl", learned)
    write_jsonl(output / "resume.jsonl", fixed)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--expected-test-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = run(args.train, args.validation, args.test, args.expected_test_sha256, args.output)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
