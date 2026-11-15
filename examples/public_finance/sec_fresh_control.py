"""Evaluate the original validation-selected lexical estimator on fresh SEC filings."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import lexical
from sec_fresh import FROZEN, digest

ORIGINAL_CONTROL = Path(__file__).parent / "results" / "lexical-receipt.json"


def run(original: Path, cases: Path, output: Path) -> dict:
    protocol = json.loads(FROZEN.read_text(encoding="utf-8"))
    original_control = json.loads(ORIGINAL_CONTROL.read_text(encoding="utf-8"))
    initial = original_control["tracks"]["sec8k"]
    if lexical.sklearn_version != original_control["scikit_learn"]:
        raise ValueError("The scikit-learn version differs from the original lexical control")
    for split in ("train", "valid"):
        name = f"sec8k-{split}.jsonl"
        if digest(original / name) != protocol["original_inputs"]["sha256"][name]:
            raise ValueError(f"Original lexical {split} file differs from the fresh-cohort receipt")
    if digest(cases) != protocol["output_sha256"]:
        raise ValueError("Fresh cases differ from the frozen SEC cohort")
    train = lexical.load_split(original, "sec8k", "train")
    valid = lexical.load_split(original, "sec8k", "valid")
    fitted = lexical.select_and_fit(train, valid)
    chosen = {"C": fitted["C"], "feature_space": fitted["space"]}
    if chosen != initial["chosen"] or fitted["search"] != initial["selection_search"]:
        raise ValueError("Refitted lexical model differs from the initial validation-selected settings")
    rows = [json.loads(line) for line in cases.read_text(encoding="utf-8").splitlines()]
    scored = lexical.evaluate(fitted["pipeline"], rows, lexical.label_space(train))
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in scored["predictions"]).encode()
    receipt = {
        "schema": 1, "baseline": original_control["baseline"], "scikit_learn": lexical.sklearn_version,
        "original_control_receipt_sha256": digest(ORIGINAL_CONTROL), "chosen": chosen,
        "train_rows": len(train), "validation_rows": len(valid), "test_rows": len(rows),
        "fresh_cases_sha256": protocol["output_sha256"], "predictions_sha256": sha256(payload).hexdigest(),
        "scores": scored["scores"], "confusion": scored["confusion"],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "lexical-predictions.jsonl").write_bytes(payload)
    (output / "lexical-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.original, args.cases, args.output), indent=2))
