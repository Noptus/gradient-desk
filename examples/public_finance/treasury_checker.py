"""Solve the fixed Treasury memo audit from its displayed sessions and claim."""

from __future__ import annotations

import argparse
from datetime import date
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import re

TRACK = "treasury_curve_audit"
CUTOFF = re.compile(r"includes session (\d{4}-\d{2}-\d{2}) and excludes")
SESSION = re.compile(
    r"^(\d{4}-\d{2}-\d{2}): two-year (-?\d+\.\d{2})%, ten-year (-?\d+\.\d{2})%$")
MEMO = re.compile(
    r"Desk memo: The (2-year|10-year) yield (rose|fell) (\d+(?:\.\d+)?) "
    r"basis points from the immediately previous session\.$")


class InvalidMemo(ValueError):
    """Raised when the prompt does not match the fixed audit template."""


def _parse(prompt: str) -> tuple[list[tuple[Decimal, Decimal]], int, Decimal]:
    cutoff = CUTOFF.search(prompt)
    if cutoff is None:
        raise InvalidMemo("prompt is missing the audit cutoff session")
    cutoff_date = date.fromisoformat(cutoff.group(1))

    sessions: list[tuple[date, tuple[Decimal, Decimal]]] = []
    for line in prompt.splitlines():
        row = SESSION.match(line.strip())
        if row is not None:
            day = date.fromisoformat(row.group(1))
            sessions.append((day, (Decimal(row.group(2)), Decimal(row.group(3)))))
    if len(sessions) != 3:
        raise InvalidMemo(f"expected three Treasury sessions, found {len(sessions)}")
    sessions.sort()
    if len({day for day, _ in sessions}) != 3:
        raise InvalidMemo("Treasury sessions must carry three distinct dates")
    if sessions[1][0] != cutoff_date:
        raise InvalidMemo("audit cutoff is not the middle session")

    memo = MEMO.search(prompt)
    if memo is None:
        raise InvalidMemo("prompt is missing a well-formed desk memo")
    tenor = 0 if memo.group(1) == "2-year" else 1
    claim = Decimal(memo.group(3))
    return [rates for _, rates in sessions], tenor, -claim if memo.group(2) == "fell" else claim


def predict(prompt: str) -> str:
    """Return the option ID whose candidate value matches the memo claim.

    Raises InvalidMemo if the template is malformed or no single candidate
    reproduces the claimed signed basis-point move.
    """
    sessions, tenor, claim = _parse(prompt)
    previous, cutoff, later = sessions
    change = cutoff[tenor] - previous[tenor]
    other_tenor = 1 - tenor
    candidates = {
        "supported": 100 * change,
        "wrong_tenor": 100 * (cutoff[other_tenor] - previous[other_tenor]),
        "wrong_session": 100 * (later[tenor] - cutoff[tenor]),
        "reversed_sign": -100 * change,
        "wrong_units": change,
    }
    matches = [option for option, value in candidates.items() if value == claim]
    if len(matches) != 1:
        raise InvalidMemo(
            f"claim {claim} matches {len(matches)} candidate values, expected 1")
    return matches[0]


def run(cases_dir: Path, output: Path) -> dict:
    receipt_path = Path(__file__).with_name("frozen-receipts.json")
    frozen = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = frozen["treasury"]["counts"]["test"]["sha256"]
    test_path = cases_dir / "treasury-test.jsonl"
    raw = test_path.read_bytes()
    observed = sha256(raw).hexdigest()
    if observed != expected:
        raise ValueError(f"digest mismatch for {test_path}: expected {expected}, got {observed}")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    predictions = [predict(row["prompt"]) for row in rows]

    scored = [{"case_id": row["case_id"], "track": TRACK, "prediction": prediction,
               "correct": row["label"] == prediction}
              for row, prediction in zip(rows, predictions)]
    correct = sum(row["correct"] for row in scored)

    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "treasury-checker-predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(p, sort_keys=True) + "\n" for p in scored), encoding="utf-8")

    receipt = {
        "baseline": "treasury_decimal_calculator",
        "track": TRACK,
        "input": {"file": test_path.name, "sha256": observed},
        "count": len(rows),
        "accuracy": round(correct / len(rows), 6),
        "correct": correct,
        "incorrect": len(rows) - correct,
        "predictions_file": predictions_path.name,
        "predictions_sha256": sha256(predictions_path.read_bytes()).hexdigest(),
    }
    (output / "treasury-checker-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    cli.add_argument("--cases-dir", type=Path, required=True,
                     help="directory with treasury-test.jsonl and adjacent files")
    cli.add_argument("--output", type=Path, required=True,
                     help="directory for the predictions JSONL and JSON receipt")
    args = cli.parse_args()
    try:
        print(json.dumps(run(args.cases_dir, args.output), indent=2))
    except (OSError, ValueError) as error:
        cli.exit(2, f"treasury_checker: {error}\n")
