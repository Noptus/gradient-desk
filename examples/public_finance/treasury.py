"""Create a date-disjoint rates memo audit from U.S. Treasury par yields."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from random import Random


TRACK = "treasury_curve_audit"
CHOICES = {
    "supported": "The claim uses the right dates, tenor, sign and basis-point conversion.",
    "wrong_tenor": "The claim substitutes the other maturity.",
    "wrong_session": "The claim uses a session beyond the cutoff.",
    "reversed_sign": "The claim reverses the direction of the change.",
    "wrong_units": "The change is in percentage points but is presented as basis points.",
}
YEARS = {
    "train": range(1990, 2014),
    "valid": range(2014, 2019),
    "test": range(2019, 2025),
}
COUNTS = {"train": 300, "valid": 100, "test": 150}
MATURITIES = ("2 Yr", "10 Yr")
FILENAME = "treasury-yield-curve-1990-2024.csv"
URL = "https://home.treasury.gov/system/files/276/yield-curve-rates-1990-2024.csv"
SOURCE_SHA256 = "258a594eb64e9d21d9e8574b0b8580b0a5230dce56bb284ec6aa49a9a260897f"


def source_rows(path: Path) -> dict:
    if sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError(f"Treasury archive hash mismatch: {path}")
    rows, dates = {}, set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"Date", *MATURITIES}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name} is missing Treasury date or maturity fields")
        for row in reader:
            trading_date = datetime.strptime(row["Date"], "%m/%d/%y").date()
            if not 1990 <= trading_date.year <= 2024 or trading_date in dates:
                raise ValueError(f"Unexpected or duplicate date: {trading_date}")
            dates.add(trading_date)
            if any(not row[name].strip() or row[name].strip() == "N/A" for name in MATURITIES):
                continue
            rates = tuple(Decimal(row[name]) for name in MATURITIES)
            if any(not rate.is_finite() for rate in rates):
                raise ValueError(f"Invalid yield at {trading_date}")
            rows.setdefault(trading_date.year, []).append((trading_date, rates))
    return {year: sorted(values) for year, values in rows.items()}


def changes(sessions: list[tuple], tenor: int) -> dict[str, Decimal]:
    cutoff_before, cutoff, later = [rates for _, rates in sessions]
    stated_delta = cutoff[tenor] - cutoff_before[tenor]
    return {
        "supported": 100 * stated_delta,
        "wrong_tenor": 100 * (cutoff[1 - tenor] - cutoff_before[1 - tenor]),
        "wrong_session": 100 * (later[tenor] - cutoff[tenor]),
        "reversed_sign": -100 * stated_delta,
        "wrong_units": stated_delta,
    }


def case(sessions: list[tuple], tenor: int, label: str) -> dict:
    date = sessions[1][0]
    maturity = "2-year" if tenor == 0 else "10-year"
    proposed_change = changes(sessions, tenor)[label]
    table = [f"{day}: two-year {rates[0]:.2f}%, ten-year {rates[1]:.2f}%" for day, rates in sessions]
    # Reordering is independent of answer and mitigates shortcuts based on input positions.
    Random(date.toordinal()).shuffle(table)
    evidence = "\n".join(table)
    memo = f"The {maturity} yield rose {abs(proposed_change):g} basis points"
    if proposed_change < 0:
        memo = f"The {maturity} yield fell {abs(proposed_change):g} basis points"
    return {
        "case_id": f"treasury-audit-{date}",
        "track": TRACK,
        "family": f"treasury-{date.year}",
        "evaluation_block": f"{date.year}-{date.month:02d}",
        "source_id": f"Treasury par yields {sessions[0][0]} to {sessions[-1][0]}",
        "prompt": (
            "Audit a memo estimating the day-over-day move in a U.S. Treasury par yield. "
            f"The evidence audit cutoff includes session {date} and excludes later observations. "
            "What is the primary defect in the claim, if any?\n\n"
            f"Treasury observations:\n{evidence}\n\nDesk memo: {memo} from the immediately previous session."
        ),
        "label": label,
        "choices": CHOICES,
    }


def prepare(raw: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    grouped = source_rows(raw / FILENAME)
    receipt = {"task": TRACK, "source": "U.S. Treasury daily par yield curve",
               "upstream": {"url": URL, "sha256": SOURCE_SHA256}, "counts": {}}
    for split, years in YEARS.items():
        candidates = []
        for year in years:
            observations = grouped.get(year, [])
            # Nonoverlapping triplets keep prices and dates distinct within and between folds.
            for offset in range(0, len(observations) - 2, 3):
                sessions = observations[offset:offset + 3]
                tenor = sessions[1][0].toordinal() % len(MATURITIES)
                outcomes = changes(sessions, tenor)
                if len(set(outcomes.values())) == len(CHOICES) and all(outcomes.values()):
                    candidates.append((sessions, tenor))
        Random(f"{TRACK}:{split}:1").shuffle(candidates)
        selected = candidates[:COUNTS[split] // len(CHOICES) * len(CHOICES)]
        # Preserve exact category balance even if this source edition has fewer eligible cases.
        selected = selected[:len(selected) // len(CHOICES) * len(CHOICES)]
        if len(selected) < 50:
            raise ValueError(f"Too few distinct {split} Treasury windows: {len(selected)}")
        labels = list(CHOICES)
        rows = [
            {**case(sessions, tenor, labels[index % len(labels)]), "split": split}
            for index, (sessions, tenor) in enumerate(selected)
        ]
        path = output / f"treasury-{split}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        receipt["counts"][split] = {
            "rows": len(rows),
            "blocks": len({row["evaluation_block"] for row in rows}),
            "eligible_nonoverlapping_windows": len(candidates),
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
    (output / "treasury-receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    print(json.dumps(prepare(args.input, args.output), indent=2))
