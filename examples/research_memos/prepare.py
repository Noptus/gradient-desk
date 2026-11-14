"""Build evaluation rows for the paper-grounded, held-out review scenarios."""

import argparse
import json
from pathlib import Path

from gradient_desk.dataset import METHODS, method_prompt, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = Path(__file__).with_name("cases.json")
    rows = json.loads(source.read_text(encoding="utf-8"))
    if len({row["id"] for row in rows}) != len(rows):
        parser.error("Duplicate research memo IDs")
    converted = []
    for row in rows:
        if row["label"] not in METHODS or row["split"] != "test":
            parser.error(f"Invalid label or split for {row['id']}")
        converted.append({
            "case_id": row["id"],
            "family": row["family"],
            "track": "methodology",
            "split": row["split"],
            "prompt": method_prompt(row["scenario"]),
            "label": row["label"],
            "rationale": row["rationale"],
            "source": row["source"],
        })
    if args.output.exists():
        parser.error("Output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, converted)
    print(f"Wrote {len(converted)} held-out research memos.")


if __name__ == "__main__":
    main()
