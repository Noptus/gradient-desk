"""Download, prepare and evaluate a local research coach."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import MODEL_REPO, MODEL_REVISION, method_prompt, prepare
from .traces import format_telemetry


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Fetch the pinned 4-bit checkpoint")
    download.add_argument("--output", type=Path, required=True)

    dataset = commands.add_parser("prepare", help="Assemble split-isolated JSONL")
    dataset.add_argument("--traces", type=Path, required=True)
    dataset.add_argument("--methods", type=Path, default=Path("data/methodology_scenarios.json"))
    dataset.add_argument("--output", type=Path, required=True)
    dataset.add_argument("--method-weight", type=_positive, default=3)

    run = commands.add_parser("evaluate", help="Score held-out cases and write raw predictions")
    run.add_argument("--cases", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--adapter", type=Path)
    run.add_argument("--batch-size", type=_positive, default=1)
    engine = run.add_mutually_exclusive_group(required=True)
    engine.add_argument("--model", type=Path)
    engine.add_argument("--baseline", choices=("resume", "rules"))

    compare = commands.add_parser("compare", help="Compare matched evaluation predictions")
    compare.add_argument("--base", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    advise = commands.add_parser("advise", help="Choose an action for one local research case")
    advise.add_argument("--model", type=Path, required=True)
    advise.add_argument("--adapter", type=Path)
    case = advise.add_mutually_exclusive_group(required=True)
    case.add_argument("--scenario")
    case.add_argument("--telemetry", type=Path)

    args = parser.parse_args(argv)
    if args.command == "download":
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            local_dir=args.output,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "LICENSE"],
        )
        print(path)
    elif args.command == "prepare":
        print(json.dumps(prepare(args.traces, args.methods, args.output, args.method_weight), indent=2))
    elif args.command == "compare":
        from .compare import write_comparison

        print(json.dumps(write_comparison(args.base, args.candidate, args.output), indent=2))
    elif args.command == "advise":
        from .evaluate import generate_predictions

        if args.scenario is not None:
            row = {"track": "methodology", "prompt": method_prompt(args.scenario)}
        else:
            telemetry = json.loads(args.telemetry.read_text(encoding="utf-8"))
            if not isinstance(telemetry, dict):
                parser.error("--telemetry must contain a JSON object")
            row = {
                "track": "training",
                "prompt": format_telemetry(telemetry),
            }
        prediction = next(generate_predictions(args.model, args.adapter, [row], progress=False))
        print(json.dumps({"action": prediction.label, "raw": prediction.raw}, indent=2))
    else:
        from .evaluate import evaluate

        if args.baseline and args.adapter:
            parser.error("--adapter requires --model")
        scores = evaluate(
            args.cases, args.output, args.model, args.adapter, args.baseline,
            batch_size=args.batch_size,
        )
        print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
