"""Download, prepare and evaluate a local research coach."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import MODEL_REPO, MODEL_REVISION, method_prompt, prepare, read_jsonl
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

    learned = commands.add_parser("learned-eval", help="Test validation-selected CPU baselines")
    learned.add_argument("--train-cases", type=Path, required=True)
    learned.add_argument("--valid-cases", type=Path, required=True)
    learned.add_argument("--cases", type=Path, required=True)
    learned.add_argument("--output", type=Path, required=True)

    paired = commands.add_parser("paired-eval", help="Evaluate base and adapter with matching settings")
    paired.add_argument("--cases", type=Path, required=True)
    paired.add_argument("--output", type=Path, required=True)
    paired.add_argument("--model", type=Path, required=True)
    paired.add_argument("--adapter", type=Path, required=True)
    paired.add_argument("--batch-size", type=_positive, default=1)
    paired.add_argument("--extra-cases", type=Path, action="append", default=[],
                        help="Independent external holdout, reported separately (repeat to add panels)")
    paired.add_argument("--validation-cases", type=Path, help="Optional fixed validation panel")
    paired.add_argument("--checkpoints", type=_positive, nargs="+", default=[],
                        help="Choose among these saved steps using the validation panel")

    compare = commands.add_parser("compare", help="Compare matched evaluation predictions")
    compare.add_argument("--base", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    advise = commands.add_parser("advise", help="Choose an action for one local research case")
    adviser = advise.add_mutually_exclusive_group(required=True)
    adviser.add_argument("--model", type=Path)
    adviser.add_argument("--policy", choices=("learned",), help="Use the validation-selected CPU policy")
    advise.add_argument("--adapter", type=Path)
    advise.add_argument("--train-cases", type=Path)
    advise.add_argument("--valid-cases", type=Path)
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
    elif args.command == "learned-eval":
        from .learned import evaluate_learned

        report = evaluate_learned(args.train_cases, args.valid_cases, args.cases, args.output)
        print(json.dumps(report, indent=2))
    elif args.command == "paired-eval":
        from .evaluate import paired_eval

        scores = paired_eval(
            args.cases, args.output, args.model, args.adapter, args.batch_size, tuple(args.extra_cases),
            args.validation_cases, tuple(args.checkpoints),
        )
        print(json.dumps(scores, indent=2))
    elif args.command == "compare":
        from .compare import write_comparison

        print(json.dumps(write_comparison(args.base, args.candidate, args.output), indent=2))
    elif args.command == "advise":
        if args.policy and (args.adapter or not args.train_cases or not args.valid_cases):
            advise.error("--policy learned requires --train-cases and --valid-cases")
        if args.model and (args.train_cases or args.valid_cases):
            advise.error("--train-cases and --valid-cases require --policy learned")
        try:
            if args.scenario is not None:
                if not args.scenario.strip():
                    raise ValueError("--scenario must have text")
                row = {"track": "methodology", "prompt": method_prompt(args.scenario)}
            else:
                telemetry = json.loads(args.telemetry.read_text(encoding="utf-8"))
                if not isinstance(telemetry, dict):
                    raise ValueError("--telemetry must contain a JSON object")
                row = {"track": "training", "prompt": format_telemetry(telemetry)}
            if args.policy:
                from .learned import make_baseline

                train, valid = (
                    [case for case in read_jsonl(path) if case.get("track") == row["track"]]
                    for path in (args.train_cases, args.valid_cases)
                )
                policy = make_baseline(train, valid)
                report = {"action": policy(row), "policy": "learned",
                          "selected_k": policy.settings.selected_k}
            else:
                from .evaluate import generate_predictions

                prediction = next(generate_predictions(args.model, args.adapter, [row], progress=False))
                report = {"action": prediction.label, "raw": prediction.raw}
        except (OSError, ValueError, KeyError, TypeError) as error:
            advise.error(str(error))
        print(json.dumps(report, indent=2))
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
