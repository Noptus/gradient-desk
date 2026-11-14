import json
from pathlib import Path

import pytest

from gradient_desk.dataset import METHODS, methods_rows, prepare, read_jsonl, validate_cases, write_jsonl
from gradient_desk.evaluate import Prediction, evaluate, parse_prediction
from gradient_desk.traces import ACTIONS


def trace_rows():
    return [
        {
            "case_id": f"t-{split}",
            "track": "training",
            "split": split,
            "family": f"f-{split}",
            "prompt": f"prompt-{split}",
            "label": "clip_grad",
            "actions": {
                action: {"validation_loss": loss}
                for action, loss in zip(ACTIONS, (1.4, 1.3, 1.0, 1.1), strict=True)
            },
        }
        for split in ("train", "valid", "test")
    ]


def test_prepare_strips_outcomes_and_never_repeats_holdouts(tmp_path):
    traces = tmp_path / "trace.jsonl"
    write_jsonl(traces, trace_rows())
    methods = Path("data/methodology_scenarios.json")
    out = tmp_path / "dataset"
    counts = prepare(traces, methods, out, 2)
    for split, stats in counts.items():
        clean = read_jsonl(out / f"{split}.jsonl")
        reference = read_jsonl(out / f"{split}-cases.jsonl")
        assert set(clean[0]) == {"prompt", "completion"}
        assert len(reference) == stats["training"] + stats["methodology"]
        assert len(clean) == (stats["training"] + 2 * stats["methodology"]
                              if split == "train" else len(reference))


def test_validate_prevents_cross_split_families():
    rows = trace_rows() + methods_rows(Path("data/methodology_scenarios.json"))
    validate_cases(rows)
    rows[0]["family"] = rows[1]["family"]
    with pytest.raises(ValueError, match="crosses splits"):
        validate_cases(rows)


def test_parser_requires_one_menu_action():
    assert parse_prediction(" `clip_grad`\n", ACTIONS) == "clip_grad"
    assert parse_prediction("clip_grad or resume", ACTIONS) is None
    assert parse_prediction("low_lr", ACTIONS) is None
    assert parse_prediction("model_costs", METHODS) == "model_costs"


def test_evaluation_counts_invalid_as_fallback_but_not_accuracy(tmp_path):
    cases = tmp_path / "cases.jsonl"
    write_jsonl(cases, trace_rows())
    output = tmp_path / "eval.jsonl"
    report = evaluate(
        cases,
        output,
        None,
        None,
        None,
        predictor=lambda *_: iter([
            Prediction(None, "A lengthy explanation", 1.0),
            Prediction("clip_grad", "clip_grad", 2.0),
            Prediction("accumulate", "accumulate", 3.0),
        ]),
    )
    scores = report["by_track"]["training"]
    assert scores["accuracy"] == pytest.approx(1 / 3)
    assert scores["valid_rate"] == pytest.approx(2 / 3)
    assert scores["mean_relative_regret"] == pytest.approx((0.4 + 0.1) / 3)
    assert json.loads(output.with_suffix(".metrics.json").read_text()) == report
