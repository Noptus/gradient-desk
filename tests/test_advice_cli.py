import json

import pytest

from gradient_desk.cli import main
from gradient_desk.dataset import method_prompt, write_jsonl


def test_learned_advice_runs_without_model_weights(tmp_path, capsys):
    train, valid = (tmp_path / name for name in ("train.jsonl", "valid.jsonl"))
    common = {"track": "methodology", "label": "audit_timing"}
    write_jsonl(train, [{
        **common, "split": "train", "family": "macro",
        "prompt": method_prompt("Calendar timestamps and a late vintage changed the original result."),
    }])
    write_jsonl(valid, [{
        **common, "split": "valid", "family": "earnings",
        "prompt": method_prompt("The historical earnings release used a revised vintage."),
    }])
    main(["advise", "--policy", "learned", "--train-cases", str(train),
          "--valid-cases", str(valid), "--scenario", "A revision changed the vintage."])
    assert json.loads(capsys.readouterr().out) == {
        "action": "audit_timing", "policy": "learned", "selected_k": 1,
    }


def test_learned_advice_rejects_ambiguous_backends_and_missing_inputs(capsys):
    with pytest.raises(SystemExit, match="2"):
        main(["advise", "--policy", "learned", "--scenario", "A proposal"])
    assert "requires --train-cases and --valid-cases" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="2"):
        main(["advise", "--model", "local-model", "--train-cases", "rows.jsonl",
              "--scenario", "A proposal"])
    assert "require --policy learned" in capsys.readouterr().err


def test_constrained_advice_scores_canonical_prompt_without_labels(tmp_path, capsys, monkeypatch):
    import gradient_desk.choice_model as choice_model
    from gradient_desk.choice_data import INTERVENTIONS, compact_intervention

    telemetry = {
        "objective": "cross-entropy",
        "optimizer": "adamw",
        "momentum": 0.9,
        "lr": 2e-4,
        "microbatch_size": 8,
        "gradient_norm_history": [1.2, 1.1, 1.4, 0.9, 0.8, 1.0, 1.3, 1.1],
        "train_loss_history": [2.1, 1.9, 1.8, 1.7, 1.6, 1.55, 1.5, 1.48],
        "validation_loss_history": [{"loss": 2.0}, {"loss": 1.7}, {"loss": 1.6}],
    }
    telemetry_path = tmp_path / "telemetry.json"
    telemetry_path.write_text(json.dumps(telemetry), encoding="utf-8")

    seen = {}

    def fake_load(model_path, adapter):
        seen["model"] = model_path
        seen["adapter"] = adapter
        return "MODEL", "TOKENIZER"

    def fake_score_rows(model, tokenizer, rows, variants, batch_size, *, progress):
        seen["model_obj"] = model
        seen["tokenizer"] = tokenizer
        seen["rows"] = rows
        seen["variants"] = variants
        seen["batch_size"] = batch_size
        seen["progress"] = progress
        (row,) = rows
        return iter([{
            "case_id": row["case_id"],
            "track": row["track"],
            "variants": 2,
            "prediction": "lower_lr",
            "probabilities": {"resume": 0.1, "lower_lr": 0.6, "clip_grad": 0.2, "accumulate": 0.1},
        }])

    monkeypatch.setattr(choice_model, "_load", fake_load)
    monkeypatch.setattr(choice_model, "_score_rows", fake_score_rows)

    main(["advise", "--constrained", "--model", str(tmp_path / "local-model"),
          "--telemetry", str(telemetry_path)])

    # No physical backend was touched; only the monkeypatched stand-ins ran.
    assert seen["model_obj"] == "MODEL" and seen["tokenizer"] == "TOKENIZER"
    # Deterministic two-variant batch-of-two scoring without stdout progress.
    assert seen["variants"] == 2 and seen["batch_size"] == 2 and seen["progress"] is False

    (row,) = seen["rows"]
    # The canonical prompt and frozen menu drive scoring; no supervision leaks.
    assert row["prompt"] == compact_intervention(telemetry)
    assert row["choices"] == INTERVENTIONS
    assert "label" not in row and "answer" not in row and "target_dist" not in row
    # The case ID is a deterministic content hash of the prompt and menu.
    assert row["case_id"] == row["case_id"].lower() and len(row["case_id"]) == 64

    assert json.loads(capsys.readouterr().out) == {
        "action": "lower_lr",
        "policy": "constrained",
        "variants": 2,
        "probabilities": {"resume": 0.1, "lower_lr": 0.6, "clip_grad": 0.2, "accumulate": 0.1},
    }


def test_constrained_advice_translates_malformed_telemetry(tmp_path, capsys):
    telemetry_path = tmp_path / "telemetry.json"
    telemetry_path.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="2"):
        main(["advise", "--constrained", "--telemetry", str(telemetry_path),
              "--model", str(tmp_path)])
    # A malformed telemetry shape is translated into a helpful user error.
    assert "advise" in capsys.readouterr().err
