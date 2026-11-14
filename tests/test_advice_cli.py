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
