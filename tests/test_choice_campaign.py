"""Pure, Metal-free tests for the v2 selection/evaluation campaign helpers.

Everything here exercises the deterministic case-selection, panel-utility,
tie-breaking and receipt-verification logic with plain Python fixtures.  No MLX
import, no model load, no tokenizer; hashing and metric math are all CPU.
"""

from __future__ import annotations

import json

import pytest

from gradient_desk.choice_campaign import (
    MAX_PER_BENCHMARK,
    _FrozenPrefix,
    _blind,
    _equal_weight_mean,
    _swap_lora_weights,
    _verify_receipt,
    case_utility,
    panel_utilities,
    select_cases,
)
from gradient_desk.choice_model import _file_sha256, candidate_logits
from gradient_desk.traces import ACTIONS


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def label_case(case_id, *, benchmark="methods", block=None, label="yes", split="valid"):
    return {
        "case_id": case_id, "track": benchmark, "benchmark": benchmark,
        "family": "fixture", "split": split,
        "evaluation_block": block if block is not None else case_id,
        "prompt": f"Question {case_id}", "label": label,
        "choices": {"yes": "Yes.", "no": "No."},
    }


def action_case(case_id, losses, *, benchmark="training", block=None, split="valid"):
    label = min(losses, key=losses.get)
    return {
        "case_id": case_id, "track": "training", "benchmark": benchmark,
        "family": "fixture", "split": split,
        "evaluation_block": block if block is not None else case_id,
        "prompt": f"Telemetry {case_id}", "label": label,
        "choices": {action: action.replace("_", " ") for action in ACTIONS},
        "actions": {action: {"validation_loss": losses[action]} for action in ACTIONS},
    }


# --------------------------------------------------------------------------- #
# select_cases: deterministic, block-diversifying, capped, label-independent
# --------------------------------------------------------------------------- #
def test_select_cases_is_deterministic_and_label_independent():
    rows = [label_case(f"c{i}", block=f"b{i % 3}") for i in range(20)]
    first = [row["case_id"] for row in select_cases(rows, limit=9)]
    # Flipping every label must not change which cases are chosen.
    for row in rows:
        row["label"] = "no"
    second = [row["case_id"] for row in select_cases(rows, limit=9)]
    assert first == second


def test_select_cases_diversifies_blocks_before_repeating():
    # Three blocks with two cases each; a limit of 3 must take one per block.
    rows = []
    for block in range(3):
        for n in range(2):
            rows.append(label_case(f"blk{block}-{n}", block=f"b{block}"))
    picked = select_cases(rows, limit=3)
    assert len({row["evaluation_block"] for row in picked}) == 3


def test_select_cases_caps_per_benchmark():
    rows = [label_case(f"a{i}", benchmark="alpha", block=f"a{i}") for i in range(50)]
    rows += [label_case(f"b{i}", benchmark="beta", block=f"b{i}") for i in range(50)]
    picked = select_cases(rows, limit=MAX_PER_BENCHMARK)
    per = {}
    for row in picked:
        per[row["benchmark"]] = per.get(row["benchmark"], 0) + 1
    assert per == {"alpha": MAX_PER_BENCHMARK, "beta": MAX_PER_BENCHMARK}


def test_select_cases_rejects_nonpositive_limit():
    with pytest.raises(ValueError):
        select_cases([label_case("c")], limit=0)


# --------------------------------------------------------------------------- #
# Blinding: scoring rows must never carry the answer fields.
# --------------------------------------------------------------------------- #
def test_blind_strips_label_target_and_actions():
    row = action_case("c", {a: i for i, a in enumerate(ACTIONS)})
    row["target_dist"] = {a: 1.0 for a in ACTIONS}
    blinded = _blind(row)
    assert "label" not in blinded and "target_dist" not in blinded and "actions" not in blinded
    assert blinded["prompt"] == row["prompt"] and blinded["choices"] == row["choices"]


# --------------------------------------------------------------------------- #
# Utility: accuracy for labels, 1 - normalized regret for measured actions.
# --------------------------------------------------------------------------- #
def test_case_utility_label_is_exact_match():
    row = label_case("c", label="yes")
    assert case_utility(row, "yes") == 1.0
    assert case_utility(row, "no") == 0.0


def test_case_utility_measured_is_one_minus_normalized_regret():
    losses = {"resume": 1.0, "lower_lr": 2.0, "clip_grad": 3.0, "accumulate": 4.0}
    row = action_case("c", losses)
    assert case_utility(row, "resume") == pytest.approx(1.0)  # best action
    assert case_utility(row, "accumulate") == pytest.approx(0.0)  # worst action
    assert case_utility(row, "lower_lr") == pytest.approx(1.0 - (2.0 - 1.0) / (4.0 - 1.0))


def test_panel_utilities_group_by_benchmark():
    rows = [
        label_case("a1", benchmark="alpha", label="yes"),
        label_case("a2", benchmark="alpha", label="yes"),
        label_case("b1", benchmark="beta", label="no"),
    ]
    predictions = {"a1": "yes", "a2": "no", "b1": "no"}
    panels = panel_utilities(rows, predictions)
    assert panels["alpha"] == pytest.approx(0.5)
    assert panels["beta"] == pytest.approx(1.0)
    assert _equal_weight_mean(panels) == pytest.approx(0.75)


def test_equal_weight_mean_ignores_case_counts():
    # One panel with many cases does not outweigh a panel with one case.
    assert _equal_weight_mean({"big": 0.0, "small": 1.0}) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Tie-break: earliest step wins an exact tie (mirrors the select() rule).
# --------------------------------------------------------------------------- #
def test_tie_break_prefers_earliest_step():
    scored = {60: 0.8, 120: 0.8, 180: 0.5}
    chosen = min(scored, key=lambda step: (-scored[step], step))
    assert chosen == 60


# --------------------------------------------------------------------------- #
# Receipt verification: frozen bytes, checkpoint, config, base digests.
# --------------------------------------------------------------------------- #
def _campaign_tree(tmp_path, *, step=120):
    cases = tmp_path / "test.jsonl"
    cases.write_text(json.dumps(label_case("t1", split="test")) + "\n", encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    checkpoint = adapter / f"{step:07d}_adapters.safetensors"
    checkpoint.write_bytes(b"fake-weights")
    config = adapter / "adapter_config.json"
    config.write_text(json.dumps({"fine_tune_type": "lora"}) + "\n", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    base = model / "model.safetensors"
    base.write_bytes(b"fake-base")
    receipt = {
        "selected_step": step,
        "selected_weights_sha256": _file_sha256(checkpoint),
        "adapter_config_sha256": _file_sha256(config),
        "base_weights_sha256": _file_sha256(base),
        "case_ids": ["v1", "v2"],
    }
    return receipt, cases, adapter, model


def test_verify_receipt_accepts_matching_tree(tmp_path):
    receipt, cases, adapter, model = _campaign_tree(tmp_path)
    frozen = _file_sha256(cases)
    step, checkpoint = _verify_receipt(receipt, cases, adapter, model, frozen)
    assert step == 120 and checkpoint.name == "0000120_adapters.safetensors"


def test_verify_receipt_rejects_wrong_frozen_bytes(tmp_path):
    receipt, cases, adapter, model = _campaign_tree(tmp_path)
    with pytest.raises(ValueError, match="Frozen test bytes"):
        _verify_receipt(receipt, cases, adapter, model, "deadbeef" * 8)


def test_verify_receipt_rejects_tampered_checkpoint(tmp_path):
    receipt, cases, adapter, model = _campaign_tree(tmp_path)
    (adapter / "0000120_adapters.safetensors").write_bytes(b"different")
    with pytest.raises(ValueError, match="checkpoint bytes"):
        _verify_receipt(receipt, cases, adapter, model, _file_sha256(cases))


def test_verify_receipt_rejects_tampered_config(tmp_path):
    receipt, cases, adapter, model = _campaign_tree(tmp_path)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="config digest"):
        _verify_receipt(receipt, cases, adapter, model, _file_sha256(cases))


def test_verify_receipt_rejects_tampered_base(tmp_path):
    receipt, cases, adapter, model = _campaign_tree(tmp_path)
    (model / "model.safetensors").write_bytes(b"swapped-base")
    with pytest.raises(ValueError, match="Base weights digest"):
        _verify_receipt(receipt, cases, adapter, model, _file_sha256(cases))


def test_verify_receipt_rejects_missing_checkpoint(tmp_path):
    receipt, cases, adapter, model = _campaign_tree(tmp_path)
    (adapter / "0000120_adapters.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        _verify_receipt(receipt, cases, adapter, model, _file_sha256(cases))


def test_swap_loads_complete_partial_adapter(tmp_path):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(2, 2)
            self.base.freeze()
            self.lora_a = mx.zeros((2, 1))
            self.lora_b = mx.zeros((1, 2))

    model = Model()
    path = tmp_path / "adapter.safetensors"
    mx.save_safetensors(str(path), {"lora_a": mx.ones((2, 1))})
    with pytest.raises(ValueError, match="expected LoRA tensors"):
        _swap_lora_weights(model, path)
    mx.save_safetensors(str(path), {"lora_a": mx.ones((2, 1)), "lora_b": mx.ones((1, 2))})
    _swap_lora_weights(model, path)
    assert model.lora_a.tolist() == [[1.0], [1.0]]


def test_frozen_prefix_matches_qwen_with_right_padding_and_adapters():
    mx = pytest.importorskip("mlx.core")
    qwen = pytest.importorskip("mlx_lm.models.qwen3")
    tuner = pytest.importorskip("mlx_lm.tuner.utils")
    model = qwen.Model(qwen.ModelArgs(
        model_type="qwen3", hidden_size=16, num_hidden_layers=4, intermediate_size=32,
        num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=24, num_key_value_heads=2,
        max_position_embeddings=1024, rope_theta=10000, head_dim=4, tie_word_embeddings=False,
    ))
    options = {"rank": 2, "scale": 2.0, "dropout": 0.0}
    config = {"fine_tune_type": "lora", "num_layers": 1, "lora_parameters": options}
    cache = _FrozenPrefix(model, config)
    token_batch = mx.array([[1, 3, 2, 0, 0], [1, 6, 5, 8, 9]])
    lengths = mx.array([3, 5])
    choices = mx.array([[20, 22], [20, 21]])
    tensors = (token_batch, lengths, choices)
    base = candidate_logits(model, *tensors)
    assert mx.allclose(base, cache(model, *tensors), atol=1e-7, rtol=1e-6).item()

    model.freeze()
    tuner.linear_to_lora_layers(model, 1, options)
    q_proj = model.layers[-1].self_attn.q_proj
    q_proj.lora_b = mx.full(q_proj.lora_b.shape, 0.1)
    tuned = candidate_logits(model, *tensors)
    assert not mx.array_equal(base, tuned).item()
    assert mx.allclose(tuned, cache(model, *tensors), atol=1e-7, rtol=1e-6).item()
