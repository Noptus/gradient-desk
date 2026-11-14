import copy
import math

import pytest

from gradient_desk.choice_data import layout, target_distribution, validate
from gradient_desk.choice_report import compare


def example(key, split="train"):
    return {
        "case_id": key, "track": "fixture", "family": "fixture", "split": split,
        "prompt": f"Question {key}", "label": "yes", "choices": {"yes": "Yes.", "no": "No."},
    }


def test_layout_ignores_labels_and_rotates_answers():
    row = example("r")
    first = layout(row)
    changed = copy.deepcopy(row)
    changed["label"] = "no"
    changed["target_dist"] = {"yes": 0.0, "no": 1.0}
    assert layout(changed) == first
    assert layout(row, variant=1)[1] == first[1][::-1]
    assert sum(target_distribution(changed, first[1])) == 1


def test_validate_fails_for_group_crossing_and_bad_soft_labels():
    one, two = example("one"), example("two", "valid")
    one["group_id"] = two["group_id"] = "same"
    with pytest.raises(ValueError, match="Source group"):
        validate([one, two])
    two["target_dist"] = {"yes": math.nan, "no": 1}
    with pytest.raises(ValueError, match="finite"):
        validate([two])


def test_paired_report_rejects_missing_ids_and_counts_blocks():
    cases = [example(str(i), "test") for i in range(12)]
    base, tuned = [], []
    for i, row in enumerate(cases):
        row["evaluation_block"] = i // 2
        base.append({"case_id": row["case_id"], "prediction": "no"})
        tuned.append({"case_id": row["case_id"], "prediction": "yes"})
    panel = compare(cases, base, tuned, replications=100)["panels"]["fixture"]
    assert panel["dependence_blocks"] == 6
    assert panel["gain"] == 1
    assert panel["p_block_one_sided"] == 1 / 64
    assert panel["paired_block_bootstrap95"] == [1, 1]
    with pytest.raises(ValueError, match="same IDs"):
        compare(cases, base[:-1], tuned, replications=100)
