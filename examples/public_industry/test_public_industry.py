"""Date, label, and counterfactual invariants for the external panel."""

from dataclasses import replace
from datetime import date, timedelta
import json
import unittest

import torch

from gradient_desk.traces import ACTIONS, FUTURE_MICROBATCHES, WARMUP_STEPS

from benchmark import ANCHORS, MONITOR_DAYS, _case, _phases, window
from panel import Day, build_panel


def raw(n=168):
    start = date(2020, 1, 1)
    return [
        (start + timedelta(days=i), tuple(float((i + j) % 11) / 20 for j in range(12)))
        for i in range(n)
    ]


class PanelTests(unittest.TestCase):
    def test_next_date_only_changes_label(self):
        data = raw(24)
        baseline = build_panel(data)[0]
        label_values = (99.0,) * 12
        changed = [*data[:20], (data[20][0], label_values), *data[21:]]
        updated = build_panel(changed)[0]
        self.assertEqual(updated.features, baseline.features)
        self.assertEqual(updated.targets, label_values)
        self.assertEqual(updated.date, data[19][0])
        self.assertEqual(updated.label_date, data[20][0])

    def test_missing_day_does_not_shift_targets(self):
        data = raw(50)
        data[24] = (data[24][0], (float("nan"),) * 12)
        days = build_panel(data)
        decision_dates = {day.date for day in days}
        for i in range(23, 44):
            self.assertNotIn(data[i][0], decision_dates)
        self.assertIn(data[44][0], decision_dates)
        self.assertTrue(all(day.label_date - day.date == timedelta(days=1) for day in days))

    def test_window_is_embargoed_and_future_cannot_fit_scaler_or_prompt(self):
        torch.set_num_threads(1)
        days = build_panel(raw())
        phases = window(days, days[79].date)
        self.assertEqual(len(phases.monitor), MONITOR_DAYS)
        self.assertEqual(len(phases.warmup), WARMUP_STEPS)
        self.assertEqual(len(phases.future), FUTURE_MICROBATCHES)
        sequence = list(vars(phases).values())
        self.assertTrue(all(a[-1].label_date < b[0].date for a, b in zip(sequence, sequence[1:])))
        shock = list(phases.final)
        last = shock[-1]
        shock[-1] = Day(last.date, last.label_date,
                        tuple(tuple(x + 500.0 for x in row) for row in last.features),
                        tuple(y + 500.0 for y in last.targets))
        shifted = replace(phases, final=shock)
        for measured, control in zip(_phases(torch, phases)[:3], _phases(torch, shifted)[:3]):
            self.assertTrue(torch.equal(measured[0], control[0]))
            self.assertTrue(torch.equal(measured[1], control[1]))
        codes = [f"i{i}" for i in range(12)]
        original = _case(torch, codes, phases, ANCHORS[0], 0, 1)
        counterfactual = _case(torch, codes, shifted, ANCHORS[0], 0, 1)
        self.assertEqual(original["prompt"], counterfactual["prompt"])
        self.assertNotEqual(original["actions"]["resume"]["validation_loss"],
                            counterfactual["actions"]["resume"]["validation_loss"])
        self.assertEqual(set(original["actions"]), set(ACTIONS))
        for action, outcome in original["actions"].items():
            self.assertEqual(outcome["future_sample_digest"], original["budget"]["future_sample_digest"])
            self.assertEqual(outcome["examples_seen"], 12 * FUTURE_MICROBATCHES)
            self.assertEqual(outcome["forward_backward_calls"], FUTURE_MICROBATCHES)
            self.assertEqual(outcome["optimizer_steps"], 4 if action == "accumulate" else 8)
        self.assertEqual(len({result["rng_start_digest"] for result in original["actions"].values()}), 1)
        self.assertNotIn("future_sample_ids", json.loads(original["prompt"].split("\nTelemetry: ")[1]))


if __name__ == "__main__":
    unittest.main()
