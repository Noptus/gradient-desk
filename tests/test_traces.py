"""Standard-library tests; torch-dependent tests skip when torch is absent.

Run after installing the package or with src on PYTHONPATH:
    python -m unittest discover -s tests -p 'test_traces.py'
"""

import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from gradient_desk import traces


HAS_TORCH = importlib.util.find_spec("torch") is not None


def without_times(record):
    result = copy.deepcopy(record)
    for action in result["actions"].values():
        action.pop("elapsed_seconds")
        action.pop("elapsed_ratio_to_resume")
    return result


class OracleTests(unittest.TestCase):
    def test_resume_near_tie(self):
        losses = dict(zip(traces.ACTIONS, (1.0005, 1.0, 1.2, 1.3)))
        self.assertEqual(traces.oracle_label(losses), "resume")

    def test_clear_winner_and_fixed_secondary_order(self):
        losses = dict(zip(traces.ACTIONS, (2.0, 1.0005, 1.0, 1.5)))
        self.assertEqual(traces.oracle_label(losses), "lower_lr")
        losses["accumulate"] = 0.5
        self.assertEqual(traces.oracle_label(losses), "accumulate")

    def test_bad_losses(self):
        with self.assertRaises(ValueError):
            traces.oracle_label({"resume": 1.0})
        for invalid in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                traces.oracle_label({a: invalid for a in traces.ACTIONS})

    def test_invalid_seed_counts(self):
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                next(traces.generate_records(value))

    def test_missing_optional_dependency_message(self):
        import builtins
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("intentionally absent")
            return original(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded):
            with self.assertRaisesRegex(RuntimeError, "optional PyTorch"):
                traces._torch()

    def test_declared_splits(self):
        self.assertEqual(len(traces.FAMILY_SPLITS), 6)
        groups = [
            {f for f, s in traces.FAMILY_SPLITS.items() if s == split}
            for split in ("train", "valid", "test")
        ]
        self.assertTrue(all(groups))
        for i, group in enumerate(groups):
            for other in groups[i + 1:]:
                self.assertTrue(group.isdisjoint(other))
        self.assertEqual(len(set(traces.SPLIT_SEED_BASES.values())), 3)


@unittest.skipUnless(HAS_TORCH, "Optional torch is not installed")
class MeasuredTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = list(traces.generate_records(seeds_per_case=1))

    def test_schema_and_mixed_families(self):
        self.assertEqual(len(self.records), 36)
        self.assertEqual(len({r["case_id"] for r in self.records}), 36)
        self.assertEqual(len({r["seed"] for r in self.records}), 36)
        self.assertEqual({r["family"] for r in self.records}, set(traces.FAMILY_SPLITS))
        for row in self.records:
            self.assertIsInstance(row["case_id"], str)
            self.assertEqual(row["track"], "training")
            self.assertEqual(row["split"], traces.FAMILY_SPLITS[row["family"]])
            self.assertIn(row["label"], traces.ACTIONS)
            self.assertEqual(set(row["actions"]), set(traces.ACTIONS))
            self.assertEqual(json.loads(json.dumps(row, allow_nan=False)), row)
            for result in row["actions"].values():
                self.assertIsInstance(result["validation_loss"], float)
                self.assertTrue(math.isfinite(result["validation_loss"]))
                self.assertGreaterEqual(result["elapsed_seconds"], 0)

    def test_determinism_all_profiles_and_families(self):
        repeated = list(traces.generate_records(seeds_per_case=1))
        self.assertEqual(
            [without_times(r) for r in self.records],
            [without_times(r) for r in repeated],
        )

    def test_measured_oracle_and_regrets(self):
        for row in self.records:
            losses = {a: row["actions"][a]["validation_loss"] for a in traces.ACTIONS}
            self.assertEqual(row["label"], traces.oracle_label(losses))
            best = min(losses.values())
            self.assertAlmostEqual(row["label_regret"], losses[row["label"]] - best)
            for action, result in row["actions"].items():
                self.assertAlmostEqual(result["regret"], losses[action] - best)
                self.assertGreaterEqual(result["regret"], 0)

    def test_matched_future_examples_and_cadence(self):
        for row in self.records:
            budget = row["budget"]
            begin = budget["checkpoint_cursor"]
            expected_ids = list(range(begin, begin + budget["future_examples"]))
            for action, result in row["actions"].items():
                self.assertEqual(result["future_sample_ids"], expected_ids)
                self.assertEqual(result["examples_seen"], budget["future_examples"])
                self.assertEqual(result["future_sample_digest"], budget["future_sample_digest"])
                self.assertEqual(result["cursor_start"], begin)
                self.assertEqual(result["cursor_end"], begin + len(expected_ids))
                self.assertEqual(result["rng_start_digest"], budget["checkpoint_rng_digest"])
                self.assertEqual(result["rng_end_digest"], result["rng_start_digest"])
                self.assertEqual(result["forward_backward_calls"], 8)
                accumulation = 2 if action == "accumulate" else 1
                self.assertEqual(result["optimizer_steps"], 8 // accumulation)
                self.assertEqual(result["effective_batch_size"], budget["microbatch_size"] * accumulation)
            base_lr = row["actions"]["resume"]["lr"]
            self.assertEqual(row["actions"]["lower_lr"]["lr"], base_lr * 0.25)
            self.assertEqual(row["actions"]["clip_grad"]["clip_grad_norm"], 0.5)

    def test_actions_have_actual_effects(self):
        for action in traces.ACTIONS[1:]:
            self.assertTrue(any(
                r["actions"][action]["validation_loss"] != r["actions"]["resume"]["validation_loss"]
                for r in self.records
            ), action)

    def test_prompt_is_pre_action_only(self):
        forbidden = ("label", "regret", "actions", "family", "split", "seed", "profile_index",
                     "elapsed_seconds", "future_sample_ids", "injection", "scenario")
        for row in self.records:
            instruction, line = row["prompt"].split("\nTelemetry: ")
            self.assertEqual(instruction, traces.INSTRUCTION)
            self.assertNotIn("\n", line)
            telemetry = json.loads(line)
            for key in forbidden:
                self.assertNotIn('"' + key + '"', line)
            self.assertNotIn(row["family"], row["prompt"])
            self.assertEqual(telemetry["training_steps"], traces.WARMUP_STEPS)
            self.assertEqual(len(telemetry["train_loss_history"]), traces.WARMUP_STEPS)
            self.assertEqual(telemetry["data"]["observed_examples"], row["budget"]["checkpoint_cursor"])
            self.assertEqual(telemetry["validation_loss_history"][-1]["step"], traces.WARMUP_STEPS)

    def test_action_execution_order_does_not_change_losses(self):
        # Same case/seed, reversed trial ordering: detects missing restoration
        # of model parameters, optimizer momentum, cursor, or Torch RNG.
        torch = traces._torch()
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]):
                baseline = traces._case(torch, "nonlinear_regression", 2, 0)
                with mock.patch.object(traces, "ACTIONS", tuple(reversed(traces.ACTIONS))):
                    reversed_order = traces._case(torch, "nonlinear_regression", 2, 0)
            self.assertEqual(baseline["actions"].keys(), reversed_order["actions"].keys())
            for action in traces.ACTIONS:
                self.assertEqual(
                    without_times(baseline)["actions"][action],
                    without_times(reversed_order)["actions"][action],
                )
        finally:
            torch.set_num_threads(threads)

    def test_caller_rng_and_settings_restored_before_yield(self):
        torch = traces._torch()
        rng = torch.get_rng_state().clone()
        threads = torch.get_num_threads()
        deterministic = torch.are_deterministic_algorithms_enabled()
        iterator = traces.generate_records(1)
        next(iterator)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(threads, torch.get_num_threads())
        self.assertEqual(deterministic, torch.are_deterministic_algorithms_enabled())
        iterator.close()

    def test_cli_jsonl_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.jsonl"
            # Generation is tested above; isolate export mechanics cheaply.
            with mock.patch.object(traces, "generate_records", return_value=iter(self.records)), mock.patch("builtins.print"):
                self.assertEqual(traces.main(["--output", str(path), "--seeds-per-case", "1"]), 0)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows, self.records)
            with self.assertRaises(FileExistsError):
                traces.main(["--output", str(path), "--seeds-per-case", "1"])


if __name__ == "__main__":
    unittest.main()
