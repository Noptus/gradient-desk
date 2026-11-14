"""Focused, deterministic leakage and budget tests for the Treasury generator.

No raw Treasury data is bundled; the panel-shaping, split, and embargo tests use
a synthetic ascending yield table. Torch-dependent measurement tests skip when
torch is absent.

Run with src on PYTHONPATH:
    python -m unittest discover -s tests -p 'test_treasury_traces.py'
"""

import copy
from datetime import date, timedelta
import importlib.util
import json
import math
import unittest

from gradient_desk import treasury_traces as tt
from gradient_desk.traces import ACTIONS


HAS_TORCH = importlib.util.find_spec("torch") is not None


def synthetic_rows(n_sessions: int, start: date = date(1991, 1, 2)) -> list:
    """Deterministic ascending (date, per-tenor level) rows, all finite.

    One business-ish session per weekday; levels are a smooth tenor-indexed
    function of the session number so every lookback window is well-defined.
    """
    rows = []
    day = start
    for k in range(n_sessions):
        while day.weekday() >= 5:  # skip weekends, keep "sessions" plausible
            day += timedelta(days=1)
        levels = tuple(
            1.0 + 0.3 * i + 0.05 * math.sin(0.11 * k + 0.2 * i) + 0.002 * k
            for i in range(tt.TENOR_COUNT)
        )
        rows.append((day, levels))
        day += timedelta(days=1)
    return rows


class ParsingAndPanelTests(unittest.TestCase):
    def test_columns_and_tenors_as_specified(self):
        self.assertEqual(tt.CSV_COLUMNS[0], "Date")
        self.assertEqual(
            tt.CSV_COLUMNS,
            ("Date", "1 Mo", "2 Mo", "3 Mo", "4 Mo", "6 Mo", "1 Yr", "2 Yr",
             "3 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr"),
        )
        self.assertEqual(
            tt.TENORS, ("6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "30 Yr")
        )
        self.assertEqual(tt.TENOR_COUNT, 8)
        for tenor in tt.TENORS:
            self.assertIn(tenor, tt.CSV_COLUMNS)

    def test_date_parse_pivot_matches_two_digit_year(self):
        self.assertEqual(tt._parse_date("01/02/90"), date(1990, 1, 2))
        self.assertEqual(tt._parse_date("12/06/21"), date(2021, 12, 6))
        self.assertEqual(tt._parse_date("06/30/24"), date(2024, 6, 30))

    def test_pinned_source_recorded_not_embedded(self):
        self.assertTrue(tt.SOURCE_URL.startswith("https://home.treasury.gov/"))
        self.assertEqual(tt.SOURCE_SHA256,
                         "258a594eb64e9d21d9e8574b0b8580b0a5230dce56bb284ec6aa49a9a260897f")
        self.assertEqual(tt.METHOD_BREAK, date(2021, 12, 6))

    def test_targets_are_next_session_change_in_basis_points(self):
        rows = synthetic_rows(tt.LOOKBACK_DAYS + 3)
        panel = tt.build_panel(rows)
        self.assertTrue(panel)
        first = panel[0]
        # Find the row pair feeding the first panel session.
        t = next(i for i, (d, _) in enumerate(rows) if d == first.date)
        for i in range(tt.TENOR_COUNT):
            expected = 100.0 * (rows[t + 1][1][i] - rows[t][1][i])
            self.assertAlmostEqual(first.targets[i], expected, places=6)
        self.assertEqual(first.label_date, rows[t + 1][0])

    def test_panel_uses_only_backward_rows(self):
        # Decision date features must not depend on any row strictly after t.
        rows = synthetic_rows(tt.LOOKBACK_DAYS + 5)
        panel = tt.build_panel(rows)
        perturbed = list(rows)
        t = next(i for i, (d, _) in enumerate(rows) if d == panel[0].date)
        # Corrupt all rows strictly after t; the first session's features must not move.
        for j in range(t + 2, len(perturbed)):
            perturbed[j] = (perturbed[j][0], tuple(99.0 for _ in range(tt.TENOR_COUNT)))
        panel2 = tt.build_panel(perturbed)
        self.assertEqual(panel[0].features, panel2[0].features)


class SplitAndEmbargoTests(unittest.TestCase):
    def test_splits_are_disjoint_whole_chronology(self):
        years = tt.SPLIT_YEARS
        self.assertEqual(years["train"], (1990, 2013))
        self.assertEqual(years["valid"], (2014, 2018))
        self.assertEqual(years["test"], (2019, 2024))
        ranges = list(years.values())
        for (lo1, hi1), (lo2, hi2) in zip(ranges, ranges[1:]):
            self.assertLess(hi1, lo2)  # strictly ordered, no shared year
        self.assertEqual(len(set(tt.SPLIT_SEED_BASES.values())), 3)

    def test_split_membership_by_year(self):
        self.assertTrue(tt._in_split(date(2013, 12, 31), "train"))
        self.assertFalse(tt._in_split(date(2014, 1, 1), "train"))
        self.assertTrue(tt._in_split(date(2014, 1, 1), "valid"))
        self.assertTrue(tt._in_split(date(2019, 6, 1), "test"))

    def test_block_phase_layout_and_size(self):
        self.assertEqual(
            tt.BLOCK_PHASE_DAYS,
            tt.LOOKBACK_DAYS + tt.SCALER_DAYS + tt.MONITOR_DAYS
            + tt.WARMUP_STEPS + tt.FUTURE_MICROBATCHES + tt.FINAL_DAYS,
        )
        self.assertEqual(tt.BLOCK_EMBARGO_DAYS, 5 * tt.EMBARGO)
        self.assertEqual(tt.BLOCK_SESSIONS, tt.BLOCK_PHASE_DAYS + tt.BLOCK_EMBARGO_DAYS + 1)
        # Whole block falls in the documented ~72-83 trading-session band.
        self.assertGreaterEqual(tt.BLOCK_SESSIONS, 72)
        self.assertLessEqual(tt.BLOCK_SESSIONS, 83)
        self.assertEqual(tt.WARMUP_STEPS, 12)
        self.assertEqual(tt.FUTURE_MICROBATCHES, 8)

    def test_carved_phases_isolated_with_embargo(self):
        panel = tt.build_panel(synthetic_rows(tt.BLOCK_SESSIONS + tt.LOOKBACK_DAYS))
        block = tt.carve_block(panel, 0)
        phases = [getattr(block, name) for name in tt._phase_names()]
        sizes = (tt.LOOKBACK_DAYS, tt.SCALER_DAYS, tt.MONITOR_DAYS,
                 tt.WARMUP_STEPS, tt.FUTURE_MICROBATCHES, tt.FINAL_DAYS)
        for part, size in zip(phases, sizes):
            self.assertEqual(len(part), size)
        # No session is shared between any two phases.
        all_dates = [s.date for part in phases for s in part]
        self.assertEqual(len(all_dates), len(set(all_dates)))
        # Each phase's last label strictly precedes the next phase's first feature.
        for left, right in zip(phases, phases[1:]):
            self.assertLess(left[-1].label_date, right[0].date)
            # Embargo: at least one observed session sits in the gap.
            self.assertNotEqual(left[-1].label_date, right[0].date)

    def test_blocks_do_not_reuse_sessions_and_stay_in_split(self):
        # Enough synthetic sessions, all in 1991 -> train split, to carve >=2 blocks.
        panel = tt.build_panel(synthetic_rows(2 * tt.BLOCK_SESSIONS + 2 * tt.LOOKBACK_DAYS + 10, start=date(1991, 1, 2)))
        blocks = tt._blocks_for_split(panel, "train")
        self.assertGreaterEqual(len(blocks), 2)
        used = []
        for _, block in blocks:
            for name in tt._phase_names():
                used.extend(s.date for s in getattr(block, name))
            for name in tt._phase_names():
                for s in getattr(block, name):
                    self.assertTrue(tt._in_split(s.date, "train"))
        self.assertEqual(len(used), len(set(used)))
        for (_, previous), (_, current) in zip(blocks, blocks[1:]):
            self.assertLess(previous.final[-1].label_date, current.lookback[0].history_start_date)
        for _, block in blocks:
            self.assertTrue(tt._in_split(block.lookback[0].history_start_date, "train"))
            self.assertTrue(tt._in_split(block.final[-1].label_date, "train"))


class OracleAndFormatReuseTests(unittest.TestCase):
    def test_reuses_canonical_actions_and_oracle(self):
        from gradient_desk import traces
        self.assertEqual(tt.ACTIONS, traces.ACTIONS)
        self.assertIs(tt.oracle_label, traces.oracle_label)
        self.assertIs(tt.format_telemetry, traces.format_telemetry)

    def test_profiles_count_in_four_to_six(self):
        self.assertGreaterEqual(len(tt.PROFILES), 4)
        self.assertLessEqual(len(tt.PROFILES), 6)


def without_wall_time(record):
    result = copy.deepcopy(record)
    for action in result["actions"].values():
        action.pop("elapsed_seconds")
    return result


@unittest.skipUnless(HAS_TORCH, "Optional torch is not installed")
class MeasuredBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        cls.panel = tt.build_panel(synthetic_rows(tt.BLOCK_SESSIONS + tt.LOOKBACK_DAYS))
        cls.block = tt.carve_block(cls.panel, 0)
        with torch.random.fork_rng(devices=[]):
            cls.record = tt._case(torch, "train", cls.block, 0, 0)

    def test_schema_and_json_roundtrip(self):
        row = self.record
        self.assertEqual(row["track"], "training")
        self.assertEqual(row["family"], "treasury_yield_mlp")
        self.assertEqual(row["split"], "train")
        self.assertEqual(row["split_id"], "treasury:train:1990-2013")
        self.assertIn(row["label"], ACTIONS)
        self.assertEqual(set(row["actions"]), set(ACTIONS))
        self.assertEqual(json.loads(json.dumps(row, allow_nan=False)), row)
        # evaluation_block names the first branch (future) session.
        self.assertEqual(row["evaluation_block"], self.block.branch[0].date.isoformat())

    def test_eight_forward_backward_calls_and_shared_future(self):
        budget = self.record["budget"]
        self.assertEqual(budget["forward_backward_calls"], 8)
        self.assertEqual(budget["microbatches"], 8)
        self.assertEqual(budget["microbatch_size"], tt.TENOR_COUNT)
        for action, result in self.record["actions"].items():
            self.assertEqual(result["forward_backward_calls"], 8)
            self.assertEqual(result["examples_seen"], budget["future_examples"])
            self.assertEqual(result["future_sample_digest"], budget["future_sample_digest"])
            accumulation = 2 if action == "accumulate" else 1
            self.assertEqual(result["optimizer_steps"], 8 // accumulation)
            self.assertEqual(result["effective_batch_size"], tt.TENOR_COUNT * accumulation)

    def test_equal_samples_and_exact_restoration(self):
        actions = self.record["actions"]
        # Every action consumes identical ordered tenor-session examples.
        sample_sets = {tuple(a["future_sample_ids"]) for a in actions.values()}
        self.assertEqual(len(sample_sets), 1)
        for result in actions.values():
            # RNG restored exactly: start digest equals the checkpoint across actions.
            self.assertEqual(result["rng_start_digest"], result["rng_end_digest"])
        self.assertEqual(len({a["rng_start_digest"] for a in actions.values()}), 1)
        base_lr = actions["resume"]["lr"]
        self.assertAlmostEqual(actions["lower_lr"]["lr"], base_lr * 0.25)
        self.assertEqual(actions["clip_grad"]["clip_grad_norm"], 0.5)
        self.assertEqual(actions["clip_grad"]["clipping_calls"], 8)

    def test_label_matches_oracle_on_terminal_losses(self):
        losses = {a: self.record["actions"][a]["validation_loss"] for a in ACTIONS}
        self.assertEqual(self.record["label"], tt.oracle_label(losses))
        best = min(losses.values())
        self.assertAlmostEqual(self.record["label_regret"], losses[self.record["label"]] - best)
        for action, result in self.record["actions"].items():
            self.assertAlmostEqual(result["regret"], losses[action] - best)
            self.assertGreaterEqual(result["regret"], 0)
            self.assertTrue(math.isfinite(result["validation_loss"]))

    def test_prompt_is_pre_action_only(self):
        forbidden = ("label", "regret", "actions", "split", "seed", "profile_index",
                     "validation_loss", "elapsed_seconds", "future_sample_ids", "y_scale_bp")
        instruction, line = self.record["prompt"].split("\nTelemetry: ")
        self.assertNotIn("\n", line)
        for key in forbidden:
            self.assertNotIn('"' + key + '"', line)
        # No raw dates leak into the prompt.
        self.assertNotIn(self.block.branch[0].date.isoformat(), self.record["prompt"])
        telemetry = json.loads(line)
        self.assertEqual(telemetry["training_steps"], tt.WARMUP_STEPS)
        self.assertEqual(len(telemetry["train_loss_history"]), tt.WARMUP_STEPS)

    def test_determinism_excluding_wall_time(self):
        with self.torch.random.fork_rng(devices=[]):
            again = tt._case(self.torch, "train", self.block, 0, 0)
        self.assertEqual(without_wall_time(self.record), without_wall_time(again))

    def test_branch_order_does_not_change_losses(self):
        from unittest import mock
        with self.torch.random.fork_rng(devices=[]):
            baseline = tt._case(self.torch, "train", self.block, 0, 0)
            with mock.patch.object(tt, "ACTIONS", tuple(reversed(ACTIONS))):
                flipped = tt._case(self.torch, "train", self.block, 0, 0)
        for action in ACTIONS:
            self.assertEqual(
                without_wall_time(baseline)["actions"][action],
                without_wall_time(flipped)["actions"][action],
            )

    def test_measurement_hash_excludes_wall_time(self):
        row = copy.deepcopy(self.record)
        slow = copy.deepcopy(self.record)
        for action in slow["actions"].values():
            action["elapsed_seconds"] = action["elapsed_seconds"] + 123.0
        self.assertEqual(tt._measurement_hash([row]), tt._measurement_hash([slow]))


if __name__ == "__main__":
    unittest.main()
