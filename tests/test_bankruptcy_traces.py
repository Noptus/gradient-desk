"""Focused, deterministic leakage/budget tests for the bankruptcy generator.

No raw UCI data is bundled. Parsing, canonicalization, shuffle, grouping, and
phase tests use a synthetic in-memory table. Torch-dependent measurement tests
skip when torch is absent; nothing here loads a language model.

Run with src on PYTHONPATH:
    python -m unittest discover -s tests -p 'test_bankruptcy_traces.py'
"""

import copy
import hashlib
import importlib.util
import io
import json
import math
import unittest
import zipfile

from gradient_desk import bankruptcy_traces as bt
from gradient_desk.traces import ACTIONS


HAS_TORCH = importlib.util.find_spec("torch") is not None


def synthetic_csv(n_rows: int, n_features: int = 20, positive_every: int = 7) -> str:
    """Deterministic comma table: ``Bankrupt?`` first, then numeric feature cols.

    Feature cells are a smooth function of (row, column) so every vector is
    distinct; roughly one positive per ``positive_every`` rows.
    """
    header = ["Bankrupt?"] + [f" X{j}" for j in range(n_features)]
    lines = [",".join(header)]
    for k in range(n_rows):
        label = 1 if k % positive_every == 0 else 0
        cells = [str(label)] + [
            f"{0.01 * k + 0.1 * j + 0.001 * (k * j % 17):.6f}" for j in range(n_features)
        ]
        lines.append(",".join(cells))
    return "\n".join(lines) + "\n"


def synthetic_table(n_rows: int = 600) -> bt.SourceTable:
    lines = synthetic_csv(n_rows).splitlines()
    header = lines[0].split(",")
    rows = []
    labels = []
    for line in lines[1:]:
        cells = [c.strip() for c in line.split(",")]
        labels.append(int(cells[0]))
        rows.append(tuple(round(float(c), 10) for c in cells[1:1 + bt.FEATURE_COUNT]))
    return bt.SourceTable(
        feature_names=tuple(h.strip() for h in header[1:1 + bt.FEATURE_COUNT]),
        rows=tuple(rows), labels=tuple(labels),
        raw_row_count=n_rows, duplicate_groups=0,
        conflicting_groups=0, conflicting_rows=0,
    )


def zipped(csv_text: str, member: str = bt.ARCHIVE_MEMBER) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, csv_text)
    return buffer.getvalue()


class PinnedSourceTests(unittest.TestCase):
    def test_source_recorded_not_embedded(self):
        self.assertTrue(bt.SOURCE_URL.endswith("taiwanese+bankruptcy+prediction.zip"))
        self.assertEqual(bt.SOURCE_DOI, "10.24432/C5004D")
        self.assertEqual(bt.SOURCE_LICENSE, "CC BY 4.0")
        self.assertEqual(
            bt.SOURCE_SHA256,
            "c346f5ad2618cb198e7ed8306cf2f31fe3bb2ec60acdbbe1736788d50f269aac",
        )
        self.assertEqual(bt.ARCHIVE_MEMBER, "data.csv")
        self.assertEqual(bt.TARGET_COLUMN, "Bankrupt?")
        self.assertEqual(bt.FEATURE_COUNT, 16)

    def test_limits_are_bounded(self):
        self.assertEqual(bt.MAX_ZIP_BYTES, 10 * 1024 * 1024)
        self.assertEqual(bt.MAX_DECOMPRESSED_BYTES, 18 * 1024 * 1024)

    def test_hash_mismatch_refuses_to_decode(self):
        with self.assertRaises(ValueError):
            bt._verify_hash(b"not the archive")


class ArchiveDecodeTests(unittest.TestCase):
    def test_rejects_extra_members(self):
        with self.assertRaises(ValueError):
            bt._read_member(_two_member())

    def test_single_member_decodes(self):
        text = bt._read_member(zipped("Bankrupt?,X0\n0,1.0\n"))
        self.assertIn("Bankrupt?", text)

    def test_oversized_zip_rejected(self):
        with self.assertRaises(ValueError):
            bt._read_member(b"0" * (bt.MAX_ZIP_BYTES + 1))


def _two_member() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.csv", "Bankrupt?,X0\n0,1.0\n")
        archive.writestr("extra.csv", "x\n")
    return buffer.getvalue()


class CanonicalizationTests(unittest.TestCase):
    def _parse(self, csv_text: str) -> bt.SourceTable:
        payload = zipped(csv_text)
        # Patch the pinned hash so the synthetic archive verifies.
        original = bt.SOURCE_SHA256
        bt.SOURCE_SHA256 = hashlib.sha256(payload).hexdigest()
        try:
            class _P:
                def open(self, mode):
                    return io.BytesIO(payload)

            digest, table = bt.parse_archive(_P())
        finally:
            bt.SOURCE_SHA256 = original
        return table

    def test_selects_target_and_first_sixteen_features(self):
        table = self._parse(synthetic_csv(40))
        self.assertEqual(len(table.feature_names), 16)
        self.assertEqual(len(table.rows[0]), 16)
        self.assertNotIn("Bankrupt?", table.feature_names)
        self.assertGreater(sum(table.labels), 0)

    def test_exact_duplicate_feature_rows_collapse(self):
        # FEATURE_COUNT is 16; build a table with enough columns.
        header = ["Bankrupt?"] + [f"c{i}" for i in range(16)]
        a = ["0"] + ["1.0"] * 16
        dup = ["0"] + ["1.0"] * 16
        other = ["1"] + ["3.0"] * 16
        text = "\n".join(",".join(r) for r in (header, a, dup, other)) + "\n"
        table = self._parse(text)
        self.assertEqual(table.raw_row_count, 3)
        self.assertEqual(len(table.rows), 2)
        self.assertEqual(table.duplicate_groups, 1)
        self.assertEqual(table.conflicting_groups, 0)

    def test_conflicting_labels_are_skipped_and_counted(self):
        header = ["Bankrupt?"] + [f"c{i}" for i in range(16)]
        a = ["0"] + ["1.0"] * 16
        conflict = ["1"] + ["1.0"] * 16   # same features, different label
        keep = ["1"] + ["9.0"] * 16
        text = "\n".join(",".join(r) for r in (header, a, conflict, keep)) + "\n"
        table = self._parse(text)
        self.assertEqual(table.conflicting_groups, 1)
        self.assertEqual(table.conflicting_rows, 1)
        # Only the non-conflicting vector survives.
        self.assertEqual(len(table.rows), 1)
        self.assertEqual(table.labels, (1,))

    def test_non_binary_target_rejected(self):
        with self.assertRaises(ValueError):
            bt._parse_label("2")
        self.assertEqual(bt._parse_label("1"), 1)

    def test_non_numeric_feature_rejected(self):
        with self.assertRaises(ValueError):
            bt._canonical("abc")
        with self.assertRaises(ValueError):
            bt._canonical("inf")


class ShuffleAndGroupingTests(unittest.TestCase):
    def test_shuffle_is_deterministic_permutation(self):
        first = bt._shuffle_indices(50)
        second = bt._shuffle_indices(50)
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(50)))

    def test_group_size_band_and_profile_budget(self):
        self.assertGreaterEqual(bt.MAX_PROFILE_ROWS, bt.GROUP_MIN)
        self.assertLessEqual(bt.MAX_PROFILE_ROWS, bt.GROUP_MAX)
        self.assertEqual(bt.GROUP_MIN, 192)
        self.assertEqual(bt.GROUP_MAX, 240)

    def test_groups_are_disjoint_with_positives(self):
        table = synthetic_table(bt.MAX_PROFILE_ROWS * 2 + 30)
        groups = bt.carve_groups(table)
        self.assertGreaterEqual(len(groups), 1)
        seen = set()
        for group in groups:
            self.assertEqual(len(group), bt.MAX_PROFILE_ROWS)
            self.assertGreaterEqual(sum(table.labels[i] for i in group), 1)
            self.assertGreaterEqual(sum(table.labels[i] for i in group[-bt.TERMINAL_ROWS:]), 1)
            for index in group:
                self.assertNotIn(index, seen)  # strictly disjoint
                seen.add(index)

    def test_grouping_rejects_too_small(self):
        table = synthetic_table(10)
        with self.assertRaises(ValueError):
            bt.carve_groups(table)

    def test_group_size_out_of_band_rejected(self):
        table = synthetic_table(bt.MAX_PROFILE_ROWS * 2)
        with self.assertRaises(ValueError):
            bt.carve_groups(table, group_size=bt.GROUP_MAX + 1)


class PhaseLayoutTests(unittest.TestCase):
    def test_phase_slices_are_disjoint_and_sized(self):
        group = list(range(bt.MAX_PROFILE_ROWS))
        for profile in bt.PROFILES:
            phases = bt.slice_phases(group, profile)
            mb = profile["microbatch"]
            self.assertEqual(len(phases.scaler), bt.SCALER_ROWS)
            self.assertEqual(len(phases.monitor), bt.MONITOR_ROWS)
            self.assertEqual(len(phases.warmup), bt.WARMUP_STEPS * mb)
            self.assertEqual(len(phases.branch), bt.FUTURE_MICROBATCHES * mb)
            self.assertEqual(len(phases.terminal), bt.TERMINAL_ROWS)
            self.assertGreaterEqual(len(phases.terminal), 24)
            self.assertEqual(phases.terminal, group[-bt.TERMINAL_ROWS:])
            flat = (phases.scaler + phases.monitor + phases.warmup
                    + phases.branch + phases.terminal)
            self.assertEqual(len(set(flat)), len(flat))

    def test_profiles_count_and_settings(self):
        self.assertGreaterEqual(len(bt.PROFILES), 4)
        self.assertLessEqual(len(bt.PROFILES), 5)
        for profile in bt.PROFILES:
            self.assertIn(profile["microbatch"], (4, 6))
            self.assertIn(profile["pos_weight"], bt.POS_WEIGHT_CHOICES)
            self.assertIn(profile["optimizer"], ("sgd", "adam"))


class ReuseTests(unittest.TestCase):
    def test_reuses_canonical_actions_and_helpers(self):
        from gradient_desk import traces
        self.assertEqual(bt.ACTIONS, traces.ACTIONS)
        self.assertIs(bt.oracle_label, traces.oracle_label)
        self.assertIs(bt.format_telemetry, traces.format_telemetry)
        self.assertEqual(bt.WARMUP_STEPS, traces.WARMUP_STEPS)
        self.assertEqual(bt.FUTURE_MICROBATCHES, traces.FUTURE_MICROBATCHES)


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
        cls.table = synthetic_table(bt.MAX_PROFILE_ROWS + 40)
        cls.group = bt.carve_groups(cls.table)[0]
        with torch.random.fork_rng(devices=[]):
            cls.record = bt._case(torch, cls.table, cls.group, 0, 0)

    def test_schema_fields(self):
        row = self.record
        self.assertEqual(row["track"], "training")
        self.assertEqual(row["family"], "taiwan_bankruptcy_mlp")
        self.assertEqual(row["benchmark"], "taiwan_bankruptcy_intervention")
        self.assertEqual(row["split"], "test")
        self.assertIn(row["label"], ACTIONS)
        self.assertEqual(set(row["actions"]), set(ACTIONS))
        self.assertIn("evaluation_block", row)
        self.assertEqual(row["evaluation_block"], row["group_metadata"]["canonical_group"])
        self.assertIn("data_source", row)
        self.assertIn("evaluation_receipts", row)
        self.assertEqual(json.loads(json.dumps(row, allow_nan=False)), row)

    def test_eight_forward_backward_calls_and_shared_future(self):
        budget = self.record["budget"]
        self.assertEqual(budget["forward_backward_calls"], 8)
        self.assertEqual(budget["microbatches"], 8)
        for action, result in self.record["actions"].items():
            self.assertEqual(result["forward_backward_calls"], 8)
            self.assertEqual(result["examples_seen"], budget["future_examples"])
            self.assertEqual(result["future_sample_digest"], budget["future_sample_digest"])
            accumulation = 2 if action == "accumulate" else 1
            self.assertEqual(result["optimizer_steps"], 8 // accumulation)

    def test_equal_samples_and_exact_restoration(self):
        actions = self.record["actions"]
        sample_sets = {tuple(a["future_sample_ids"]) for a in actions.values()}
        self.assertEqual(len(sample_sets), 1)
        self.assertEqual(len({a["rng_start_digest"] for a in actions.values()}), 1)
        base_lr = actions["resume"]["lr"]
        self.assertAlmostEqual(actions["lower_lr"]["lr"], base_lr * 0.25)
        self.assertEqual(actions["clip_grad"]["clip_grad_norm"], 0.5)
        self.assertEqual(actions["clip_grad"]["clipping_calls"], 8)

    def test_label_matches_oracle_on_terminal_losses(self):
        losses = {a: self.record["actions"][a]["validation_loss"] for a in ACTIONS}
        self.assertEqual(self.record["label"], bt.oracle_label(losses))
        best = min(losses.values())
        self.assertAlmostEqual(self.record["label_regret"], losses[self.record["label"]] - best)
        for action, result in self.record["actions"].items():
            self.assertAlmostEqual(result["regret"], losses[action] - best)
            self.assertGreaterEqual(result["regret"], 0)
            self.assertTrue(math.isfinite(result["validation_loss"]))

    def test_prompt_is_pre_action_only_no_outcome_or_prediction(self):
        forbidden = ("label", "regret", "actions", "validation_loss", "elapsed_seconds",
                     "future_sample_ids", "Bankrupt?", "bankrupt", "prediction")
        instruction, line = self.record["prompt"].split("\nTelemetry: ")
        self.assertNotIn("\n", line)
        payload = json.loads(line)
        for key in forbidden:
            self.assertNotIn(key, payload)
        telemetry = json.loads(line)
        self.assertEqual(telemetry["training_steps"], bt.WARMUP_STEPS)
        self.assertEqual(len(telemetry["train_loss_history"]), bt.WARMUP_STEPS)
        self.assertEqual(telemetry["objective"], "binary_cross_entropy")
        self.assertIn(telemetry["pos_weight"], bt.POS_WEIGHT_CHOICES)

    def test_no_raw_ratios_or_source_indices_in_trace(self):
        line = json.dumps(self.record, allow_nan=False)
        # Raw source-row indices and raw ratio values must not leak as sample ids.
        for action in self.record["actions"].values():
            for sample in action["future_sample_ids"]:
                self.assertTrue(sample.startswith("g0/branch/"))

    def test_determinism_excluding_wall_time(self):
        with self.torch.random.fork_rng(devices=[]):
            again = bt._case(self.torch, self.table, self.group, 0, 0)
        self.assertEqual(without_wall_time(self.record), without_wall_time(again))

    def test_branch_order_does_not_change_losses(self):
        from unittest import mock
        with self.torch.random.fork_rng(devices=[]):
            baseline = bt._case(self.torch, self.table, self.group, 0, 0)
            with mock.patch.object(bt, "ACTIONS", tuple(reversed(ACTIONS))):
                flipped = bt._case(self.torch, self.table, self.group, 0, 0)
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
        self.assertEqual(bt._measurement_hash([row]), bt._measurement_hash([slow]))


if __name__ == "__main__":
    unittest.main()
