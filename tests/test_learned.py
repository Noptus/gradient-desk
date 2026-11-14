"""Standard-library tests for the transparent data-driven baselines.

These validate both tracks on small synthetic records, split rejection,
deterministic tie-breaking, and an independently read-through of the
selection/retrieval logic (no reliance on the module's own helpers for the
oracle computations).
"""

import json
import math
import unittest

from gradient_desk.dataset import METHODS, method_prompt
from gradient_desk.learned import (
    METHODOLOGY_K_CANDIDATES,
    TRAINING_K_CANDIDATES,
    make_baseline,
    training_signature,
)
from gradient_desk.traces import ACTIONS, INSTRUCTION


# --------------------------------------------------------------------------- #
# Synthetic record builders
# --------------------------------------------------------------------------- #
def training_prompt(*, lr, batch, grad_norms, train_losses, val_losses,
                    objective="mean_squared_error"):
    telemetry = {
        "objective": objective,
        "lr": lr,
        "microbatch_size": batch,
        "batch_size": batch,
        "gradient_norm_history": grad_norms,
        "train_loss_history": train_losses,
        "validation_loss_history": [{"step": i, "loss": v} for i, v in enumerate(val_losses)],
    }
    return INSTRUCTION + "\nTelemetry: " + json.dumps(telemetry)


def training_row(case_id, family, split, label_losses, **telemetry_kwargs):
    return {
        "case_id": case_id,
        "track": "training",
        "family": family,
        "split": split,
        "prompt": training_prompt(**telemetry_kwargs),
        "label": min(label_losses, key=label_losses.get),
        "actions": {a: {"validation_loss": label_losses[a]} for a in ACTIONS},
    }


def methodology_row(case_id, family, split, scenario, label):
    return {
        "case_id": case_id,
        "track": "methodology",
        "family": family,
        "split": split,
        "prompt": method_prompt(scenario),
        "label": label,
    }


# Two latent "regimes": high gradient norm -> clip_grad is best; high lr ->
# lower_lr is best. Validation cases mirror the regimes so k selection is
# well-posed and the nearest neighbour is unambiguous.
def clip_case(cid, split):
    return training_row(
        cid, f"clipfam-{cid}", split,
        {"resume": 1.0, "lower_lr": 0.9, "clip_grad": 0.2, "accumulate": 0.95},
        lr=0.03, batch=16, grad_norms=[1.0, 2.0, 3.5, 4.2],
        train_losses=[2.0, 1.8, 1.6, 1.5], val_losses=[2.0, 1.9, 1.83 if split == "valid" else 1.85],
    )


def lr_case(cid, split):
    return training_row(
        cid, f"lrfam-{cid}", split,
        {"resume": 1.2, "lower_lr": 0.3, "clip_grad": 1.1, "accumulate": 1.0},
        lr=0.45, batch=16, grad_norms=[0.5, 0.6, 0.7, 0.8],
        train_losses=[1.0, 1.1, 1.3, 1.6], val_losses=[1.0, 1.2, 1.53 if split == "valid" else 1.5],
    )


METHOD_LABELS = list(METHODS)


def method_scenario(keyword):
    # Distinctive vocabulary per label makes TF-IDF retrieval deterministic.
    return f"A fictional research case about {keyword} {keyword} and review."


class TrainingTrackTests(unittest.TestCase):
    def setUp(self):
        self.train = [clip_case(f"c{i}", "train") for i in range(4)]
        self.train += [lr_case(f"l{i}", "train") for i in range(4)]
        self.valid = [clip_case("cv", "valid"), lr_case("lv", "valid")]

    def test_signature_is_scale_normalized_tuple(self):
        sig = training_signature(self.train[0]["prompt"])
        self.assertEqual(len(sig), 10)
        self.assertTrue(all(math.isfinite(x) for x in sig))

    def test_predicts_nearest_action_per_regime(self):
        predict = make_baseline(self.train, self.valid)
        clip_query = clip_case("q", "valid")
        lr_query = lr_case("q", "valid")
        self.assertEqual(predict(clip_query), "clip_grad")
        self.assertEqual(predict(lr_query), "lower_lr")

    def test_k_selected_from_predeclared_candidates(self):
        predict = make_baseline(self.train, self.valid)
        settings = predict.settings
        self.assertEqual(settings.track, "training")
        self.assertEqual(settings.candidate_k, TRAINING_K_CANDIDATES)
        self.assertIn(settings.selected_k, TRAINING_K_CANDIDATES)
        self.assertEqual(set(settings.validation_regret), set(TRAINING_K_CANDIDATES))

    def test_deterministic_resume_first_tie(self):
        # All actions equal -> normalized losses all zero -> resume wins.
        flat = training_row(
            "flat", "flatfam", "train",
            {a: 0.5 for a in ACTIONS},
            lr=0.03, batch=16, grad_norms=[1.0, 1.0, 1.0, 1.0],
            train_losses=[1.0, 1.0], val_losses=[1.0, 1.0],
        )
        predict = make_baseline([flat] + self.train, self.valid)
        self.assertEqual(predict(flat), "resume")
        self.assertEqual(predict.settings.tie_order, ACTIONS)

    def test_deterministic_repeated_fit(self):
        a = make_baseline(self.train, self.valid)
        b = make_baseline(self.train, self.valid)
        query = lr_case("q", "valid")
        self.assertEqual(a(query), b(query))
        self.assertEqual(a.settings.selected_k, b.settings.selected_k)

    def test_independent_nearest_neighbour_oracle(self):
        # Recompute k=1 prediction entirely independently of module internals,
        # using raw normalized losses and Euclidean distance on z-scored sigs.
        sigs = [training_signature(r["prompt"]) for r in self.train]
        width = len(sigs[0])
        means = [sum(s[i] for s in sigs) / len(sigs) for i in range(width)]
        stds = []
        for i in range(width):
            var = sum((s[i] - means[i]) ** 2 for s in sigs) / len(sigs)
            stds.append(math.sqrt(var) or 1.0)

        def z(sig):
            return [(sig[i] - means[i]) / stds[i] for i in range(width)]

        def norm_losses(row):
            vals = {a: row["actions"][a]["validation_loss"] for a in ACTIONS}
            lo, hi = min(vals.values()), max(vals.values())
            span = hi - lo
            return {a: (vals[a] - lo) / span if span else 0.0 for a in vals}

        train_pts = [z(s) for s in sigs]
        train_nl = [norm_losses(r) for r in self.train]
        query = clip_case("q", "valid")
        qpt = z(training_signature(query["prompt"]))
        nearest = min(range(len(train_pts)),
                      key=lambda i: sum((a - b) ** 2 for a, b in zip(qpt, train_pts[i])))
        scores = train_nl[nearest]
        best = min(scores.values())
        expected = next(a for a in ACTIONS if scores[a] <= best)
        policy = make_baseline(self.train, self.valid)
        self.assertEqual(policy.settings.selected_k, 1)
        self.assertEqual(expected, policy(query))


class MethodologyTrackTests(unittest.TestCase):
    def setUp(self):
        self.train = []
        for rep in range(3):
            for label in METHOD_LABELS:
                self.train.append(methodology_row(
                    f"{label}-{rep}", f"fam-{label}", "train",
                    method_scenario(label), label,
                ))
        self.valid = [
            methodology_row(f"{label}-v", f"validation-{label}", "valid",
                            method_scenario(label) + " independent example", label)
            for label in METHOD_LABELS
        ]

    def test_retrieves_matching_label(self):
        predict = make_baseline(self.train, self.valid)
        for label in METHOD_LABELS:
            query = methodology_row("q", "qfam", "valid", method_scenario(label), label)
            self.assertEqual(predict(query), label)

    def test_k_and_vocab_metadata(self):
        predict = make_baseline(self.train, self.valid)
        settings = predict.settings
        self.assertEqual(settings.track, "methodology")
        self.assertEqual(settings.candidate_k, METHODOLOGY_K_CANDIDATES)
        self.assertIn(settings.selected_k, METHODOLOGY_K_CANDIDATES)
        self.assertGreater(settings.vocabulary_size, 0)
        self.assertEqual(set(settings.validation_accuracy), set(METHODOLOGY_K_CANDIDATES))
        self.assertEqual(settings.tie_order, tuple(METHODS))

    def test_no_overlap_falls_back_to_frequent_label(self):
        predict = make_baseline(self.train, self.valid)
        # Scenario text with only unseen tokens -> no cosine overlap.
        query = methodology_row("q", "qfam", "valid", "zzz qqq xyzzy", METHOD_LABELS[0])
        result = predict(query)
        self.assertIn(result, METHODS)

    def test_vocabulary_is_training_only(self):
        predict = make_baseline(self.train, self.valid)
        baseline_vocab = predict.settings.vocabulary_size
        # Adding an exotic token to a *validation* row must not grow the vocab.
        noisy_valid = list(self.valid)
        noisy_valid[0] = methodology_row(
            noisy_valid[0]["case_id"], noisy_valid[0]["family"], "valid",
            method_scenario(METHOD_LABELS[0]) + " zzzxyzzyqqq", METHOD_LABELS[0],
        )
        again = make_baseline(self.train, noisy_valid)
        self.assertEqual(again.settings.vocabulary_size, baseline_vocab)

    def test_deterministic_repeated_fit(self):
        a = make_baseline(self.train, self.valid)
        b = make_baseline(self.train, self.valid)
        query = methodology_row("q", "qfam", "valid", method_scenario(METHOD_LABELS[2]),
                                METHOD_LABELS[2])
        self.assertEqual(a(query), b(query))
        self.assertEqual(a.settings.selected_k, b.settings.selected_k)


class SplitHygieneTests(unittest.TestCase):
    def training_pair(self):
        train = [clip_case(f"c{i}", "train") for i in range(3)]
        valid = [lr_case("lv", "valid")]
        return train, valid

    def test_rejects_prompt_overlap(self):
        train, _ = self.training_pair()
        overlap = clip_case("c0", "valid")
        overlap["prompt"] = train[0]["prompt"]  # identical prompt crosses splits
        with self.assertRaisesRegex(ValueError, "both train and validation"):
            make_baseline(train, [overlap])

    def test_rejects_mislabelled_training_split(self):
        train, valid = self.training_pair()
        train[0]["split"] = "valid"
        with self.assertRaisesRegex(ValueError, "non-train split"):
            make_baseline(train, valid)

    def test_rejects_mislabelled_validation_split(self):
        train, valid = self.training_pair()
        valid[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "non-valid split"):
            make_baseline(train, valid)

    def test_rejects_mixed_tracks(self):
        train = [clip_case("c0", "train")]
        valid = [methodology_row("m", "fam", "valid", method_scenario("audit_timing"),
                                 "audit_timing")]
        with self.assertRaisesRegex(ValueError, "different tracks"):
            make_baseline(train, valid)

    def test_rejects_empty_collections(self):
        train, valid = self.training_pair()
        with self.assertRaises(ValueError):
            make_baseline([], valid)
        with self.assertRaises(ValueError):
            make_baseline(train, [])


if __name__ == "__main__":
    unittest.main()
