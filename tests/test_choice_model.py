"""Focused tests for the constrained QLoRA backend."""

from __future__ import annotations

import math

import pytest

from gradient_desk.choice_model import (
    CANDIDATE_LETTERS,
    MAX_CHOICES,
    MIN_CHOICES,
)

try:  # MLX-dependent tests degrade to skips; pure-Python tests always run.
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from gradient_desk.choice_model import (
        candidate_logits,
        candidate_probabilities,
        masked_soft_cross_entropy,
    )

    HAVE_MLX = True
except ImportError:  # pragma: no cover - exercised only without the backend
    HAVE_MLX = False

requires_mlx = pytest.mark.skipif(not HAVE_MLX, reason="MLX backend not installed")


# --------------------------------------------------------------------------- #
# Pure-Python helpers
# --------------------------------------------------------------------------- #
def test_candidate_letters_are_the_a_through_h_contract():
    assert CANDIDATE_LETTERS == ("A", "B", "C", "D", "E", "F", "G", "H")
    assert (MIN_CHOICES, MAX_CHOICES) == (2, 8)


def soft_cross_entropy(logits, targets):
    peak = max(logits)
    normalizer = math.log(math.fsum(math.exp(value - peak) for value in logits))
    return -math.fsum(target * (logit - peak - normalizer)
                      for logit, target in zip(logits, targets, strict=True))


# --------------------------------------------------------------------------- #
# Tiny synthetic MLX model mirroring the Qwen surface used by the backend
# --------------------------------------------------------------------------- #
def _build_tiny_model_classes():
    """Define the synthetic model classes lazily so import works without MLX."""

    class _Body(nn.Module):
        """Causal body: hidden[b, t] depends only on tokens <= t (running mean).

        A genuine causal dependency means the last *real* position's hidden
        state is invariant to right padding, which is exactly the property the
        backend relies on.  No attention needed; a cumulative mean over
        per-token embeddings is strictly causal.
        """

        def __init__(self, vocab, hidden):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab, hidden)

        def __call__(self, inputs):
            emb = self.embed_tokens(inputs)  # [B, T, H]
            cumsum = mx.cumsum(emb, axis=1)
            counts = mx.arange(1, inputs.shape[1] + 1, dtype=emb.dtype).reshape(1, -1, 1)
            return cumsum / counts

    class _TinyModel(nn.Module):
        def __init__(self, vocab, hidden):
            super().__init__()
            self.args = type("Args", (), {"tie_word_embeddings": False})()
            self.model = _Body(vocab, hidden)
            self.lm_head = nn.Linear(hidden, vocab, bias=False)

        def __call__(self, inputs):
            return self.lm_head(self.model(inputs))

    return _TinyModel


def _deterministic_tiny_model(vocab=11, hidden=6):
    mx.random.seed(1234)
    model = _build_tiny_model_classes()(vocab, hidden)
    mx.eval(model.parameters())
    return model


@requires_mlx
def test_right_padding_matches_unpadded_last_position():
    """Padded-batch candidate logits equal per-row unpadded logits."""
    model = _deterministic_tiny_model()
    candidate_ids = mx.array([[1, 2, 3], [1, 2, 3]], dtype=mx.int32)
    candidate_mask = mx.ones((2, 3), dtype=mx.float32)

    rows = [[4, 5, 6, 7], [8, 9]]  # different real lengths
    pad = 0
    width = max(len(r) for r in rows)
    padded = mx.array([r + [pad] * (width - len(r)) for r in rows], dtype=mx.int32)
    lengths = mx.array([len(r) for r in rows], dtype=mx.int32)
    batched = candidate_logits(model, padded, lengths, candidate_ids)
    mx.eval(batched)

    for i, row in enumerate(rows):
        single_ids = mx.array([row], dtype=mx.int32)
        single_len = mx.array([len(row)], dtype=mx.int32)
        single = candidate_logits(
            model, single_ids, single_len,
            mx.array([[1, 2, 3]], dtype=mx.int32),
        )
        mx.eval(single)
        assert mx.allclose(batched[i], single[0], atol=1e-5).item()


@requires_mlx
def test_masked_soft_cross_entropy_matches_pure_python():
    """The differentiable loss equals the reference formula on real slots."""
    logits = mx.array([[1.5, -0.5, 2.0, 9.9]])  # last slot is padding
    mask = mx.array([[1.0, 1.0, 1.0, 0.0]])
    targets = mx.array([[0.5, 0.2, 0.3, 0.0]])
    loss = masked_soft_cross_entropy(logits, targets, mask)
    mx.eval(loss)
    expected = soft_cross_entropy([1.5, -0.5, 2.0], [0.5, 0.2, 0.3])
    assert float(loss.item()) == pytest.approx(expected, abs=1e-5)


@requires_mlx
def test_masked_probabilities_ignore_padded_slots():
    logits = mx.array([[0.1, 0.2, 50.0]])  # third slot huge but masked off
    mask = mx.array([[1.0, 1.0, 0.0]])
    probs = candidate_probabilities(logits, mask)
    mx.eval(probs)
    assert float(probs[0, 2].item()) == pytest.approx(0.0, abs=1e-6)
    assert float(probs[0, 0].item() + probs[0, 1].item()) == pytest.approx(1.0, abs=1e-6)


@requires_mlx
def test_loss_has_finite_value_and_gradients():
    """Gradients through the candidate loss are finite and non-trivial."""
    model = _deterministic_tiny_model()
    batch_ids = mx.array([[4, 5, 6, 0], [8, 9, 0, 0]], dtype=mx.int32)
    lengths = mx.array([3, 2], dtype=mx.int32)
    candidate_ids = mx.array([[1, 2, 3], [1, 2, 3]], dtype=mx.int32)
    mask = mx.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], dtype=mx.float32)
    targets = mx.array([[0.6, 0.3, 0.1], [0.7, 0.3, 0.0]], dtype=mx.float32)

    def loss_fn(model):
        logits = candidate_logits(model, batch_ids, lengths, candidate_ids)
        return masked_soft_cross_entropy(logits, targets, mask)

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(loss, grads)
    assert math.isfinite(float(loss.item()))
    flat = [g for _, g in tree_flatten(grads)]
    assert flat, "expected trainable gradients"
    assert all(bool(mx.all(mx.isfinite(g)).item()) for g in flat)
    assert any(float(mx.max(mx.abs(g)).item()) > 0 for g in flat)
