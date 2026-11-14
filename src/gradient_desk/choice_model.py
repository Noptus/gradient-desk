"""Train and score a Qwen3 QLoRA adapter for a constrained answer menu.

Cross-entropy uses the legal option-token logits and either a hard label or a
soft target. The language-model head runs only at each row's last real token.
Right padding follows that token and cannot change its causal hidden state.
Scoring averages canonical probabilities across deterministic answer layouts.
Saved tensors use the standard MLX LoRA format.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

from .choice_data import CODES, layout, target_distribution, validate
from .dataset import MODEL_REPO, MODEL_REVISION, read_jsonl
from .train import EXPECTED_WEIGHTS_SHA256

# Single-character option slots.  Each must encode to exactly one token id under
# the Qwen tokenizer (verified by ``candidate_token_ids``).
CANDIDATE_LETTERS = tuple(CODES)
MIN_CHOICES = 2
MAX_CHOICES = len(CANDIDATE_LETTERS)
TEMPERATURE = 1.0

# Default LoRA shape mirrors the v1 pilot defaults so packaged adapters remain
# comparable.  All are overridable from the CLI.
DEFAULT_RANK = 8
DEFAULT_SCALE = 20.0
DEFAULT_DROPOUT = 0.0
DEFAULT_NUM_LAYERS = 8


def candidate_token_ids(tokenizer, count: int) -> list[int]:
    """Return the single-token ids for the first ``count`` option letters.

    Verifies that each letter encodes to exactly one token id with special
    tokens disabled, as required for constrained single-token scoring.
    """
    if not MIN_CHOICES <= count <= MAX_CHOICES:
        raise ValueError(f"Unsupported option count: {count}")
    ids = []
    for letter in CANDIDATE_LETTERS[:count]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"Option letter {letter!r} does not encode to a single token "
                f"(got {encoded!r}); this tokenizer cannot drive constrained scoring"
            )
        ids.append(int(encoded[0]))
    if len(set(ids)) != len(ids):
        raise ValueError("Option letters collide onto the same token id")
    return ids


def _encode_prompt(tokenizer, row: dict, variant: int):
    """Build the chat tokens and choice ordering without consulting labels."""
    text, order = layout(row, variant=variant)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return list(prompt_ids), order


def _lm_head(model, hidden):
    """Apply the (possibly LoRA-wrapped) language-model head to hidden states.

    Mirrors ``mlx_lm.models.qwen3.Model.__call__``: an untied model exposes
    ``lm_head``; a tied model reuses ``model.model.embed_tokens.as_linear``.
    """
    if getattr(model.args, "tie_word_embeddings", False):
        return model.model.embed_tokens.as_linear(hidden)
    return model.lm_head(hidden)


def candidate_logits(model, batch_ids, lengths, candidate_matrix):
    """Return masked candidate logits at each row's last real position.

    Parameters
    ----------
    batch_ids : int32 ``[B, T]`` right-padded token ids.
    lengths   : int32 ``[B]`` real (unpadded) prompt length per row.
    candidate_matrix : int32 ``[B, C]`` option token ids; padded option slots
        repeat a valid id and are masked out via ``candidate_mask``.
    """
    import mlx.core as mx

    # Transformer body only -> hidden states [B, T, H]; never the full vocab.
    hidden = model.model(batch_ids)
    last = (lengths - 1).reshape(-1, 1, 1)
    last = mx.broadcast_to(last, (hidden.shape[0], 1, hidden.shape[-1]))
    final_hidden = mx.take_along_axis(hidden, last, axis=1)  # [B, 1, H]
    full = _lm_head(model, final_hidden)  # [B, 1, V]
    full = full.reshape(full.shape[0], full.shape[-1])  # [B, V]
    # Gather only the option token ids per row -> [B, C].
    logits = mx.take_along_axis(full, candidate_matrix, axis=1)
    return logits.astype(mx.float32) / TEMPERATURE


def masked_soft_cross_entropy(logits, targets, candidate_mask):
    """Differentiable soft cross-entropy with explicit padded-candidate masking.

    ``candidate_mask`` is 1.0 for real option slots and 0.0 for padding.  Padded
    logits are pushed to ``-inf`` before a numerically stable log-softmax so they
    receive zero probability; padded targets are zero and contribute nothing.
    Returns the mean loss over the batch.
    """
    import mlx.core as mx

    neg_inf = mx.array(-1e9, dtype=logits.dtype)
    masked_logits = mx.where(candidate_mask > 0, logits, neg_inf)
    logp = masked_logits - mx.logsumexp(masked_logits, axis=-1, keepdims=True)
    per_row = -mx.sum(targets * candidate_mask * logp, axis=-1)
    return mx.mean(per_row)


def candidate_probabilities(logits, candidate_mask):
    """Softmax over masked candidate logits -> probabilities ``[B, C]``."""
    import mlx.core as mx

    neg_inf = mx.array(-1e9, dtype=logits.dtype)
    masked_logits = mx.where(candidate_mask > 0, logits, neg_inf)
    logp = masked_logits - mx.logsumexp(masked_logits, axis=-1, keepdims=True)
    return mx.exp(logp)


def _pad_batch(rows_tokens, pad_id):
    """Right-pad a list of token-id lists; return (ids[B,T], lengths[B])."""
    import mlx.core as mx

    lengths = [len(t) for t in rows_tokens]
    width = max(lengths)
    padded = [t + [pad_id] * (width - len(t)) for t in rows_tokens]
    return mx.array(padded, dtype=mx.int32), mx.array(lengths, dtype=mx.int32)


def _pad_candidates(orders, candidate_ids_by_count):
    """Build candidate id matrix and mask for a batch of option orders.

    ``candidate_ids_by_count[n]`` gives the ``n`` single-token option ids.  Each
    row uses the first ``len(order)`` of those; shorter rows pad with the first
    id and mask it off.
    """
    import mlx.core as mx

    width = max(len(o) for o in orders)
    ids, mask = [], []
    for order in orders:
        count = len(order)
        row_ids = list(candidate_ids_by_count[count])
        pad = width - count
        ids.append(row_ids + [row_ids[0]] * pad)
        mask.append([1.0] * count + [0.0] * pad)
    return mx.array(ids, dtype=mx.int32), mx.array(mask, dtype=mx.float32)


def _load(model_path: Path, adapter: Path | None):
    try:
        from mlx_lm import load
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("Install the 'mlx' extra for Apple Silicon inference") from exc
    return load(
        str(model_path.resolve()),
        adapter_path=str(adapter.resolve()) if adapter else None,
        tokenizer_config={"trust_remote_code": False},
    )


def _candidate_table(tokenizer):
    """Precompute option-token ids for every supported choice count."""
    return {n: candidate_token_ids(tokenizer, n) for n in range(MIN_CHOICES, MAX_CHOICES + 1)}


def _pad_id(tokenizer):
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    if pad is None:  # pragma: no cover - every Qwen tokenizer defines eos
        raise ValueError("Tokenizer exposes neither a pad nor an eos token id")
    return int(pad)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _balanced_batches(rows, batch_size, seed):
    """Round-robin tracks; draw labels using inverse-square-root row weighting."""
    strata = defaultdict(lambda: defaultdict(list))
    for row in rows:
        strata[row["track"]][row["label"]].append(row)
    rng = random.Random(seed)
    tracks = sorted(strata)
    while True:
        rng.shuffle(tracks)
        for track in tracks:
            buckets = strata[track]
            names = sorted(buckets)
            weights = [math.sqrt(len(buckets[name])) for name in names]
            labels = rng.choices(names, weights=weights, k=batch_size)
            yield [rng.choice(buckets[name]) for name in labels]


def _prepare_batch(tokenizer, batch, table, pad_id, step=0, max_tokens=768):
    import mlx.core as mx
    token_lists, orders, target_lists = [], [], []
    for offset, row in enumerate(batch):
        prompt_ids, order = _encode_prompt(tokenizer, row, variant=step + offset)
        if not prompt_ids or len(prompt_ids) > max_tokens:
            raise ValueError(f"Training prompt {row['case_id']} has {len(prompt_ids)} tokens; limit is {max_tokens}")
        token_lists.append(prompt_ids)
        orders.append(order)
        target_lists.append(target_distribution(row, order))
    batch_ids, lengths = _pad_batch(token_lists, pad_id)
    candidate_matrix, candidate_mask = _pad_candidates(orders, table)
    width = candidate_matrix.shape[1]
    targets = mx.array(
        [target + [0.0] * (width - len(target)) for target in target_lists], dtype=mx.float32
    )
    return batch_ids, lengths, candidate_matrix, candidate_mask, targets


def fit(
    train: Path,
    valid: Path | None,
    model_path: Path,
    output: Path,
    *,
    steps: int,
    save_every: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    num_layers: int = DEFAULT_NUM_LAYERS,
    rank: int = DEFAULT_RANK,
    scale: float = DEFAULT_SCALE,
    dropout: float = DEFAULT_DROPOUT,
    resume: Path | None = None,
    max_tokens: int = 768,
) -> dict:
    """Fine-tune constrained candidate soft targets and write an MLX adapter.

    Only the final-state candidate logits enter the loss, so each step runs one
    short right-padded forward/backward over the prompt tokens.  Adapter weights
    and ``adapter_config.json`` are written in the standard MLX LoRA layout.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import linear_to_lora_layers

    if steps <= 0 or batch_size <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("steps, batch-size and learning-rate must be positive")
    if output.exists():
        raise FileExistsError(f"Adapter output already exists: {output}")

    train_rows = read_jsonl(train)
    if not train_rows:
        raise ValueError("No training rows supplied")
    if {r.get("split") for r in train_rows} - {"train"}:
        raise ValueError("fit trains on the train split only")
    valid_rows = read_jsonl(valid) if valid else []
    if not valid_rows or {r["split"] for r in valid_rows} != {"valid"}:
        raise ValueError("fit requires a separate validation split")
    validate(train_rows + valid_rows)
    base_digest = _file_sha256(model_path / "model.safetensors")
    if base_digest != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("Base checkpoint digest does not match the pinned Qwen snapshot")
    started = time.monotonic()
    print(f"Loading pinned {MODEL_REPO}@{MODEL_REVISION}", flush=True)
    mx.random.seed(seed)
    model, tokenizer = _load(model_path, None)
    table = _candidate_table(tokenizer)
    pad_id = _pad_id(tokenizer)
    lengths = {}
    for row in train_rows:
        # Token count variation by option position is negligible for these short
        # menus; each actual candidate batch still has a strict length guard.
        tokens, _ = _encode_prompt(tokenizer, row, variant=0)
        previous = lengths.get(row["track"], 0)
        lengths[row["track"]] = max(previous, len(tokens))
        if not tokens or len(tokens) > max_tokens:
            raise ValueError(f"Training prompt {row['case_id']} has {len(tokens)} tokens; limit is {max_tokens}")
    print(f"Train rows {len(train_rows)}; validation rows {len(valid_rows)}; max train tokens {lengths}", flush=True)

    lora_parameters = {"rank": rank, "scale": scale, "dropout": dropout}
    model.freeze()
    linear_to_lora_layers(model, num_layers, lora_parameters)
    if resume is not None:
        # Weights-only continuation: load previously trained LoRA tensors into
        # the freshly attached layers, then keep training the same shapes.
        model.load_weights(str((resume / "adapters.safetensors").resolve()), strict=False)

    def loss_fn(model, batch_ids, lengths, candidate_matrix, candidate_mask, targets):
        logits = candidate_logits(model, batch_ids, lengths, candidate_matrix)
        return masked_soft_cross_entropy(logits, targets, candidate_mask)

    optimizer = optim.Adam(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    output.mkdir(parents=True)
    adapter_file = output / "adapters.safetensors"
    config = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": lora_parameters,
        "target": "constrained_candidate_soft_cross_entropy",
        "temperature": TEMPERATURE,
        "sampler": "round-robin bench; square-root class counts",
    }
    (output / "adapter_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = {
        "base_repo": MODEL_REPO,
        "base_revision": MODEL_REVISION,
        "weights_sha256": base_digest,
        "config_sha256": _file_sha256(output / "adapter_config.json"),
        "dataset_sha256": {"train": _file_sha256(train), "valid": _file_sha256(valid)},
        "resume_adapter_sha256": _file_sha256(resume / "adapters.safetensors") if resume else None,
        "hyperparameters": {
            "steps": steps, "batch_size": batch_size, "learning_rate": learning_rate,
            "seed": seed, "num_layers": num_layers, "max_tokens": max_tokens, **lora_parameters,
        },
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    def save_checkpoint(step):
        weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(adapter_file), weights)
        mx.save_safetensors(str(output / f"{step:07d}_adapters.safetensors"), weights)

    receipts = []
    model.train()
    stream = _balanced_batches(train_rows, batch_size, seed)
    for step in range(1, steps + 1):
        batch = next(stream)
        tensors = _prepare_batch(tokenizer, batch, table, pad_id, step, max_tokens)
        loss, grads = loss_and_grad(model, *tensors)
        grads, norm = optim.clip_grad_norm(grads, max_norm=1.0)
        mx.eval(loss, norm)
        value, norm_value = float(loss.item()), float(norm.item())
        if not math.isfinite(value) or not math.isfinite(norm_value):
            raise RuntimeError(f"Nonfinite loss or gradient at step {step}")
        optimizer.update(model, grads)
        mx.eval(model.trainable_parameters(), optimizer.state)
        receipts.append({"step": step, "loss": value, "gradient_norm": norm_value,
                         "batch": len(batch), "track": batch[0]["track"]})
        if step % max(1, save_every) == 0 or step == steps:
            save_checkpoint(step)
        if step % 10 == 0 or step == steps or step == 1:
            window = receipts[-10:]
            mean_loss = math.fsum(row["loss"] for row in window) / len(window)
            print(f"step {step}/{steps} mean loss {mean_loss:.4f} "
                  f"peak {mx.get_peak_memory()/1e9:.2f} GB; elapsed {time.monotonic()-started:.1f}s", flush=True)

    report = {"steps": steps, "final_loss": receipts[-1]["loss"], "receipts": receipts,
              "peak_mlx_bytes": mx.get_peak_memory(), "elapsed_seconds": time.monotonic() - started}
    (output / "training-receipts.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# --------------------------------------------------------------------------- #
# Scoring (no labels, deterministic permutations)
# --------------------------------------------------------------------------- #
def score(
    cases: Path,
    model_path: Path,
    output: Path,
    *,
    adapter: Path | None = None,
    variants: int = 2,
    batch_size: int = 1,
) -> list[dict]:
    """Score cases over deterministic permutations and write probabilities.

    For each case we evaluate ``variants`` distinct deterministic option-order
    permutations, map each permutation's candidate probabilities back onto the
    option IDs, average per ID, and report the argmax ID and the full
    distribution.  Scoring never reads ``label`` from the rows.
    """
    if variants < 1:
        raise ValueError("variants must be at least 1")
    rows = read_jsonl(cases)
    if not rows:
        raise ValueError("No scoring cases supplied")
    for row in rows:
        layout(row)

    model, tokenizer = _load(model_path, adapter)
    results = list(_score_rows(model, tokenizer, rows, variants, batch_size))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in results:
            stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
    return results


def _score_rows(model, tokenizer, rows, variants, batch_size, *, progress=True):
    """Yield one scored record per row, averaging over permutation variants."""
    import mlx.core as mx

    table = _candidate_table(tokenizer)
    pad_id = _pad_id(tokenizer)

    # One (row, variant) work item per permutation; batch across items.
    work = []
    denominators = []
    for index, row in enumerate(rows):
        count = min(variants, len(row["choices"]))
        denominators.append(count)
        for variant in range(count):
            token_ids, order = _encode_prompt(tokenizer, row, variant)
            work.append((index, token_ids, order))
    work.sort(key=lambda item: len(item[1]))
    # ID -> accumulated probability mass, per row.
    accum = [{} for _ in rows]

    for start in range(0, len(work), batch_size):
        chunk = work[start:start + batch_size]
        token_lists, orders = [], []
        for _, prompt_ids, order in chunk:
            token_lists.append(prompt_ids)
            orders.append(order)
        batch_ids, lengths = _pad_batch(token_lists, pad_id)
        candidate_matrix, candidate_mask = _pad_candidates(orders, table)
        logits = candidate_logits(model, batch_ids, lengths, candidate_matrix)
        probs = candidate_probabilities(logits, candidate_mask)
        mx.eval(probs)
        for row_pos, (index, _, _) in enumerate(chunk):
            order = orders[row_pos]
            for slot, option_id in enumerate(order):
                accum[index][option_id] = accum[index].get(option_id, 0.0) + float(probs[row_pos, slot])

        if progress and ((start // batch_size) % 20 == 0 or start + batch_size >= len(work)):
            print(f"Scored {min(start + batch_size, len(work))}/{len(work)} permutations", flush=True)

    for index, row in enumerate(rows):
        totals = accum[index]
        averaged = {oid: mass / denominators[index] for oid, mass in totals.items()}
        norm = math.fsum(averaged.values()) or 1.0
        distribution = {oid: value / norm for oid, value in averaged.items()}
        prediction = max(distribution, key=lambda oid: (distribution[oid], oid))
        record = {
            "case_id": row.get("case_id"),
            "track": row.get("track"),
            "family": row.get("family"),
            "split": row.get("split"),
            "variants": denominators[index],
            "prediction": prediction,
            "probabilities": distribution,
        }
        if "label" in row:
            # Reported for downstream analysis only; never used while scoring.
            record["label"] = row["label"]
        yield record


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    fit_cmd = commands.add_parser("fit", help="Fine-tune constrained soft targets into a LoRA adapter")
    fit_cmd.add_argument("--train", type=Path, required=True)
    fit_cmd.add_argument("--valid", type=Path)
    fit_cmd.add_argument("--model", type=Path, required=True)
    fit_cmd.add_argument("--output", type=Path, required=True)
    fit_cmd.add_argument("--steps", type=_positive_int, required=True)
    fit_cmd.add_argument("--save-every", type=_positive_int, default=100)
    fit_cmd.add_argument("--batch-size", type=_positive_int, default=4)
    fit_cmd.add_argument("--max-tokens", type=_positive_int, default=768)
    fit_cmd.add_argument("--learning-rate", type=float, default=1e-5)
    fit_cmd.add_argument("--seed", type=int, default=0)
    fit_cmd.add_argument("--num-layers", type=_positive_int, default=DEFAULT_NUM_LAYERS)
    fit_cmd.add_argument("--rank", type=_positive_int, default=DEFAULT_RANK)
    fit_cmd.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    fit_cmd.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    fit_cmd.add_argument(
        "--resume", type=Path,
        help="Optional adapter dir to continue from (weights-only continuation)",
    )

    score_cmd = commands.add_parser("score", help="Score cases over deterministic permutations")
    score_cmd.add_argument("--cases", type=Path, required=True)
    score_cmd.add_argument("--model", type=Path, required=True)
    score_cmd.add_argument(
        "--adapter", type=Path,
        help="Optional adapter dir; omit to score the base model (the baseline)",
    )
    score_cmd.add_argument("--output", type=Path, required=True)
    score_cmd.add_argument("--variants", type=_positive_int, default=2)
    score_cmd.add_argument("--batch-size", type=_positive_int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "fit":
        report = fit(
            args.train, args.valid, args.model, args.output,
            steps=args.steps, save_every=args.save_every, batch_size=args.batch_size,
            learning_rate=args.learning_rate, seed=args.seed, num_layers=args.num_layers,
            rank=args.rank, scale=args.scale, dropout=args.dropout, resume=args.resume, max_tokens=args.max_tokens,
        )
        print(json.dumps({"steps": report["steps"], "final_loss": report["final_loss"]}, indent=2))
    else:
        results = score(
            args.cases, args.model, args.output,
            adapter=args.adapter, variants=args.variants, batch_size=args.batch_size,
        )
        print(json.dumps({"scored": len(results), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
