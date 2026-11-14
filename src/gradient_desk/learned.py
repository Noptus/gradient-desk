"""Transparent, data-driven CPU baselines for the public benchmark.

Two tracks, one interface. ``make_baseline(training_rows, validation_rows)``
returns a deterministic ``Callable[[dict], str]`` predictor that also carries
read-only interface metadata on ``.settings``.

Training track
    Features come only from the pre-action JSON telemetry embedded in the
    prompt (the object after ``"\\nTelemetry: "``). We build a compact,
    scale-normalized signature (gradient-norm level/trend, train and
    validation trends, learning rate and its log, objective and batch size).
    We fit nothing beyond storing train-only signatures and their per-action
    measured validation losses, normalized within each case to ``[0, 1]`` so
    that incomparable loss scales do not dominate. At inference we take the k
    nearest training cases and pick the action with the lowest mean normalized
    measured loss; ties resolve to the menu order (resume first). The single
    hyperparameter k is chosen from a predeclared candidate list by minimizing
    normalized regret on the validation split.

Methodology track
    A retrieval baseline over the scenario text after the ``"Case: "`` marker.
    Word/token TF-IDF vectors are built from the training vocabulary only; a
    k-nearest weighted vote over cosine similarity chooses the label. k is
    chosen from predeclared candidates by maximizing validation accuracy.

Guarantees
    * Standard library only.
    * Features, vocabulary, IDF, and candidate labels are learned from the
      training rows exclusively; validation rows are used only to *select* k.
    * Case identifiers are never read, and no label parsing of IDs occurs.
    * Overlapping or mis-split train/validation inputs are rejected up front:
      a row may not appear in both, and the two collections must not carry
      conflicting split tags.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import json
import math
import re

from .dataset import METHODS
from .traces import ACTIONS

# Predeclared candidate neighbourhood sizes (selected on validation).
TRAINING_K_CANDIDATES = (1, 3, 5, 7, 9)
METHODOLOGY_K_CANDIDATES = (1, 3, 5, 7)

TELEMETRY_MARKER = "\nTelemetry: "
CASE_MARKER = "Case: "
_TOKEN = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------- #
# Shared split hygiene
# --------------------------------------------------------------------------- #
def _prompt_digest(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _reject_bad_splits(training_rows: list[dict], validation_rows: list[dict]) -> None:
    """Reject overlapping rows or conflicting split tags between collections."""
    if not training_rows:
        raise ValueError("Expected at least one training row")
    if not validation_rows:
        raise ValueError("Expected at least one validation row")

    train_digests = {}
    for row in training_rows:
        digest = _prompt_digest(row["prompt"])
        train_digests[digest] = row.get("split")
        if row.get("split") not in (None, "train"):
            raise ValueError(f"Training row carries non-train split: {row.get('split')!r}")

    for row in validation_rows:
        digest = _prompt_digest(row["prompt"])
        if digest in train_digests:
            raise ValueError("A prompt appears in both train and validation collections")
        if row.get("split") not in (None, "valid"):
            raise ValueError(f"Validation row carries non-valid split: {row.get('split')!r}")


def _track_of(rows: list[dict]) -> str:
    tracks = {row["track"] for row in rows}
    if len(tracks) != 1:
        raise ValueError(f"Expected exactly one track, saw {sorted(tracks)}")
    track = next(iter(tracks))
    if track not in ("training", "methodology"):
        raise ValueError(f"Unknown track: {track!r}")
    return track


# --------------------------------------------------------------------------- #
# Training track: telemetry signature + nearest-neighbour measured loss
# --------------------------------------------------------------------------- #
def _telemetry(prompt: str) -> dict:
    parts = prompt.split(TELEMETRY_MARKER, maxsplit=1)
    if len(parts) != 2:
        raise ValueError("Training prompt lacks pre-action telemetry")
    return json.loads(parts[1])


def _trend(series: list[float]) -> float:
    """Scale-normalized last-minus-first slope of a short history."""
    values = [float(v) for v in series if v is not None]
    if len(values) < 2:
        return 0.0
    scale = max(abs(values[0]), abs(values[-1]), 1e-9)
    return (values[-1] - values[0]) / scale


def _val_losses(info: dict) -> list[float]:
    return [float(point["loss"]) for point in info.get("validation_loss_history", [])]


def training_signature(prompt: str) -> tuple[float, ...]:
    """Compact scale-normalized feature vector from pre-action telemetry only."""
    info = _telemetry(prompt)
    norms = [float(v) for v in info.get("gradient_norm_history", [])]
    recent = norms[-4:] if norms else [0.0]
    grad_level = math.log1p(max(recent))
    grad_spread = (max(recent) - min(recent)) / max(max(recent), 1e-9)
    grad_trend = _trend(norms)

    train_trend = _trend([float(v) for v in info.get("train_loss_history", [])])
    val_trend = _trend(_val_losses(info))

    lr = float(info.get("lr", 0.0))
    objective = info.get("objective", "")
    is_classification = 1.0 if objective == "binary_cross_entropy" else 0.0
    is_ranking = 1.0 if objective == "pairwise_rank_cross_entropy" else 0.0

    batch = float(info.get("microbatch_size", info.get("batch_size", 0)) or 0)
    log_batch = math.log1p(batch)

    return (
        grad_level,
        grad_spread,
        grad_trend,
        train_trend,
        val_trend,
        lr,
        math.log(max(lr, 1e-12)),
        is_classification,
        is_ranking,
        log_batch,
    )


def _normalized_action_losses(row: dict) -> dict[str, float]:
    """Per-case measured validation losses normalized to [0, 1] across actions."""
    losses = {action: float(row["actions"][action]["validation_loss"]) for action in ACTIONS}
    best, worst = min(losses.values()), max(losses.values())
    span = worst - best
    if span <= 0:
        return {action: 0.0 for action in ACTIONS}
    return {action: (losses[action] - best) / span for action in losses}


def _standardize(vectors: list[tuple[float, ...]]):
    """Return per-feature mean and std (std floored) for z-scoring signatures."""
    width = len(vectors[0])
    means = [math.fsum(v[i] for v in vectors) / len(vectors) for i in range(width)]
    stds = []
    for i in range(width):
        var = math.fsum((v[i] - means[i]) ** 2 for v in vectors) / len(vectors)
        stds.append(math.sqrt(var) or 1.0)
    return means, stds


def _zscore(vector, means, stds):
    return tuple((vector[i] - means[i]) / stds[i] for i in range(len(vector)))


def _sq_distance(a, b) -> float:
    return math.fsum((x - y) ** 2 for x, y in zip(a, b, strict=True))


@dataclass(frozen=True)
class _TrainingModel:
    track: str
    selected_k: int
    candidate_k: tuple
    feature_names: tuple
    n_train: int
    n_validation: int
    tie_order: tuple
    validation_regret: dict = field(default_factory=dict)


def _fit_training(training_rows, validation_rows):
    signatures = [training_signature(row["prompt"]) for row in training_rows]
    means, stds = _standardize(signatures)
    train_points = [_zscore(sig, means, stds) for sig in signatures]
    train_losses = [_normalized_action_losses(row) for row in training_rows]

    def predict_with_k(point, k):
        order = sorted(range(len(train_points)),
                       key=lambda i: (_sq_distance(train_points[i], point), i))[:k]
        scores = {
            action: math.fsum(train_losses[i][action] for i in order) / len(order)
            for action in ACTIONS
        }
        best = min(scores.values())
        return next(a for a in ACTIONS if scores[a] <= best)  # resume-first ties

    # Select k on validation normalized regret (lower is better).
    val_points = [_zscore(training_signature(r["prompt"]), means, stds) for r in validation_rows]
    val_norm = [_normalized_action_losses(r) for r in validation_rows]
    regret_by_k = {}
    for k in TRAINING_K_CANDIDATES:
        kk = min(k, len(train_points))
        regret = math.fsum(val_norm[j][predict_with_k(val_points[j], kk)]
                           for j in range(len(val_points))) / len(val_points)
        regret_by_k[k] = regret
    selected_k = min(TRAINING_K_CANDIDATES, key=lambda k: (regret_by_k[k], k))
    effective_k = min(selected_k, len(train_points))

    settings = _TrainingModel(
        track="training",
        selected_k=selected_k,
        candidate_k=TRAINING_K_CANDIDATES,
        feature_names=(
            "grad_level", "grad_spread", "grad_trend", "train_trend", "val_trend",
            "lr", "log_lr", "is_classification", "is_ranking", "log_batch",
        ),
        n_train=len(train_points),
        n_validation=len(val_points),
        tie_order=ACTIONS,
        validation_regret=dict(regret_by_k),
    )

    def predict(row: dict) -> str:
        point = _zscore(training_signature(row["prompt"]), means, stds)
        return predict_with_k(point, effective_k)

    predict.settings = settings
    return predict


# --------------------------------------------------------------------------- #
# Methodology track: TF-IDF retrieval over scenario text
# --------------------------------------------------------------------------- #
def _scenario_text(prompt: str) -> str:
    idx = prompt.find(CASE_MARKER)
    if idx < 0:
        raise ValueError("Methodology prompt lacks a 'Case:' marker")
    return prompt[idx + len(CASE_MARKER):]


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _term_counts(text: str) -> Counter:
    return Counter(_tokenize(text))


def _tfidf_vector(counts: Counter, vocab, idf) -> dict:
    vector = {}
    total = sum(counts.values()) or 1
    for term, count in counts.items():
        if term in vocab:
            vector[term] = (count / total) * idf[term]
    norm = math.sqrt(math.fsum(w * w for w in vector.values())) or 1.0
    return {term: w / norm for term, w in vector.items()}


def _cosine(a: dict, b: dict) -> float:
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    return math.fsum(weight * large.get(term, 0.0) for term, weight in small.items())


@dataclass(frozen=True)
class _MethodologyModel:
    track: str
    selected_k: int
    candidate_k: tuple
    vocabulary_size: int
    labels: tuple
    n_train: int
    n_validation: int
    tie_order: tuple
    validation_accuracy: dict = field(default_factory=dict)


def _fit_methodology(training_rows, validation_rows):
    # Vocabulary and IDF come from training documents only.
    docs = [_term_counts(_scenario_text(row["prompt"])) for row in training_rows]
    document_frequency = Counter()
    for counts in docs:
        document_frequency.update(counts.keys())
    vocab = set(document_frequency)
    n_docs = len(docs)
    idf = {term: math.log((1 + n_docs) / (1 + df)) + 1.0 for term, df in document_frequency.items()}

    train_vectors = [_tfidf_vector(counts, vocab, idf) for counts in docs]
    train_labels = [row["label"] for row in training_rows]
    label_order = tuple(METHODS)  # deterministic menu order for tie-breaking

    def predict_with_k(vector, k):
        scored = sorted(
            range(len(train_vectors)),
            key=lambda i: (-_cosine(vector, train_vectors[i]), i),
        )[:k]
        votes = Counter()
        for i in scored:
            votes[train_labels[i]] += max(_cosine(vector, train_vectors[i]), 0.0)
        if not votes or all(weight == 0 for weight in votes.values()):
            # No overlap: fall back to the most frequent training label.
            votes = Counter(train_labels)
        top = max(votes.values())
        tied = {label for label, weight in votes.items() if weight == top}
        return next(label for label in label_order if label in tied)

    val_vectors = [_tfidf_vector(_term_counts(_scenario_text(r["prompt"])), vocab, idf)
                   for r in validation_rows]
    val_labels = [r["label"] for r in validation_rows]
    accuracy_by_k = {}
    for k in METHODOLOGY_K_CANDIDATES:
        kk = min(k, len(train_vectors))
        correct = sum(predict_with_k(val_vectors[j], kk) == val_labels[j]
                      for j in range(len(val_vectors)))
        accuracy_by_k[k] = correct / len(val_vectors)
    selected_k = max(METHODOLOGY_K_CANDIDATES, key=lambda k: (accuracy_by_k[k], -k))
    effective_k = min(selected_k, len(train_vectors))

    settings = _MethodologyModel(
        track="methodology",
        selected_k=selected_k,
        candidate_k=METHODOLOGY_K_CANDIDATES,
        vocabulary_size=len(vocab),
        labels=tuple(sorted(set(train_labels))),
        n_train=len(train_vectors),
        n_validation=len(val_vectors),
        tie_order=label_order,
        validation_accuracy=dict(accuracy_by_k),
    )

    def predict(row: dict) -> str:
        vector = _tfidf_vector(_term_counts(_scenario_text(row["prompt"])), vocab, idf)
        return predict_with_k(vector, effective_k)

    predict.settings = settings
    return predict


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #
def make_baseline(
    training_rows: list[dict],
    validation_rows: list[dict],
) -> Callable[[dict], str]:
    """Fit a transparent baseline and return a deterministic predictor.

    Both collections must belong to the same track ("training" or
    "methodology"). The returned callable maps one case dict to a single
    action label and exposes selection metadata on ``.settings``.
    """
    _reject_bad_splits(training_rows, validation_rows)
    track = _track_of(training_rows)
    if _track_of(validation_rows) != track:
        raise ValueError("Train and validation rows belong to different tracks")
    if track == "training":
        return _fit_training(training_rows, validation_rows)
    return _fit_methodology(training_rows, validation_rows)


def evaluate_learned(
    train_path,
    valid_path,
    case_path,
    output_path,
) -> dict:
    """Run the validation-selected baselines on a disjoint test panel."""
    from dataclasses import asdict
    from pathlib import Path
    import time

    from .dataset import read_jsonl
    from .evaluate import Prediction, evaluate

    training_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(valid_path)
    heldout_rows = read_jsonl(case_path)
    tagged = {"train": training_rows, "valid": validation_rows, "test": heldout_rows}
    earlier = set()
    families = set()
    for split, rows in tagged.items():
        if not rows or {row["split"] for row in rows} != {split}:
            raise ValueError(f"The {split} file must contain only {split} cases")
        current = {row["prompt"] for row in rows}
        group = {(row["track"], row["family"]) for row in rows}
        if earlier.intersection(current) or families.intersection(group):
            raise ValueError(f"Rows or workload families overlap in the {split} file")
        earlier.update(current)
        families.update(group)
    available = sorted({row["track"] for row in heldout_rows})
    predictors = {
        track: make_baseline(
            [row for row in training_rows if row["track"] == track],
            [row for row in validation_rows if row["track"] == track],
        )
        for track in available
    }

    def predictor(_base, _adapter, rows):
        for row in rows:
            started = time.perf_counter()
            label = predictors[row["track"]](row)
            yield Prediction(label, label, time.perf_counter() - started)

    report = evaluate(case_path, output_path, None, None, None, predictor, custom_engine="learned")
    report["validation_selection"] = {track: asdict(policy.settings) for track, policy in predictors.items()}
    path = Path(output_path).with_suffix(".metrics.json")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
