"""Fit validation-selected TF-IDF logistic baselines for the frozen panels.

Only the prompt supplies features. Train labels fit the model and validation
labels select a fixed feature and regularization grid. The held-out reports
record case IDs, predictions and probabilities without copying source text.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from prepare import RECEIPT, verify

# Source-grouped tracks keyed by their JSONL filename stem prefix.
TRACKS = ("finqa", "sec8k", "treasury")
SPLITS = ("train", "valid", "test")
# Fixed, non-adaptive grid; identical for every track.
C_GRID = (0.25, 1.0, 4.0, 16.0)
# Two deterministic feature spaces; chosen per track on valid alongside C.
FEATURE_SPACES = ("word", "char_word")
SEED = 0


def load_split(cases_dir: Path, track: str, split: str) -> list[dict]:
    path = cases_dir / f"{track}-{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def label_space(rows: list[dict]) -> list[str]:
    # Option-ID space taken from the choices mapping, never from labels/ids.
    return sorted(rows[0]["choices"])


def build_pipeline(space: str, c_value: float) -> Pipeline:
    """Deterministic TF-IDF -> multinomial logistic regression."""
    word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), lowercase=True,
        min_df=2, sublinear_tf=True, strip_accents="unicode",
    )
    if space == "word":
        features = word
    else:
        char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), lowercase=True,
            min_df=2, sublinear_tf=True, strip_accents="unicode",
        )
        features = FeatureUnion([("word", word), ("char", char)])
    model = LogisticRegression(
        C=c_value, max_iter=2000, random_state=SEED,
        class_weight="balanced",
    )
    return Pipeline([("tfidf", features), ("clf", model)])


def select_and_fit(train: list[dict], valid: list[dict]) -> dict:
    """Grid-search C and feature space on valid accuracy only, then return the
    chosen train-only fit and the selection trace. The returned pipeline is the
    one fit on train during the search; it is not refit."""
    x_train = [r["prompt"] for r in train]
    y_train = [r["label"] for r in train]
    x_valid = [r["prompt"] for r in valid]
    y_valid = [r["label"] for r in valid]

    trace, best = [], None
    for space in FEATURE_SPACES:
        for c_value in C_GRID:
            pipe = build_pipeline(space, c_value)
            pipe.fit(x_train, y_train)
            acc = accuracy_score(y_valid, pipe.predict(x_valid))
            trace.append({"feature_space": space, "C": c_value, "valid_accuracy": round(acc, 6)})
            key = (acc, -FEATURE_SPACES.index(space), -C_GRID.index(c_value))
            if best is None or key > best[0]:
                best = (key, space, c_value, pipe)

    return {"space": best[1], "C": best[2], "pipeline": best[3], "search": trace}


def evaluate(pipe: Pipeline, rows: list[dict], labels: list[str]) -> dict:
    prompts = [r["prompt"] for r in rows]
    y_true = [r["label"] for r in rows]
    y_pred = list(pipe.predict(prompts))
    proba = pipe.predict_proba(prompts)
    order = list(pipe.classes_)
    predictions = []
    for row, pred, dist in zip(rows, y_pred, proba):
        predictions.append({
            "case_id": row["case_id"],
            "prediction": pred,
            "probabilities": {lab: round(float(dist[order.index(lab)]), 6) for lab in labels},
        })
    scores = {
        "accuracy": round(accuracy_score(y_true, y_pred), 6),
        "macro_f1": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 6),
    }
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {"predictions": predictions, "scores": scores,
            "confusion": {"labels": labels, "matrix": matrix}}


def run(cases_dir: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    digests = verify(cases_dir, json.loads(RECEIPT.read_text(encoding="utf-8")))
    receipt = {"baseline": "tfidf_logreg_lexical", "scikit_learn": sklearn_version, "sklearn_grid": {"C": list(C_GRID),
               "feature_spaces": list(FEATURE_SPACES)}, "tracks": {}}
    all_predictions: list[dict] = []

    for track in TRACKS:
        train = load_split(cases_dir, track, "train")
        valid = load_split(cases_dir, track, "valid")
        labels = label_space(train)

        chosen = select_and_fit(train, valid)
        valid_eval = evaluate(chosen["pipeline"], valid, labels)

        # Parse test labels only after choosing C and the feature space.
        test = load_split(cases_dir, track, "test")
        test_eval = evaluate(chosen["pipeline"], test, labels)
        for prediction in test_eval["predictions"]:
            all_predictions.append({"track": track, **prediction})

        receipt["tracks"][track] = {
            "track": train[0]["track"],
            "labels": labels,
            "chosen": {"C": chosen["C"], "feature_space": chosen["space"]},
            "selection_search": chosen["search"],
            "counts": {"train": len(train), "valid": len(valid), "test": len(test)},
            "digests": {split: digests[f"{track}-{split}.jsonl"] for split in SPLITS},
            "valid": {"scores": valid_eval["scores"], "confusion": valid_eval["confusion"]},
            "test": {"scores": test_eval["scores"], "confusion": test_eval["confusion"]},
        }

    predictions_path = output / "lexical-predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(p, sort_keys=True) + "\n" for p in all_predictions), encoding="utf-8")
    receipt["predictions_file"] = predictions_path.name
    receipt["predictions_sha256"] = sha256(predictions_path.read_bytes()).hexdigest()

    receipt_path = output / "lexical-receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    cli.add_argument("--cases-dir", type=Path, required=True,
                     help="directory with <track>-{train,valid,test}.jsonl")
    cli.add_argument("--output", type=Path, required=True,
                     help="directory for predictions JSONL and JSON receipt")
    args = cli.parse_args()
    summary = run(args.cases_dir, args.output)
    print(json.dumps({t: summary["tracks"][t]["test"]["scores"] for t in summary["tracks"]}, indent=2))
