"""Prepare an operator-choice routing benchmark from FinQA single-step programs.

This standard-library command-line tool turns the FinQA question-answering
corpus (czyssrs/FinQA, MIT licence, pinned at commit
0f16e2867befa6840783e58be38c9efb9229d742) into a classification benchmark under
``track = "finqa_operator"``.  Each retained question is one whose original
annotated program is a *single* arithmetic step drawn from the four semantic
operators ``add``, ``subtract``, ``multiply`` and ``divide``.  The model is
shown the question together with a short exact slice of the supporting evidence
(the values of ``qa.gold_inds``) and asked which arithmetic operation answers
it.  The label is the operator of the original expert program.

Design constraints honoured here (see README-style notes in ``build``):

* No corpus text is copied to stdout or into the receipt; only the derived
  JSONL rows contain evidence, and the tool never prints that evidence.
* Annotated programs, intermediate steps and gold answers never appear in a
  prompt -- the prompt carries the question plus evidence sentences only.
* Rows whose evidence slice would exceed the character budget are skipped under
  a single, documented, deterministic length rule.
* Issuers (the ticker prefix of the FinQA ``id``) are allocated to folds by a
  stable SHA256 hash into 60/20/20 buckets, *independent of the label
  distribution*.  A row survives only when its hash-allocated fold matches the
  fold the original publisher assigned it to (publisher ``dev`` maps to our
  ``valid``).  This simultaneously blocks issuers across folds and respects the
  upstream train/dev/test repartition.
* Final folds are built by deterministic class-balanced sampling up to the
  capacities (320 / 100 / 140), preferring at most one question per source
  document while enough rows remain.

Only this file is used; it depends on the Python standard library only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

TRACK = "finqa_operator"
SCHEMA_VERSION = 1

SOURCE_REPO = "czyssrs/FinQA"
SOURCE_COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
SOURCE_LICENSE = "MIT"

# The four semantic operators the benchmark routes between.  The mapping is a
# constant presented verbatim as the row-level ``choices`` dict so that every
# row offers the identical option set regardless of its label.
CHOICES = {
    "add": "Add the two quantities together (a + b).",
    "subtract": "Subtract the second quantity from the first (a - b).",
    "multiply": "Multiply the two quantities (a * b).",
    "divide": "Divide the first quantity by the second (a / b).",
}
OPERATORS = tuple(CHOICES)

# Publisher split (file stem) -> our fold name.
SPLIT_FROM_FILE = {"train": "train", "dev": "valid", "test": "test"}
FOLDS = ("train", "valid", "test")

# Hash-allocation buckets, in hundredths, independent of labels.
ALLOCATION = (("train", 60), ("valid", 20), ("test", 20))

# Fold capacities for the final class-balanced sample.
CAPACITY = {"train": 320, "valid": 100, "test": 140}

# Deterministic evidence length rule: the assembled prompt must not exceed this
# many characters.  Rows whose prompt would be longer are skipped.  2600 is the
# documented ideal ceiling in the specification.
MAX_PROMPT_CHARS = 2600

# A minimal count below which the held-out test fold is considered inadequate.
MIN_ADEQUATE_TEST = 80

OUTPUT_NAMES = {
    "train": "finqa-train.jsonl",
    "valid": "finqa-valid.jsonl",
    "test": "finqa-test.jsonl",
}

# A single top-level arithmetic call: op(arg, arg) with no chaining / refs.
_SINGLE_STEP = re.compile(
    r"^(add|subtract|multiply|divide)\(\s*([^(),#]+?)\s*,\s*([^(),#]+?)\s*\)$"
)


# --------------------------------------------------------------------------- #
# Deterministic hashing helpers
# --------------------------------------------------------------------------- #

def _digest(*parts: str) -> str:
    """Stable SHA256 hex digest over newline-joined string parts."""
    payload = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def issuer_fold(issuer: str) -> str:
    """Allocate an issuer to a fold via a stable hash, independent of labels.

    The issuer string is hashed and reduced modulo 100; the resulting bucket is
    mapped onto the 60/20/20 allocation.  Because only the issuer identity feeds
    the hash, the allocation cannot depend on the label mix an issuer carries.
    """
    bucket = int(_digest("finqa-issuer-allocation", issuer), 16) % 100
    cursor = 0
    for fold, width in ALLOCATION:
        cursor += width
        if bucket < cursor:
            return fold
    return ALLOCATION[-1][0]  # pragma: no cover - widths sum to 100


def normalize_question(question: str) -> str:
    """Normalise a question for cross-fold duplicate detection."""
    return re.sub(r"\s+", " ", question.strip().lower())


# --------------------------------------------------------------------------- #
# Parsing the raw corpus
# --------------------------------------------------------------------------- #

def single_step_operator(program: str) -> str | None:
    """Return the operator if ``program`` is a single-step arithmetic call."""
    if program is None:
        return None
    match = _SINGLE_STEP.match(program.strip())
    if match is None:
        return None
    return match.group(1)


def issuer_of(identifier: str) -> str:
    """Ticker/issuer prefix of a FinQA id such as ``ETR/2016/page_23.pdf-2``."""
    return identifier.split("/", 1)[0]


def document_of(identifier: str) -> str:
    """Source document of an id, i.e. the path with the trailing ``-N`` cut."""
    return identifier.rsplit("-", 1)[0]


def evidence_slice(gold_inds: dict) -> list[str]:
    """Exact evidence sentences (gold_inds values) in a deterministic order.

    ``gold_inds`` keys look like ``text_1`` or ``table_8``; we sort by the
    (section, index) pair so the slice order is stable and reproducible.
    """

    def sort_key(item: tuple[str, str]) -> tuple[str, int, str]:
        key = item[0]
        section, _, number = key.rpartition("_")
        try:
            index = int(number)
        except ValueError:
            index = 0
        return (section or key, index, key)

    return [value for _, value in sorted(gold_inds.items(), key=sort_key)]


def build_prompt(question: str, evidence: list[str]) -> str:
    """Assemble the operator-routing prompt from the question and evidence."""
    lines = [
        "You are routing a financial question to a single arithmetic operation.",
        "Which operation in the choices would answer the question from this evidence?",
        "",
        f"Question: {question.strip()}",
        "",
        "Evidence:",
    ]
    lines.extend(f"- {sentence.strip()}" for sentence in evidence)
    return "\n".join(lines)


def load_records(path: Path) -> list[dict]:
    """Read one raw FinQA split file into a list of record dicts."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON array")
    return data


def candidate_rows(records: list[dict], publisher_split: str) -> list[dict]:
    """Yield benchmark rows from one split that pass all deterministic filters.

    Filters, in order: valid single-step operator program; non-empty evidence;
    hash-allocated issuer fold matches ``publisher_split``; assembled prompt
    within the documented character budget.
    """
    rows: list[dict] = []
    for record in records:
        qa = record.get("qa") or {}
        identifier = record.get("id")
        if not identifier:
            continue
        operator = single_step_operator(qa.get("program"))
        if operator is None:
            continue
        evidence = evidence_slice(qa.get("gold_inds") or {})
        if not evidence:
            continue
        issuer = issuer_of(identifier)
        if issuer_fold(issuer) != publisher_split:
            # Blocks issuers and honours the publisher repartition at once.
            continue
        question = (qa.get("question") or "").strip()
        if not question:
            continue
        prompt = build_prompt(question, evidence)
        if len(prompt) > MAX_PROMPT_CHARS:
            # Documented deterministic length rule: drop over-long evidence.
            continue
        case_id = f"finqa-{_digest(TRACK, identifier)[:16]}"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_id": case_id,
                "track": TRACK,
                "family": f"issuer-{issuer}",
                "split": publisher_split,
                "prompt": prompt,
                "label": operator,
                "choices": dict(CHOICES),
                "evaluation_block": f"issuer-{issuer}",
                "source_id": identifier,
                # Bookkeeping used by sampling / validation; not part of schema.
                "_issuer": issuer,
                "_document": document_of(identifier),
                "_norm_question": normalize_question(question),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Deterministic class-balanced sampling
# --------------------------------------------------------------------------- #

def _sort_candidates(rows: list[dict]) -> list[dict]:
    """Deterministic, label-independent ordering of candidate rows."""
    return sorted(rows, key=lambda row: _digest("finqa-order", row["source_id"]))


def class_balanced_sample(rows: list[dict], capacity: int) -> list[dict]:
    """Pick up to ``capacity`` rows, as class-balanced as capacities allow.

    Two passes over a deterministic ordering:

    1. Prefer at most one question per source document, cycling through the
       operator classes round-robin so that counts stay as even as the
       available rows permit.
    2. If the per-document preference leaves the fold short of capacity (and
       rows remain), relax the one-per-document rule and continue the same
       round-robin top-up.
    """
    by_op: dict[str, list[dict]] = {op: [] for op in OPERATORS}
    for row in _sort_candidates(rows):
        by_op[row["label"]].append(row)

    chosen: list[dict] = []
    used_documents: set[str] = set()
    taken_ids: set[str] = set()

    def drain(respect_documents: bool) -> None:
        cursors = {op: 0 for op in OPERATORS}
        progressed = True
        while len(chosen) < capacity and progressed:
            progressed = False
            for op in OPERATORS:
                if len(chosen) >= capacity:
                    break
                pool = by_op[op]
                index = cursors[op]
                while index < len(pool):
                    row = pool[index]
                    index += 1
                    if row["source_id"] in taken_ids:
                        continue
                    if respect_documents and row["_document"] in used_documents:
                        continue
                    chosen.append(row)
                    taken_ids.add(row["source_id"])
                    used_documents.add(row["_document"])
                    progressed = True
                    break
                cursors[op] = index

    drain(respect_documents=True)
    if len(chosen) < capacity:
        drain(respect_documents=False)
    return chosen


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate(folds: dict[str, list[dict]]) -> None:
    """Assert disjoint issuers, no duplicate questions, and adequate test."""
    issuers = {fold: {row["_issuer"] for row in rows} for fold, rows in folds.items()}
    for left in FOLDS:
        for right in FOLDS:
            if left < right:
                overlap = issuers[left] & issuers[right]
                if overlap:
                    raise ValueError(
                        f"Issuer overlap between {left} and {right}: {sorted(overlap)}"
                    )

    questions = {
        fold: {row["_norm_question"] for row in rows} for fold, rows in folds.items()
    }
    for left in FOLDS:
        for right in FOLDS:
            if left < right:
                overlap = questions[left] & questions[right]
                if overlap:
                    raise ValueError(
                        f"Duplicate normalized question across {left}/{right}"
                    )

    if len(folds["test"]) < MIN_ADEQUATE_TEST:
        raise ValueError(
            f"Test fold has {len(folds['test'])} rows; need >= {MIN_ADEQUATE_TEST}"
        )


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #

_PUBLIC_FIELDS = (
    "schema_version",
    "case_id",
    "track",
    "family",
    "split",
    "prompt",
    "label",
    "choices",
    "evaluation_block",
    "source_id",
)


def _public(row: dict) -> dict:
    """Project a row onto the published schema (drop private bookkeeping)."""
    return {field: row[field] for field in _PUBLIC_FIELDS}


def write_jsonl(path: Path, rows: list[dict]) -> str:
    """Atomically write rows as JSONL; return the file's SHA256 digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(_public(row), sort_keys=True, ensure_ascii=False) + "\n"
        for row in rows
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build(input_dir: Path, output_dir: Path) -> dict:
    """Prepare the three folds and the receipt; return the receipt metadata."""
    inputs = {
        "train": input_dir / "finqa-train.json",
        "valid": input_dir / "finqa-dev.json",
        "test": input_dir / "finqa-test.json",
    }
    candidates: dict[str, list[dict]] = {}
    raw_counts: dict[str, int] = {}
    for fold, path in inputs.items():
        records = load_records(path)
        raw_counts[fold] = len(records)
        candidates[fold] = candidate_rows(records, publisher_split=fold)

    # Guarantee cross-fold question uniqueness deterministically before
    # sampling: a normalized question seen in an earlier fold (train, then
    # valid, then test) is dropped from later folds.  Issuers are already
    # disjoint by construction, so this only removes coincidental re-phrasings.
    seen_questions: set[str] = set()
    for fold in FOLDS:
        kept = []
        for row in candidates[fold]:
            if row["_norm_question"] in seen_questions:
                continue
            kept.append(row)
        candidates[fold] = kept
        seen_questions.update(row["_norm_question"] for row in kept)

    folds = {
        fold: class_balanced_sample(candidates[fold], CAPACITY[fold])
        for fold in FOLDS
    }

    validate(folds)

    digests = {}
    for fold in FOLDS:
        digests[fold] = write_jsonl(output_dir / OUTPUT_NAMES[fold], folds[fold])

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "track": TRACK,
        "source": {
            "repo": SOURCE_REPO,
            "commit": SOURCE_COMMIT,
            "license": SOURCE_LICENSE,
            "files": {fold: path.name for fold, path in inputs.items()},
            "sha256": {fold: hashlib.sha256(path.read_bytes()).hexdigest() for fold, path in inputs.items()},
        },
        "method": {
            "task": (
                "route a FinQA single-step question to one of four arithmetic "
                "operators given its exact gold_inds evidence slice"
            ),
            "operators": list(OPERATORS),
            "single_step_rule": (
                "retain only questions whose qa.program is exactly one top-level "
                "call op(a, b) with op in {add, subtract, multiply, divide}, no "
                "chained steps and no '#' back-references"
            ),
            "evidence_rule": (
                "prompt carries the question and the exact values of qa.gold_inds "
                "(section-ordered); no programs, steps, annotations or answers"
            ),
            "length_rule": (
                f"skip any row whose assembled prompt exceeds {MAX_PROMPT_CHARS} "
                "characters (deterministic)"
            ),
            "issuer_allocation": (
                "issuer = id ticker prefix; stable SHA256(issuer) mod 100 mapped "
                "to 60/20/20 train/valid/test, independent of labels; a row is "
                "kept only if its allocated fold equals its publisher split "
                "(publisher dev -> valid), blocking issuers and honouring the "
                "upstream repartition"
            ),
            "sampling_rule": (
                "deterministic hash-ordered, round-robin class-balanced selection "
                "up to capacity; prefer at most one question per source document, "
                "relaxing that only if the fold is still short of capacity"
            ),
            "capacities": dict(CAPACITY),
        },
        "counts": {
            "raw_records": raw_counts,
            "candidates": {fold: len(candidates[fold]) for fold in FOLDS},
            "selected": {fold: len(folds[fold]) for fold in FOLDS},
            "label_distribution": {
                fold: dict(Counter(row["label"] for row in folds[fold]))
                for fold in FOLDS
            },
            "issuers": {fold: len({row["_issuer"] for row in folds[fold]}) for fold in FOLDS},
            "documents": {fold: len({row["_document"] for row in folds[fold]}) for fold in FOLDS},
        },
        "outputs": {
            fold: {"file": OUTPUT_NAMES[fold], "sha256": digests[fold]}
            for fold in FOLDS
        },
        "validation": {
            "issuers_disjoint": True,
            "questions_unique_across_folds": True,
            "test_adequate": len(folds["test"]) >= MIN_ADEQUATE_TEST,
        },
        "limits": (
            "Operator-routing proxy over single-step FinQA items only; no corpus "
            "text is printed and no financial advice is implied."
        ),
    }

    receipt_path = output_dir / "finqa-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory holding finqa-{train,dev,test}.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory to receive finqa-{train,valid,test}.jsonl and the receipt",
    )
    options = parser.parse_args(argv)
    receipt = build(options.input, options.output)
    # Only derived counts/metadata are printed -- never corpus text.
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
