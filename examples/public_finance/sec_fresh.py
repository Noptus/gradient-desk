"""Freeze an issuer-disjoint follow-up cohort for SEC section routing."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
from hashlib import file_digest, sha256
import json
from pathlib import Path

import sec_items
from prepare import RECEIPT

START = date(2026, 10, 1)
END = date(2026, 11, 13)
MAX_PER_LABEL = 48
MIN_PER_LABEL = 20
SHINGLE_WIDTH = 5
MIN_SHARED_SHINGLES = 20
CONTAINMENT_NUMERATOR = 17
CONTAINMENT_DENOMINATOR = 20
OUTPUT_NAME = "sec8k-fresh-test.jsonl"
FROZEN = Path(__file__).with_name("fresh-sec-receipt.json")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def normalize(excerpt: str) -> tuple[str, ...]:
    return tuple(sec_items.DUPLICATE_HASH.sub(" ", excerpt.casefold()).split())


class TemplateIndex:
    """Detect templates using exact text and normalized shingle containment."""

    def __init__(self):
        self.exact = set()
        self.sizes = []
        self.postings = defaultdict(list)

    @staticmethod
    def shingles(words):
        return set(zip(*(words[offset:] for offset in range(SHINGLE_WIDTH))))

    def contains(self, words) -> bool:
        if words in self.exact:
            return True
        grams = self.shingles(words)
        overlap = Counter(index for gram in grams for index in self.postings.get(gram, ()))
        return any(
            n >= MIN_SHARED_SHINGLES
            and CONTAINMENT_DENOMINATOR * n
            >= CONTAINMENT_NUMERATOR * min(len(grams), self.sizes[index])
            for index, n in overlap.items()
        )

    def add(self, words) -> None:
        if words in self.exact:
            return
        self.exact.add(words)
        grams = self.shingles(words)
        index = len(self.sizes)
        self.sizes.append(len(grams))
        for gram in grams:
            self.postings[gram].append(index)


def issuer(cik: str) -> str:
    if not cik or not cik.strip().isascii() or not cik.strip().isdigit():
        raise ValueError("Missing or invalid source CIK")
    return cik.strip().zfill(10)


def original_inputs(directory, expected):
    issuers, accessions, case_ids = set(), set(), set()
    templates = TemplateIndex()
    inputs, counts = {}, {}
    for split, entry in sorted(expected["splits"].items()):
        path = directory / f"sec8k-{split}.jsonl"
        computed = digest(path)
        if computed != entry["sha256"]:
            raise ValueError(f"Original SEC {split} bytes differ from the published protocol")
        inputs[path.name] = computed
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        counts[split] = len(rows)
        for row in rows:
            prompt = row["prompt"]
            if row["split"] != split or not prompt.startswith(sec_items.PROMPT_PREFIX):
                raise ValueError(f"Invalid original SEC split or prompt: {row['case_id']}")
            issuers.add(issuer(row["evaluation_block"].removeprefix("cik-")))
            accessions.add(row["source_id"])
            case_ids.add(row["case_id"])
            templates.add(normalize(prompt.removeprefix(sec_items.PROMPT_PREFIX)))
    return issuers, accessions, case_ids, templates, inputs, counts


def _sample(source, original_issuers, original_accessions, original_templates):
    candidates, rejected = defaultdict(list), Counter()
    for event, cik, accession, day, label, body, title, company, link in source:
        if not event or not accession or not link or not link.startswith("https://www.sec.gov/"):
            rejected["unlinked"] += 1
            continue
        try:
            cik = issuer(cik)
        except ValueError:
            rejected["invalid_issuer"] += 1
            continue
        if cik in original_issuers or accession in original_accessions:
            rejected["original_issuer_or_accession"] += 1
            continue
        excerpt, _ = sec_items.snippet(body, title, company)
        if len(excerpt) < 140:
            rejected["short_excerpt"] += 1
            continue
        words = normalize(excerpt)
        if original_templates.contains(words):
            rejected["original_template"] += 1
            continue
        candidates[label].append((event, cik, accession, day, excerpt, words, link))

    available = {label: len({row[1] for row in candidates[label]}) for label in sec_items.CHOICES}
    used_issuers, used_accessions = set(), set()
    fresh_templates = TemplateIndex()
    chosen = []
    for label in sorted(available, key=lambda label: (available[label], label)):
        rows = sorted(candidates[label], key=lambda row: (
            sha256(f"sec8k-fresh-v1:{row[0]}".encode()).digest(), row[0],
        ))
        count = 0
        for event, cik, accession, day, excerpt, words, link in rows:
            if cik in used_issuers or accession in used_accessions:
                rejected["fresh_issuer_or_accession"] += 1
                continue
            if fresh_templates.contains(words):
                rejected["fresh_template"] += 1
                continue
            group = f"cik-{cik}"
            chosen.append({
                "case_id": f"sec8k-fresh-{sha256(event.encode()).hexdigest()[:24]}",
                "track": sec_items.TRACK, "family": group, "evaluation_block": group,
                "group_id": group, "split": "test", "label": label,
                "choices": sec_items.CHOICES, "source_id": accession, "source_url": link,
                "filing_date": day.isoformat(), "prompt": sec_items.PROMPT_PREFIX + excerpt,
            })
            used_issuers.add(cik)
            used_accessions.add(accession)
            fresh_templates.add(words)
            count += 1
            if count == MAX_PER_LABEL:
                break
        if count < MIN_PER_LABEL:
            raise ValueError(f"Insufficient cases for fixed label {label}: {count}")
    return sorted(chosen, key=lambda row: row["case_id"]), rejected, available


def build(raw: Path, original: Path, output: Path, verified: Path | None) -> dict:
    import duckdb

    old_protocol = json.loads(RECEIPT.read_text(encoding="utf-8"))["sec8k"]
    paths = [raw / f"sec-item-{part}.parquet" for part in sec_items.PARTS]
    raw_sha = {f"data/items/part-{part:05d}.parquet": digest(path)
               for part, path in zip(sec_items.PARTS, paths, strict=True)}
    if raw_sha != old_protocol["source_files"] or sec_items.REVISION != old_protocol["revision"]:
        raise ValueError("SEC raw sources differ from the published revision")
    blocked, accessions, old_ids, templates, old_sha, old_counts = original_inputs(original, old_protocol)
    source = sec_items.source_rows(paths, START, END)
    rows, rejected, available = _sample(source, blocked, accessions, templates)
    new_ids = {row["case_id"] for row in rows}
    if len(new_ids) != len(rows) or new_ids & old_ids:
        raise ValueError("Generated SEC IDs are not unique across the two cohorts")
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    dates = [row["filing_date"] for row in rows]
    receipt = {
        "schema": 1, "task": sec_items.TRACK, "rating_source": "issuer-chosen Form 8-K item",
        "source": {
            "publisher": old_protocol["publisher"], "revision": old_protocol["revision"],
            "license": "Apache-2.0 dataset code; original issuer text retained in local working data",
            "parquet_sha256": raw_sha, "filing_window_inclusive": [START.isoformat(), END.isoformat()],
            "first_filing_selected": min(dates), "last_filing_selected": max(dates),
            "form_reference": sec_items.PRIMARY_DOC,
        },
        "original_inputs": {"sha256": old_sha, "rows": old_counts, "excluded_issuers": len(blocked),
                            "excluded_accessions": len(accessions), "templates_indexed": len(templates.exact)},
        "sampling": {
            "min_per_label": MIN_PER_LABEL, "max_per_label": MAX_PER_LABEL, "max_per_issuer": 1,
            "max_per_accession": 1, "eligible_rows_before_disjoint_filters": len(source),
            "new_issuers_available_by_label": available,
            "case_order": "Rarest eligible label first, ties by item code. Stable SHA-256 event order within label.",
            "near_duplicate": {
                "normalization": "Casefold, replace numbers and punctuation, collapse whitespace.",
                "width": SHINGLE_WIDTH, "minimum_shared": MIN_SHARED_SHINGLES,
                "shorter_text_containment": [CONTAINMENT_NUMERATOR, CONTAINMENT_DENOMINATOR],
            },
            "rejected": dict(sorted(rejected.items())),
        },
        "cases": len(rows), "issuer_blocks": len({row["evaluation_block"] for row in rows}),
        "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
        "output_file": OUTPUT_NAME, "output_sha256": sha256(payload).hexdigest(),
        "generator_sha256": digest(Path(__file__)),
        "extractor_sha256": digest(Path(sec_items.__file__)),
        "duckdb": duckdb.__version__,
    }
    if verified is not None:
        expected = json.loads(verified.read_text(encoding="utf-8"))
        if receipt != expected:
            raise ValueError("Fresh SEC reconstruction differs from the frozen receipt")
    output.mkdir(parents=True, exist_ok=False)
    (output / OUTPUT_NAME).write_bytes(payload)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--raw", type=Path, required=True, help="Directory with the six pinned Parquet files")
    cli.add_argument("--original", type=Path, required=True, help="Directory with original prepared SEC splits")
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--verify-against", type=Path, default=FROZEN if FROZEN.exists() else None)
    args = cli.parse_args()
    print(json.dumps(build(args.raw, args.original, args.output, args.verify_against), indent=2))
