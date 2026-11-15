"""Extract a grouped SEC Form 8-K section classification panel from pinned Parquet."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import re


TRACK = "sec8k_item_routing"
REVISION = "dff0c3ef84097f39c97ab450465f47e9ceed3870"
PARTS = (25, 26, 30, 31, 34, 35)
CHOICES = {
    "1.01": "Entry into a material definitive agreement.",
    "1.02": "Termination of a material definitive agreement.",
    "2.02": "Results of operations and financial condition.",
    "2.03": "Creation of a direct financial obligation or off-balance-sheet obligation.",
    "3.02": "Unregistered sales of equity securities.",
    "5.02": "Changes involving directors, principal officers, or compensatory arrangements.",
    "7.01": "Regulation FD disclosure.",
    "8.01": "Other events disclosed by the registrant.",
}
CAP = {"train": 320, "valid": 96, "test": 160}
TRAIN_END = date(2026, 7, 31)
DEV_END = date(2026, 9, 30)
PRIMARY_DOC = "https://www.sec.gov/files/form8-k.pdf"
DUPLICATE_HASH = re.compile(r"\d+|[^\w\s]+", flags=re.UNICODE)
HEADING = re.compile(r"^\s*Item\s*\d{1,2}\.\d{2}(?:\s|[.:—-])*", re.I)
LEAK = re.compile(r"\bitem\s+\d{1,2}\.\d{2}\b", re.I)
PROMPT_PREFIX = (
    "Which Form 8-K section did the issuer use for the following excerpt? "
    "The original section heading and any numeric references to other items were removed.\n\n"
)


def fold(cik: str) -> str:
    bucket = int(sha256(f"edgar-2026-v2:{cik}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 60 else "valid" if bucket < 80 else "test"


def snippet(body: str, heading: str, company: str) -> tuple[str, str]:
    candidate = HEADING.sub("", " ".join(body.split())).lstrip(" .:-—")
    title = " ".join((heading or "").split())
    if title and candidate.casefold().startswith(title.casefold()):
        candidate = candidate[len(title):].lstrip(" .:-—")
    candidate = LEAK.sub("[section reference]", candidate)
    company = " ".join((company or "").split())
    if company and len(company) > 3:
        candidate = re.sub(re.escape(company), "[issuer]", candidate, flags=re.I)
    excerpt = candidate if len(candidate) <= 1250 else candidate[:1250].rsplit(" ", 1)[0] + " …"
    template = " ".join(DUPLICATE_HASH.sub(" ", excerpt.casefold()).split())
    return excerpt, sha256(template.encode()).hexdigest()


def source_rows(paths: list[Path], start: date, end: date) -> list[tuple]:
    import duckdb

    columns = "event_id, cik, accession_no, filing_date, item_code, item_text, item_title, company_name, primary_document_url"
    labels = list(CHOICES)
    return duckdb.execute(
        f"""SELECT {columns}
        FROM read_parquet(?)
        WHERE status = 'parsed' AND form = '8-K' AND NOT is_amendment
          AND item_code IN ({", ".join("?" for _ in labels)})
          AND length(item_text) BETWEEN 180 AND 15000
          AND filing_date BETWEEN ? AND ?
        ORDER BY filing_date, cik, accession_no, item_code""",
        [[str(path) for path in paths], *labels, start, end],
    ).fetchall()


def prepare(raw: Path, output: Path) -> dict:
    import duckdb

    output.mkdir(parents=True, exist_ok=True)
    labels = list(CHOICES)
    paths = [raw / f"sec-item-{part}.parquet" for part in PARTS]
    source = source_rows(paths, date(2026, 1, 2), date(2026, 11, 13))
    candidates = defaultdict(list)
    template_folds = defaultdict(set)
    rejected = Counter()
    for event, cik, accession, day, label, body, title, company, link in source:
        if not cik or not accession or not event or not link or not link.startswith("https://www.sec.gov/"):
            rejected["unlinked_row"] += 1
            continue
        source_fold = "train" if day <= TRAIN_END else "valid" if day <= DEV_END else "test"
        if fold(cik) != source_fold:
            rejected["issuer_or_period_mismatch"] += 1
            continue
        excerpt, template = snippet(body, title, company)
        if len(excerpt) < 140:
            rejected["short_excerpt"] += 1
            continue
        template_folds[template].add(source_fold)
        candidates[source_fold, label].append((event, template, cik, accession, excerpt, link))

    receipt = {
        "task": TRACK, "publisher": "ZipLime/sec-8k-events",
        "revision": REVISION,
        "form_reference": PRIMARY_DOC,
        "source_files": {
            f"data/items/part-{part:05d}.parquet": sha256(path.read_bytes()).hexdigest()
            for part, path in zip(PARTS, paths, strict=True)
        },
        "duckdb": duckdb.__version__, "splits": {},
    }
    issuers = {}
    for split, size in CAP.items():
        selected = []
        used_accessions, used_templates = set(), set()
        for label in labels:
            values = sorted(
                candidates[split, label],
                key=lambda item: (sha256(f"{TRACK}:{item[0]}".encode()).digest(), item[0]),
            )
            class_count = 0
            for event, template, cik, accession, excerpt, url in values:
                if len(template_folds[template]) > 1:
                    rejected["cross_split_template"] += 1
                    continue
                if accession in used_accessions or template in used_templates:
                    continue
                identifier = sha256(event.encode()).hexdigest()[:24]
                selected.append({
                    "case_id": f"sec8k-{identifier}",
                    "track": TRACK, "family": f"cik-{cik}", "evaluation_block": f"cik-{cik}",
                    "split": split, "label": label, "choices": CHOICES,
                    "source_id": accession, "source_url": url,
                    "prompt": PROMPT_PREFIX + excerpt,
                })
                used_accessions.add(accession)
                used_templates.add(template)
                class_count += 1
                if class_count >= size // len(labels):
                    break
        if len(selected) < 64 or set(row["label"] for row in selected) != set(labels):
            raise ValueError(f"Insufficient {split} data: {dict(Counter(row['label'] for row in selected))}")
        issuers[split] = {row["family"] for row in selected}
        selected.sort(key=lambda row: row["case_id"])
        path = output / f"sec8k-{split}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected), encoding="utf-8")
        receipt["splits"][split] = {
            "counts": dict(Counter(row["label"] for row in selected)),
            "issuer_blocks": len(issuers[split]),
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
    if sum(map(len, issuers.values())) != len(set.union(*issuers.values())):
        raise ValueError("Issuer overlap between folds")
    receipt["rejected"] = dict(rejected)
    (output / "sec8k-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    print(json.dumps(prepare(args.input, args.output), indent=2))
