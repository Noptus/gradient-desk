"""Download pinned sources and reproduce the pre-registered finance panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import finqa
import sec_items
import treasury


ROOT = Path(__file__).resolve().parent
RECEIPT = ROOT / "frozen-receipts.json"


def download(path: Path, url: str, digest: str, limit: int = 120_000_000) -> None:
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Cached input digest differs: {path.name}")
        return
    temporary = path.with_suffix(path.suffix + ".part")
    checksum = hashlib.sha256()
    size = 0
    try:
        request = Request(url, headers={"User-Agent": "Gradient Desk reproducibility/0.2"})
        with urlopen(request, timeout=45) as response, temporary.open("wb") as file:
            while block := response.read(1_048_576):
                size += len(block)
                if size > limit:
                    raise ValueError(f"Source response exceeds the download limit: {url}")
                checksum.update(block)
                file.write(block)
        if checksum.hexdigest() != digest:
            raise ValueError(f"Upstream bytes differ from frozen source: {url}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def verify(output: Path, receipts: dict) -> dict:
    outputs = {
        **{value["file"]: value["sha256"] for value in receipts["finqa"]["outputs"].values()},
        **{f"sec8k-{split}.jsonl": value["sha256"] for split, value in receipts["sec8k"]["splits"].items()},
        **{f"treasury-{split}.jsonl": value["sha256"] for split, value in receipts["treasury"]["counts"].items()},
    }
    for name, digest in outputs.items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Reconstructed benchmark digest differs: {name}")
    return outputs


def build(cache: Path, output: Path) -> dict:
    receipts = json.loads(RECEIPT.read_text(encoding="utf-8"))
    cache.mkdir(parents=True, exist_ok=True)
    source = receipts["finqa"]["source"]
    for split, file in source["files"].items():
        url = (f"https://raw.githubusercontent.com/{source['repo']}/{source['commit']}"
               f"/dataset/{file.removeprefix('finqa-')}")
        download(cache / file, url, source["sha256"][split])
    sec = receipts["sec8k"]
    for part in sec_items.PARTS:
        upstream = f"data/items/part-{part:05d}.parquet"
        url = f"https://huggingface.co/datasets/{sec['publisher']}/resolve/{sec['revision']}/{upstream}"
        download(cache / f"sec-item-{part}.parquet", url, sec["source_files"][upstream])
    download(cache / treasury.FILENAME, treasury.URL, treasury.SOURCE_SHA256)
    output.mkdir(parents=True, exist_ok=True)
    finqa.build(cache, output)
    sec_items.prepare(cache, output)
    treasury.prepare(cache, output)
    return verify(output, receipts)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cache", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    print(json.dumps(build(args.cache, args.output), indent=2))
