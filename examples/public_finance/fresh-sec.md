# Independent SEC Form 8-K follow-up

The first paired QLoRA test measured a gain of 0.629 percentage points on 159 SEC excerpts. The follow-up uses 374 disclosures filed after the original development cutoff, with one case per new issuer. It retains the eight original section labels, issuer-chosen references, text extraction, masking, and prompt. [First test evidence](../../reports/constrained-v2/heldout-first/README.md) · [SEC Form 8-K](https://www.sec.gov/files/form8-k.pdf)

The input window runs from October 1 through November 13, 2026, after the initial development cutoff. The generator excludes each of the 525 distinct CIKs and every accession in the original 320 fitting, 96 development, and 159 test cases. It compares the masked excerpts against all three initial splits and against the fresh cases already selected. The template rule rejects an exact normalized match or at least 20 shared five-word sequences covering 85% of the smaller set. Casefolding and punctuation and number removal help filter repeated text with changed amounts or formatting. The sampler starts with the rarest category by eligible issuer count, chooses candidates by deterministic event hash, and caps each code at 48. [Exact source, filtering, and output receipt](fresh-sec-receipt.json)

The cohort contains 48 cases for seven labels and 38 for Item 1.02. The source is the same six pinned Parquet shards used by the original SEC benchmark. The test uses unseen issuers within that source. The initial and fresh cohort results are reported separately. Source text is stored in the local workspace. [Pinned ZipLime extraction](https://huggingface.co/datasets/ZipLime/sec-8k-events/tree/dff0c3ef84097f39c97ab450465f47e9ceed3870)

## Reproduce and compare

Reconstruct the original files with the original finance protocol if needed. Keep the six Parquet files in the raw cache. Use DuckDB 2.0.0 when rebuilding the follow-up. [Original dataset construction](README.md)

```sh
python examples/public_finance/sec_fresh.py \
  --raw .work/v2/public-finance/cache \
  --original .work/v2/public-finance/reconstructed \
  --output .work/v2/sec-fresh
```

The test protocol fixes exact accuracy against the pinned unadapted Qwen3 base, with paired issuer-level uncertainty, a 0.02 observed-gain threshold, and a declared single follow-up test before viewing its results. Every scorer uses the same examples and option meanings. A statistical improvement additionally requires a positive paired 95% interval and one-sided paired p below 0.05. The validation-selected lexical fit from the first study is a secondary control selected on the same old training and development rows with scikit-learn 1.9.1. [Original lexical receipt](results/lexical-receipt.json)

```sh
python examples/public_finance/sec_fresh_control.py \
  --original .work/v2/public-finance/reconstructed \
  --cases .work/v2/sec-fresh/sec8k-fresh-test.jsonl \
  --output .work/v2/sec-fresh-lexical
```
