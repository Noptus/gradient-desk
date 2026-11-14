# Finance benchmark protocol

This suite tests three different kinds of financial reasoning. Labels come from the original operator annotations, issuer filing sections, and an exact arithmetic oracle. Training and development files are reconstructed locally from pinned sources. The test files are frozen by SHA-256 in [the data receipt](frozen-receipts.json). Third-party texts stay in the local cache.

| Test | Likely use | Train / development / test | Held-out units |
| --- | --- | ---: | --- |
| FinQA operator routing | Choose a calculator operation from a question and its gold evidence | 320 / 72 / 103 | 15 issuers |
| SEC 8-K section routing | Classify eight disclosure sections from their text | 320 / 96 / 159 | 136 issuers |
| Treasury memo audit | Find an error in the dates, tenor, sign or units of a yield claim | 300 / 100 / 150 | 60 months |

## Freeze and compare

The frozen base and adapter see the same inputs and shuffled answer layout. A second layout rotates the options, and the test uses their mean class probabilities. The fixed primary metric for each of these three panels is exact accuracy. Macro accuracy, calibration loss, and confusion counts show how much of an overall change comes from class mix. Any neural intervention panel uses matched outcome regret as its primary metric.

Select training configuration and checkpoint on development rows only. Open the test reports for a single frozen selection. A panel shows an observed improvement at a gain of at least 0.02 against the frozen base. Treat gains as confirmed only if a paired block-bootstrap interval excludes zero and the one-sided paired block sign-flip test passes Holm correction at 0.05. Blocks are issuer or calendar month as listed above. The Treasury classification and Treasury intervention results share an archive and should count as one independent source in an improvement claim. Previously scored panels can diagnose failures and carry less weight after another round of tuning.

### FinQA

Use the original FinQA train, development and test files from a pinned MIT-licensed repository. Only exact single-step add, subtract, multiply, and divide annotations qualify. Each model gets the same short slice of original gold evidence and is asked for the operation. This is an operator-routing projection, so its scores cannot be compared with the original full-program or execution benchmarks. A stable issuer fold must agree with the original publisher split. The sampler balances classes where eligible examples permit it and retains the frozen evaluation class proportions. Group inference by issuer. Potential exposure of an older public benchmark in base-model pretraining is unknown. [Original paper and dataset](https://aclanthology.org/2021.emnlp-main.300/)

### SEC Form 8-K

Use the pinned ZipLime extraction of issuer-authored SEC filings and the official form definitions. Retain parsed original filings from 2026, with train ending July 31 and development ending September 30. Require an independent CIK hash split to agree with the period. Remove item headings, code references and explicit issuer names, and discard repeated templates crossing groups. Hold at most one section per filing and a cap per class. Judge the actual section used by the issuer. Filing guidance can leave more than one reasonable categorization. Dataset code is Apache-2.0. The source's notice distinguishes SEC metadata from issuer-authored text and gives the original document URL. [Source dataset](https://huggingface.co/datasets/ZipLime/sec-8k-events) · [SEC Form 8-K](https://www.sec.gov/files/form8-k.pdf)

### Treasury

Use the official daily par curve archive. At each date, select nonoverlapping three-session windows where a correct basis-point change and four named error computations differ. Shuffle display order and apply equal error frequencies. Splits use disjoint dates and are shared with the intervention benchmark. The rate rows are observed Treasury estimates. The cutoff scenarios and audit defects are generated, and this test does not establish release-time trading experience. An exact calculator is the task ceiling. [U.S. Treasury interest rate data](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics)

### Lexical comparator

A TF-IDF logistic baseline uses only the case prompt. Development accuracy selects word n-grams with or without additional character n-grams, and inverse regularization strength from 0.25, 1, 4 and 16. The program checks the reconstructed inputs against [the frozen digests](frozen-receipts.json) before use. Held-out accuracy was 68.93% on FinQA, 86.16% on SEC and 32.00% on Treasury.

Reproduce it with the pinned solver after rebuilding the inputs.

```sh
python3 -m pip install scikit-learn==1.9.1
python3 examples/public_finance/lexical.py \
  --cases-dir .work/v2/public-finance/reconstructed \
  --output .work/v2/public-finance/lexical
```

See the [aggregate receipt](results/lexical-receipt.json) and [case-level predictions](results/lexical-predictions.jsonl).

### Treasury calculator

The fixed-format checker reads the displayed dates, yields and claim. It evaluates each of the five possible discrepancies using exact decimal arithmetic. It identified all 150 held-out labels. This score measures what a task-specific calculator can do on the generated memo template.

```sh
python3 examples/public_finance/treasury_checker.py \
  --cases-dir .work/v2/public-finance/reconstructed \
  --output .work/v2/public-finance/treasury-checker
```

See the [calculator receipt](results/treasury-checker-receipt.json) and [case-level predictions](results/treasury-checker-predictions.jsonl).

## Reproduce the inputs

Run with Python 3.12 or newer. DuckDB 2.0.0 is only required for the SEC Parquet extraction.

```sh
python3 -m pip install duckdb==2.0.0
python3 examples/public_finance/prepare.py \
  --cache .work/v2/public-finance/cache \
  --output .work/v2/public-finance/reconstructed
```

The command checks the raw source bytes and generated JSONL digests against the frozen receipt. Its inputs are about 160 MB. Source availability and benchmark licensing stay with each upstream dataset.
