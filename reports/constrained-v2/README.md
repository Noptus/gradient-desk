# Constrained QLoRA evaluation protocol

Gradient Desk tests whether an adapter improves financial research decisions and selection of an optimizer intervention from checkpoint telemetry. The reference protocol combines [three public finance text panels](../../examples/public_finance/README.md) with two measured intervention panels. The [frozen receipt](protocol.json) binds all five test panels and the scheduled checkpoints before transformer holdout inference. Earlier synthetic runs and fictional review cases contribute training examples and serve as diagnostics.

| Panel | Primary endpoint | Test cases | Dependence blocks |
| --- | --- | ---: | ---: |
| FinQA operator routing | Accuracy | 103 | 15 issuers |
| SEC 8-K item routing | Accuracy | 159 | 136 issuers |
| Treasury yield claim audit | Accuracy | 150 | 60 calendar months |
| Treasury yield MLP intervention | Normalized terminal MSE regret | 85 | 17 date blocks |
| Taiwanese bankruptcy MLP intervention | Normalized terminal BCE regret | 140 | 28 source-row groups |

For an intervention, each case restores identical model, optimizer, and random state. Each action consumes the same eight ordered microbatches. Accumulation averages pairs and makes four optimizer updates, while the other branches each make eight. Clipping adds a gradient-reduction operation. The monitor and terminal datasets use separate rows. The prompt includes pre-action telemetry only. The train objective assigns half of the measured intervention target mass to the near-tie-resolved oracle label and distributes half with a softmax of negative normalized regret at temperature 0.18. Categorical finance rows use one-hot labels.

Treasury measures generalization to later years with a six-input yield-change MLP. Historical features and one-session-ahead labels are included in the date-boundary checks. The [U.S. Treasury daily par yields archive](https://home.treasury.gov/system/files/276/yield-curve-rates-1990-2024.csv) includes a documented December 2021 methodology change. The audit and intervention both use this archive and count as one source. FinQA and SEC use different corpora and label sources with potential overlap in issuers.

The [UCI Taiwanese Bankruptcy Prediction dataset](https://doi.org/10.24432/C5004D) provides a test-only transfer panel. Liang and coauthors contributed the anonymous company ratios and outcomes under CC BY 4.0. The generator uses 16 ratios and a small BCE classifier, with no UCI examples entering adapter fitting or checkpoint selection. It uses 28 disjoint groups of 236 rows, five profiles per group, and shared terminal rows within each group. Issuer identifiers are unavailable, so group inference assumes dependence is adequately represented by disjoint row groups. The experiment evaluates an optimizer decision and makes no claim about investment performance.

## Fixed training and scoring

Train 240 Adam steps on a single four-row batch per step, using rank 8, scale 16, zero dropout, and adapters on the final eight Qwen3 layers. The learning rate is 0.00015, seed 1777, and input token limit 1024. Cycle through the five task tracks with a square-root class-frequency sampler. Shuffle answer positions using the case ID and rotate them across training. Checkpoints are at steps 60, 120, 180, and 240.

Use up to 32 development cases per panel, picked across the declared blocks by stable hashes of IDs before measuring outcomes. Score the three public language panels and the Treasury intervention panel. Maximize mean panel utility with equal panel weights, using accuracy for language and 1 minus normalized regret for intervention. Early steps break exact ties. Record the frozen base on the same development inputs. Evaluate only the selected checkpoint on the frozen holdout.

Score both models with the same Qwen3 chat template, option meanings, and pair of layout orders. Restrict the final-token probability distribution to the legal menu and average order-specific probabilities in option-ID space. The primary comparator is the unadapted base checkpoint. Report transparent CPU controls separately. An observed improvement requires a gain of at least 0.02. A confirmed improvement also requires a positive paired block-bootstrap interval and a one-sided paired block test with Holm-adjusted p below 0.05. Intervention regret divides excess terminal loss by the within-case range of measured losses. Report raw regret alongside it.

## Generate data and run

Use Python 3.12 and the repository's pinned [Apple Silicon environment](../../requirements-apple.txt). The [base checkpoint](https://huggingface.co/Qwen/Qwen3-8B-MLX-4bit/tree/383413e909f3bc5303ce195ebbdf0339c5a1a2a3) is obtained with `gradient-desk download --output models/Qwen3-8B-MLX-4bit`. Follow the [finance source protocol](../../examples/public_finance/README.md) to build `finqa-*.jsonl`, `sec8k-*.jsonl`, and `treasury-*.jsonl` under `.work/v2/public-finance`. Fetch the verified public sources before running the CPU generators.

```sh
mkdir -p .work/v2
curl -fsSL -o .work/v2/treasury-yield-curve-1990-2024.csv \
  https://home.treasury.gov/system/files/276/yield-curve-rates-1990-2024.csv
curl -fsSL -o .work/v2/uci-taiwan-bankruptcy.zip \
  'https://archive.ics.uci.edu/static/public/572/taiwanese+bankruptcy+prediction.zip'
python -m gradient_desk.treasury_traces \
  --csv .work/v2/treasury-yield-curve-1990-2024.csv --output .work/v2/treasury
python -m gradient_desk.bankruptcy_traces \
  --archive .work/v2/uci-taiwan-bankruptcy.zip --output .work/v2/bankruptcy/test.jsonl
python -m gradient_desk.traces --output data/generated/traces.jsonl --seeds-per-case 12
gradient-desk prepare --traces data/generated/traces.jsonl --output data/generated/mixed
python -m gradient_desk.choice_data --output .work/v2/legacy-choice \
  --input data/generated/mixed/train-cases.jsonl \
  --input data/generated/mixed/valid-cases.jsonl
python -m gradient_desk.choice_data --output .work/v2/campaign-01-data \
  --input .work/v2/legacy-choice/train.jsonl \
  --input .work/v2/legacy-choice/valid.jsonl \
  --input .work/v2/treasury/treasury-train.jsonl \
  --input .work/v2/treasury/treasury-valid.jsonl \
  --input .work/v2/public-finance/finqa-train.jsonl \
  --input .work/v2/public-finance/finqa-valid.jsonl \
  --input .work/v2/public-finance/sec8k-train.jsonl \
  --input .work/v2/public-finance/sec8k-valid.jsonl \
  --input .work/v2/public-finance/treasury-train.jsonl \
  --input .work/v2/public-finance/treasury-valid.jsonl
python -m gradient_desk.choice_data --freeze-test-against .work/v2/campaign-01-data \
  --input .work/v2/public-finance/finqa-test.jsonl \
  --input .work/v2/public-finance/sec8k-test.jsonl \
  --input .work/v2/public-finance/treasury-test.jsonl \
  --input .work/v2/treasury/treasury-test.jsonl \
  --input .work/v2/bankruptcy/test.jsonl --output .work/v2/frozen
```

The test projection excludes wall-clock timings. Retained training diagnostics contain elapsed times, so the byte digests in the fit receipt identify the original run. When regenerating data, use the new fit's provenance with its exact validation bytes. Compare the stable measured-source and frozen holdout hashes on the reference runtime. Confirm the test digest before the one held-out evaluation. Small numerical changes across CPU runtimes can affect regenerated test bytes.

```sh
export HF_HUB_OFFLINE=1
python -u -m gradient_desk.choice_model fit \
  --train .work/v2/campaign-01-data/train.jsonl --valid .work/v2/campaign-01-data/valid.jsonl \
  --model models/Qwen3-8B-MLX-4bit --output adapters/choice-v2-campaign-01 \
  --steps 240 --save-every 60 --batch-size 4 --learning-rate 0.00015 \
  --seed 1777 --num-layers 8 --rank 8 --scale 16 --max-tokens 1024
python -u -m gradient_desk.choice_campaign select \
  --validation .work/v2/campaign-01-data/valid.jsonl \
  --model models/Qwen3-8B-MLX-4bit --adapter adapters/choice-v2-campaign-01 \
  --steps 60 120 180 240 --limit-per-benchmark 32 --variants 2 --batch-size 8 \
  --benchmark finqa_operator --benchmark sec8k_item_routing \
  --benchmark treasury_curve_audit --benchmark treasury_intervention \
  --output .work/v2/choice-selection.json
python -u -m gradient_desk.choice_campaign evaluate \
  --cases .work/v2/frozen/test.jsonl --expected-cases-sha256 \
  b29b8149e140374b1a1a8e58e4b228b77ab6e2ee346725fcec8e5075735eba05 \
  --model models/Qwen3-8B-MLX-4bit --adapter adapters/choice-v2-campaign-01 \
  --selection .work/v2/choice-selection.json --variants 2 --batch-size 8 \
  --output .work/v2/choice-heldout
```

Run `gradient_desk.choice_cpu` with the same training, development, and frozen-test paths to reproduce the fixed-resume and validation-selected telemetry controls. The language controls are described in the [finance source protocol](../../examples/public_finance/README.md).
