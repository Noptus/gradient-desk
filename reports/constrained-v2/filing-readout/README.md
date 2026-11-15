# SEC filing routing follow-up

A supervised ridge head uses the final real-token hidden state from frozen Qwen3 in two answer layouts. It fitted 320 original training cases. Development selection on 32 locked original cases chose regularization 0.1 and full head weight, raising accuracy from 75% to 84.375%. The readout, model, inputs, and scoring procedure were fixed before opening the new issuer cases. [Fit receipt](fit-receipt.json) · [Development](development.json) · [Frozen follow-up plan](../filing-readout-plan.json)

| Policy | Correct / cases | Accuracy |
| --- | ---: | ---: |
| Frozen Qwen3 | 254 / 374 | 67.9144% |
| Supervised Qwen3 readout | 295 / 374 | 78.8770% |
| Validation-selected TF-IDF | 294 / 374 | 78.6096% |

The head gained 10.9626 percentage points against the frozen base, with a paired issuer-block bootstrap interval of [6.4171, 15.5080] points and one-sided p = 0.000167 under the registered resampling procedure. It corrected 61 base errors and lost 20 previously correct predictions. The exact discordance test agrees with the conclusion at p = 0.00000283. [Primary report](heldout/paired-report.json) · [Blinded base predictions](heldout/base.jsonl) · [Blinded head predictions](heldout/readout.jsonl)

Against the fixed TF-IDF control, the gain was 0.2674 percentage points with a paired interval of [−4.0107, 4.5455] points. The head corrected 35 lexical errors and lost 34 previously correct predictions. This comparison shows similar observed accuracy for the two learned policies. [Paired lexical comparison and exact audit](heldout/secondary-controls.json) · [Lexical predictions and fit receipt](../../../examples/public_finance/results/fresh-sec-lexical/lexical-receipt.json)

The 374 issuers were absent from the original fitting, development, and test data. The follow-up uses the same SEC section labels and input masking with newly selected issuers. The initial 159-case pure QLoRA comparison remains a separate experiment with a 0.629-point directional gain. The FinQA and Treasury intervention observations use pure QLoRA and the initial frozen suite. [Fresh cohort source and reconstruction](../../../examples/public_finance/fresh-sec.md) · [First blind evaluation](../heldout-first/README.md)

## Reproduce the head and scores

From a checkout of v0.2.0, install the Apple Silicon dependencies and download the pinned base as shown in the [project setup](../../../README.md). Reconstruct the original finance panels and the fresh SEC cohort using the [data instructions](../../../examples/public_finance/fresh-sec.md). The SEC-only train and validation files provide the same section rows, in the same order, as the original combined training files.

```sh
finance_cases=.work/v2/public-finance/reconstructed
readout_plan=reports/constrained-v2/filing-readout-plan.json
python -m gradient_desk.filing_readout fit \
  --model models/Qwen3-8B-MLX-4bit \
  --train "$finance_cases/sec8k-train.jsonl" \
  --validation "$finance_cases/sec8k-valid.jsonl" \
  --selection reports/constrained-v2/campaign-warmup/selection.json \
  --plan "$readout_plan" --output .work/v2/filing-head-replay

python -m gradient_desk.filing_readout score \
  --model models/Qwen3-8B-MLX-4bit \
  --head .work/v2/filing-head-replay \
  --cases .work/v2/sec-fresh/sec8k-fresh-test.jsonl \
  --plan "$readout_plan" --output .work/v2/filing-head-score-replay
```

To score the exact released weights, use the extracted `gradient-desk-sec-filing-head` directory as `--head`. The scorer verifies the locked plan, case digest and weights before inference. [Release archive](https://github.com/Noptus/gradient-desk/releases/tag/v0.2.0)
