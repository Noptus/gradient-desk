# Reference experiment protocol

This protocol freezes the reference comparison before any base or tuned LLM scores on the test cases. Small public panels are being used for a reproducible pilot.

## Inputs and checkpoints

- Base model: [Qwen3-8B-MLX-4bit](https://huggingface.co/Qwen/Qwen3-8B-MLX-4bit/tree/383413e909f3bc5303ce195ebbdf0339c5a1a2a3), pinned publisher revision.
- Run: [M4 configuration](configs/m4-pilot.yaml), 180 supervised LoRA updates, 8 final transformer layers, rank 8. Qwen thinking mode is disabled in training and evaluation. Only the completion contributes to the training loss.
- Training: 144 measured neural intervention cases and 32 methodology cases, with the methodology cases represented three times each. Each of six synthetic workload families and 16 fictional methodology families belongs to exactly one split.
- Primary synthetic test: all 144 measured neural cases and all 16 held-out methodology cases. The published investment methodology labels are reviewer judgments.
- External tests: same frozen models and scoring on the independent industry-return replay and sourced research memo panels. Each panel is reported separately. These add realistic inputs and do not establish investment performance.
- Untuned controls: always resume, prespecified telemetry rules, and the identical base model and prompt. An additional CPU policy selects neighborhood sizes on validation only, using measured losses for neural cases and lexical TF-IDF for methodology cases.

## Checkpoint selection

The candidate steps are **60, 120, and 180**. The fixed developer panel contains all 16 validation methodology cases plus the 12 training-diagnosis validation cases with seed index 0, one per learning-rate and batch profile in each validation family. The criterion averages two quantities on this panel: one minus mean normalized neural regret, and methodology macro accuracy. Ties select the earlier checkpoint. Candidate scores and the selected step will be published, along with all 180 training steps. The final-step adapter is a distinct checkpoint when a previous step wins.

Periodic MLX training logs contain completion-token cross-entropy from 16 draws of the validation loader at each interval. Those draws can differ across intervals. Selection uses the fixed panel and decision metrics above.

## Scoring

The two model conditions get identical prompts, deterministic greedy decoding, a 16-token response cap, and the same inference batch size. Paired evaluation checks scalar-versus-batched generation on a small fixed subset before using batched inference. Scoring counts an unsupported output as a classification error and uses resume for its counterfactual loss. The scorer reports exact-match and macro accuracy, valid response rates, and action counts.

For neural training the primary endpoint is **mean normalized regret at a fixed exposure budget**. For a case, this is (chosen loss minus best loss) divided by (worst loss minus best loss). All actions consume the same future examples and forward/backward microbatches. Accumulation uses fewer optimizer updates; clipping adds gradient work. A zero span gives zero regret. We retain mean and median relative regret and mean log-relative regret as sensitivity metrics. Resampling paired seed blocks within a synthetic family describes sampling uncertainty for that family. It does not support an inference about the population of workload families. External market snapshots are grouped by anchor date and small panels are descriptive.

The experiment measures a one-step intervention recommendation from a bounded menu. It does not establish that an LLM improves full training runs, profit, overall investment judgment, or generalizes to a production market dataset.

## Reproduce the fixed selection panel and paired test

After preparing trace data, the following command makes the exact selection panel:

```sh
python - <<'PY'
from pathlib import Path
from gradient_desk.dataset import read_jsonl, write_jsonl

root = Path("data/generated/mixed")
rows = read_jsonl(root / "valid-cases.jsonl")
panel = [r for r in rows if r["track"] == "methodology" or r.get("seed_index") == 0]
assert len(panel) == 28
write_jsonl(root / "selection.jsonl", panel)
PY

gradient-desk paired-eval --cases data/generated/mixed/test-cases.jsonl \
  --validation-cases data/generated/mixed/selection.jsonl --checkpoints 60 120 180 \
  --model models/Qwen3-8B-MLX-4bit --adapter adapters/m4-pilot \
  --batch-size 4 --output reports/paired
```

Pass `--extra-cases` once per external panel to score them during the same model load. The full A/B output contains raw labels and scorer totals. The batch runtime is an amortized per-case wall-time estimate and should not be treated as latency for an individual request.
