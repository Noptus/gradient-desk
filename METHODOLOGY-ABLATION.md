# Matched-exposure methodology ablation

The mixed adapter learns neural interventions and investment research methodology together. This ablation asks whether the methodology scores change when the neural training examples are removed while the methodology examples and their order stay fixed.

The control is the mixed adapter at step **120**. The intervention is methodology-only QLoRA at step **49**. Both start from the same pinned Qwen3-8B, adapter seed, 4-bit checkpoint, rank, learning rate and batch size. The actual pinned MLX-LM 0.31.3 iterator yields 49 methodology exposures and 71 neural examples in the first 120 mixed updates. The driver verifies that a permuted input file reproduces exactly those same 49 method records in the same order on the ablated run. The exposure IDs, original update positions, and order digest are frozen in [the exposure plan](reports/methodology-exposure-plan.json).

The two fixed checkpoints are compared on all 16 synthetic methodology test cases and all eight case memos derived from public sources. Checkpoint 49 is fixed in advance. The reports also compare with the untouched base and with the validation-selected reference adapter. All models use the same prompt, greedy decode, response cap and scoring rules as the primary test. Report exact and macro accuracy, unsupported responses, and paired wins and losses, with raw predictions.

Removing 71 examples changes the optimizer clock and update history. The resulting comparison identifies the effect of deleting the particular auxiliary examples from this trajectory. It does not identify a universal interference mechanism, an optimal dataset mixture, or a stable difference in investment skill.

## Reproduce

Run the reference setup and mixed training from [the main protocol](EXPERIMENT.md), including its matched test with the published memo panel. Then run:

```sh
python -m gradient_desk.methodology_ablation prepare \
  --mixed-data data/generated/mixed --output data/generated/methodology-ablation
HF_HUB_OFFLINE=1 python -m gradient_desk.train --config configs/m4-methodology.yaml

python -m gradient_desk.methodology_ablation score \
  --model models/Qwen3-8B-MLX-4bit --mixed-adapter adapters/m4-pilot \
  --ablation-adapter adapters/m4-methodology --main-report reports/paired \
  --panel data/generated/mixed/test-cases.jsonl \
  --panel examples/research_memos/frozen.jsonl --output reports/methodology-ablation
```

The reference evaluator and ablation command match external reports by input basename. Use the same memo filename for both commands.
