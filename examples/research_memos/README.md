# Paper-grounded research review

Eight original evaluation scenarios translate methodological issues from public finance papers into decisions a researcher could face. Each memo links the primary source and records an independently reviewable label and rationale. The actual papers supply context. They make no claims about these fictional case results.

This external panel was written after the adapter's train and validation datasets were frozen and before any model predictions on the panel. The prompts include only the scenario and shared action menu. Paper links, rationales, and labels remain in the evaluation metadata. This small authored panel measures priority agreement with a reviewer. It cannot establish investment returns or provide an independently adjudicated human benchmark.

```sh
python examples/research_memos/prepare.py data/generated/research-memos.jsonl
gradient-desk paired-eval --cases data/generated/research-memos.jsonl \
  --model models/Qwen3-8B-MLX-4bit --adapter adapters/m4-pilot \
  --batch-size 4 --output reports/research-memos
```

An independent Claude Opus 4.8 editorial pass agreed with all eight proposed labels. It flagged the vintage question as having a plausible second answer because timing errors invalidate the backtest. This is a model-based audit of the authored panel.
