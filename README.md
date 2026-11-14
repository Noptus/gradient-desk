# Gradient Desk

A local QLoRA research coach that chooses bounded neural-training interventions from checkpoint telemetry and prioritizes the next step in fictional investment research reviews. Each training label comes from running the candidate actions from the same checkpoint and measuring their effect.

The coach outputs a single action for a researcher to inspect and test.

## Start with the learned CPU policy

In the primary comparison, the CPU telemetry and retrieval policies scored higher than the frozen base and mixed adapter across both tracks. They select neighborhood sizes on development data before applying the resulting policy to a new case.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[traces]'
mkdir -p data/generated
python -m gradient_desk.traces --output data/generated/traces.jsonl --seeds-per-case 12
gradient-desk prepare --traces data/generated/traces.jsonl --output data/generated/mixed
gradient-desk advise --policy learned \
  --train-cases data/generated/mixed/train-cases.jsonl \
  --valid-cases data/generated/mixed/valid-cases.jsonl \
  --telemetry data/telemetry.example.json
```

For a research proposal, use `--scenario 'Your fictional research case'` in place of `--telemetry`. The JSON output records the action, policy, and selected neighborhood size. The `learned-eval` report includes validation scores for the candidate sizes. Prepare once and reuse the train and validation manifests for later requests.

## Inspect the local QLoRA experiments

The [model card](MODEL_CARD.md) reports the mixed adapter and a second methodology-only run. On the 144 synthetic training cases the mixed adapter always selected `resume`. The separate industry-return benchmark exposed a large increase in regret that the synthetic average concealed.

### Try the experimental adapter

Requirements are Apple Silicon with Metal, Python 3.12, and about 5 GB of free disk space for the 4-bit base checkpoint. The reference machine is an Apple M4 with 16 GB of unified memory.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[mlx,traces,test]' -r requirements-apple.txt

gradient-desk download --output models/Qwen3-8B-MLX-4bit
```

Download `gradient-desk-mixed-step-180.tar.gz` from the [GitHub release](https://github.com/Noptus/gradient-desk/releases/tag/v0.1.0). Run a fictional research case through the packaged mixed adapter with:

```sh
mkdir -p adapters
tar -xzf gradient-desk-mixed-step-180.tar.gz -C adapters
gradient-desk advise \
  --model models/Qwen3-8B-MLX-4bit --adapter adapters/m4-mixed-step-180 \
  --scenario 'A signal uses the latest vintage of a series and credits it with the original release timestamp.'
```

Use `--telemetry path/to/pre-action.json` instead of `--scenario` to evaluate a training snapshot using [this example telemetry schema](data/telemetry.example.json). The same release ships mixed step 120 and methodology-only step 49 for the [matched-exposure ablation](METHODOLOGY-ABLATION.md).

## Reproduce the experiments

```sh
mkdir -p data/generated
python -m gradient_desk.traces --output data/generated/traces.jsonl --seeds-per-case 12
gradient-desk prepare --traces data/generated/traces.jsonl --output data/generated/mixed
HF_HUB_OFFLINE=1 python -u -m gradient_desk.train --config configs/m4-pilot.yaml

gradient-desk evaluate --cases data/generated/mixed/test-cases.jsonl \
  --model models/Qwen3-8B-MLX-4bit --output reports/base.jsonl
gradient-desk evaluate --cases data/generated/mixed/test-cases.jsonl \
  --model models/Qwen3-8B-MLX-4bit --adapter adapters/m4-pilot --output reports/adapter.jsonl
gradient-desk evaluate --cases data/generated/mixed/test-cases.jsonl \
  --baseline rules --output reports/rules.jsonl
gradient-desk evaluate --cases data/generated/mixed/test-cases.jsonl \
  --baseline resume --output reports/resume.jsonl
python -m pytest -q
```

The checkpoints, trace outputs, and prediction files contain no private market data. The downloader pins the publisher's [Qwen3-8B MLX 4-bit revision](https://huggingface.co/Qwen/Qwen3-8B-MLX-4bit/tree/383413e909f3bc5303ce195ebbdf0339c5a1a2a3). Training reads a verified local snapshot with model-side code loading disabled. A Mac that takes a long time to initialize binary dependencies from an external drive can use its internal disk for the Python environment while keeping models and working artifacts on the external drive.

The [reference protocol](EXPERIMENT.md) gives the validation-only checkpoint selection rule, the matched evaluation command, and the evidence required to interpret the scores.

## What the two tracks measure

**Neural training.** Each row starts from an identical model, optimizer, data cursor, and random state. Keep settings, reduce learning rate, clip global gradient norm, or accumulate gradients. Every branch sees the same ordered eight microbatches. Accumulation takes four optimizer updates and the other interventions take eight. The label reflects the smallest measured post-intervention validation loss, with a fixed near-tie threshold and first preference for resuming. Outcome values, durations, IDs, labels, and split names never enter the model prompt.

The fixtures use six controlled synthetic workload families and one small architecture. The test families are a tabular regime shift and cross-sectional ranking. Baselines and the adapter use identical test rows. Primary normalized regret is excess validation loss over the best measured action divided by the span between the best and worst actions in that case. Its range is 0 to 1. The scorer also records relative regret and its log transform so a few divergent runs cannot hide the typical case. An invalid model response scores as an exact-match miss, with `resume` used for its regret. Exact-match accuracy, family breakdowns and per-case predictions make the score inspectable.

**Investment research.** Fictional proposals test whether to audit data availability, repair validation, model costs, limit risk, specify a falsifiable thesis, or begin a small monitored pilot. The 64 cases cover 16 distinct scenario families assigned before tuning. Train, validation, and test contain 32, 16, and 16 cases respectively. The labels express a reviewer judgment about the most urgent research action. The initial labels were written with Claude Code and are published with their rationales for inspection. The score measures agreement with those recorded review judgments.

## License

Code, scenarios, and the distributed adapter use Apache-2.0. The Qwen team publishes the base checkpoint under Apache-2.0 and hosts its weights separately. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
