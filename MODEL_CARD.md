# Gradient Desk model card

## Version 0.2.0

Gradient Desk uses the pinned `Qwen/Qwen3-8B-MLX-4bit` snapshot. The checkpoint advisor scores four bounded optimizer actions from pre-action telemetry. The SEC classifier uses an eight-class linear readout fitted on average normalized Qwen hidden states. Both return distributions over fixed candidate menus. Their recorded probabilities depend on the input and answer layout.

The final LoRA run used 180 Adam updates with a 20-step learning-rate ramp to 0.00003. It took 29 minutes 59 seconds at 12.58 GB peak MLX allocation on an Apple M4 with 16 GB. An eligibility-first rule was adopted after reviewing the development results and published before initial test inference. Step 180 was the only checkpoint that improved all three development sources at the registered threshold. The [training protocol](reports/constrained-v2/README.md) preserves the original failed run, the mean-first checkpoint and the residual blend experiment.

In the 637-case blind trial, LoRA improved FinQA operator accuracy by 2.91 points and reduced Treasury intervention regret by 0.0799 against frozen Qwen. The Treasury paired 95% interval for the regret gain was 0.0161 to 0.1477, with Holm-adjusted p of 0.0858. The original five comparisons fell short of the confirmation rule. Taiwan bankruptcy intervention regret increased from 0.4244 to 0.5273 across 28 source-row groups. The exact calculator identified all 150 Treasury audit cases, and the CPU telemetry policy achieved 0.1321 regret on the Treasury intervention panel. The [complete matched report](reports/constrained-v2/heldout-first/README.md) gives all five endpoints and the controls.

The SEC follow-up fitted the head on 320 original filing sections and selected its hyperparameters on 32 original development cases. On 374 fresh sections from distinct issuers, accuracy improved from 67.91% to 78.88%, with a paired 95% interval of 6.42 to 15.51 points for the gain and preregistered p of 0.0001666. The same-filing TF-IDF baseline got 78.61%. The [SEC follow-up report](reports/constrained-v2/filing-readout/README.md) includes the source cutoffs, paired predictions and fit receipt.

## Version 0.1.0 reference run

The Apple M4 host had 16 GB of unified memory. The 4-bit base weights, datasets, checkpoints and evidence were kept on Tsunami. An internal temporary Python environment avoided initialization delays from binary packages on the external disk. The run adapted 4.85 million parameters in the last eight transformer layers. Wall time is measured after base load, includes validation and checkpoint writes, and is separate from GPU utilization. From prepared mixed-training file to final checkpoint took 1106s in this run.

| Run | Updates | Wall after load | Peak MLX allocation |
| --- | --- | --- | --- |
| Mixed | 180 | 1004s | 5.57 GB |
| Methodology-only | 49 | 145s | 5.20 GB |

The full mixed run reached 180 updates. The published primary candidate is **step 180**. A fixed 28-case development panel chose it before any held-out model scores. The same no-thinking prompt, 16-token cap and greedy decoder were used everywhere. Batch size 4 passed the scalar-versus-batched check on 4 development cases. The evaluator also restored the exact original module graph after candidate selection before scoring the frozen base.

| Mixed step | Neural mean regret ↓ | Methodology macro | Selection score ↑ |
| --- | --- | --- | --- |
| 60 | 0.547 | 22.2% | 0.338 |
| 120 | 0.380 | 22.2% | 0.421 |
| 180 | 0.380 | 61.1% | 0.615 |

## Held-out results

The paired protocol is in [EXPERIMENT.md](EXPERIMENT.md). Probe data and family breakdowns are linked in the raw result folders. Mean normalized training regret ranges from 0 to 1, with lower values indicating better outcomes. An invalid model response counts as an exact-match miss and uses resume for outcome scoring. Macro accuracy gives each observed action label equal weight.


### Synthetic neural training

144 measured counterfactuals across two held-out workload families.

| Policy | Exact action | Macro accuracy | Valid output | Mean normalized regret ↓ |
| --- | --- | --- | --- | --- |
| Qwen3-8B base | 50/144 (34.7%) | 19.5% | 100.0% | 0.468 |
| Mixed LoRA, step 180 | 70/144 (48.6%) | 25.0% | 100.0% | 0.366 |
| Always resume | 70/144 (48.6%) | 25.0% | 100.0% | 0.366 |
| Telemetry rules | 43/144 (29.9%) | 20.3% | 100.0% | 0.425 |
| Learned telemetry policy | 87/144 (60.4%) | 33.3% | 100.0% | 0.285 |

Exact-action change from base is +13.9 points. Mean normalized regret changes by -0.102.


### Historical industry-return training

24 measured cases across six historical anchors. This is an archival stress replay.

| Policy | Exact action | Macro accuracy | Valid output | Mean normalized regret ↓ |
| --- | --- | --- | --- | --- |
| Qwen3-8B base | 6/24 (25.0%) | 10.5% | 100.0% | 0.278 |
| Mixed LoRA, step 180 | 3/24 (12.5%) | 33.3% | 100.0% | 0.724 |
| Learned telemetry policy | 20/24 (83.3%) | 44.4% | 100.0% | 0.028 |

Exact-action change from base is -12.5 points. Mean normalized regret changes by +0.446. The best fixed action identified on this test panel, `lower_lr`, got 19/24 and 0.042 regret. The [industry-replay receipt](examples/public_industry/) shows all four fixed actions. The learned policy selected its settings on separate development data.


### Fictional methodology

16 scenarios from four disjoint families.

| Policy | Exact action | Macro accuracy | Valid output |
| --- | --- | --- | --- |
| Qwen3-8B base | 7/16 (43.8%) | 47.2% | 100.0% |
| Mixed LoRA, step 180 | 8/16 (50.0%) | 50.0% | 100.0% |
| TF-IDF retrieval | 13/16 (81.25%) | 77.8% | 100.0% |

Exact-action change from base is +6.2 points.


### Public-source methodology memos

Eight sourced cases. Labels describe reviewer judgments.

| Policy | Exact action | Macro accuracy | Valid output |
| --- | --- | --- | --- |
| Qwen3-8B base | 4/8 (50.0%) | 58.3% | 100.0% |
| Mixed LoRA, step 180 | 4/8 (50.0%) | 50.0% | 100.0% |
| TF-IDF retrieval | 6/8 (75.0%) | 83.3% | 100.0% |

Exact-action change from base is +0.0 points.


The mixed adapter emitted `resume` on all 144 synthetic neural cases and all 24 historical industry-return cases. Its lower synthetic regret versus the base reflects that default action. On the industry panel, the same behavior increased regret from 0.278 to 0.724. The learned CPU telemetry policy attained 0.285 and 0.028 respectively. TF-IDF retrieval led both methodology panels in the primary comparison. For exploring similar decisions locally, the [learned policy CLI](README.md#start-with-the-learned-cpu-policy) is the best tested starting point.

## Matched-exposure methodology ablation

The methodology-only adapter replays the 49 methodology examples used in the first 120 mixed updates, in the same order. The comparison drops 71 neural examples, changing the optimizer step count and trajectory. Checkpoint 49 and the mixed comparator at step 120 were fixed in advance.

| Panel | Cases | Mixed exact | Methods-only exact | Mixed macro | Methods-only macro |
| --- | --- | --- | --- | --- | --- |
| Fictional methodology | 16 | 37.5% | 12.5% | 33.3% | 16.7% |
| Public-source memos | 8 | 62.5% | 12.5% | 66.7% | 16.7% |

The methodology-only adapter got 2 of 16 fictional cases and 1 of 8 sourced memos right. The exposure-matched mixed comparator got 6 of 16 and 5 of 8. The specialist also produced one unsupported action on the memo panel. Removing neural examples changed outcomes in the opposite direction to the simplest interference hypothesis.

These small panels characterize this run, and differences of one or two predictions should be read as counts. The benchmark scope is checkpoint decisions and prioritization of proposed research steps. Deployment evidence would require independent architectures, objectives and budgets, plus evaluation on real prospective training traces.

## Artifacts

- [Public release archives](https://github.com/Noptus/gradient-desk/releases/tag/v0.1.0)
- [Training receipt](reports/mixed-run-receipt.json), [secondary receipt](reports/methodology-run-receipt.json), [input digests](reports/reference-inputs.json)
- [Primary predictions and paired comparisons](reports/paired), [label confusions](reports/label-confusion.json), [matched-exposure evidence](reports/methodology-ablation)
- [Reference software and source provenance](NOTICE)

The raw industry return archive remains with its original publisher. The public example downloads the pinned archive and rebuilds the measured case data locally. Optimizer decisions and finite loss comparisons are reproducible under the reference stack. Raw clocks and file hashes that include timing can differ across replays.
