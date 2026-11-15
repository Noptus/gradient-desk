# First frozen QLoRA evaluation

A single development-selected warmup checkpoint and the frozen Qwen3 base scored 637 cases under identical constraints. The selection used the eligibility ordering documented before the first transformer test. The previous release scored the same cases as a secondary comparator. [Selection receipt](../campaign-warmup/holdout-selection.json) · [Test protocol](../protocol.json)

| Panel | Cases / blocks | Metric | Base | QLoRA | Gain | Holm p |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| FinQA operation | 103 / 15 | Accuracy | 86.408% | 89.320% | +2.913 pp | 0.500 |
| SEC Form 8-K | 159 / 136 | Accuracy | 70.440% | 71.069% | +0.629 pp | 1.000 |
| Treasury claim audit | 150 / 60 | Accuracy | 18.667% | 18.667% | 0.000 pp | 1.000 |
| Treasury yield MLP | 85 / 17 | Normalized regret | 0.46689 | 0.38698 | +0.07992 | 0.086 |
| Taiwanese bankruptcy MLP | 140 / 28 | Normalized regret | 0.42441 | 0.52726 | −0.10285 | 1.000 |

FinQA and Treasury met the 0.02 observed-gain threshold. Zero panels met both confirmation criteria. The Treasury intervention's paired block interval on gain was [0.01612, 0.14771], and the Taiwan transfer panel's was [−0.16501, −0.04500]. The positive direction indicates higher accuracy or lower regret. [Paired report and intervals](paired-report.json)

The lexical control scored 86.164% on SEC, and the exact Treasury checker scored all 150 claims. A telemetry nearest-neighbor policy produced 0.13206 normalized regret on the Treasury intervention. Fixed resume produced 0.33690 on Taiwan. The preceding released adapter produced respective exact accuracies of 88.350% on FinQA, 69.811% on SEC, and 16.667% on the claim audit, and normalized regrets of 0.52316 on Treasury and 0.40100 on Taiwan. [Case-aligned CPU control comparisons](secondary-controls.json) · [Historical model result](paired-report.json)

All scored distributions and case IDs are available for reconciliation. The direct and cached computations agreed on the mixed-length parity probes for all three models, with zero probability discrepancy. The protocol retains aggregate source definitions, dependence units, device measurements, and exact file digests. [Frozen base](base.jsonl) · [Warmup QLoRA](tuned.jsonl) · [Previous release](prior.jsonl)
