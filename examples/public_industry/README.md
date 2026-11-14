# External industry-return intervention benchmark

This benchmark tests whether a training coach transfers from synthetic tasks to a small neural network trained on historical U.S. industry portfolios. It measures training interventions on observed returns. It does not estimate trading performance and does not assume that an archived data vintage was available historically.

## Reproduce

Run from the Gradient Desk checkout. A CPU-only PyTorch run is sufficient.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[traces]'
mkdir -p .work/public-industry
curl --fail --location \
  'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp_202412/12_Industry_Portfolios_daily_CSV.zip' \
  --output .work/public-industry/12-industry-202412.zip
python examples/public_industry/benchmark.py \
  --archive .work/public-industry/12-industry-202412.zip \
  --output .work/public-industry/verified-cases.jsonl
python -m unittest discover -s examples/public_industry -v
gradient-desk evaluate --cases .work/public-industry/verified-cases.jsonl \
  --baseline resume --output .work/public-industry/resume.jsonl
gradient-desk evaluate --cases .work/public-industry/verified-cases.jsonl \
  --baseline rules --output .work/public-industry/rules.jsonl
```

The versioned archive is checked against its expected SHA-256 before decoding. The generated JSONL is an evaluation-only set with `track=training` and `split=test`. Use it as the case input for rules, resume, base, or adapter evaluation. Do not use these rows for training or model selection. Pass `--extra-cases .work/public-industry/verified-cases.jsonl` to `paired-eval` to evaluate base and adapter on this panel separately. The raw archive and full per-row source-derived trace outputs are local. [Run receipt](run-receipt.json).

## Design

A row contains one industry’s returns at the close of session *t*, then its moving mean over 5 and 20 sessions and moving standard deviation over 20 sessions. The remaining inputs are the equal-weight mean across industries at *t* and the industry’s return minus that mean. The label is the same industry’s value-weighted return for the next observed session. Both are computed from the archived value-weighted table. Targets stay in percentage points.

Each case starts with a fixed fit-only normalization window. A separate, earlier monitoring window provides the pre-action loss history from repeated evaluations during warmup. Warmup then branches into exactly eight later date-ordered microbatches, one 12-industry cross-section per microbatch. A subsequent 16-session period measures terminal mean squared error. A trading-session embargo between phases prevents a label from overlapping the next phase’s first feature date.

Every action restores the same model, optimizer and RNG state, sees the same date-indexed samples in order, and makes eight forward/backward calls. Accumulation makes four updates instead of eight. The four learning rates share a seed and fit windows within each of six non-overlapping historical blocks. The anchor dates and learning rates were chosen before inspecting the intervention outcomes or model policies. The dates are convenience selections spanning several years rather than a random or regime-stratified sample. Each anchor starts at the first complete feature date on or after the listed date. A strict block non-overlap assertion and complete-session check run before any training. Treat the six blocks as the sampling units and the results as descriptive.

## CPU pilot

The local run produced 24 cases and 96 measured continuations. A second run matched the first run’s measurement hash after excluding wall-clock timing. Raw output file hashes can differ across reruns because elapsed time is included. Exact floating-point reproduction was checked on the recorded PyTorch version and machine.

| Fixed action | Accuracy | Mean normalized regret |
| --- | ---: | ---: |
| resume | 12.5% | 0.7241 |
| lower_lr | 79.2% | 0.0421 |
| clip_grad | 0% | 0.5050 |
| accumulate | 8.3% | 0.1486 |

The 19-of-24 concentration in `lower_lr` makes the ex post best fixed action a strong reference. All four fixed policies are shown because comparison only with resume can exaggerate the value of a learned policy on this panel. Outcomes reflect this model, schedule, archive and set of windows. Historical returns may contain revisions.

The source is the Kenneth R. French Data Library’s December 2024 CRSP vintage. The published example code has no third-party data bundled. [Portfolio definitions](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_12_ind_port.html) · [Archived data library](https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/data_library_202412_archive.html).
