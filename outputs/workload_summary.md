# Phase 1 -- workload summary (live parse)

Source: `Inferact/codex_swebenchpro_traces` (610 trials, 20230 calls).
Tokens estimated at **4.065 chars/token** (no tokenizer
installed). This ratio is an **ANCHOR, not a validation**: it is fit so the mean
input equals the documented 68,329, so the Input-tokens row below is
tautological (✱) — do not read it as "reproduced".

| Metric (tokens/call unless noted) | Mean | P50 | P90 | P99 | Doc mean |
|---|---|---|---|---|---|
| LLM calls per trial | 33 | 30 | 57 | 90 | | doc 33
| Input tokens ✱(anchor) | 68,329 | 64,053 | 114,897 | 165,454 | | doc 68,329
| Cached tokens | 66,280 | 62,892 | 114,183 | 164,464 | | doc 64,338
| Computed uncached | 2,049 | 320 | 6,365 | 15,513 | | doc 3,991
| Output tokens | 512 | 242 | 1,114 | 4,766 | | doc 520

- **Cache hit rate:** 97.0%  (doc 94.2%)
- **Uncached fraction:** 3.0%  (doc 5.8%)
- **Input:output ratio:** 133.6:1  (doc 131:1)
- **Uncached heavy tail:** top 10% of calls carry 55.7% of all uncached compute.
- **Output heavy tail:** top 10% of calls carry 51.4% of all output tokens.

## Caveats
- The chars/token ratio is an anchor (see above). **Sensitivity:** holding it at
  a literature value of 4.0 instead of the fitted 4.065
  shifts mean input by only ~1.6%, so the magnitude is robust to the choice.
- **Cache-hit is OPTIMISTIC:** the segmentation assumes the entire previous
  request is a perfect cached prefix, giving 97.0%
  vs the doc's 94.2%. Real block-boundary rounding / eviction lowers it — read
  the hit rate as "~95% ± a few points", and `s_cached` as an upper bound.
- `inter_call_delay` is **NOT** in the public dataset (no timestamps); the
  plan's 10.5 s mean came from an internal trace. Kept as `null`.
- gpt turns are length-preserving Lorem-ipsum placeholders -> output token
  *counts* are meaningful, content is not.

## What actually validates the segmentation model
Input is an anchor (fit), so it proves nothing. The **non-calibrated** metrics —
**calls/trial (33), output (512 vs doc
520), uncached, and cache-hit** — are independent and land close to the dataset
card, which is the real corroboration that the per-call segmentation is right.
