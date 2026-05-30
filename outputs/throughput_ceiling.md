# Theoretical serving throughput ceiling (roofline)

Output tokens/sec ceiling = 1 / [ max(decode-KV/BW, decode-compute) + prefill-compute/O ], per GPU, eta=0.5, Codex/SWEBenchPro mean call. This is the best the hardware could do (batch-saturated, no SLO/HBM-cap). The model's operating points are shown as % of this ceiling.

## dense_70b
- **Theoretical ceiling: 150 out-tok/s per GPU** (10802 for 72 GPUs). Binding: **decode_KV_bandwidth**.
- Breakdown per output token: decode-KV 5.60ms, decode-compute 0.112ms, prefill-amortized 1.07ms.
- Admission-bound operating point (HBM-capped B=1): 25 tok/s = **16% of ceiling** — the headline regime is far below the wall (HBM capacity, not bandwidth, is what binds it).
- Decode-only rate at large batch (HBM cap + prefill IGNORED): 186 tok/s — note this *exceeds* the 150 unified ceiling precisely because it drops the prefill term: that is the `tps()` decode-only caveat made visible.

Raising the ceiling itself (not just utilization):
| lever | f | new ceiling tok/s/gpu |
|---|---|---|
| CPU offload f=0.0 | 0.0 | 150 |
| CPU offload f=0.3 | 0.3 | 201 |
| CPU offload f=0.5 | 0.5 | 259 |
| CPU offload f=0.9 | 0.9 | 614 |

## moe_large_mla
- **Theoretical ceiling: 354 out-tok/s per GPU** (25454 for 72 GPUs). Binding: **prefill_compute_amortized**.
- Breakdown per output token: decode-KV 1.20ms, decode-compute 0.059ms, prefill-amortized 1.63ms.
- Admission-bound operating point (HBM-capped B=1): 0 tok/s = **0% of ceiling** — the headline regime is far below the wall (HBM capacity, not bandwidth, is what binds it).
- Decode-only rate at large batch (HBM cap + prefill IGNORED): 826 tok/s — note this *exceeds* the 354 unified ceiling precisely because it drops the prefill term: that is the `tps()` decode-only caveat made visible.

Raising the ceiling itself (not just utilization):
| lever | f | new ceiling tok/s/gpu |
|---|---|---|
| CPU offload f=0.0 | 0.0 | 354 |
| CPU offload f=0.3 | 0.3 | 405 |
| CPU offload f=0.5 | 0.5 | 449 |
| CPU offload f=0.9 | 0.9 | 572 |

## Verdict: does the model reflect the theoretical upper bound?
- **Yes, the ceiling is encoded and reachable:** decode TPS converges to BW_HBM/KV as batch grows, and this module makes the unified prefill+decode ceiling explicit. Offload/MLA/sparsity RAISE the ceiling by cutting KV-per-token (see table); batch/CPUs only move you toward it.
- **Caveat the core metric:** `analytical.tps()` is decode-only (omits prefill), so for this 131:1 prefill-heavy workload it slightly OVERstates the per-token decode rate vs the unified ceiling here. Use `serving_roofline()` for the true bound and `utilization()` to place any operating point.
- The ceiling is the THEORETICAL bound at eta; real serving sits at eta*utilization below it (validation: eta~0.4, and admission/SLO keep utilization well under 100%).
