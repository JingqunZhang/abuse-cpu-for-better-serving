> **⚠️ SUPERSEDED** — this report's numbers (e.g. ~3.3×) predate the
> CPU-attention-compute cost and the rate-path overlap fixes. Current
> authoritative result: **`concurrent_mix.md`** (and `SUMMARY.md`). Kept for
> history; see `README.md` Outputs map and `model_revisions.md`.

# Verdict: does CPU offloading core attention help SYSTEM throughput?

Using the two-resource (GPU compute ∥ HBM-bandwidth) contention model
(`model/contention.py`), prefill-heavy mean call, dense-70B, eta=0.5. Best f per
config; gain = best-f throughput / f=0 throughput.

| cpus/gpu | sparse | compress | best f | **gain** | bottleneck@best |
|---|---|---|---|---|---|
| 0.5 (NVL72) | dense | none | 0.05 | **1.05×** | gpu_hbm |
| 0.5 (NVL72) | dense | int4 | 0.05 | 1.05× | gpu_hbm |
| 0.5 (NVL72) | 10% | none | 0.50 | **1.45×** | cpu_dram |
| 0.5 (NVL72) | 10% | int4 | 0.30 | 1.42× | gpu_hbm |
| 1 | dense | none | 0.20 | **1.25×** | gpu_hbm |
| 1 | 10% | none | 0.70 | **3.31×** | gpu_hbm |
| 2 | dense | none | 0.50 | 1.99× | gpu_hbm |
| ≥4 | any | any | 0.70 | **~3.3×** | gpu_hbm |

## Verdict: YES — in the prefill-heavy / HBM-bandwidth-bound regime, it helps.

This **updates** the earlier "no help on NVL72" conclusion, which was based on a
model that did NOT account for prefill/decode contention on HBM bandwidth. Once
that contention is modeled, the picture changes:

- **Why it helps:** at f=0 the GPU is **HBM-bandwidth-bound** (decode KV + weight
  streaming saturate HBM; GPU compute ~idle, so prefill is starved). Offloading
  core attention moves KV streaming to CPU DRAM, **freeing HBM bandwidth +
  capacity for prefill** — the benefit is relief, not CPU speed.
- **Stock NVL72 (0.5 Grace/GPU):** modest with dense attention (~**1.05×**, small
  feasible f before CPU DRAM becomes the new bottleneck), but **~1.45× with
  sparse (10%) attention** — because sparse shrinks the offloaded bytes so the
  weak CPU bandwidth keeps up.
- **More CPU (≥1–2 Grace/GPU) or sparse/compressed KV:** up to **~2–3.3×**, with
  the bottleneck staying on the (relieved) HBM rather than CPU DRAM.

## The crossover (when does it help / stop helping)
Offload helps while the offloaded KV stream fits CPU bandwidth faster than it was
clogging HBM:  `f·sparse·compress·KV / BW_CPU_pool  <  gpu_hbm_freed`. So:
- **too little CPU bandwidth** (stock NVL72, dense) → CPU DRAM becomes the
  bottleneck almost immediately → only a small f helps (~1.05×);
- **sparse/compressed KV** shrinks the offloaded bytes → CPU keeps up → larger f,
  bigger gain even on 0.5 Grace;
- **more CPU bandwidth** → bottleneck stays on HBM → gain rises to the
  HBM-relief ceiling (~3.3× here, set by the residual weight-streaming floor).

## Honest caveats
- This is the prefill-heavy, **admission-bound (small batch)** regime where the
  GPU is genuinely HBM-bound at f=0. If the GPU could already batch decode large
  (KV fit in HBM), it would not be HBM-starved and the benefit shrinks.
- Two-resource fluid model, optimistic overlap, no inter-node fabric, mean call.
- Gains assume the CPU attention overlaps GPU prefill (ScoutAttention layer-ahead
  style); under no overlap they shrink (cf. the opt..cons band elsewhere).
- Sparse attention carries a ~2–2.5% accuracy cost (not modeled).
