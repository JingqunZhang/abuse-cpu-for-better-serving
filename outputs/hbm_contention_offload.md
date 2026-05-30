# Prefill/decode HBM-bandwidth contention + core-attention offload

> **⚠️ SUPERSEDED** — `serving_2res` is the single-mean-call precursor to the concurrent fluid model. Current authoritative result: `concurrent_mix.md` (and `SUMMARY.md`). Kept as a building-block cross-check; see `README.md` Outputs map.

GPU = two resources (compute F_G ∥ HBM bandwidth). Prefill is compute-bound; decode core-attention is HBM-bandwidth-bound. Offloading fraction f of core attention to CPU removes its KV streaming from HBM, freeing HBM bandwidth for prefill. dense-70B, eta=0.5.

## NVL72 stock (0.5 Grace/GPU)
| f (offload) | system tok/s | bottleneck | GPU compute (ms) | GPU HBM (ms) | CPU DRAM (ms) | C2C (ms) |
|---|---|---|---|---|---|---|
| 0.0 | 39 (1.00x) | gpu_hbm | 505 | 13493 | 0 | 0 |
| 0.1 | 28 (0.72x) | cpu_dram | 505 | 12148 | 18629 | 1 |
| 0.2 | 14 (0.36x) | cpu_dram | 505 | 10802 | 37257 | 1 |
| 0.3 | 9 (0.24x) | cpu_dram | 505 | 9456 | 55886 | 2 |
| 0.5 | 6 (0.14x) | cpu_dram | 505 | 6764 | 93143 | 3 |

## 4 Grace/GPU
| f (offload) | system tok/s | bottleneck | GPU compute (ms) | GPU HBM (ms) | CPU DRAM (ms) | C2C (ms) |
|---|---|---|---|---|---|---|
| 0.0 | 39 (1.00x) | gpu_hbm | 505 | 13493 | 0 | 0 |
| 0.1 | 43 (1.11x) | gpu_hbm | 505 | 12148 | 291 | 1 |
| 0.2 | 48 (1.25x) | gpu_hbm | 505 | 10802 | 582 | 1 |
| 0.3 | 55 (1.43x) | gpu_hbm | 505 | 9456 | 873 | 2 |
| 0.5 | 77 (1.99x) | gpu_hbm | 505 | 6764 | 1455 | 3 |

## Reading it
- At **f=0** the GPU is **HBM-bandwidth bound** (gpu_hbm >> gpu_compute): decode KV streaming saturates HBM, GPU compute idles -> prefill starved.
- Raising **f** moves KV streaming off HBM: gpu_hbm drops, throughput rises UNTIL the bottleneck moves to **CPU DRAM** (or C2C) -- exactly the regime you described. The optimum f is where gpu_hbm ≈ max(gpu_compute, cpu_dram).
- On stock NVL72 (0.5 Grace) the CPU bottleneck appears almost immediately (tiny CPU bandwidth), so the feasible f is small; more CPU (or sparse/compressed KV to shrink the offloaded bytes) widens it.
- **This is a real win mechanism the time-share models understated**: the benefit is freeing HBM BANDWIDTH for prefill, not making attention faster.
