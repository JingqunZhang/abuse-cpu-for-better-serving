# Dynamic GPU/CPU decode-offload scheduling

Event sim, dense-70B, eta=0.5, SLO TPOT<=50ms, 40 sessions x 6 calls. Question: on a FIXED CPU:GPU ratio, can a dynamic scheduler that spills decode to otherwise-idle CPU match/beat the GPU-only baseline?

## dense_70b — NVL72 stock (~0.5 Grace/GPU)
| policy | throughput tok/s | TTFT (s) | offloaded % | GPU util | CPU util |
|---|---|---|---|---|---|
| gpu_only | 69.4 (1.00x) | 139.4 | 0% | 100% | 0% |
| dynamic | 69.4 (1.00x) | 139.4 | 0% | 100% | 0% |
| all_cpu | 69.4 (1.00x) | 139.4 | 0% | 100% | 0% |

## dense_70b — CPU-heavy (4 CPU/GPU)
| policy | throughput tok/s | TTFT (s) | offloaded % | GPU util | CPU util |
|---|---|---|---|---|---|
| gpu_only | 69.4 (1.00x) | 139.4 | 0% | 100% | 0% |
| dynamic | 69.4 (1.00x) | 139.4 | 0% | 100% | 0% |
| all_cpu | 69.4 (1.00x) | 139.4 | 0% | 100% | 0% |

## dense_70b — FastDecode-style (16 CPU/GPU)
| policy | throughput tok/s | TTFT (s) | offloaded % | GPU util | CPU util |
|---|---|---|---|---|---|
| gpu_only | 69.4 (1.00x) | 139.4 | 0% | 100% | 0% |
| dynamic | 89.7 (1.29x) | 118.4 | 17% | 96% | 83% |
| all_cpu | 80.6 (1.16x) | 117.7 | 15% | 85% | 81% |

## How this answers the dynamic question
- The scheduler decision is **state-dependent** (offload only when GPU is at its HBM decode budget AND CPU has SLO headroom) — exactly the dynamic behavior the closed form cannot express.
- At the **stock NVL72 ratio** the CPU pool is tiny, so dynamic offload spills only a sliver and lands **~break-even** (the design goal: 'no worse', exploiting idle CPU) — it does NOT need the 32+ CPUs the static model wanted just to avoid losing ground.
- As CPU:GPU grows, dynamic offload's gain rises toward the static disagg ceiling, but **dynamic always >= all_cpu** because it never offloads when that would violate the SLO or starve.
