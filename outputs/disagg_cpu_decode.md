# Prefill-on-GPU / decode-on-CPU disaggregation

out_tps = O * min(GPU prefill feed, CPU decode capacity). The GPU runs at its prefill roofline while the CPU pool decodes in parallel. eta=0.5, SLO TPOT<=50ms, Codex/SWEBenchPro mean call.

## dense_70b
- Baseline (GPU does all, admission-bound small batch): **24 out-tok/s** — the GPU wastes most time on memory-bound single-stream decode.

| cpus/gpu | out tok/s | gain | CPU batch | CPU TPOT (ms) | bottleneck |
|---|---|---|---|---|---|
| 1 | 0 | 0.00x | - | - | cpu_infeasible(SLO) |
| 2 | 0 | 0.00x | - | - | cpu_infeasible(SLO) |
| 4 | 0 | 0.00x | - | - | cpu_infeasible(SLO) |
| 8 | 0 | 0.00x | - | - | cpu_infeasible(SLO) |
| 16 | 22 | 0.91x | 1 | 45.5 | cpu_decode |
| 32 | 104 | 4.29x | 4 | 38.6 | cpu_decode |
| 64 | 207 | 8.57x | 4 | 19.3 | cpu_decode |
| 128 | 415 | 17.14x | 4 | 9.6 | cpu_decode |
| 256 | 829 | 34.29x | 4 | 4.8 | cpu_decode |

## moe_large_mla
- Baseline (GPU does all, admission-bound small batch): **47 out-tok/s** — the GPU wastes most time on memory-bound single-stream decode.

| cpus/gpu | out tok/s | gain | CPU batch | CPU TPOT (ms) | bottleneck |
|---|---|---|---|---|---|
| 1 | 0 | 0.00x | - | - | cpu_infeasible(SLO) |
| 2 | 0 | 0.00x | - | - | cpu_infeasible(SLO) |
| 4 | 0 | 0.00x | - | - | cpu_infeasible(SLO) |
| 8 | 0 | 0.00x | - | - | cpu_infeasible(SLO) |
| 16 | 29 | 0.61x | 1 | 34.6 | cpu_decode |
| 32 | 63 | 1.34x | 2 | 31.8 | cpu_decode |
| 64 | 133 | 2.83x | 4 | 30.0 | cpu_decode |
| 128 | 282 | 5.99x | 8 | 28.4 | cpu_decode |
| 256 | 614 | 13.06x | 32 | 47.5 | gpu_prefill |

## Why there is a gain — three compounding effects
1. **GPU runs in parallel on prefill.** While the CPU decodes session A, the GPU prefills session B. The GPU stops wasting ~all its time on slow memory-bound B~1 decode and runs at its compute-bound prefill roofline.
2. **KV leaves HBM.** Decode KV now lives in CPU DRAM, so the GPU's HBM holds only weights + transient prefill working set — the decode-KV *admission stall* that capped the baseline at batch~1 disappears, and the prefill feed becomes purely compute-bound (the ceiling rows above).
3. **CPU can batch decode cheaply.** Because KV fits in big, cheap CPU DRAM, the CPU pool decodes a large batch; its limit is CPU *bandwidth* (weight + KV streaming under the SLO), not capacity.

## When (the crossover)
- **SLO gate:** CPU decode TPOT ~= weight_bytes/(cpus x BW_cpu) <= SLO. Large models need many CPUs (dense-70B: SLO-infeasible below ~16 CPUs).
- **Attention gate:** at long context, CPU decode batch is capped by KV streaming (B x KV/(cpus x BW_cpu) <= SLO), limiting capacity below the weight-only estimate.
- **Balance:** out_tps = O x min(GPU prefill feed, CPU capacity); gain saturates at the GPU prefill ceiling once the CPU pool is big enough.

**Bottom line:** gain starts around **~32 Grace-class CPUs/GPU** under a 50ms SLO and grows to **~13–34x** the admission-bound baseline as the CPU pool approaches the GPU prefill feed. Below ~16 CPUs it is SLO-infeasible. This is the FastDecode 'aggregate many CPUs' regime — practical with CPU-heavy nodes, or with **quantization / small / low-active models**, which slash the required CPU count (the gate is weight-bytes / bandwidth). All numbers are optimistic-overlap, no inter-node fabric cost — ceilings.
