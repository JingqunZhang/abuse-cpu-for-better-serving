# Phase 5 -- parallelism: recommended NVL72 configs

Model: 671B MoE (MLA KV), workload = Codex/SWEBenchPro mean call, eta=0.5, SLO TPOT<=50ms / TTFT<=10s. System throughput = DP x (B_d / TPOT). Offload = sparse CPU co-attention f=0.3.

**All tok/s are reported as an OPTIMISTIC..CONSERVATIVE overlap band**; and they omit real inter-node fabric cost, so treat them as ceilings (plausibly ~2-4x above a production stack).

## Ranked feasible deployments (by optimistic system tokens/s)

| rank | layout | GPUs | offload | B_d/replica | TPOT (ms) | system tok/s (opt..cons) |
|---|---|---|---|---|---|---|
| 1 | 4x18 (TP6,PP3) | 72 | f=0.3 | 512 | 43.0 | 25049..20517 |
| 2 | 4x18 (TP6,PP3) | 72 | none | 384 | 44.7 | 20823..20823 |
| 3 | 2x36 (TP8,PP4) | 64 | f=0.3 | 512 | 24.3 | 19186..16037 |
| 4 | EP 2x32 (TP8,PP4,EP8) | 64 | f=0.3 | 512 | 24.5 | 19108..15983 |
| 5 | 2x36 (TP8,PP4) | 64 | none | 512 | 30.1 | 17308..17308 |
| 6 | EP 2x32 (TP8,PP4,EP8) | 64 | none | 512 | 30.3 | 17244..17244 |
| 7 | 1x72 VPP (TP8,PP9,VPP4) | 72 | f=0.3 | 512 | 10.8 | 12846..11504 |
| 8 | 1x72 (TP8,PP9) | 72 | f=0.3 | 512 | 10.9 | 12805..11458 |
| 9 | 8x9  (TP3,PP3) | 72 | f=0.3 | 96 | 46.7 | 12550..11321 |
| 10 | EP-heavy 1x64 (TP8,PP8,EP8) | 64 | f=0.3 | 512 | 12.4 | 12357..10961 |
| 11 | 1x72 VPP (TP8,PP9,VPP4) | 72 | none | 512 | 13.3 | 12069..12069 |
| 12 | 1x72 (TP8,PP9) | 72 | none | 512 | 13.5 | 12025..12025 |

## Is expert parallelism actually mandatory? (round-2 correction)

- Non-EP layout `TP8,PP4, ep=1` (32 GPUs) with **TP also sharding experts**: fits=True, feasible=True, 8654 tok/s.
- Same layout with **EP-only expert sharding** (the old assumption): fits=False (experts only /PP=4 -> too big per rank).

**Verdict: EP is NOT strictly mandatory to *fit*** -- TP (×PP) shards the expert FFNs just as well, so a non-EP replica fits the 671B MoE. The earlier 'EP mandatory' was a modeling artifact (experts were sharded only by EP). EP still helps *throughput* by replacing per-layer TP all-reduce with all-to-all routing and reducing per-rank expert compute, but it is a performance choice, not a fitting requirement.

## CAUTION: the sparse-offload gain depends on the overlap assumption

For `1x72 (TP8,PP9)`: f=0 gives 12025 tok/s (opt==cons, no CPU path to overlap). f=0.3 gives **12805 optimistic** but only **11458 conservative**.

-> Under OPTIMISTIC overlap, sparse offload helps (+6%). Under CONSERVATIVE overlap it **HURTS** (-5%). So the MoE offload benefit is real only if CPU attention + C2C genuinely overlap GPU work (ScoutAttention layer-ahead). Treat the offload uplift as best-case.

## Decision: bigger replica vs more DP vs more batch?

- **Best feasible config:** 4x18 (TP6,PP3) (72 GPUs, sparse offload), 25049..20517 system tok/s (opt..cons) at B_d=512/replica.
- **Bigger replica (more TP/PP)** is needed to fit 1.34 TB of weights (~7+ GPUs minimum) and to shrink per-rank weight-read (the TPOT floor); beyond the fit/TPOT minimum, extra TP/PP mostly adds comm.
- **More DP replicas** multiply throughput linearly *once each replica meets the SLO* (this model assumes zero cross-replica cost, so DP linearity is an upper bound) -- the place to spend leftover GPUs.
- **More decode batch capacity** is the cheapest lever but is capped by HBM and the TPOT SLO; sparse CPU offload raises the cap.

**Rule of thumb:** size the replica to the smallest TP*PP that fits weights and meets the TPOT floor; add sparse CPU offload to grow batch; spend remaining rack GPUs on DP. (tok/s are optimistic ceilings.)
