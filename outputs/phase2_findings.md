> **⚠️ SUPERSEDED** — early closed-form findings (e.g. ~2.2×) that predate the
> concurrent fluid model, the CPU-attention-compute cost, and the overlap-band
> fixes. Current authoritative result: **`concurrent_mix.md`** (and `SUMMARY.md`).
> Kept for history; see `README.md` Outputs map and `model_revisions.md`.

# Phase 2 — analytical model first findings

Config: dense-70B (GQA) on GB200 per-GPU view, workload = Codex/SWEBenchPro
*mean* call (S_c=64338, A=3991, O=520). Reference slice eta=0.5, r=1.0,
B_d_target=256. Full grid in `sweep_results.csv`.

## Headline result

**Scope: single GB200-class GPU + its Grace CPU, interactive TPOT SLO.** Within
this regime, whether CPU attention offload helps **throughput** hinges on **how
much KV the CPU has to read**, i.e. sparsity — not on f alone. (Out of regime —
e.g. FastDecode's multi-node CPU-bandwidth aggregation at loose latency — dense
CPU attention *can* win; that is not this model's regime.)

| Regime | Gain(f) shape | Best gain | Why |
|---|---|---|---|
| Dense CPU attention (sparse=1.0) | monotonically **decreasing** | 1.0 (at f=0) | one CPU's DRAM ~16x slower than HBM; batch headroom (~1/(1-f)) can't pay for it |
| Sparse CPU attention (sparse=0.1, ScoutAttention) | **rises above 1** | ~2.2x | CPU touches 10x less KV; bandwidth gap shrinks to ~1.6x, batch headroom wins |

> **Accuracy caveat:** `sparse=0.1` means the CPU attends only ~10% of KV blocks
> (ScoutAttention-style), which costs ~2.1–2.5% model accuracy — not modeled
> here. The sparse-regime gain is therefore a quality/throughput trade, not a
> free lunch; lossless (dense) CPU attention gives no throughput win in this regime.

## What binds

- The baseline (f=0) is **HBM-capacity bound**: one 64k-context sequence's KV is
  ~21 GB on dense-70B, so after 140 GB weights only ~1-2 sequences fit. The GPU
  is starved of batch, not bandwidth.
- Offload's lever is **capacity → batch**: `B_HBM(f) ~ 1/(1-f)`.
- The cost is **CPU attention time** (`max(KV_bytes/BW_CPU, FLOPs/F_C)`), which
  dense-mode makes dominate TPOT.

## Caveats / what this does NOT yet say

- The f=1 sparse point only "wins" because batch jumps to 256 while TPOT hits
  ~4.6 s/token — **unusable under any TPOT SLO**. The interior optimum f* only
  appears once the SLO constraint (section 11) is imposed. **Next iteration.**
- Append-prefill old-KV materialization (~21 GB over C2C) is included but C2C
  utilization stays low here because decode TPOT is long; under tighter TPOT or
  bigger B_p this flips — to be stressed in the event sim (Phase 4).
- Single-device only. The 671B MoE / NVL72 story needs Phase-5 sharding.

## Tie-back to the section-11 questions (partial answers)

- *Is CPU backing useful?* Yes for capacity unconditionally (frees HBM, enables
  larger batch); useful for **throughput** only with sparse CPU attention.
- *Best f\**? Without an SLO: f→1 under sparsity, f=0 under dense. With an SLO it
  will be interior — pending next iteration.
- *Does partial CPU attention improve TPS or only HBM pressure?* Only HBM pressure
  in the dense regime; both in the sparse regime.
