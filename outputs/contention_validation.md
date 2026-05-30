# Validation of the two-resource (HBM-contention) offload model

The contention model (`model/contention.py`) flipped a conclusion ("no help" ->
"helps"), so it needs scrutiny. Verdict: **directionally sound (physics +
mechanism + literature), but the magnitudes are optimistic bounds** — the
specific gain needs real prefill/decode co-execution data to pin down.

## (1) Physics sanity — PASS
`serving_2res(f=0)` never exceeds the independent `serving_roofline` ceiling:

| context S | f=0 serving tps | roofline ceiling | OK? |
|---|---|---|---|
| 4,000 | 436 | 660 | ✅ |
| 16,000 | 214 | 461 | ✅ |
| 68,329 | 39 | 150 | ✅ |

## (2) Mechanism — defensible
The gain comes from two real effects, both captured:
- **Capacity:** offloading f frees HBM -> decode batch B rises (1.7 -> 3.5 at
  f=0.5) -> model weights amortize over more sequences (less HBM/token).
- **Bandwidth:** the offloaded KV streaming leaves HBM (goes to CPU DRAM),
  freeing HBM bandwidth for prefill.
The f=0 baseline being HBM-bound is NOT a strawman: at 68k context one 70B GPU
fits B≈2, ~26 ms/token, 39 tok/s — the genuine long-context single-GPU regime.

## (3) Real-workload (context distribution) — holds, modest
Gain is ~uniform across the Codex context range (which is long throughout —
mean/P50 input ≈ 61-68k), not a single-point artifact:

| context S | best gain (dense, 1 Grace/GPU) |
|---|---|
| 2,000 | 1.10× |
| 8,000 | 1.11× |
| 32,000 | 1.11× |
| 68,329 | 1.11× |
| 130,000 | 1.02× |

So distribution-weighted, the dense/1-Grace gain is ~1.1×; sparse or more CPU
push it to ~1.4–3.3× (see `offload_verdict.md`).

## (4) Literature cross-check — consistent
- **Adrenaline** (offload decode attention to prefill GPU instances): +1.68×
  output throughput. Our dense offload (1.1–2×) is in range (they use HBM↔HBM,
  ~1:1 BW; we use slower CPU, so less for dense, similar with sparse).
- **FastDecode** (attention on aggregated CPUs): 1.88–5×. Our high-CPU/sparse
  cases (~3.3×) sit inside this. The mechanism (free GPU for the compute-bound
  part) is the same.
Our predicted range does not contradict published attention-offload results.

## Honest limitations (what is NOT validated)
1. **No real co-execution measurement.** The two-resource model is a fluid
   ROOFLINE: it assumes GPU compute and HBM bandwidth overlap perfectly and that
   prefill/decode interfere only through shared HBM bandwidth. Real chunked-
   prefill scheduling, kernel interference, and cache effects make actual
   contention messier — so treat the gains as an **upper bound**. Pinning the
   real number needs measured prefill+decode co-run throughput on the target HW.
2. **Overlap assumption.** Gains assume CPU attention hides behind GPU work
   (ScoutAttention layer-ahead). No-overlap shrinks them (cf. opt..cons band).
3. **PD-disaggregation alternative.** A real stack might run prefill on separate
   instances; then the "free HBM bandwidth for prefill" benefit changes form
   (this model is single-GPU prefill+decode co-resident).
4. **Mean/percentile call, optimistic eta, no inter-node fabric, sparse accuracy
   cost (~2–2.5%) not modeled.**

## Bottom line
The contention model is **physically consistent, mechanistically sound, and
within the literature's measured range** — the "offload helps in the prefill-
heavy/HBM-bound regime" conclusion is trustworthy *as a direction and an upper
bound*. The exact multiplier (1.1× dense NVL72 → 3.3× with CPU/sparse) should be
confirmed against real co-execution data before being quoted as a prediction.
