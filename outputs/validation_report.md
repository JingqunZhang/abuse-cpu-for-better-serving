# Phase 3 — validation report

Three layers of validation: (A) limiting-case unit tests, (B) qualitative
agreement with the related work, (C) SLO-constrained f* behavior and the
section-11 decision questions.

## A. Limiting-case tests (`tests/test_model_limits.py`, 14/14 pass)

| Case | Expectation | Result |
|---|---|---|
| f=0 partial == gpu_hot | identical TPOT; CPU/C2C terms = 0 | ✅ |
| f=1 | GPU attention term = 0; all attn on CPU | ✅ |
| BW_CPU, F_C → ∞ | CPU attention time → 0 | ✅ |
| M_HBM → ∞ | decode batch capacity → ∞ (offload benefit ↓) | ✅ |
| BW_C2C → 0 | CPU-backing append-prefill unusable (T→∞) | ✅ |
| dense CPU attn | Gain(f) ≤ 1 ∀f>0 (capacity-only benefit) | ✅ |
| sparse CPU attn | Gain(f) > 1 at some f | ✅ |
| HBM capacity vs f | larger f → larger fit-batch | ✅ |
| cpu_backing append | flush ∝ A (new KV), not S_c (old KV) | ✅ |
| TTFT vs A | more append tokens → larger TTFT | ✅ |
| f* under SLO (sparse) | interior f*>0, gain>1.05 | ✅ |
| f* under SLO (dense) | gain ≤ 1.05 (no win) | ✅ |
| more HBM | offload gain shrinks → ~1 | ✅ |
| eta=0.3 | 50ms TPOT infeasible | ✅ |

## B. Relationship to related work

**Scope first.** Our model is a SINGLE-NODE (one GB200-class GPU + its attached
Grace), INTERACTIVE-SLO model. Several papers operate in a *different* regime;
"shares qualitative shape" ≠ "reproduces". The legend below is deliberate:
🟰 = shares qualitative shape; ⚠️ = different regime, NOT reproduced; ⏳ = not yet.

| Work | Its claim | Relationship |
|---|---|---|
| **FastDecode** | dense CPU attention raises throughput (1.88–5.04×) | ⚠️ **Different regime — NOT reproduced.** FastDecode aggregates the DRAM bandwidth of MANY CPU nodes and runs at loose latency / huge batch, so its effective CPU bandwidth ≈ the GPU rate. Our single-CPU, ~16× slower, tight-SLO model *cannot* and *should not* reproduce it — and indeed predicts dense CPU attention loses **in our regime**. The two are consistent once scoped; we do not claim to reproduce FastDecode. |
| **Lamina** | more memory capacity → larger batch → higher throughput, until compute saturates | 🟰 HBM sweep: gain falls once HBM ≥288GB (compute/SLO binds, not capacity) |
| **Adrenaline** | moderate offload helps; too much overloads the attention side | 🟰 shape only — Adrenaline offloads to **prefill GPUs** (HBM↔HBM, ~1:1 BW), not CPU. Our interior f* (0.22–0.44) matches the *shape*, not the mechanism |
| **ScoutAttention** | CPU co-attention only wins when sparse + overlapped | 🟰 dense never beats baseline **in our regime**; sparse=0.1 does, and ONLY under the optimistic-overlap branch (its layer-ahead). See accuracy caveat below |
| **KVPR** | CPU-backing + GPU recompute helps when transfer is the bottleneck | 🟰 policy B modeled; BW_C2C→0 limiting test confirms transfer-bound failure (note KVPR is PCIe-era; we use NVLink-C2C, far faster) |
| **Frontier** | event dependencies (KV transfer, sync) matter for accuracy | ⏳ closed-form is a bound; the Phase-4 event sim quantifies the gap |
| **PPD** | turn-2+ old-KV location dominates; avoid re-transfer | 🟰 append_time charges old-KV C2C load; cpu_backing skips old-KV re-flush |

**No contradictions once each claim is scoped to its regime.** The headline
"dense CPU attention does not help throughput" is a statement about THIS regime
(single CPU at ~1/16 HBM bandwidth, tight interactive TPOT), not a universal
claim — FastDecode shows the opposite holds with multi-node bandwidth at loose
latency.

> **Accuracy caveat (sparse offload).** The sparse-CPU-attention win assumes
> block-selected attention (ScoutAttention-style, `sparse≈0.1`). That is **not
> free**: ScoutAttention reports ~2.1–2.5% accuracy drop at its budgets. Our
> throughput model has **no accuracy term**, so every "sparse offload gives
> N×" result trades that ~2–2.5% quality. Dense (lossless) CPU attention does
> not help throughput here at all.

## C. SLO-constrained f* and the section-11 decisions

Default SLO: TPOT ≤ 50 ms, TTFT ≤ 10 s, C2C util ≤ 1. Dense-70B on GB200,
workload = Codex/SWEBenchPro mean call. Joint (f, B_d) optimization.

| eta | sparsity | f* | B_d* | Gain | baseline B_d |
|---|---|---|---|---|---|
| 0.3 | any | — | — | infeasible | — |
| 0.5 | dense | 0.02 | 1 | 1.00 | 1 |
| 0.5 | sparse 0.1 | **0.22** | 2 | **1.86** | 1 |
| 0.7 | dense | 0.02 | 1 | 1.00 | 1 |
| 0.7 | sparse 0.1 | **0.44** | 4 | **2.06** | 1 |

**Answers to the section-11 questions (this single-device regime):**

1. *Is CPU backing useful?* **Yes, unconditionally for capacity** (frees HBM,
   needed for long idle sessions). For **throughput**: only with sparse CPU attn.
2. *Best f\*?* **0.22–0.44** under the interactive SLO with sparse attention;
   **~0 (no offload)** with dense attention.
3. *Partial CPU attn → TPS or only HBM pressure?* **Only HBM pressure** when
   dense; **both** when sparse.
4. *At what BW_CPU does CPU attn become useful?* In this SLO-tight, long-context
   regime, batch is **latency-capped at 1–2**, so raw BW_CPU is **secondary** —
   sparsity (how much KV the CPU touches) is the real lever. BW_CPU matters only
   under looser TPOT SLOs that permit large batches.
5. *At what HBM does CPU attn stop helping?* Around **≥288 GB** here: once the
   GPU-only baseline already fits the SLO-limited batch (~2), offload gain
   collapses from 1.86× to ~1.06×.
6. *Is C2C a bottleneck for append old-KV load?* Not at the mean call (C2C util
   ≪ 1 because decode TPOT is long); becomes binding under tighter TPOT, bigger
   B_p, or P90/P99 context — to be stressed in the event sim.

## Bottom line

The model is internally consistent (limiting cases pass) and, **once each paper
is scoped to its regime**, does not contradict the literature — but it
*reproduces* none of them quantitatively (different hardware regimes; FastDecode
in particular is explicitly out of scope). The headline design recommendation it
supports, **within the single-node interactive regime**: CPU as KV backing store
+ GPU keeps hot KV + *sparse* CPU co-attention at small-moderate f, with the
throughput win gated on (a) attention sparsity — which costs ~2–2.5% accuracy —
and (b) genuine CPU/GPU overlap. Remaining risk (queueing/contention) is the
Phase-4 event-sim's job; absolute magnitudes are optimistic bounds.
