> **✅ AUTHORITATIVE** — narrative overview of the current conclusions. The
> headline numbers table lives in `concurrent_mix.md`. See `README.md` Outputs map.

# CPU-GPU Attention / KV Offload — study summary

When does offloading KV storage and a fraction of decode attention from GPU HBM
to CPU DRAM improve **system output-token throughput** for long-context agentic
serving? Built as a 5-phase modular model (closed-form → event-sim → parallelism),
validated against the real Codex/SWEBenchPro trace and 7 systems papers.

> **Latest refinement — concurrent multi-scenario model** (`model/concurrent.py`,
> `outputs/concurrent_mix.md`; runnable per-config via `scenario.py --mix`). The
> single-mean-call analysis below was generalized to a **fluid steady-state where a
> MIX of workload classes contend concurrently** for GPU-compute ∥ HBM-bandwidth +
> CPU/C2C, with a **TPOT-SLO batch cap** and a **calibratable compute/HBM overlap
> fraction** (`fit_overlap`). It sharpened two things and is the current best view:
> (1) the **real "no-help" boundary is COMPUTE-bound, not short-context** — offload
> helps whenever GPU HBM binds (even short context, via weight-amortization), and
> does nothing only when GPU compute binds; (2) once **CPU attention compute** is
> counted (~L·S·d_attn/token, compute-bound on a weak CPU), the **bankable
> (no-overlap) gain is ≈1.0–1.1×** and the **perfect-overlap gain is ~1.18× dense /
> ~2.5× sparse-10% on a 1-Grace GB200 @50 ms TPOT** (at the realistic Grace bf16
> peak f_cpu≈14 TF — an upper bound; irregular sparse kernels realize less) — the offload win *lives or
> dies by CPU/GPU overlap* (high `ov`) **or** (a loose latency budget **and**
> sparsity **and** ≥~2 Grace/GPU). The reported default is the conservative floor,
> not the optimistic end; `sparse` is applied to the GPU baseline too
> (apples-to-apples), so the gain isolates the offload effect. Three rounds of
> adversarial multi-agent audit moved the verdict mildly-optimistic → honest →
> mildly-pessimistic (i.e. it now errs slightly *against* offload); 81/81 tests.
> The phase-by-phase findings below remain valid in direction; these are the
> corrected, current magnitudes.

## What was built
- **Closed-form analytical model** (`model/analytical.py`): KV/HBM/CPU capacity,
  append-prefill, TPOT (optimistic/conservative overlap), TTFT, C2C util, TPS,
  gain, and a joint **(f\*, B_d) SLO-constrained optimizer**.
- **Live trace parser** (`model/trace_parser.py`): reconstructs the workload from
  the public dataset; matches the dataset-card means (input/output/calls exact-ish,
  cache-hit 97% vs 94%).
- **Event-driven simulator** (`sim/event_sim.py`): batched decode engine + HBM
  admission + FIFO C2C, replaying the trace.
- **Parallelism model** (`model/parallelism.py`): DP/TP/PP/VPP/EP/CP over NVL72.
- **Concurrent multi-scenario model** (`model/concurrent.py`): the current best
  view (see callout above); wired into `scenario.py --mix`.
- **72 tests** (limiting cases, SLO behavior, sim sanity, sharding, concurrent-mix
  invariants + the two review-pass fixes) + plots/CSVs.

## The central findings

1. **The workload is prefill-dominated and HBM-admission-bound.** 131:1
   input:output, 94% cache hit, long idle gaps. One 64k-context sequence's KV is
   ~21 GB on dense-70B; with weights you fit ~1 resident session per GPU. The
   event sim shows TTFT dominated by capacity-queueing, almost independent of
   arrival rate. **This is the strongest case in the study — for CPU KV backing
   store**: park idle/warm sessions in CPU DRAM (long idle gaps make this nearly
   free) and shrink resident hot KV so more sessions are admitted.

2. **CPU as KV backing store: useful unconditionally** (capacity), needed for the
   idle-gap reuse pattern. Append-prefill should stay on GPU; with a CPU backing
   copy you flush only *new* KV, not old (PPD-style).

3. **Partial CPU attention helps throughput only when it's sparse — in this
   single-node, interactive-SLO regime.** Dense CPU attention does not beat
   baseline here: one CPU's DRAM is ~16× slower than HBM, so the batch-capacity
   gain (~1/(1-f)) can't pay for it. **ScoutAttention-style sparse CPU
   co-attention (≈10% blocks)** shrinks that gap and yields gains — **but at a
   ~2.1–2.5% accuracy cost (not modeled here), and only if the CPU path truly
   overlaps GPU work** (it can hurt under conservative/serialized overlap; see
   #7). Out of this regime (e.g. FastDecode's multi-node CPU-bandwidth
   aggregation at loose latency), *dense* CPU attention can win — we do not model
   or contradict that.

4. **Under an interactive TPOT SLO, the optimum f\* is interior — but only at the
   optimistic end.** Bankable (no-overlap): f\*≈0, gain ≈1.0× (offloading the slow
   CPU attention onto the critical path doesn't pay). With perfect overlap:
   interior f\*~0.15–0.60 giving ~1.18× dense / ~2.5× sparse over the best GPU-only
   batch (apples-to-apples — the GPU baseline gets the same sparsity, so this is
   the offload effect alone, not a sparsity benefit denied to the baseline). The
   SLO is what creates the interior optimum (unconstrained f→1 just trades latency
   for batch).

5. **CPU bandwidth is a secondary lever; sparsity is primary.** In the SLO-tight,
   long-context regime decode batch is latency-capped at 1–2, so *how much KV the
   CPU touches* matters more than raw BW_CPU. Offload's benefit fades once HBM is
   large enough (≥~288 GB here) that the baseline already fits the SLO batch.

6. **`f` can't lower the append-prefill floor.** Old-KV materialization (~21 GB,
   set by `r`) bounds how many sessions fit regardless of decode offload — to
   admit more you must also lower `r` (materialize/recompute less old KV, KVPR-style).

> **Headline metric is now SERVING output tokens/sec** (prefill burst included),
> not decode-only — `serving_tps()` / `optimize_f(objective="serving")`. For this
> 131:1 prefill-heavy workload decode-only over-states throughput, so all headline
> numbers fold in the prefill compute. The theoretical ceiling is in
> `model/roofline.py` (`serving_roofline()`): dense-70B = 150 out-tok/s/GPU
> (KV-bandwidth bound), MoE-MLA = 354 (prefill-compute bound); admission-bound
> operating points sit ~16% of ceiling.

7. **For the 671B MoE on NVL72 (MLA KV; serving throughput):**
   - **EP is NOT mandatory to fit.** Once TP (×PP) is allowed to shard expert
     FFNs — as real systems do — non-EP replicas fit the 671B model (e.g. a
     TP8,PP4 replica fits with expert-TP but not under EP-only sharding). The
     earlier "EP mandatory" was a modeling artifact. EP remains a throughput
     *choice* (all-to-all vs TP all-reduce), not a fitting requirement.
   - **The sparse-offload gain is overlap-dependent.** For 1×72: f=0.3 gives
     **+24% under optimistic overlap but −13% under conservative** (fully
     serialized) overlap. So the MoE offload benefit is real only if CPU
     attention + C2C genuinely hide behind GPU work (ScoutAttention layer-ahead).
   - **Serving (prefill-inclusive) throughput**: EP-heavy 1×64 ≈ **11.5k tok/s
     (f=0) → 12.4k (sparse f=0.3)** opt, ~11k conservative. (The earlier
     ~33–41k were decode-only optimistic; folding in the prefill burst drops them
     to a realistic ~11–12k, in line with DeepSeek's published ~14.8k/node.)
   - Still **optimistic-overlap, no inter-node fabric cost** — treat as a ceiling.
     See `recommended_configs.md`.

## Spend-the-GPUs guidance
Size the replica to the **smallest TP·PP·EP that fits weights and meets the TPOT
floor**; add **sparse CPU offload** to grow per-replica batch; spend **all
remaining rack GPUs on DP** (linear throughput once each replica meets SLO). PP
bubbles and EP all-to-all make over-large replicas counterproductive past the
fit/TPOT threshold.

## Recommended design
> CPU stores idle/warm KV (authoritative); GPU materializes old KV for
> append-prefill and keeps hot/recent KV; CPU handles a **small, sparse** fraction
> of cold attention **only under HBM pressure and a loose-enough TPOT budget**.

This matches the going-in hypothesis (plan §12) and is now quantified.

## Caveats / next steps
- Token counts are chars/4.065 estimates (no tokenizer); `inter_call_delay` and
  true prefix-cache boundaries aren't in the public trace.
- Hardware specs are approximate GB200 figures; calibrate `Coeffs`/`eta` against
  real microbenchmarks (KVPR-style profiler) before quantitative use.
- Event sim batches decode but interleaves prefill coarsely; a chunked-prefill
  refinement and CP ring-attention comm would sharpen TTFT and the MoE numbers.
- TP is modeled as sharding only dense/attention params (experts via EP); real
  systems also TP-shard expert FFNs.
