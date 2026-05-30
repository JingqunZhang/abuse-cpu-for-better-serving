# How the analytical model works — flow, bottlenecks, parallelism

## A. Data flow (inputs -> answer)

```
 INPUTS (4 config objects)
   WorkloadConfig : S_c (cached), A (append/uncached), O (output), T_idle
   ModelConfig    : L, P_act, P_total, d_kv (GQA/MLA), d_attn, expert fracs
   HardwareConfig : F_G, BW_HBM, M_HBM, F_C, BW_CPU, M_CPU, BW_C2C, BW_NVLink,
                    n_gpus, cpus_per_gpu   (x efficiency eta_compute/eta_mem)
   PolicyConfig   : f (CPU attn frac), r (old-KV materialize), B_d, sparse, ...
        |
        v   per call x = (S_c, A, O)
  (1) sizing        kv_size(S), weight_read_bytes(B)        [bytes]
        |
        v
  (2) per-phase TIME models (each = a roofline max(); see B)
        T_append (prefill)   tpot (one decode step)
        |
        v
  (3) capacity coupling   B_eff = min(B_target, B_HBM(f)),  B_HBM ~ 1/(1-f)
        |
        v
  (4) system throughput   TPS = B_eff / TPOT   (output tokens/sec)   <-- the gain metric
        |
        v
  (5) SLO-constrained optimize   max TPS  s.t. TPOT<=SLO, TTFT<=SLO, HBM fits
        -> f*, B_d*, gain
```

Layers above this: **event sim** (real queueing/contention), **parallelism**
(shard over GPUs), **disagg** (GPU prefill ∥ CPU decode), **validate** (vs HW).

## B. How ONE decode token's time (TPOT) is computed — and the bottleneck at each spot

`tpot()` builds the per-token time from sub-terms. Each sub-term is itself a
**roofline = max(compute_time, memory_time)** — i.e. an operator is as slow as
its binding resource (compute OR bandwidth), the "短板 / weakest link" within
that op:

```
gpu_nonattn = max( 2*P_act*B / F_G ,          # compute: FLOPs / FLOP-rate
                   weight_read_bytes(B)/BW_HBM ) # memory: weight bytes / HBM BW
              ^ at small batch the weight-read (memory) dominates; at large batch
                compute does. The max() picks whichever is the bottleneck.

gpu_attn    = max( (1-f)*B*KV(S)/BW_HBM ,      # memory: stream KV from HBM
                   attn_FLOPs / F_G )           # compute floor
              ^ decode attention is memory-bound -> KV-bytes/BW usually wins.

cpu_attn    = max( sparse*f*B*KV(S)/BW_CPU ,   # CPU memory
                   sparse*f*B*L*S*d_attn / F_C )# CPU compute
c2c_decode  = L*B*(Q+O)*q / BW_C2C + 2L*t_sync # transfer + per-layer sync
```

## C. The "短板效应" appears at THREE different levels

1. **Inside one operator (compute vs memory):** `max(compute, memory)` — roofline.
   Captures e.g. "decode is memory-bound", "prefill is compute-bound".
2. **Across the parallel paths of one decode step (GPU vs CPU):** see D — `max`
   of the two paths that run *at the same time* (the slower path gates the step).
3. **Across the whole system (which resource caps throughput):** the binding
   constraint:
   - **HBM capacity** caps the batch: `B_HBM(f) = free_HBM / ((1-f)*KV(S))` ->
     `B_eff = min(B_target, B_HBM)`. This is why offload (f) helps: it lifts the cap.
   - **C2C bandwidth** can throttle: `TPS /= append_c2c_util` if oversubscribed.
   - **SLO** caps the usable batch (bigger B -> bigger TPOT -> may break SLO).
   The system answer is bounded by **whichever of these binds first** (a min /
   feasibility filter), not the sum.

So "短板" is handled by **max() within an op**, **max() across simultaneous
paths**, and **min()/feasibility across system resources** — three distinct uses.

## D. How parallelism / overlap is handled

The model has an explicit knob for what runs in parallel vs serially:

```
# OPTIMISTIC overlap (things that truly run concurrently -> take the slower):
TPOT_opt  = gpu_nonattn  +  max( gpu_attn , cpu_attn + c2c )  +  merge + t_dispatch
                            \_________________________________/
              GPU attention over its (1-f) KV  runs IN PARALLEL with
              CPU attention over its f KV (+ the C2C transfer). The step is
              gated by the SLOWER of the two paths -> max(), not sum.

# CONSERVATIVE overlap (assume everything serializes -> sum):
TPOT_cons = gpu_nonattn + gpu_attn + cpu_attn + c2c + merge + t_dispatch
```

- `gpu_nonattn` (weight/FFN) is **serial before** attention (`+`), because the
  next layer's attention needs the projected Q/K/V.
- The GPU-attn vs (CPU-attn + C2C) is the **overlap** — ScoutAttention's
  layer-ahead idea: CPU computes its KV slice while the GPU computes its slice.
- The **opt..cons band** we report = best-case (perfect overlap) to worst-case
  (no overlap). Real systems sit between; this is why we report BOTH and never
  hide behind the optimistic number.

Parallelism at larger scales:
- **Disaggregation (model/disagg.py):** GPU prefill engine and CPU decode engine
  run as a pipeline. Two parallel stages -> system rate = `min(GPU prefill feed,
  CPU decode capacity)` — the pipeline is gated by its slower stage (another
  short-board, this time across engines).
- **Tensor/Pipeline/Expert parallel (model/parallelism.py):** per-rank work is
  sharded; comm terms (TP all-reduce, EP all-to-all) are added; PP overlaps
  micro-batches with a bubble factor `phi(PP,VPP,m)`; per-token time =
  `bubble * (nonattn + max(attn paths) + comm)`.
- **Event sim (sim/event_sim.py, dynamic_sched.py):** the *real* parallelism —
  GPU and CPU are separate resources processing their own queues concurrently;
  contention/queueing emerge from the discrete-event schedule rather than a
  closed-form max/min. This is where the closed form's optimism is corrected.

## E. From one token to the gain

```
TPOT (one token, one batch) ---B_eff = min(B_target, HBM cap)---> 
   TPS = B_eff / TPOT  (system output tokens/sec)  --ratio vs baseline--> GAIN
   subject to: TPOT<=SLO_TPOT, TTFT<=SLO_TTFT, HBM fits, C2C util<=1
```

The **gain lives in TPS** (system output tokens/sec) — more concurrent sessions
served per second — NOT in per-token latency (TPOT/TTFT), which offload keeps
within the SLO but does not improve.
```
