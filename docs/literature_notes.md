# Phase 0 — Literature notes & model justification

For each related work: what is modeled, the resource bottleneck it identifies,
the (explicit or implicit) model, validation method, reported gain, limitation,
and **which term of our model it justifies**.

## Per-paper summary

### FastDecode (arXiv 2403.11421)
- **Models:** transformer inference split into an *S-part* (model/non-attention,
  weight-heavy, kept on GPU) and an *R-part* (attention over KV, memory-bound,
  pushed to CPU "R-workers").
- **Bottleneck:** KV cache caps batch size; naive host offload is bound by
  CPU↔GPU bandwidth; aggregated CPU memory capacity/BW/compute can absorb the
  memory-bound R-part.
- **Model:** resource-ratio matching between GPU S-part and CPU R-part.
- **Validation:** end-to-end vs vLLM, same GPU count.
- **Gain:** 1.88×–5.04× throughput.
- **Limitation:** assumes you can scale CPU workers freely; CPU↔GPU BW is the
  binding constraint they engineer around.
- **Justifies in our model:** treating CPU as a separate resource pool;
  `T_CPUattn` and the throughput `min()` over GPU vs CPU vs C2C bounds;
  evaluating *system* throughput under batch-capacity change, not single-request
  TPOT.

### Lamina (model–attention disaggregation)
- **Models:** memory-optimized devices hold KV + run attention; compute devices
  run non-attention operators; staggered pipelining across batches.
- **Bottleneck:** memory capacity limits batch size; throughput rises with more
  memory devices until compute devices saturate.
- **Model:** choose memory-device count so attention time ≈ model-slice interval.
- **Validation:** comparable hardware cost.
- **Gain:** +16.1%–90.1% throughput.
- **Limitation:** needs careful staggering; benefit is batch-size-driven.
- **Justifies in our model:** the **balance condition** `T_CPUattn ≈ T_GPU_model`;
  the staggered/overlap term (optimistic TPOT); sweeping the GPU↔CPU device
  ratio (later: parallelism phase).

### Adrenaline (decode-attention offload to prefill instances)
- **Models:** offload a *fraction* of decode attention to PD-disaggregated
  prefill instances.
- **Bottleneck:** decode instances limited by HBM capacity / KV; offload raises
  decode batch and compute utilization, but too much offload overloads the
  attention side.
- **Model:** load-aware offload fraction.
- **Validation:** serving benchmarks.
- **Gain:** up to 1.68× output-token throughput.
- **Limitation:** offload ratio must be tuned to load.
- **Justifies in our model:** sweeping **fraction f** (not assuming f=1);
  the expected **concave Gain(f)** curve — rises at small f, peaks, then falls
  when the CPU/attention side bottlenecks (`cpu_attn_bound`, `c2c_bound`).

### ScoutAttention (layer-ahead CPU KV offload)
- **Models:** unimportant KV blocks on CPU DRAM, digests + important blocks on
  GPU, GPU-CPU cooperative *sparse* attention, layer-ahead CPU pre-computation.
- **Bottleneck:** CPU-side latency — hidden by computing layer i+1 on CPU while
  GPU works on window i.
- **Model:** `T_stall,i = max(0, T_CPU,i+1 − T_GPU,window,i)`.
- **Validation:** accuracy + speed vs full / offloading attention.
- **Gain:** up to 5.1× vs full attention, 2.1× vs offloading; ~2.1–2.4% acc drop.
- **Limitation:** relies on sparsity; CPU fraction must be small and overlapped.
- **Justifies in our model:** full CPU attention is *not* the only design point
  (partial/sparse f); the **layer-ahead overlap term** `T_stall` (optimistic
  overlap branch of TPOT); CPU helps only when its fraction is small + hidden.

### KVPR (I/O-aware KV partial recompute, arXiv 2411.17089)
- **Models:** KV stored in CPU DRAM; GPU *recomputes* part of KV and *transfers*
  the rest; split point chosen to overlap recompute and transfer. CPU does **not**
  compute attention.
- **Bottleneck:** CPU↔GPU transfer vs GPU recompute.
- **Model:** `T = max(T_recompute, T_transfer)`; profiler/scheduler/runtime.
- **Validation:** profiler-driven.
- **Gain:** up to −35.8% latency, +46.2% throughput.
- **Limitation:** recompute costs GPU FLOPs.
- **Justifies in our model:** **profiler-driven coefficients** (`Coeffs`,
  `eta`) instead of pure peak specs; the **CPU-backing + GPU-materialize policy
  (B)** as a first-class alternative to CPU attention; future split-point
  optimization on `append_time` (recompute vs C2C load of old KV).

### Frontier (LLM inference simulator)
- **Models:** event-driven serving with explicit dependencies — pipeline stages,
  PDD KV transfer, AFD activation transfer, MoE EP sync.
- **Bottleneck:** missing KV-transfer dependencies cause large sim error.
- **Model:** discrete-event with request state / queueing / KV transfer / memory
  admission / scheduler feedback.
- **Validation:** per-phase latency + end-to-end; PDD error ~9–11%.
- **Gain:** (simulator, not a system) — accuracy claim.
- **Limitation:** heavier than closed-form.
- **Justifies in our model:** keep an **event-driven extension path** (Phase 4);
  model queueing, KV transfer, memory admission; validate both per-phase and
  end-to-end. Our closed-form is the lower bound; the event sim explains the gap.

### PPD (multi-turn append-prefill disaggregation)
- **Models:** later conversation turns routed locally so decode-node KV is reused
  instead of recomputed / re-transferred.
- **Bottleneck:** turn-2+ old KV location — prefill node can't see decode node's KV.
- **Model:** session affinity + KV locality.
- **Validation:** multi-turn serving.
- **Gain:** ~75% KV-transfer reduction at 3.1 avg turns/conversation.
- **Limitation:** requires affinity routing.
- **Justifies in our model:** **append-prefill accounts for old-KV location**
  (`append_time`: old-KV load over C2C, GPU append, new-KV flush only);
  the `cpu_backing` flag (authoritative old KV on CPU → don't re-flush old KV);
  session affinity / KV locality treated as first-order, not an afterthought.

## Mapping table: model term → justifying work(s)

| Model term / file location | Equation | Justified by |
|---|---|---|
| `kv_size` (config/analytical) | `L·S·d_KV·q_KV` | all (KV is the shared currency) |
| CPU as separate resource pool | `min(TPS_GPU, TPS_CPU, TPS_C2C, …)` | FastDecode, Lamina |
| `decode_batch_cap` `B_HBM(f)=free/((1-f)M_KV)` | section 5 | Adrenaline, Lamina (capacity→batch) |
| `gpu_attn_time` `(1-f)B_d M_KV/BW_HBM` | section 7.1 | FastDecode (memory-bound attn) |
| `cpu_attn_time` `max(mem/BW_CPU, flops/F_C)` | section 7.2 | FastDecode, ScoutAttention |
| `c2c_decode_time` `+2L·t_sync` | section 7.2 | FastDecode (BW bottleneck), Frontier (sync) |
| optimistic overlap / `T_stall` | section 7.3 | ScoutAttention (layer-ahead), Lamina (staggered) |
| `append_time` old-KV load + new-KV flush | section 6 | PPD, KVPR |
| `cpu_backing` (no old-KV re-flush) | section 6 rule | PPD |
| policy B (CPU backing + GPU recompute) | section 4.4 | KVPR |
| sweep over `f` → concave `Gain(f)` | section 8 | Adrenaline |
| `Coeffs` / `eta` profiler calibration | `R_eff=η·R_peak` | KVPR |
| event-driven extension (Phase 4) | — | Frontier |

**Exit criterion check:** every modeling term above has ≥1 related-work
justification. ✅

## Sources
- Inferact/codex_swebenchpro_traces — HuggingFace Datasets
- FastDecode — arXiv 2403.11421
- Lamina (heterogeneous LLM decoding, model–attention disaggregation)
- Adrenaline — boosting resource utilization via decode-attention offload
- ScoutAttention — arXiv 2603.27138
- KVPR — arXiv 2411.17089
- Frontier — LLM inference simulation
- PPD — "Not All Prefills Are Equal", multi-turn PD disaggregation
