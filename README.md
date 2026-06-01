# CPU-GPU Attention / KV Offload Analytical Model

Analytical (and, later, event-driven) model for deciding **when** offloading KV
storage and a fraction of decode attention from GPU HBM to CPU DRAM improves
*system output-token throughput* for long-context agentic LLM serving.

Workload: `Inferact/codex_swebenchpro_traces` (prefill-dominated, 131:1
input:output, 94% cache hit, long idle gaps). See `docs/literature_notes.md` for
the related-work map. **Current results live in `outputs/concurrent_mix.md`
(authoritative headline) and `outputs/SUMMARY.md` (narrative)** — see the
[Outputs map](#outputs-map-which-report-is-current) below for which report
supersedes which.

## Start here — the mental model (60 seconds)

The model answers one question: **does moving KV (and a fraction of decode
attention) to the CPU raise system output-tok/s — and if so, when?** Three ideas
make every result readable:

1. **It's a bottleneck ladder, not a single number.** Find what binds, relieve
   it, repeat — until you hit the hard floor (GPU prefill compute). For
   long-context dense LLM serving the *first* thing that binds is **HBM
   capacity**: 70B weights eat ~140 GB, leaving room for only ~2–3 resident 64k
   sessions, so the weight read is barely amortized. The levers that climb the
   ladder, in order of leverage: **tensor-parallel (pool HBM) → sparse attention
   → KV quantization (int4) → CPU offload.** On 2×GPU this takes Codex from ~100
   to ~970 tok/s; at the top you're GPU-compute-bound and only more GPUs help.

2. **Every gain is a `[conservative .. optimistic]` band**, set by one overlap
   knob `ov` (how much the CPU's work hides behind the GPU's). `gain@con` (ov=0,
   no overlap) is the **bankable floor**; `gain@opt` (ov=1, a ScoutAttention-style
   layer-ahead pipeline) is the **upper bound that requires that pipeline**. A
   bare call returns the conservative floor — we never headline the rosy end.
   Read `gain@con` first; treat `gain@opt` as upside you must earn.

3. **The CPU relieves CAPACITY (bytes), not COMPUTE (FLOPs).** Decode is
   memory-bound, so the GPU's spare compute can't help; the CPU's value is being a
   second, cheaper place to *store* KV. Its roles, ranked for a CPU-scarce node:
   **(a) backing store for idle sessions** — park the 10.5s-idle sessions' KV in
   CPU DRAM to admit more concurrent sessions; unconditional, needs no overlap,
   the best use at a low CPU:GPU ratio. **(b) core-attention offload** — frees HBM
   capacity for a bigger batch, but only bankable (no-overlap) at **≥~2 Grace/GPU**
   and otherwise needs the layer-ahead pipeline; it competes with quantization,
   which does the same capacity job without the CPU. **(c) full decode engine
   (disaggregation)** — huge gains but needs ~32 Grace/GPU to balance the pipeline.

> **Honest bottom line** (realistic Grace `f_cpu`≈14 TF peak, apples-to-apples
> sparse baseline, 1 Grace/GPU): dense core-attention offload ≈**1.0× bankable /
> ≈1.18× with overlap**; sparse-10% ≈**1.0–1.10× bankable / ≈2.5× with overlap**.
> The win lives or dies by overlap (and by the CPU's *achievable* sparse-attention
> FLOP/s — these use the bf16 peak, an upper bound), and **KV quantization often
> beats offload outright** on this capacity-bound workload. (Hardened by three
> rounds of adversarial multi-agent audit + a hardware-settings review; see
> `outputs/model_revisions.md`.)

## Two offload architectures (don't confuse them)

The repo models **two different ways** the CPU can help — they answer different
questions and give very different gains:

- **Partial offload** (`scenario.py` / `concurrent.py`, model `serving_mix`): the
  GPU still does *all* prefill **and** decode, but hands a fraction `f` of the
  *core-attention* KV to the CPU. Gain is a band: ≈**1.0× bankable** (no overlap)
  up to ≈**1.18× dense / ≈2.5× sparse-10×** with a perfect layer-ahead pipeline,
  on a 1-Grace GB200 (at the realistic CPU bf16 peak; irregular sparse kernels
  realize less). Needs sparsity **and** overlap (≥~1 Grace/GPU makes sparse
  bankable). Right for a single interactive node with a few CPUs.
- **Full disaggregation** (`disagg.py`): the GPU does **only prefill**, a *pool*
  of CPUs does **only decode**, run as a pipeline. Large gain (≈ 13–34× over the
  admission-bound baseline) but needs **~32+ Grace-class CPUs per GPU**
  (FastDecode-style aggregation). `out_tps = O · min(GPU prefill feed, CPU
  decode capacity)`.

## Which command answers which question?

| Your question | Command | Architecture |
|---|---|---|
| For MY hardware + workload: offload yes/no, optimal `f`, gain, binding resource? (one-stop **VERDICT**) | `python -m model.scenario --workload "…" [--mix "…"] [--measured-tps X]` | partial |
| Partial-offload throughput band for a concurrent class **mix** (authoritative headline) | `python -m model.concurrent` → `concurrent_mix.md` | partial |
| **Full disaggregation**: GPU-prefill / CPU-decode pipeline throughput, and how many CPUs it needs | `python -m model.disagg` → `disagg_cpu_decode.md` | disagg |
| Best multi-GPU parallel deployment (TP/PP/EP) on NVL72 | `python -m model.parallelism` → `recommended_configs.md` | — |
| Sanity-check the closed form against real queueing/contention | `python -m sim.event_sim` / `python -m sim.dynamic_sched` | — |
| Calibrate / error against real hardware measurements | `python -m model.validate` (edit `hw_measurements.json`) | — |
| How does the model work internally? | read `outputs/MODEL_WALKTHROUGH.md` | — |

**Start here:** `python -m model.scenario --gpus 1 --cpus-per-gpu 4` prints a
one-line VERDICT for the default workload; add `--workload` for your own.

## Layout

```
model/config.py      Workload / Model / Hardware / Policy / Coeffs configs (plan §4)
model/analytical.py  Closed-form model: kv_size, hbm/cpu capacity, append_time,
                     tpot, c2c_util, tps, gain, crossover  (plan §5-8)
model/contention.py  Two-resource (GPU-compute ∥ HBM-bw) single-mean-call model
                     -- prefill/decode contend for HBM bandwidth (serving_2res)
model/concurrent.py  CURRENT best model: fluid steady-state for a MIX of workload
                     classes contending concurrently for GPU-compute ∥ HBM-bw +
                     CPU/C2C; latency-aware (TPOT-SLO batch cap), calibratable
                     overlap fraction (serving_mix, serving_band, fit_overlap)
model/sweep.py       Sweep f x B_d x r x eta x sparsity -> sweep_results.csv + plots
tests/test_model_limits.py   Limiting-case + regime tests (plan §3 Phase-3 checks)
docs/literature_notes.md     Phase-0 paper notes + model-term justification table
docs/external_validation.md  llm-emu cross-check plan: what it can/can't validate,
                     the oracle-injection integration paths, and prerequisites
outputs/             sweep_results.csv, *.png, workload_stats.json, findings
```

## Run

```bash
python3 tests/test_model_limits.py      # analytical limiting-case + SLO checks (20/20)
python3 tests/test_event_sim.py         # event-sim sanity checks (4/4)
python3 -m model.sweep                  # CSV + 4 plots into outputs/
python3 -m model.trace_parser           # download + parse trace -> Phase-1 outputs
python3 -m sim.event_sim                # event sim vs analytical -> outputs/
python3 tests/test_parallelism.py       # parallelism checks
python3 -m model.parallelism            # NVL72 sweep -> parallelism_sweep.csv + recs
python3 -m model.concurrent             # FIXED reference report -> concurrent_mix.md
                                        # (for YOUR case use model.scenario, below)
python3 -m model.disagg                 # GPU-prefill / CPU-decode pipeline -> disagg_cpu_decode.md
python3 -m sim.dynamic_sched            # dynamic GPU/CPU scheduling sim -> dynamic_scheduling.md
python3 -m model.validate               # roofline-floor + HW calibration -> validation_vs_hardware.md
python3 -m pytest tests/ -q             # full suite (75/75)
```

## Tunable hardware scenarios

Every hardware feature is a CLI flag in `model/scenario.py` — set them and get
single-GPU f*/TPOT/throughput + the best multi-GPU deployment:

```bash
python3 -m model.scenario                       # default GB200 NVL72, dense-70B
python3 -m model.scenario --model moe_large_mla # 671B MoE on 72 GPUs
python3 -m model.scenario --gpus 8 --gpu-flops 2e15 --hbm-cap 80 --hbm-bw 3.35 \
    --c2c-bw 64 --cpu-dram-bw 400 --cpus-per-gpu 4 --model moe_large_mla
```

Flags: `--gpus --gpu-flops --hbm-cap --hbm-bw --nvlink-bw --c2c-bw
--cpu-dram-bw --cpu-flops --cpu-mem --cpus-per-gpu --eta --model --sparse
--slo-tpot --slo-ttft --mix --workload --measured-tps`. `--cpus-per-gpu` scales CPU
bandwidth/memory/compute per GPU (FastDecode-style aggregation). Programmatic:
`HardwareConfig.system(...)`.

**Model YOUR workload** (not the Codex mean) with `--workload`, and get a
one-line **VERDICT** + the binding-resource meaning:

```bash
python3 -m model.scenario --gpus 1 --cpus-per-gpu 4 \
    --workload "s_cached=16000,a_append=2000,o_output=400"
# ... [3] CONCURRENT MIX ...
#   binding @ f=0: gpu_hbm = HBM-bandwidth-bound -> offload CAN help
#   VERDICT: OFFLOAD f=0.85 (sparse=0.1) -- 1.39x (no overlap) .. 5.99x (perfect overlap); ...
```

**Calibrate the band to a single number** with one measured co-execution
throughput: `--measured-tps 42` fits `ov` via `fit_overlap` and prints a single
calibrated gain (or tells you the measurement is outside the band — the honest
signal that something other than overlap is off).

`--mix` drives section **[3] CONCURRENT MIX** (the `model/concurrent.py` fluid
model): a concurrent workload mix like `--mix "long:0.7,short:0.3"` (classes:
`long short mid prefill`). It reports the f=0 `[conservative..optimistic]`
throughput band, the binding resource, the best offload `f`, and the gain at
**both** overlap ends (`gain@con≈1` means the offload win requires CPU/GPU
overlap):

```bash
python3 -m model.scenario --gpus 1 --cpus-per-gpu 4 --mix "long:0.7,short:0.3"
```

## Validating against real hardware

```bash
python3 -m model.validate          # roofline-floor checks + calibration/LOO error
```

`outputs/validation_vs_hardware.md`: (1) **roofline floor checks** — the model's
TPOT never falls below the physical weights+KV/HBM floor (a hard correctness
guarantee); (2) **calibration + leave-one-out MAPE** against measured points in
`hw_measurements.json`. Seeded with literature-derived Llama-3.1-8B/H100 ITL
points → fitted η≈0.4, **LOO error ~13–18%** (comparable to analytical sims like
Frontier's ~9–11%). **Drop your own tt-stack / vLLM measurements into
`hw_measurements.json`** (schema in `model/validate.py`) and re-run for a
hardware-specific error number.

## Key idea

Throughput uses a **capacity-bounded batch**: `B_d_eff = min(B_d_target,
B_HBM(f))` where `B_HBM(f) ~ 1/(1-f)`. Offload raises batch capacity but also
TPOT (CPU attention + C2C). Their ratio is the section-8 crossover. Whether it
pays off depends on two things, both now in the concurrent model:
- **CPU attention sparsity.** Dense CPU attention is ~L·S·d_attn FLOP/token
  (~178 ms/token at 68k on one Grace) — compute-bound, not just bandwidth-bound;
  ScoutAttention-style sparse (~10%) is what makes offload viable.
- **CPU/GPU overlap.** The offload gain *lives or dies by overlap*. The default
  reported number is the **conservative** (no-overlap) floor; with **perfect
  overlap** the concurrent model shows ~**1.18× (dense) / ~2.5× (sparse 10%)** on a
  1-Grace GB200 under a 50 ms TPOT SLO (at the realistic CPU bf16 peak — an upper
  bound; irregular sparse kernels realize less); with **no overlap the gain is
  ≈1.0–1.1×** (the
  slow CPU attention lands on the per-token critical path). The optimistic numbers
  require a layer-ahead pipeline **or** (a loose latency budget **and** sparsity
  **and** ≥~2 Grace/GPU). `sparse` is applied to the GPU baseline too, so the gain
  isolates the offload effect — it is not a sparsity benefit denied to the baseline.

`eta` caveat: the headline uses `eta=0.5`; the hardware-fit is ≈0.40, where the
dense single-GPU baseline is SLO-infeasible (needs tensor-parallel first) — so
`eta=0.5` is the more offload-favorable choice, not less. See
`outputs/validation_vs_hardware.md`.

### The "短板 / weakest-link" rule, applied at three levels

The whole model is roofline `max()` / bottleneck `min()` reasoning. The same
short-board idea shows up three times — knowing which one you're looking at is
the key to reading any result:

1. **Inside one operator** — `max(compute_time, memory_time)`: an op runs as slow
   as its binding resource (e.g. decode attention is memory-bound).
2. **Across the two parallel paths of a decode step** — `max(GPU_attn,
   CPU_attn + C2C)`: the slower of the GPU and offloaded paths gates the step;
   how much they overlap is the `ov` knob.
3. **Across the whole system** — the binding resource caps throughput
   (`Λ_max = 1 / max_resource(Σ p·util_sec)`), and feasibility filters
   (HBM capacity, TPOT SLO) cap the batch. The system answer is whichever binds
   first, not the sum.

### How a result reads: the `[conservative .. optimistic]` band

Every throughput/gain is reported as a **band**, governed by one overlap knob
`ov ∈ [0,1]` = the fraction of offloaded CPU work that hides behind GPU work:

| end | `ov` | meaning |
|---|---|---|
| **conservative** *(default)* | 0 | offloaded CPU attention fully **serialized** onto the critical path (and the call rate) — a true no-overlap lower bound. `serving_mix`/`best_f` return this unless you ask otherwise |
| **optimistic** | 1 | offloaded work **perfectly overlaps** GPU work (ScoutAttention layer-ahead pipeline) — the loosest upper bound |

Both ends come from one primitive, `combine(a,b,ov) = a + b − ov·min(a,b)` (ov=0 →
sum/serial, ov=1 → max/overlap), applied with the **same `ov`** to both the
per-token TPOT (which caps the batch via the SLO) and the call-rate util-seconds
(which caps Λ) — so the latency and throughput channels can't disagree about what
overlaps. `serving_mix(..., overlap=ov)` takes a float in between; `fit_overlap(...,
measured_tps)` bisects `ov` to match one measured co-execution point, collapsing
the band into a single calibrated prediction (returns `None` if the measurement
is outside the band — the honest signal that something other than overlap is off).
The truth sits inside the band; the default is the conservative floor, never the
optimistic number.

## Glossary (canonical symbols)

| symbol | meaning | units | set in |
|---|---|---|---|
| `S_c` / `A` / `O` | cached-prefix / append (uncached) / output tokens per call | tokens | `WorkloadConfig` |
| `S` | total context attended at decode = `S_c + A` | tokens | derived |
| `f` | fraction of decode-attention KV handled by CPU (`1−f` on GPU) | — | `PolicyConfig` |
| `sparse` | fraction of context KV the CPU actually reads/token (ScoutAttention) | — | `PolicyConfig` |
| `r` | fraction of old KV materialized to GPU for append-prefill | — | `PolicyConfig` |
| `ov` | overlap fraction (0 = serialized, 1 = perfect) — the band knob | — | `overlap=` arg |
| `B` (≡ `B_d`, `B_d_eff`) | resident decode batch actually run (can be fractional in the fluid model) | seqs | computed |
| `b_cap` / `b_slo` | batch ceiling from HBM capacity / from the TPOT SLO; `B = min(b_cap, b_slo)` | seqs | computed |
| `η` (`eta`) | achievable-efficiency factor on FLOP/s & bandwidth (`eta_compute`/`eta_mem`) | — | `HardwareConfig.effective` |
| `util_sec` | per-resource utilization-seconds per normalized call (diagnostic) | s | computed |
| `Λ_max` | max sustainable call rate = `1 / max_resource(Σ p·util_sec)` | calls/s | computed |
| TPOT / TTFT | time-per-output-token / time-to-first-token | s | computed |
| `tps()` | decode-only throughput — **superseded**, over-states on prefill-heavy work | tok/s | `analytical` |
| `serving_tps()` | prefill-inclusive throughput (real-world headline) | tok/s | `analytical` |
| `serving_mix()` | fluid steady-state throughput for a concurrent class **mix** — **current best** | tok/s | `concurrent` |

## Outputs map (which report is current)

`outputs/` holds the full history; only two are the current headline:

| report | status |
|---|---|
| **`concurrent_mix.md`** | **AUTHORITATIVE** — current headline (fluid mix, latency-aware, overlap band, CPU-attention compute cost) |
| **`SUMMARY.md`** | **AUTHORITATIVE** — narrative overview of all conclusions |
| `MODEL_WALKTHROUGH.md` | how the model works (flow, bottlenecks, parallelism) |
| `disagg_cpu_decode.md` | **full disaggregation** (GPU-prefill / CPU-decode pipeline) — the 13–34× line |
| `recommended_configs.md` | NVL72 / MoE multi-GPU deployment recommendations |
| `throughput_ceiling.md` | physical roofline ceiling (sanity floor) |
| `event_sim_vs_analytical.md`, `dynamic_scheduling.md`, `validation_vs_hardware.md` | discrete-event / scheduling sims + hardware-calibration cross-checks |
| `model_revisions.md` | changelog of correctness fixes across review rounds |
| `offload_verdict.md`, `phase2_findings.md`, `hbm_contention_offload.md`, … | **SUPERSEDED** — earlier-round numbers (e.g. 3.3×/2.2×) that predate the CPU-attention-compute and overlap fixes; kept for history, see the banner at the top of each |

## Status (phase progress)

- [x] Phase 0 — literature notes + model justification
- [x] Phase 1 — live trace parser (`model/trace_parser.py`) → `workload_stats_live.json`,
      `workload_summary.md`, `workload_distributions.png`; reproduces dataset-card means
- [x] Phase 2 — closed-form model, sweep, plots
- [x] Phase 3 — TTFT + SLO-constrained joint (f*, B_d) optimizer (`optimize_f`),
      20/20 limiting-case + SLO tests, `outputs/validation_report.md` (paper-trend checks)
- [x] Phase 4 — event-driven simulator (`sim/event_sim.py`): batched decode engine,
      HBM-admission pool, FIFO C2C; `outputs/event_sim_vs_analytical.md` (workload is
      admission-bound; closed-form is a throughput-upper/latency-lower bound)
- [x] Phase 5 — DP/TP/PP/VPP/EP/CP (`model/parallelism.py`): NVL72 sharding, PP bubble,
      TP/EP comm; `outputs/parallelism_sweep.csv` + `recommended_configs.md`
- [x] Refinement — two-resource HBM-bandwidth contention (`model/contention.py`) and
      the **concurrent multi-scenario fluid model** (`model/concurrent.py`):
      heterogeneous class mix, latency-aware batch cap, calibratable overlap;
      two independent soundness-review passes; `outputs/concurrent_mix.md`,
      `outputs/contention_validation.md`, `outputs/offload_verdict.md`

**Overall conclusions:** `outputs/SUMMARY.md`. All phases complete; 75/75 tests pass.
```
