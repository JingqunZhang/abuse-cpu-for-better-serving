# External validation via llm-emu — what it can and cannot prove

> Status: plan / assessment (not yet executed). Source reviewed:
> [`AKafakA/llm-emu`](https://github.com/AKafakA/llm-emu) `main`, files
> `vllm_emulator/oracle/{base,gpu_cost_oracle}.py`,
> `vllm_emulator/hooks/executor_hook.py`,
> `vllm_patches/overrides/vllm/v1/core/sched/scheduler.py`.

This document records, honestly and with scope, whether and how the external
project **llm-emu** can serve as an independent cross-check for *this* repo's
CPU-GPU attention/KV-offload analytical model (`model/concurrent.py`,
`model/analytical.py`). It is deliberately explicit about the boundary between
"llm-emu validates X" and "llm-emu cannot say anything about X", because the
whole point of this model is to be honest about what is and isn't established.

---

## 1. What llm-emu actually is

llm-emu is a **profile-driven online emulator for vLLM**, not an analytical
model and not a cycle-level simulator. It runs the **real, unmodified vLLM v1
control plane** — scheduler, KV-cache block manager, admission/preemption, HTTP
stack, tokenizer, sampling/output pipeline — and **replaces only the GPU forward
pass** with a latency draw:

```
real vLLM scheduler decides the batch  ──►  executor hook intercepts the step
                                            ──►  oracle.estimate_step_latency_us(...)
                                            ──►  sleep / Timer that long, return fake tokens
```

So the *timing of one GPU step* is faked; **everything that decides which
requests are in that step is the genuine vLLM scheduler** running against a real
GPU KV-block budget. Confirmed from the patched scheduler: a request enters the
running batch only if `kv_cache_manager.allocate_slots(...)` succeeds, else it
preempts; batch is also capped by `max_num_seqs`.

The default latency source (`ProfileGpuCostOracle`) interpolates a step latency
from a **profile pack** captured once on a real GPU (adaptive-K Shepard
inverse-distance pooling over buckets indexed by token count and concurrency).
Pre-built pack: **Qwen3-8B on A40**. Validated to ±10% of real GPU
TTFT/TPOT/throughput.

### The oracle interface (the injection point)

```python
class BaseGpuCostOracle(ABC):
    @abstractmethod
    def estimate_step_latency_us(self, total_tokens: int, has_prefill=False,
                                 num_requests=0, **kwargs) -> float: ...
```

The executor hook calls it once per scheduler step with features it derives from
`SchedulerOutput`:

| llm-emu feature        | meaning                                            | maps to our model |
|------------------------|----------------------------------------------------|-------------------|
| `total_tokens`         | Σ scheduled tokens this step (prefill chunks + decode) | mixed step token count |
| `has_prefill`          | any new (prefill) request in the step              | prefill present flag |
| `num_requests`         | requests in the step (batch concurrency)           | ≈ `B` (prefill+decode) |
| `num_new_reqs`         | new/prefill requests                               | prefill stream count |
| `sum_kv`               | Σ `num_computed_tokens` over decoding requests     | `B_decode · avg_S` ⇒ `avg_S = sum_kv / num_decode` |

This is exactly the information `model/concurrent.py::_decode_tpot(B, f, avg_S,
…)` consumes (it already works in `avg_S`, the residency-weighted mean context).
The information granularity is sufficient to drive our step model.

---

## 2. What llm-emu CAN validate for us

### 2.1 The baseline (f=0) continuous-batching dynamics

Our fluid model abstracts the scheduler into a closed-form `Λ_max = 1 /
max_resource(Σ p·util_sec)` and a TPOT-SLO batch cap `b_slo`. llm-emu runs the
**real scheduler**, so it can check whether our closed-form throughput, the
resident decode batch `B`, and the TPOT-vs-batch relationship match what real
admission/preemption produces under a request-rate sweep. This is a strictly
better no-overlap / queueing cross-check than our home-grown
`sim/event_sim.py`, because the queueing is the genuine vLLM scheduler, not our
approximation of it.

### 2.2 The overlap fraction `ov` (calibration of the band)

Our headline honesty caveat is the `[conservative..optimistic]` band from the
`ov ∈ [0,1]` compute∥HBM overlap knob (`serving_band`, `fit_overlap`). A
measured prefill+decode **co-execution** step latency is exactly a measurement
of realized overlap. Two routes:

- **Profile-pack route (true HW):** capture a pack on the target GPU; the mixed
  prefill+decode buckets in it *are* co-execution measurements. Feed the
  resulting throughput into `fit_overlap(classes, model, hw, measured_tps)` to
  collapse our band into a single calibrated curve.
- **Injected-oracle route (see §4):** run our own step model as the oracle and
  compare the *emergent* end-to-end throughput under the real scheduler against
  our closed-form — the gap quantifies what the fluid `Λ_max` over-counts
  relative to real serialization.

### 2.3 The latency/critical-path channel of the offload claim

Our model says the offload gain "lives or dies by overlap": with `ov=0` the CPU
attention lands on the per-token critical path, breaks the 50 ms TPOT SLO, and
forces `B` down (`gain@con ≈ 1.00×`). If we inject an offload-aware oracle (§4,
Channel A), llm-emu's real scheduler will react to the inflated per-step latency
exactly as a real engine would (it caps the running batch by TPOT indirectly via
queue build-up). That tests whether our `b_slo` mechanism is the right shape.

---

## 3. What llm-emu CANNOT validate (the boundary — read this before claiming anything)

1. **It has no model of CPU offload at all.** Vanilla vLLM does not offload core
   attention to CPU DRAM. There is no notion of CPU-attention compute
   (`coeffs.beta · L·S·d_attn`), CPU DRAM bandwidth (`bw_cpu`), or the C2C
   Q/O exchange (`bw_c2c`, `t_sync`). Out of the box, llm-emu can validate the
   **f=0 GPU-only baseline only** — it cannot measure the offload gain that is
   the entire subject of this repo. Any offload validation requires us to *inject*
   that physics ourselves (§4), at which point we are validating "our physics +
   real scheduler", **not** "real hardware".

2. **The stock profile pack is the wrong hardware and model.** A40 / RTX 8000 +
   Qwen3/Llama-8B. Our regime is **GB200-class node + dense-70B / 671B-MoE-MLA +
   Grace CPU + NVLink-C2C** (`HardwareConfig` defaults). None of our
   conclusions are about A40. A stock-pack run validates llm-emu's *own* fidelity,
   not our model's regime.

3. **The oracle latency is a black box per step.** It returns one synthesized
   step latency; it does not decompose into compute vs HBM vs CPU vs C2C. So it
   can calibrate the *aggregate* overlap `ov`, but it cannot independently confirm
   our internal roofline split (the `max(compute, memory)` inside each operator).
   That split must be checked against microbenchmarks or `roofline.py`, not here.

4. **Capacity channel is invisible to a stock oracle.** Even with an
   offload-aware *latency* oracle (Channel A), the **vLLM KV-block manager still
   accounts KV on GPU**. Offloading KV off-GPU should let the real scheduler admit
   a larger running batch `B` (our capacity→weight-amortization gain channel). A
   latency-only injection does **not** make that happen — the scheduler does not
   know KV left HBM. Capturing this requires Channel B (§4): telling the block
   manager the offloaded KV does not consume GPU blocks.

5. **It is an emulator of one node's serving loop, not a hardware truth oracle.**
   Even on true hardware, the profile pack itself carries ±10% and assumes the
   captured buckets cover the operating point. Extrapolation beyond captured
   (token, concurrency, sum_kv) buckets is interpolation, not measurement.

---

## 4. Integration paths (engineering effort + prerequisites)

### Channel A — inject an offload-aware latency oracle (LOW effort, ~1–2 days)

Implement `OffloadCostOracle(BaseGpuCostOracle)` whose
`estimate_step_latency_us(total_tokens, has_prefill, num_requests, num_new_reqs,
sum_kv)` reconstructs the step composition and applies **our** step model:

- `num_decode = num_requests − num_new_reqs`; `avg_S = sum_kv / max(num_decode,1)`.
- decode part → `model.concurrent._decode_tpot`-style roofline over the
  `(1−f)` GPU KV share;
- prefill part → `prefill GEMM + prefill attn` from the `num_new_reqs` chunks;
- offload part → `cpu_attn (α/β) + c2c (q_act, t_sync)` for the `f` share,
  combined with the GPU path via the `ov` overlap knob.

Injection mechanism: `create_oracle_from_profile_pack(...)` builds the oracle;
add an env-var branch (e.g. `VLLM_EMULATOR_ORACLE_CLASS`) or monkeypatch in
`executor_hook._initialize`. We supply a `HardwareConfig` (target GPU) instead of
a profile pack, so **no real-hardware pack is needed** — but the result is
"our physics under real scheduling", explicitly not a hardware measurement.

**Validates:** §2.1 (baseline batching), §2.3 (latency/critical-path channel),
the shape of `b_slo`. **Does not validate:** absolute hardware latency, or the
capacity channel (§3.4).

**Prerequisites:** vLLM v0.18.1 (version-locked to the patch set); a Python env
that can import vLLM with the CUDA-mock; one `HardwareConfig` for the target GPU.

### Channel B — model offloaded KV as freeing GPU blocks (MEDIUM effort)

To make the **capacity→bigger-`B`** gain emerge from the real scheduler, tell the
KV-cache accounting that the `f`-share of KV does not live in GPU blocks. vLLM
already has the native hook: the **`KVConnector`** interface
(`get_num_new_matched_tokens`, async load) that the patched scheduler calls.
Either (a) implement a connector that reports the offloaded tokens as externally
held so `allocate_slots` consumes fewer GPU blocks, or (b) patch the block budget
directly in `vllm_patches/overrides/.../scheduler.py`. Then the real scheduler
admits the larger batch on its own, and end-to-end throughput reflects both gain
channels.

**Validates:** the full offload gain (latency AND capacity channels) emerging
from real admission control. **Cost:** version-coupled to vLLM 0.18.1, more
invasive, needs careful checking that block bookkeeping stays consistent.

**Prerequisites:** all of Channel A, plus familiarity with vLLM's
`kv_cache_manager` / `KVConnector`; a regression check that f=0 reproduces stock
behavior.

---

## 5. Honest verdict

| Use | Verdict |
|-----|---------|
| Calibrate our `ov` / collapse the band via `fit_overlap` | ✅ strong — *if* a relevant-HW profile pack or Channel-A run is available |
| Replace/augment `sim/event_sim.py` as the no-overlap, real-scheduler cross-check | ✅ strong (baseline f=0 side) |
| Validate the **offload gain itself** out of the box | ❌ no — vanilla vLLM has no CPU offload; it is llm-emu's blind spot and our subject |
| Use directly in the GB200 / 70B-MoE regime | ❌ no stock profile pack; needs HW capture or injected analytic oracle |
| Inject our offload step model as a custom oracle (Channel A) | ✅ best low-cost first step — tests latency/overlap channel under real scheduling |
| Recover the capacity→batch gain via KVConnector (Channel B) | ✅ feasible, native hook exists — needed for the *complete* offload validation |

**One sentence:** llm-emu cannot, by itself, confirm our offload conclusions
(its blind spot is exactly our subject), but it is the most ready-made substrate
to (a) calibrate the dangling `ov` overlap fraction and (b) replace our hand-
rolled event sim with the real vLLM scheduler — and with a modest custom-oracle
(+optional KVConnector) it can turn our closed-form gain into a
real-scheduler-driven prediction. Neither path is a substitute for a real
prefill+decode co-execution measurement on the target hardware; that remains the
one open hook.
