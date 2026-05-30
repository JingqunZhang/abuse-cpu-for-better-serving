# Comparison with LIFE (arXiv:2508.00904) and improvements adopted

*LIFE — "Forecasting LLM Inference Performance via Hardware-Agnostic Analytical
Modeling"* is the closest published peer to our model. Read in full; key points
and what we changed below.

## What LIFE is
- Lightweight, modular, **operator-level** analytical model of TTFT / TPOT / TPS,
  parameterized only by hardware specs (TOPS + memory bandwidth) — no dataset
  benchmarking. Supports quantization, KV compression, LoRA, chunked prefill,
  different attentions, operator fusion.
- Core decode equation (their eq. 4), decode being memory-bound (t_compute ≪
  t_mem):  **TPOT = Σ_ops MEM_op /(BW · em_op) + Σ_ops t_dispatch_op** —
  per-operator memory traffic over (bandwidth × per-op memory efficiency), plus a
  per-op dispatch overhead.

## LIFE's OWN reported accuracy (Table 10, Llama2-7B decode)
| Hardware | prompt | measured TPS | forecast TPS | error |
|---|---|---|---|---|
| NVIDIA V100 fp16 | 512 | 40.0 | 32.6 | ~18% |
| NVIDIA V100 fp16 | 1024 | 36.9 | 30.3 | ~18% |
| NVIDIA V100 fp16 | 2048 | 32.1 | 26.7 | ~17% |
| Ryzen iGPU int4 | 128 | 34.5 | 33.4 | ~3% |
| Ryzen iGPU int4 | 1536 | 32.8 | 27.2 | ~17% |
| Ryzen CPU bf16 | 2048 | 0.45 | 1.62 | ~2.6× (!) |

**Takeaways:**
1. A peer analytical model is **~17–18% off on GPU decode**, worse on CPU at long
   prompts. So **our ~13–18% leave-one-out error is in the normal band**, not a
   red flag.
2. The shared failure mode is exactly **assuming a fixed efficiency**: LIFE's CPU
   forecast blows up at long prompts because it held efficiency flat (10%) while
   measured efficiency collapsed (9%→3%). Their central message: *real efficiency
   varies "beyond a simple roofline" — compute and memory hit very different,
   size-dependent fractions of peak.*

## What we adopted (this round)
1. **Separate compute vs memory efficiency.** `HardwareConfig.effective()` now
   takes `eta_compute` (scales FLOP/s — binds prefill/append) and `eta_mem`
   (scales all bandwidths — binds decode). `effective(eta)` still sets both
   (back-compat). A single scalar cannot match a compute-bound TTFT *and* a
   memory-bound TPOT at once; the split can. (Also the round-1 reviewer's request.)
2. **Dispatch-overhead term.** `Coeffs.t_dispatch` adds a fixed, batch-independent
   per-decode-step latency to TPOT (LIFE's `t_dispatch_op`) — the kernel-launch /
   scheduling floor pure roofline misses at small batch / short context. Default 0
   (neutral); set it (or fit it) when you have real data.

Both default to neutral, so all prior results/tests are unchanged (35/35 pass).

## What we deliberately did NOT do
- We did **not** auto-fit `t_dispatch` / a 2-parameter efficiency on the 3
  approximate literature points — that would overfit (those points even show a
  context slope steeper than memory traffic explains, i.e. they include
  scheduling/prefix effects the closed form shouldn't absorb). The batch-32 point
  remains **event-sim territory** (concurrency/continuous-batching), not a closed-
  form decode step.

## Path to LIFE-grade accuracy on your hardware
LIFE gets its GPU numbers by **calibrating per-operator, size-dependent memory
efficiency curves** from a measurement database (their "statistics database"),
not a single η. To match that here:
1. Collect controlled **decode-only** TPOT at several (batch, context) points on
   your tt-stack / target GPU.
2. Fit `eta_mem` (and optionally a size-dependent em curve) + `t_dispatch` per
   operator instead of one global η.
3. Report leave-one-out MAPE (the harness in `model/validate.py` already does the
   single-η version; the split-η/dispatch knobs are now in place to extend it).

## Sources
- *Forecasting LLM Inference Performance via Hardware-Agnostic Analytical
  Modeling* (LIFE), arXiv:2508.00904 — eq. 4, Table 10, "beyond a simple roofline".
