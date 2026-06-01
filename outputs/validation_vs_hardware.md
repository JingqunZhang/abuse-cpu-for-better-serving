# Validation vs. real hardware

Two-part validation: (1) physical roofline-floor checks (rigorous, no data needed); (2) calibration + leave-one-out MAPE against real measurements. Status: 2 real (literature-derived) measurement point(s) loaded from `hw_measurements.json`.

## (1) Roofline floor checks (model TPOT >= physical HBM/FLOP floor)

| case | model TPOT @eta=1 (ms) | physical floor (ms) | >= floor? |
|---|---|---|---|
| dense70b_b1_64k | 20.12 | 17.50 | ✅ |
| dense70b_b32_8k | 27.99 | 17.50 | ✅ |
| moe_b16_32k | 86.89 | 82.39 | ✅ |

A pass means the model never predicts faster than physics allows -- a necessary correctness condition (not sufficient: it bounds below, not the absolute value).

## (2) Calibration harness

### Single-eta fit (robust headline)
**2 real points.** Best-fit eta = 0.40, in-sample MAPE = 5.7%.

**Leave-one-out held-out MAPE = 11.2%** (the honest generalization error).

| point | fit eta | pred TPOT (ms) | measured (ms) | error |
|---|---|---|---|---|
| llama8b_h100_b1_ctx1k | 0.40 | 12.0 | 11.0 | 9.5% |
| llama8b_h100_b1_ctx8k | 0.45 | 11.3 | 13.0 | 12.9% |

### LIFE-style joint fit (eta_compute, eta_mem, t_dispatch)
In-sample MAPE = **5.4%** at eta_mem=0.45, t_dispatch=0.5ms (eta_compute=0.1 is weakly determined for memory-bound decode). The split + dispatch terms tighten the fit vs single-eta (5.7% -> 5.4%).

_Joint leave-one-out skipped: 2 points < 5 needed for a 3-parameter CV. Add more (batch, context) points to validate the joint fit out-of-sample._

**Verdict:** held-out MAPE 11.2% — **within** a ~15% band (comparable to analytical sims like Frontier ~9–11%).

### Caveats on these points
- They are **approximate, literature-derived** single-stream ITL figures (NVIDIA TRT-LLM + vLLM H100 benchmarks), not a controlled decode-only sweep — treat the ~13–18% as indicative, not final.
- eta≈0.4 here folds in everything the closed form omits at batch 1 (kernel launch, sampling, non-peak HBM): that is what calibration is for. Replace these with your own tt-stack / vLLM measurements in `hw_measurements.json` for a hardware-specific number.

### Sources
- NVIDIA, *LLM Inference Benchmarking with TensorRT-LLM* (developer.nvidia.com) — single-stream ITL ~11–21 ms, 8B/H100.
- vLLM v0.6.0 perf blog (vllm.ai) — Llama-3 70B/8B H100 TPOT.
- *Forecasting LLM Inference Performance via Hardware-Agnostic Analytical Modeling*, arXiv:2508.00904 — comparable analytical model.
- *Frontier* LLM inference simulator — ~9–11% PDD error target.
