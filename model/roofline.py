"""Theoretical serving throughput ceiling (output tokens/sec) + utilization.

Evaluation question: does the model reflect the THEORETICAL throughput upper
bound of real serving?  The pieces exist (decode TPS converges to BW_HBM/KV;
disagg captures the prefill compute roofline) but the core tps() is decode-only
and omits prefill -- so for a prefill-heavy workload it does NOT report a single
unified serving ceiling.  This module provides that ceiling and expresses any
operating point as a fraction of it (MBU/MFU-style).

Ceiling derivation (one GPU, batch-saturated, ignore SLO & HBM capacity -- i.e.
the best the hardware could ever do for this model+workload):

  per OUTPUT token, unavoidable GPU time =
      max( (1-f)*KV(S)/BW_HBM ,   # decode KV streaming (memory wall)
           2*P_act/F_G )          # decode FFN compute
    + T_prefill / O               # prefill compute, amortized over the O outputs

  ceiling_tps_per_gpu = 1 / (that)        ;  system = x n_gpus

Whichever of the three terms dominates is the binding roofline. KV streaming
usually dominates at long context -> the only ways to RAISE the ceiling are to
cut KV-per-token (MLA, KV quant, sparse/offload f, shorter context) or add HBM
bandwidth; adding batch/CPUs only moves you TOWARD the ceiling, not past it.

Run:  python -m model.roofline
"""

from __future__ import annotations

import os
from dataclasses import replace

from . import analytical as an
from .config import (Coeffs, HardwareConfig, ModelConfig, PolicyConfig,
                     WorkloadConfig)

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))


def serving_roofline(model, hw, work, *, f=0.0, coeffs=Coeffs()):
    """Theoretical output-token/s ceiling for this model+workload+hardware."""
    kv_gpu = (1.0 - f) * an.kv_size(work.s_context, model)
    decode_kv = kv_gpu / hw.bw_hbm                      # s/token (memory)
    decode_cmp = 2 * model.p_act / hw.f_gpu             # s/token (compute)
    t_prefill = an.append_time(PolicyConfig.policy("gpu_hot", b_p=1), work, hw,
                               model, coeffs).gpu_compute
    prefill_amort = t_prefill / work.o_output           # s/token
    decode_term = max(decode_kv, decode_cmp)
    per_tok = decode_term + prefill_amort
    per_gpu = 1.0 / per_tok if per_tok > 0 else float("inf")
    # binding factor
    parts = {"decode_KV_bandwidth": decode_kv, "decode_compute": decode_cmp,
             "prefill_compute_amortized": prefill_amort}
    binding = max(parts, key=parts.get)
    return {
        "ceiling_tps_per_gpu": per_gpu,
        "ceiling_tps_system": per_gpu * hw.n_gpus,
        "decode_kv_ms": decode_kv * 1e3,
        "decode_compute_ms": decode_cmp * 1e3,
        "prefill_amort_ms": prefill_amort * 1e3,
        "binding": binding,
        "n_gpus": hw.n_gpus,
    }


def utilization(operating_tps_per_gpu, model, hw, work, *, f=0.0):
    """Operating point as a fraction of the theoretical per-GPU ceiling."""
    rl = serving_roofline(model, hw, work, f=f)
    c = rl["ceiling_tps_per_gpu"]
    return operating_tps_per_gpu / c if c > 0 else 0.0


def report():
    hw = HardwareConfig().effective(0.5)
    work = WorkloadConfig()
    coeffs = Coeffs()
    lines = ["# Theoretical serving throughput ceiling (roofline)\n",
        "Output tokens/sec ceiling = 1 / [ max(decode-KV/BW, decode-compute) + "
        "prefill-compute/O ], per GPU, eta=0.5, Codex/SWEBenchPro mean call. This "
        "is the best the hardware could do (batch-saturated, no SLO/HBM-cap). The "
        "model's operating points are shown as % of this ceiling.\n"]
    for mn in ("dense_70b", "moe_large_mla"):
        model = ModelConfig.preset(mn)
        rl = serving_roofline(model, hw, work)
        # operating points: admission-bound baseline (HBM-capped)
        base = an.tps(PolicyConfig.policy("gpu_hot", b_d=256), work, hw, model)
        # uncapped large-batch DECODE-ONLY rate (ignores HBM cap AND prefill)
        w0 = replace(work, a_append=0.0)
        big_tpot = an.tpot(PolicyConfig.policy("gpu_hot", b_d=4096), w0, hw,
                           model, coeffs).optimistic
        big_decode_only = 4096 / big_tpot
        lines += [f"## {model.name}",
            f"- **Theoretical ceiling: {rl['ceiling_tps_per_gpu']:.0f} out-tok/s "
            f"per GPU** ({rl['ceiling_tps_system']:.0f} for {rl['n_gpus']} GPUs). "
            f"Binding: **{rl['binding']}**.",
            f"- Breakdown per output token: decode-KV {rl['decode_kv_ms']:.2f}ms, "
            f"decode-compute {rl['decode_compute_ms']:.3f}ms, prefill-amortized "
            f"{rl['prefill_amort_ms']:.2f}ms.",
            f"- Admission-bound operating point (HBM-capped B={base.batch}): "
            f"{base.tps:.0f} tok/s = **{utilization(base.tps, model, hw, work)*100:.0f}% "
            f"of ceiling** — the headline regime is far below the wall (HBM "
            f"capacity, not bandwidth, is what binds it).",
            f"- Decode-only rate at large batch (HBM cap + prefill IGNORED): "
            f"{big_decode_only:.0f} tok/s — note this *exceeds* the {rl['ceiling_tps_per_gpu']:.0f} "
            f"unified ceiling precisely because it drops the prefill term: that is "
            f"the `tps()` decode-only caveat made visible.",
            "",
            "Raising the ceiling itself (not just utilization):",
            "| lever | f | new ceiling tok/s/gpu |",
            "|---|---|---|"]
        for f in (0.0, 0.3, 0.5, 0.9):
            r = serving_roofline(model, hw, work, f=f)
            lines.append(f"| CPU offload f={f} | {f} | {r['ceiling_tps_per_gpu']:.0f} |")
        lines.append("")
        print(f"{model.name}: ceiling {rl['ceiling_tps_per_gpu']:.0f} tok/s/gpu "
              f"(binding {rl['binding']}); admission op {base.tps:.0f} "
              f"({utilization(base.tps,model,hw,work)*100:.0f}% of ceiling)")
    lines += ["## Verdict: does the model reflect the theoretical upper bound?",
        "- **Yes, the ceiling is encoded and reachable:** decode TPS converges to "
        "BW_HBM/KV as batch grows, and this module makes the unified prefill+decode "
        "ceiling explicit. Offload/MLA/sparsity RAISE the ceiling by cutting "
        "KV-per-token (see table); batch/CPUs only move you toward it.",
        "- **Caveat the core metric:** `analytical.tps()` is decode-only (omits "
        "prefill), so for this 131:1 prefill-heavy workload it slightly OVERstates "
        "the per-token decode rate vs the unified ceiling here. Use "
        "`serving_roofline()` for the true bound and `utilization()` to place any "
        "operating point.",
        "- The ceiling is the THEORETICAL bound at eta; real serving sits at "
        "eta*utilization below it (validation: eta~0.4, and admission/SLO keep "
        "utilization well under 100%).",
    ]
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "throughput_ceiling.md")
    with open(p, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    report()
