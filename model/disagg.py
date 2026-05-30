"""Prefill-on-GPU / decode-on-CPU disaggregation throughput model.

The key idea (user's point): when the GPU hands the prefilled KV to the CPU and
the CPU continues decoding, the GPU is NOT idle -- it prefills the NEXT sessions.
GPU and CPU run as a PIPELINE. So system output throughput is a balance:

    out_tps = O * min( GPU_prefill_call_rate , CPU_decode_call_rate )

- GPU_prefill_call_rate = 1 / T_prefill   (prefill is compute-bound; the GPU runs
  at its prefill roofline once decode is off it).
- CPU_decode_call_rate  = (best feasible CPU decode tok/s) / O, where the CPU
  pool (cpus aggregated) decodes a batch Bc, bounded by the TPOT SLO, CPU memory,
  AND -- crucially -- CPU bandwidth for BOTH weight streaming and KV/attention
  streaming (long context makes attention the limiter at high Bc).

Baseline (GPU does prefill + decode), admission-bound at small batch Bg:
    base_tps = O / (T_prefill + O * TPOT_gpu(Bg) / Bg)
  -- at Bg=1 the GPU wastes ~all its time on memory-bound single-stream decode,
  which is exactly what disaggregation reclaims.

Run:  python -m model.disagg
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import replace

from . import analytical as an
from .config import (Coeffs, HardwareConfig, ModelConfig, PolicyConfig,
                     SLOConfig, WorkloadConfig)

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
BC_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def prefill_time(model, hw, work, coeffs=Coeffs()):
    """GPU compute time to prefill one call's uncached/append tokens (s)."""
    pol = PolicyConfig.policy("gpu_hot", b_p=1)
    return an.append_time(pol, work, hw, model, coeffs).gpu_compute


def gpu_decode_tpot(model, hw, work, B, coeffs=Coeffs()):
    pol = PolicyConfig.policy("gpu_hot", b_d=B)
    w = replace(work, a_append=0.0)
    return an.tpot(pol, w, hw, model, coeffs).optimistic


def baseline_out_tps(model, hw, work, coeffs=Coeffs()):
    """GPU does prefill+decode; decode batch = HBM-fit (admission-bound)."""
    cap = an.decode_batch_cap(PolicyConfig.policy("gpu_hot", b_d=1), work, hw,
                              model, coeffs)
    Bg = max(1, int(cap))
    tpot_g = gpu_decode_tpot(model, hw, work, Bg, coeffs)
    T_pf = prefill_time(model, hw, work, coeffs)
    O = work.o_output
    return O / (T_pf + O * tpot_g / Bg), Bg, T_pf, tpot_g


def cpu_decode_capacity(model, hw, work, cpus, slo, coeffs=Coeffs()):
    """Best feasible CPU-pool decode throughput (tok/s) and the batch achieving
    it, under the TPOT SLO and CPU memory. CPU pool = cpus aggregated: its
    'GPU' compute/bandwidth ARE the pooled CPU's."""
    cpu_eng = replace(hw, f_gpu=hw.f_cpu * cpus, bw_hbm=hw.bw_cpu * cpus)
    mem = hw.m_cpu * cpus
    per_seq_kv = an.kv_size(work.s_context, model)
    best_tok, best_Bc, best_tpot = 0.0, 0, float("inf")
    for Bc in BC_GRID:
        if Bc * per_seq_kv > mem:                 # KV must fit in CPU DRAM
            break
        tpot_c = gpu_decode_tpot(model, cpu_eng, work, Bc, coeffs)
        if tpot_c > slo.slo_tpot:                 # SLO gate
            continue
        tok = Bc / tpot_c
        if tok > best_tok:
            best_tok, best_Bc, best_tpot = tok, Bc, tpot_c
    return best_tok, best_Bc, best_tpot


def disagg_out_tps(model, hw, work, cpus, slo, coeffs=Coeffs()):
    T_pf = prefill_time(model, hw, work, coeffs)
    prefill_call_rate = 1.0 / T_pf
    cpu_tok, Bc, tpot_c = cpu_decode_capacity(model, hw, work, cpus, slo, coeffs)
    cpu_call_rate = cpu_tok / work.o_output
    out = work.o_output * min(prefill_call_rate, cpu_call_rate)
    bound = "gpu_prefill" if prefill_call_rate <= cpu_call_rate else "cpu_decode"
    if cpu_tok == 0:
        bound = "cpu_infeasible(SLO)"
    return {"out_tps": out, "bound": bound, "Bc": Bc, "tpot_cpu": tpot_c,
            "prefill_feed_tps": work.o_output * prefill_call_rate,
            "cpu_cap_tps": cpu_tok}


def run(model_name="dense_70b", eta=0.5):
    model = ModelConfig.preset(model_name)
    hw = HardwareConfig().effective(eta)
    work = WorkloadConfig()
    slo = SLOConfig()
    base, Bg, T_pf, tpot_g = baseline_out_tps(model, hw, work)
    rows = []
    print(f"\n===== {model.name}  eta={eta} =====")
    print(f" baseline (GPU prefill+decode, Bg={Bg}): {base:.1f} out-tok/s  "
          f"[T_prefill={T_pf*1e3:.0f}ms, GPU decode TPOT={tpot_g*1e3:.1f}ms]")
    print(f" GPU prefill-only feed ceiling: {work.o_output/T_pf:,.0f} out-tok/s "
          f"(={1/T_pf:.1f} calls/s x O={int(work.o_output)})")
    print(f" {'cpus':>5} {'out_tps':>9} {'gain':>6} {'Bc':>4} {'tpot_cpu_ms':>11} {'bound':>18}")
    cross = None
    for cpus in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        d = disagg_out_tps(model, hw, work, cpus, slo)
        gain = d["out_tps"] / base if base > 0 else 0.0
        if cross is None and gain > 1.0:
            cross = cpus
        rows.append((cpus, d, gain))
        print(f" {cpus:>5} {d['out_tps']:>9.1f} {gain:>6.2f} {d['Bc']:>4} "
              f"{(d['tpot_cpu']*1e3 if d['tpot_cpu']!=float('inf') else -1):>11.1f} "
              f"{d['bound']:>18}")
    print(f" -> first cpus_per_gpu with gain>1: {cross}")
    return base, rows


def write_report():
    lines = ["# Prefill-on-GPU / decode-on-CPU disaggregation\n",
        "out_tps = O * min(GPU prefill feed, CPU decode capacity). The GPU runs "
        "at its prefill roofline while the CPU pool decodes in parallel. eta=0.5, "
        "SLO TPOT<=50ms, Codex/SWEBenchPro mean call.\n"]
    for mn in ("dense_70b", "moe_large_mla"):
        base, rows = run(mn)
        model = ModelConfig.preset(mn)
        lines += [f"## {model.name}",
            f"- Baseline (GPU does all, admission-bound small batch): "
            f"**{base:.0f} out-tok/s** — the GPU wastes most time on memory-bound "
            f"single-stream decode.",
            "",
            "| cpus/gpu | out tok/s | gain | CPU batch | CPU TPOT (ms) | bottleneck |",
            "|---|---|---|---|---|---|"]
        for cpus, d, gain in rows:
            tp = d['tpot_cpu']*1e3 if d['tpot_cpu'] != float('inf') else None
            lines.append(f"| {cpus} | {d['out_tps']:.0f} | {gain:.2f}x | {d['Bc']} | "
                         f"{tp:.1f} | {d['bound']} |" if tp else
                         f"| {cpus} | {d['out_tps']:.0f} | {gain:.2f}x | - | - | "
                         f"{d['bound']} |")
        lines.append("")
    lines += ["## Why there is a gain — three compounding effects",
        "1. **GPU runs in parallel on prefill.** While the CPU decodes session A, "
        "the GPU prefills session B. The GPU stops wasting ~all its time on slow "
        "memory-bound B~1 decode and runs at its compute-bound prefill roofline.",
        "2. **KV leaves HBM.** Decode KV now lives in CPU DRAM, so the GPU's HBM "
        "holds only weights + transient prefill working set — the decode-KV "
        "*admission stall* that capped the baseline at batch~1 disappears, and the "
        "prefill feed becomes purely compute-bound (the ceiling rows above).",
        "3. **CPU can batch decode cheaply.** Because KV fits in big, cheap CPU "
        "DRAM, the CPU pool decodes a large batch; its limit is CPU *bandwidth* "
        "(weight + KV streaming under the SLO), not capacity.",
        "",
        "## When (the crossover)",
        "- **SLO gate:** CPU decode TPOT ~= weight_bytes/(cpus x BW_cpu) <= SLO. "
        "Large models need many CPUs (dense-70B: SLO-infeasible below ~16 CPUs).",
        "- **Attention gate:** at long context, CPU decode batch is capped by KV "
        "streaming (B x KV/(cpus x BW_cpu) <= SLO), limiting capacity below the "
        "weight-only estimate.",
        "- **Balance:** out_tps = O x min(GPU prefill feed, CPU capacity); gain "
        "saturates at the GPU prefill ceiling once the CPU pool is big enough.",
        "",
        "**Bottom line:** gain starts around **~32 Grace-class CPUs/GPU** under a "
        "50ms SLO and grows to **~13–34x** the admission-bound baseline as the CPU "
        "pool approaches the GPU prefill feed. Below ~16 CPUs it is SLO-infeasible. "
        "This is the FastDecode 'aggregate many CPUs' regime — practical with "
        "CPU-heavy nodes, or with **quantization / small / low-active models**, "
        "which slash the required CPU count (the gate is weight-bytes / bandwidth). "
        "All numbers are optimistic-overlap, no inter-node fabric cost — ceilings.",
    ]
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "disagg_cpu_decode.md")
    with open(p, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    write_report()
