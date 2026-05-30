"""Two-resource (compute vs HBM-bandwidth) serving model + attention offload.

Captures the scenario the simpler models missed: in a PREFILL-HEAVY regime the
GPU has two distinct resources -- compute (F_G) and HBM bandwidth (BW_HBM) --
that prefill and decode contend for:

  * prefill is COMPUTE-bound (weight GEMMs over A tokens),
  * decode core-attention is HBM-BANDWIDTH-bound (streaming the KV cache).

If decode KV streaming saturates HBM bandwidth, the GPU's compute units idle and
prefill throughput drops -- even though there is spare FLOP/s. Offloading a
fraction f of core attention to the CPU REMOVES that KV bandwidth demand from
HBM (the KV lives in CPU DRAM, attention runs there), freeing HBM bandwidth for
prefill. The offloaded work's new bottleneck is CPU DRAM bandwidth and the C2C
link (Q/O transfer) -- NOT HBM. So offload can help precisely when GPU is
HBM-bandwidth-bound, even though CPU attention is "slow".

Per-call GPU resource demand (steady state, decode batch B amortizes weights):
  compute-sec = [2*P_act*A (prefill) + O*2*P_act (decode FFN)] / F_G
  hbm-sec     = [W_prefill + O*((1-f)*KV(S) + W_step/B)] / BW_HBM
  cpu-sec     = O * f * sparse * KV(S) / (BW_CPU * cpus_per_gpu)   # offloaded attn
  c2c-sec     = O * f * L * (d_Q+d_O) * q_act / BW_C2C             # Q/O, not KV
GPU runs compute ∥ HBM -> GPU-sec = max(compute, hbm). The pipeline (GPU ∥ CPU
∥ C2C) call-rate = 1 / max(GPU-sec, cpu-sec, c2c-sec). throughput = O*call-rate.

Run:  python -m model.contention
"""

from __future__ import annotations

import os
from dataclasses import replace

from . import analytical as an
from .config import (Coeffs, HardwareConfig, ModelConfig, PolicyConfig,
                     WorkloadConfig)

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))


def serving_2res(model, hw, work, *, f=0.0, sparse=1.0, cpus_per_gpu=1.0,
                 coeffs=Coeffs()):
    """Two-resource serving throughput with attention offload.  B is the
    HBM-capacity-bounded decode batch (offload f raises it)."""
    A, O, S, L = work.a_append, work.o_output, work.s_context, model.layers
    pol = PolicyConfig.policy("partial_cpu_attn" if f > 0 else "gpu_hot",
                              f=f, b_d=1)
    B = max(1.0, an.decode_batch_cap(pol, work, hw, model, coeffs))

    # GPU compute-seconds per call (prefill FFN + decode FFN)
    pf_flops = 2 * model.p_act * A
    dec_flops = O * 2 * model.p_act
    compute_sec = (pf_flops + dec_flops) / hw.f_gpu

    # GPU HBM-seconds per call
    w_prefill = an.weight_read_bytes(model, A)               # prefill weight read
    w_step = an.weight_read_bytes(model, B)                  # decode weight / step
    dec_hbm = O * ((1.0 - f) * an.kv_size(S, model) + w_step / B)
    hbm_sec = (w_prefill + dec_hbm) / hw.bw_hbm

    gpu_sec = max(compute_sec, hbm_sec)

    # CPU side: offloaded attention over its KV (sparse + compressed via kv_size)
    bw_cpu = hw.bw_cpu * cpus_per_gpu
    cpu_sec = O * f * sparse * an.kv_size(S, model) / bw_cpu if f > 0 else 0.0
    # plus CPU decompress compute if compressed
    if f > 0 and model.kv_decompress_flops > 0:
        f_cpu = hw.f_cpu * cpus_per_gpu
        cpu_sec += O * f * sparse * an.kv_elems(S, model) * model.kv_decompress_flops / f_cpu

    # C2C: Q out + O back per layer (KV stays on CPU). Sequential within a layer
    # -> one C2C direction each -> bw_c2c_oneway (see analytical.c2c_decode_time).
    d_io = model.n_heads * model.head_dim
    c2c_sec = O * f * L * 2 * d_io * model.q_act / hw.bw_c2c_oneway if f > 0 else 0.0

    bottleneck_sec = max(gpu_sec, cpu_sec, c2c_sec)
    tps = O / bottleneck_sec if bottleneck_sec > 0 else 0.0
    parts = {"gpu_compute": compute_sec, "gpu_hbm": hbm_sec,
             "cpu_dram": cpu_sec, "c2c": c2c_sec}
    binding = max(parts, key=parts.get)
    return {"tps": tps, "B": B, "binding": binding,
            "gpu_compute_sec": compute_sec, "gpu_hbm_sec": hbm_sec,
            "cpu_sec": cpu_sec, "c2c_sec": c2c_sec,
            "compute_util_if_hbm_bound": compute_sec / hbm_sec if hbm_sec else 0}


def report():
    work = WorkloadConfig()
    model = ModelConfig()                       # dense-70B
    lines = ["# Prefill/decode HBM-bandwidth contention + core-attention offload\n",
        "> **⚠️ SUPERSEDED** — `serving_2res` is the single-mean-call precursor to "
        "the concurrent fluid model. Current authoritative result: "
        "`concurrent_mix.md` (and `SUMMARY.md`). Kept as a building-block "
        "cross-check; see `README.md` Outputs map.\n",
        "GPU = two resources (compute F_G ∥ HBM bandwidth). Prefill is "
        "compute-bound; decode core-attention is HBM-bandwidth-bound. Offloading "
        "fraction f of core attention to CPU removes its KV streaming from HBM, "
        "freeing HBM bandwidth for prefill. dense-70B, eta=0.5.\n"]
    # Prefill-heavy: use the mean call (A~4k uncached, O~520) -- already 131:1.
    for cpg, label in [(0.5, "NVL72 stock (0.5 Grace/GPU)"),
                       (4.0, "4 Grace/GPU")]:
        hw = HardwareConfig(cpus_per_gpu=cpg).effective(0.5)
        hw = replace(hw, bw_cpu=hw.bw_cpu * cpg, f_cpu=hw.f_cpu * cpg)
        lines += [f"## {label}",
            "| f (offload) | system tok/s | bottleneck | GPU compute (ms) | GPU HBM (ms) | CPU DRAM (ms) | C2C (ms) |",
            "|---|---|---|---|---|---|---|"]
        base = None
        for f in (0.0, 0.1, 0.2, 0.3, 0.5):
            r = serving_2res(model, hw, work, f=f, sparse=1.0, cpus_per_gpu=cpg)
            if base is None:
                base = r["tps"]
            g = r["tps"] / base if base else 1.0
            lines.append(f"| {f} | {r['tps']:.0f} ({g:.2f}x) | {r['binding']} | "
                         f"{r['gpu_compute_sec']*1e3:.0f} | {r['gpu_hbm_sec']*1e3:.0f} | "
                         f"{r['cpu_sec']*1e3:.0f} | {r['c2c_sec']*1e3:.0f} |")
            print(f"{label:28s} f={f}: tps={r['tps']:.0f} bind={r['binding']} "
                  f"(gpu_cmp={r['gpu_compute_sec']*1e3:.0f}ms gpu_hbm={r['gpu_hbm_sec']*1e3:.0f}ms "
                  f"cpu={r['cpu_sec']*1e3:.0f}ms)")
        lines.append("")
    lines += ["## Reading it",
        "- At **f=0** the GPU is **HBM-bandwidth bound** (gpu_hbm >> gpu_compute): "
        "decode KV streaming saturates HBM, GPU compute idles -> prefill starved.",
        "- Raising **f** moves KV streaming off HBM: gpu_hbm drops, throughput "
        "rises UNTIL the bottleneck moves to **CPU DRAM** (or C2C) -- exactly the "
        "regime you described. The optimum f is where gpu_hbm ≈ max(gpu_compute, "
        "cpu_dram).",
        "- On stock NVL72 (0.5 Grace) the CPU bottleneck appears almost "
        "immediately (tiny CPU bandwidth), so the feasible f is small; more CPU "
        "(or sparse/compressed KV to shrink the offloaded bytes) widens it.",
        "- **This is a real win mechanism the time-share models understated**: the "
        "benefit is freeing HBM BANDWIDTH for prefill, not making attention faster.",
    ]
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "hbm_contention_offload.md")
    with open(p, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    report()
