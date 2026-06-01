"""PD-disaggregation model (Mooncake-style), the FAST'25-critique follow-up.

Mooncake (Qin et al., FAST'25) splits the cluster into a PREFILL pool and a
DECODE pool, with a CPU/DRAM KVCache pool between them and RDMA transfer of KV
from prefill->decode. Crucially Mooncake keeps decode attention ON GPU (KV is
async-loaded back to VRAM; constraint KVCache<VRAM) -- it does NOT offload
attention compute to CPU. So this module models the DISAGGREGATION BENEFIT that
concurrent.py's single contended resource pool cannot express:

  * prefill pool: N_p GPUs, each compute-bound on the (uncached) prefill;
  * decode  pool: N_d GPUs, each HBM-bound, batch capped by VRAM + TBT SLO;
  * RDMA link : per call kv_size(S) moves prefill->decode (Messenger), capped
    by the decode pool's aggregate RDMA NIC bandwidth, and adding TTFT latency
    (layer-wise streamed -> hidden by ov, exactly as in concurrent._mix_ttft).

System throughput (disaggregated) balances the two pools over a fixed GPU
budget G = N_p + N_d:  Lambda = max_split min(N_p*lam_p, N_d*lam_d), maximized
at N_p*lam_p = N_d*lam_d -> Lambda = G * lam_p*lam_d/(lam_p+lam_d).

The honest question this answers: WHEN does disaggregation beat co-location?
In the fluid model co-location ALREADY overlaps prefill-compute with decode-HBM
(different resources -> free max() overlap at the optimistic end), so the
disaggregation win is expected to be largest in PREFILL-HEAVY regimes
(Mooncake's 720:1 trace) and small in our decode-heavy long-context regime.

Run:  python -m model.pd_disagg
"""

from __future__ import annotations

import os

from . import analytical as an
from .concurrent import _batch_slo_cap, _decode_tpot, serving_mix
from .config import Coeffs, HardwareConfig, ModelConfig, SLOConfig, WorkloadConfig

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))


def _prefill_rate_per_gpu(work, model, hw, coeffs):
    """Calls/s one prefill GPU sustains: only the UNCACHED A tokens are
    prefilled (prefix reused from pool), prefill is GEMM+attn compute vs a
    one-shot weight+KV-write HBM read -> busy time = max(compute, hbm)."""
    A, Sc, L = work.a_append, work.s_cached, model.layers
    pf_flops = 2 * model.p_act * A + coeffs.gamma * L * A * (Sc + A / 2.0) * model.d_attn
    t_comp = pf_flops / hw.f_gpu
    t_hbm = (an.weight_read_bytes(model, A) + an.kv_size(A, model)) / hw.bw_hbm
    return 1.0 / max(t_comp, t_hbm)


def _decode_rate_per_gpu(work, model, hw, slo, coeffs, sparse, f, overlap):
    """(calls/s, B) one decode GPU sustains under continuous batching: B capped
    by VRAM capacity AND TBT SLO; tok/s = B/step; calls/s = (B/step)/O."""
    S, O = work.s_context, work.o_output
    w = an.weights_bytes(model)
    free = hw.m_hbm - w - coeffs.m_runtime - coeffs.m_workspace
    per_seq = (1.0 - f) * an.kv_size(S, model)
    if free <= 0:
        return 0.0, 0.0
    b_cap = max(1.0, free / per_seq) if per_seq > 0 else 1e9
    b_slo = _batch_slo_cap(f, S, model, hw, sparse, overlap, slo, coeffs)
    B = max(1.0, min(b_cap, b_slo))
    step = _decode_tpot(B, f, S, model, hw, sparse, overlap, coeffs)
    tok_s = B / step
    return (tok_s / O if O > 0 else 0.0), B


def system_pd(work, model, hw, gpus, *, slo=SLOConfig(), coeffs=Coeffs(),
              sparse=1.0, f=0.0, overlap="optimistic", bw_rdma=200e9):
    """Disaggregated system tok/s with the prefill:decode GPU split balanced."""
    lam_p = _prefill_rate_per_gpu(work, model, hw, coeffs)
    lam_d, B = _decode_rate_per_gpu(work, model, hw, slo, coeffs, sparse, f, overlap)
    if lam_p <= 0 or lam_d <= 0:
        return {"tps": 0.0, "lam_p": lam_p, "lam_d": lam_d, "n_p": 0.0,
                "n_d": 0.0, "B": B, "binding": "infeasible", "ratio": 0.0}
    n_p = gpus * lam_d / (lam_p + lam_d)        # balance point
    n_d = gpus - n_p
    lam = min(n_p * lam_p, n_d * lam_d)
    binding = "balanced"
    # RDMA ingest ceiling on the decode pool: per call kv_size(S) must arrive.
    rdma_cap = n_d * bw_rdma / an.kv_size(work.s_context, model)
    if rdma_cap < lam:
        lam = rdma_cap
        binding = "rdma_transfer"
    return {"tps": lam * work.o_output, "lam_p": lam_p, "lam_d": lam_d,
            "n_p": n_p, "n_d": n_d, "B": B, "binding": binding,
            "ratio": n_p / n_d, "rdma_cap": rdma_cap}


def system_colocated(work, model, hw, gpus, *, slo=SLOConfig(), coeffs=Coeffs(),
                     sparse=1.0, f=0.0, overlap="optimistic", pool=True):
    """Co-located baseline: each GPU runs prefill+decode of its own requests
    (concurrent.serving_mix already contends both phases), system = G x per-GPU.
    pool=True uses the realistic Mooncake-pooled per-GPU baseline."""
    r = serving_mix([(work, 1.0)], model, hw, f=f, sparse=sparse,
                    overlap=overlap, slo=slo, coeffs=coeffs, pool=pool)
    return {"tps": gpus * r["tps"], "per_gpu": r["tps"], "binding": r["binding"]}


def _regimes():
    # Mooncake trace: avg input 7590, output 182 (720:1, prefill-heavy); ~50%
    # prefix-cache hit -> ~half the input is cached, half uncached.
    mooncake = WorkloadConfig(name="mooncake_720to1",
                              s_cached=3795, a_append=3795, o_output=182)
    # Our Codex long-context regime: 64k ctx, 520 output (decode/HBM-heavy).
    codex = WorkloadConfig(name="codex_64k",
                           s_cached=64_338, a_append=3_991, o_output=520)
    return mooncake, codex


def report():
    model = ModelConfig()
    hw = HardwareConfig().effective(0.5)
    slo = SLOConfig(slo_tpot=0.05)
    mooncake, codex = _regimes()
    lines = ["# PD-disaggregation vs co-location (Mooncake-style)\n",
             "> Generated by `python3 -m model.pd_disagg`. dense-70B, G=72, "
             "f=0 (Mooncake keeps attention on GPU), eta=0.5. Disaggregated "
             "system balances the prefill:decode GPU split; co-located runs "
             "both phases on every GPU (concurrent.serving_mix x G, pool=True). "
             "RDMA=200 GB/s.\n",
             "| regime | overlap | co-located tok/s | disagg tok/s | disagg/coloc "
             "| opt N_p:N_d | decode B | bind |",
             "|---|---|---|---|---|---|---|---|"]
    for name, w in [("Mooncake 720:1 (prefill-heavy)", mooncake),
                    ("Codex 64k/520 (decode-heavy)", codex)]:
        for ovl in ["optimistic", "conservative"]:
            co = system_colocated(w, model, hw, 72, slo=slo, overlap=ovl)
            pd = system_pd(w, model, hw, 72, slo=slo, overlap=ovl)
            ratio = pd["tps"] / co["tps"] if co["tps"] > 0 else 0.0
            lines.append(
                f"| {name} | {ovl} | {co['tps']:.0f} | {pd['tps']:.0f} | "
                f"{ratio:.2f}x | {pd['n_p']:.0f}:{pd['n_d']:.0f} | "
                f"{pd['B']:.1f} | {pd['binding']} |")
            print(f"[{name} / {ovl}] coloc={co['tps']:.0f} disagg={pd['tps']:.0f} "
                  f"({ratio:.2f}x) split {pd['n_p']:.0f}:{pd['n_d']:.0f} "
                  f"lam_p={pd['lam_p']:.3f} lam_d={pd['lam_d']:.3f}")
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "pd_disagg.md")
    with open(p, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    report()
