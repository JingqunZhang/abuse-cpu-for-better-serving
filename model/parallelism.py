"""Phase 5 -- parallelism extension (DP/TP/PP/VPP/EP/CP) over an NVL72 rack.

Goal (plan): convert rack-level resources into per-rank resources, add stage /
sharding / MoE-comm terms, make the 671B MoE fit when sharded, and sweep NVL72
deployments to answer: spend GPUs on a bigger replica, more DP replicas, or more
decode batch capacity?

Conventions:
  - A *replica* spans G = TP * PP * CP physical GPUs.  EP distributes experts
    across GPUs already inside the replica (EP <= TP*?), not extra GPUs.
  - DP replicas run independently: total GPUs = DP * G <= rack size (72).
  - Per-rank resources = one GPU's HardwareConfig (HBM capacity is physical).

Sharding (per rank):
  layers_rank   = L / PP                              (+ VPP interleave -> bubble)
  p_act_rank    = p_act(1-ea)/TP + p_act*ea/EP        (TP shards dense, EP experts)
  p_tot_rank    = p_total(1-et)/(TP*PP) + p_total*et/(EP*PP)   (resident weights)
  KV per rank   = M_KV(S) / (PP * TP_kv * CP)         (rho_KV sharding factor)
    where TP_kv = min(TP, n_kv_heads)  (GQA caps KV sharding)

Comm terms added to the per-token decode step:
  TP all-reduce  : 2(TP-1)/TP * hidden * q_act * B / BW_NVLink   per layer_rank
  EP all-to-all  : 2(EP-1)/EP * hidden * q_act * B / BW_NVLink   per MoE layer_rank
  PP bubble      : phi(PP,VPP,m) = 1 + (PP-1)/(VPP*m), m = microbatches ~ B
  EP imbalance   : expert compute * (1 + imbalance)

Run:  python -m model.parallelism
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, replace

from . import analytical as an
from .config import (Coeffs, HardwareConfig, ModelConfig, PolicyConfig,
                     SLOConfig, WorkloadConfig)

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
RACK_GPUS = 72


@dataclass(frozen=True)
class ParallelConfig:
    dp: int = 1
    tp: int = 1
    pp: int = 1
    vpp: int = 1
    ep: int = 1
    cp: int = 1
    ep_imbalance: float = 0.15      # expert-load imbalance penalty
    # Real systems also shard expert FFN matrices by TP (not only by EP). When
    # True, experts are distributed across min(ep*tp, tp*cp) in-stage GPUs, so
    # a non-EP config can still fit a huge MoE -- this is the correction that
    # tests whether "EP is mandatory" was a modeling artifact.
    shard_experts_with_tp: bool = True

    @property
    def expert_pieces(self) -> int:
        """How many in-stage GPUs the expert mass is sharded across.

        EP places whole experts on different ranks; TP-of-experts shards each
        expert's FFN.  Either way the per-rank expert capacity = mass / pieces.
        Capped by the GPUs available within a pipeline stage (tp*cp)."""
        deg = self.ep * (self.tp if self.shard_experts_with_tp else 1)
        return max(1, min(deg, self.tp * self.cp))

    @property
    def gpus_per_replica(self) -> int:
        return self.tp * self.pp * self.cp

    @property
    def total_gpus(self) -> int:
        return self.dp * self.gpus_per_replica

    def valid(self, rack=RACK_GPUS) -> bool:
        return 1 <= self.total_gpus <= rack


@dataclass
class RankView:
    layers_rank: float
    p_act_rank: float
    p_tot_rank: float
    kv_shard: float          # rho_KV: KV-size multiplier per rank (<=1)
    tp_kv: int


def shard(model: ModelConfig, p: ParallelConfig) -> RankView:
    ea, et = model.expert_frac_act, model.expert_frac_total
    layers_rank = model.layers / p.pp
    e = p.expert_pieces                      # in-stage GPUs sharing the experts
    # Active compute: non-expert sharded by TP; experts by the expert grid.
    p_act_rank = model.p_act * (1 - ea) / p.tp + model.p_act * ea / e
    # Resident weights: non-expert by TP*PP; experts by expert_pieces*PP.
    p_tot_rank = (model.p_total * (1 - et) / (p.tp * p.pp)
                  + model.p_total * et / (e * p.pp))
    tp_kv = min(p.tp, model.n_kv_heads)
    kv_shard = 1.0 / (p.pp * tp_kv * p.cp)
    return RankView(layers_rank, p_act_rank, p_tot_rank, kv_shard, tp_kv)


def pp_bubble(p: ParallelConfig, microbatches: float) -> float:
    if p.pp <= 1:
        return 1.0
    return 1.0 + (p.pp - 1) / (p.vpp * max(1.0, microbatches))


@dataclass
class RankStep:
    tpot: float               # optimistic-overlap per-token time
    tpot_cons: float          # conservative (fully serialized) per-token time
    weight_read: float
    compute: float
    gpu_attn: float
    cpu_attn: float
    comm_tp: float
    comm_ep: float
    bubble: float
    hbm_used: float
    b_cap: float


def _rank_weight_read_bytes(model: ModelConfig, p: ParallelConfig, b_d: int) -> float:
    """Per-rank weight bytes STREAMED per decode step, mirroring analytical
    weight_read_bytes (batch saturation) but on the sharded expert mass.

    Non-routed weights (sharded by TP*PP) are always read; routed experts
    (sharded by expert_pieces*PP) are read only as the batch covers them."""
    ea, et = model.expert_frac_act, model.expert_frac_total
    nonroute = model.p_total * (1 - et) / (p.tp * p.pp)
    if et <= 0 or ea <= 0:                      # dense: all resident read
        return (nonroute + model.p_total * et / (p.expert_pieces * p.pp)) * model.q_weight
    route_resident = model.p_total * et / (p.expert_pieces * p.pp)
    rho = min(1.0, (model.p_act * ea) / (model.p_total * et))
    coverage = 1.0 - (1.0 - rho) ** b_d
    return (nonroute + route_resident * coverage) * model.q_weight


def rank_step(model: ModelConfig, hw: HardwareConfig, p: ParallelConfig,
              work: WorkloadConfig, *, f: float, sparse: float, b_d: int,
              coeffs: Coeffs = Coeffs()) -> RankStep:
    """Per-rank decode TPOT (optimistic & conservative) and HBM usage."""
    rv = shard(model, p)
    kv_rank = an.kv_size(work.s_context, model) * rv.kv_shard
    kv_cached_rank = an.kv_size(work.s_cached, model) * rv.kv_shard

    # linear-layer time: compute vs weight-read (batch-saturating for MoE,
    # consistent with analytical.weight_read_bytes).
    compute = 2 * rv.p_act_rank * b_d / hw.f_gpu
    weight_read = _rank_weight_read_bytes(model, p, b_d) / hw.bw_hbm
    # EP/expert imbalance inflates the expert part of compute
    if model.expert_frac_act > 0 and p.expert_pieces > 1:
        compute_exp = (2 * (model.p_act * model.expert_frac_act / p.expert_pieces)
                       * b_d / hw.f_gpu)
        compute += compute_exp * p.ep_imbalance
    nonattn = max(compute, weight_read)

    gpu_attn = (1 - f) * b_d * kv_rank / hw.bw_hbm
    cpu_attn = 0.0
    c2c = 0.0
    if f > 0:
        cpu_attn = sparse * f * b_d * kv_rank / hw.bw_cpu
        d_io = model.n_heads * model.head_dim
        # Q-out then O-back are SEQUENTIAL within a layer -> one C2C direction
        # each -> bw_c2c_oneway (not the bidirectional aggregate). See
        # analytical.c2c_decode_time for the full rationale.
        c2c = ((rv.layers_rank * b_d * 2 * d_io * model.q_act) / hw.bw_c2c_oneway
               + 2 * rv.layers_rank * hw.t_sync)

    d_model = model.hidden
    tp_ar = 0.0
    if p.tp > 1:
        tp_ar = (2 * (p.tp - 1) / p.tp * d_model * model.q_act * b_d
                 / hw.bw_nvlink) * rv.layers_rank
    ep_a2a = 0.0
    if p.ep > 1 and model.expert_frac_act > 0:
        ep_a2a = (2 * (p.ep - 1) / p.ep * d_model * model.q_act * b_d
                  / hw.bw_nvlink) * rv.layers_rank

    bubble = pp_bubble(p, microbatches=b_d)
    merge = coeffs.t_merge if f > 0 else 0.0
    comm = tp_ar + ep_a2a
    tpot = bubble * (nonattn + max(gpu_attn, cpu_attn + c2c) + comm + merge)
    tpot_cons = bubble * (nonattn + gpu_attn + cpu_attn + c2c + comm + merge)

    # HBM per rank (full resident weights, regardless of read saturation)
    w_bytes = rv.p_tot_rank * model.q_weight
    overhead = coeffs.m_runtime + coeffs.m_workspace
    kv_decode = b_d * (1 - f) * kv_rank
    kv_append = 1 * 1.0 * kv_cached_rank      # B_p=1, r=1
    hbm_used = w_bytes + overhead + kv_decode + kv_append

    free = hw.m_hbm - w_bytes - overhead - kv_append
    per_seq = (1 - f) * kv_rank
    b_cap = (free / per_seq) if per_seq > 0 else float("inf")
    if free <= 0:
        b_cap = 0.0
    return RankStep(tpot, tpot_cons, weight_read, compute, gpu_attn, cpu_attn,
                    tp_ar, ep_a2a, bubble, hbm_used, b_cap)


@dataclass
class ConfigResult:
    pcfg: ParallelConfig
    f: float
    sparse: float
    b_d: int                  # per-replica decode batch
    tpot: float               # optimistic
    tpot_cons: float          # conservative
    ttft: float
    replica_tps: float        # tokens/s for one replica (optimistic)
    system_tps: float         # dp * replica_tps (optimistic)
    system_tps_cons: float    # dp * (b_d / tpot_cons)  -> conservative band
    feasible: bool
    fits: bool


B_GRID = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]


def best_for_config(model, hw, p, work, slo: SLOConfig, *, f, sparse,
                    coeffs=Coeffs()):
    """Search per-replica B_d to maximize system throughput under SLO."""
    rv = shard(model, p)
    best = None
    fits_any = False
    for b_d in B_GRID:
        rs = rank_step(model, hw, p, work, f=f, sparse=sparse, b_d=b_d,
                       coeffs=coeffs)
        fits = rs.hbm_used <= hw.m_hbm and b_d <= rs.b_cap + 1e-9
        if rs.hbm_used <= hw.m_hbm:
            fits_any = True
        if not fits:
            continue
        # TTFT: append-prefill compute + old-KV load over the ONE-WAY C2C link.
        t_pf = 2 * rv.p_act_rank * work.a_append / hw.f_gpu      # prefill compute / rank
        appt = t_pf + an.kv_size(work.s_cached, model) * rv.kv_shard / hw.bw_c2c_oneway
        ttft = appt + rs.tpot
        if rs.tpot <= slo.slo_tpot and ttft <= slo.slo_ttft:
            # SERVING throughput: GPU-seconds per call = prefill + O*TPOT/B
            # (decode batched). Replaces the decode-only b_d/TPOT so the number
            # reflects real serving on a prefill-heavy workload.
            O = work.o_output
            replica_tps = O / (t_pf + O * rs.tpot / b_d)
            system_tps = p.dp * replica_tps
            system_tps_cons = (p.dp * O / (t_pf + O * rs.tpot_cons / b_d)
                               if rs.tpot_cons > 0 else 0.0)
            if best is None or system_tps > best.system_tps:
                best = ConfigResult(p, f, sparse, b_d, rs.tpot, rs.tpot_cons,
                                    ttft, replica_tps, system_tps,
                                    system_tps_cons, True, True)
    if best is None:
        return ConfigResult(p, f, sparse, 0, float("inf"), float("inf"),
                            float("inf"), 0.0, 0.0, 0.0, False, fits_any)
    return best


# Reasonable NVL72 deployments (replica_gpus x dp), per the plan.
def deployment_grid(rack=RACK_GPUS):
    cfgs = []
    # one 72-GPU replica and DP fan-outs of a 72/N-GPU replica
    layouts = [
        ("1x72 (TP8,PP9)",  ParallelConfig(dp=1, tp=8, pp=9)),
        ("2x36 (TP8,PP4)",  ParallelConfig(dp=2, tp=8, pp=4)),  # 2*32=64<=72
        ("4x18 (TP6,PP3)",  ParallelConfig(dp=4, tp=6, pp=3)),  # 4*18=72
        ("8x9  (TP3,PP3)",  ParallelConfig(dp=8, tp=3, pp=3)),  # 8*9=72
        ("1x72 VPP (TP8,PP9,VPP4)", ParallelConfig(dp=1, tp=8, pp=9, vpp=4)),
        ("EP-heavy 1x64 (TP8,PP8,EP8)", ParallelConfig(dp=1, tp=8, pp=8, ep=8)),
        ("EP 2x32 (TP8,PP4,EP8)", ParallelConfig(dp=2, tp=8, pp=4, ep=8)),
        ("TP-only 9x8 (TP8)", ParallelConfig(dp=9, tp=8, pp=1)),  # 9*8=72
    ]
    return [(name, c) for name, c in layouts if c.valid(rack)]


def run_sweep(model=None, hw=None, work=None, slo=None):
    model = model or ModelConfig.preset("moe_large_mla")
    hw = hw or HardwareConfig().effective(0.5)
    work = work or WorkloadConfig()
    slo = slo or SLOConfig()
    rows = []
    results = []
    for name, p in deployment_grid(hw.n_gpus):
        for f, sparse in [(0.0, 1.0), (0.3, 0.1)]:
            r = best_for_config(model, hw, p, work, slo, f=f, sparse=sparse)
            results.append((name, r))
            rows.append(dict(
                layout=name, dp=p.dp, tp=p.tp, pp=p.pp, vpp=p.vpp, ep=p.ep,
                cp=p.cp, gpus=p.total_gpus, offload_f=f, sparse=sparse,
                fits=r.fits, feasible=r.feasible, b_d_per_replica=r.b_d,
                tpot_ms=r.tpot * 1e3 if r.feasible else "",
                ttft_s=round(r.ttft, 3) if r.feasible else "",
                replica_tps=round(r.replica_tps, 1),
                system_tps_optimistic=round(r.system_tps, 1),
                system_tps_conservative=round(r.system_tps_cons, 1),
            ))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "parallelism_sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.join(OUT, 'parallelism_sweep.csv')} ({len(rows)} rows)")
    write_recommendations(results, model=model, hw=hw, work=work, slo=slo)
    return results


def write_recommendations(results, model=None, hw=None, work=None, slo=None):
    feas = [(n, r) for n, r in results if r.feasible]
    feas.sort(key=lambda x: -x[1].system_tps)
    lines = ["# Phase 5 -- parallelism: recommended NVL72 configs\n",
             "Model: 671B MoE (MLA KV), workload = Codex/SWEBenchPro mean call, "
             "eta=0.5, SLO TPOT<=50ms / TTFT<=10s. System throughput = "
             "DP x (B_d / TPOT). Offload = sparse CPU co-attention f=0.3.\n",
             "**All tok/s are reported as an OPTIMISTIC..CONSERVATIVE overlap band**; "
             "and they omit real inter-node fabric cost, so treat them as ceilings "
             "(plausibly ~2-4x above a production stack).\n",
             "## Ranked feasible deployments (by optimistic system tokens/s)\n",
             "| rank | layout | GPUs | offload | B_d/replica | TPOT (ms) | "
             "system tok/s (opt..cons) |", "|---|---|---|---|---|---|---|"]
    for i, (n, r) in enumerate(feas[:12], 1):
        off = f"f={r.f}" if r.f > 0 else "none"
        lines.append(f"| {i} | {n} | {r.pcfg.total_gpus} | {off} | {r.b_d} | "
                     f"{r.tpot*1e3:.1f} | {r.system_tps:.0f}..{r.system_tps_cons:.0f} |")
    # --- "Is EP mandatory?" test: does a non-EP config fit once TP also shards
    # experts?  Compare expert-TP ON vs OFF for a non-EP layout.
    import dataclasses as _dc
    model = model or ModelConfig.preset("moe_large_mla")
    hw = hw or HardwareConfig().effective(0.5)
    work = work or WorkloadConfig()
    slo = slo or SLOConfig()
    # Use a LOW-PP non-EP layout (pp alone can't shard the experts enough) so
    # the expert-TP-vs-EP-only distinction actually shows.
    nonep = ParallelConfig(dp=1, tp=8, pp=4, ep=1)               # expert-TP ON (default)
    nonep_off = _dc.replace(nonep, shard_experts_with_tp=False)  # EP-only sharding
    r_on = best_for_config(model, hw, nonep, work, slo, f=0.0, sparse=1.0)
    r_off = best_for_config(model, hw, nonep_off, work, slo, f=0.0, sparse=1.0)
    lines += ["",
        "## Is expert parallelism actually mandatory? (round-2 correction)",
        "",
        f"- Non-EP layout `TP8,PP4, ep=1` (32 GPUs) with **TP also sharding "
        f"experts**: fits={r_on.fits}, feasible={r_on.feasible}"
        + (f", {r_on.system_tps:.0f} tok/s." if r_on.feasible else "."),
        f"- Same layout with **EP-only expert sharding** (the old assumption): "
        f"fits={r_off.fits} (experts only /PP=4 -> too big per rank).",
        "",
        ("**Verdict: EP is NOT strictly mandatory to *fit*** -- TP (×PP) shards the "
         "expert FFNs just as well, so a non-EP replica fits the 671B MoE. The "
         "earlier 'EP mandatory' was a modeling artifact (experts were sharded only "
         "by EP). EP still helps *throughput* by replacing per-layer TP all-reduce "
         "with all-to-all routing and reducing per-rank expert compute, but it is a "
         "performance choice, not a fitting requirement."
         if (r_on.fits and not r_off.fits) else
         "**Verdict: in this configuration the expert-TP vs EP-only distinction did "
         "not change feasibility** (both "
         f"fit={r_on.fits}/{r_off.fits}); see the sweep for the throughput effect.")]

    # --- Offload benefit is overlap-dependent (the band exposes this) ---
    pp9 = ParallelConfig(dp=1, tp=8, pp=9)
    f0 = best_for_config(model, hw, pp9, work, slo, f=0.0, sparse=1.0)
    f3 = best_for_config(model, hw, pp9, work, slo, f=0.3, sparse=0.1)
    if f0.feasible and f3.feasible:
        lines += ["",
            "## CAUTION: the sparse-offload gain depends on the overlap assumption",
            "",
            f"For `1x72 (TP8,PP9)`: f=0 gives {f0.system_tps:.0f} tok/s (opt==cons, "
            f"no CPU path to overlap). f=0.3 gives **{f3.system_tps:.0f} optimistic** "
            f"but only **{f3.system_tps_cons:.0f} conservative**.",
            "",
            (f"-> Under OPTIMISTIC overlap, sparse offload helps "
             f"(+{100*(f3.system_tps/f0.system_tps-1):.0f}%). Under CONSERVATIVE "
             f"overlap it **{'helps' if f3.system_tps_cons>f0.system_tps else 'HURTS'}** "
             f"({100*(f3.system_tps_cons/f0.system_tps-1):+.0f}%). So the MoE offload "
             "benefit is real only if CPU attention + C2C genuinely overlap GPU work "
             "(ScoutAttention layer-ahead). Treat the offload uplift as best-case."),
        ]

    if feas:
        best_name, best = feas[0]
        lines += ["",
            "## Decision: bigger replica vs more DP vs more batch?",
            "",
            f"- **Best feasible config:** {best_name} ({best.pcfg.total_gpus} GPUs, "
            f"{'sparse offload' if best.f>0 else 'no offload'}), "
            f"{best.system_tps:.0f}..{best.system_tps_cons:.0f} system tok/s "
            f"(opt..cons) at B_d={best.b_d}/replica.",
            "- **Bigger replica (more TP/PP)** is needed to fit 1.34 TB of weights "
            "(~7+ GPUs minimum) and to shrink per-rank weight-read (the TPOT floor); "
            "beyond the fit/TPOT minimum, extra TP/PP mostly adds comm.",
            "- **More DP replicas** multiply throughput linearly *once each replica "
            "meets the SLO* (this model assumes zero cross-replica cost, so DP "
            "linearity is an upper bound) -- the place to spend leftover GPUs.",
            "- **More decode batch capacity** is the cheapest lever but is capped by "
            "HBM and the TPOT SLO; sparse CPU offload raises the cap.",
            "",
            "**Rule of thumb:** size the replica to the smallest TP*PP that fits "
            "weights and meets the TPOT floor; add sparse CPU offload to grow batch; "
            "spend remaining rack GPUs on DP. (tok/s are optimistic ceilings.)",
        ]
    else:
        lines.append("\n**No feasible config under the default SLO** -- loosen "
                     "TPOT/TTFT or raise efficiency.")
    with open(os.path.join(OUT, "recommended_configs.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.join(OUT, 'recommended_configs.md')}")


if __name__ == "__main__":
    run_sweep()
