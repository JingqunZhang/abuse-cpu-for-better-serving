"""Closed-form analytical model (Phase 2, sections 5-8 of the plan).

Every public function takes the four config objects (policy, workload, hardware,
model) plus optional Coeffs, and returns either a scalar (seconds / bytes /
tokens-per-second) or a small result dataclass.  Hardware passed in is expected
to already be efficiency-scaled via HardwareConfig.effective(eta) by the caller;
the sweep driver does this.

The structure mirrors the plan exactly:

    kv_size            section 4.2
    hbm_capacity       section 5
    cpu_capacity       section 5
    decode_batch_cap   section 5
    append_time        section 6
    tpot               section 7  (optimistic & conservative overlap)
    c2c_util           section 7.2
    tps / gain         section 8
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import (Coeffs, HardwareConfig, ModelConfig, PolicyConfig,
                     SLOConfig, WorkloadConfig)


# --------------------------------------------------------------------------
# 4.2  KV size
# --------------------------------------------------------------------------
def kv_size(s_tokens: float, model: ModelConfig) -> float:
    """M_KV(S) = L * S * d_KV * q_KV * kv_compress  (bytes ACTUALLY stored/read).

    kv_compress<1 models KV compression/quantization: fewer bytes to store,
    transfer (C2C), and stream for attention. The compute cost of decompressing
    is charged separately in the attention rooflines (see kv_elems)."""
    return model.layers * s_tokens * model.d_kv * model.q_kv * model.kv_compress


def kv_elems(s_tokens: float, model: ModelConfig) -> float:
    """Uncompressed KV element count (L*S*d_KV) -- what must be DECOMPRESSED to
    compute attention, regardless of how few bytes were stored."""
    return model.layers * s_tokens * model.d_kv


# --------------------------------------------------------------------------
# 5  Capacity model
# --------------------------------------------------------------------------
def weights_bytes(model: ModelConfig) -> float:
    """Resident model weights in HBM (total params; MoE keeps all experts hot)."""
    return model.p_total * model.q_weight


def weight_read_bytes(model: ModelConfig, b_d: float) -> float:
    """Weight bytes STREAMED from HBM per decode step at batch b_d.

    Dense: all active params (= all params) every step, batch-independent.
    MoE: non-routed params (attention/shared, p_act*(1-ea)) are always read;
    routed experts (p_total*et) are read only insofar as the batch touches them.
    A token activates fraction rho = (p_act*ea)/(p_total*et) of expert mass; a
    batch of b_d tokens covers 1-(1-rho)^b_d of it, so the read grows from
    ~p_act at b_d=1 toward ~p_total as the batch saturates all experts. This
    corrects the old P_act-only term that under-counted MoE at large batch.
    """
    ea, et = model.expert_frac_act, model.expert_frac_total
    if et <= 0 or ea <= 0:                     # dense
        return model.p_act * model.q_weight
    nonroute = model.p_act * (1.0 - ea)        # always read
    route_total = model.p_total * et
    rho = min(1.0, (model.p_act * ea) / route_total)
    coverage = 1.0 - (1.0 - rho) ** b_d
    return (nonroute + route_total * coverage) * model.q_weight


@dataclass
class HBMUsage:
    used: float
    free: float
    weights: float
    kv_decode: float
    kv_append: float
    overhead: float
    fits: bool


def hbm_capacity(pol: PolicyConfig, work: WorkloadConfig,
                 hw: HardwareConfig, model: ModelConfig,
                 coeffs: Coeffs = Coeffs()) -> HBMUsage:
    """GPU HBM usage (section 5).

        M_used = M_weights + M_runtime + M_workspace
                 + B_d (1-f) M_KV(S)            <- hot decode KV on GPU
                 + B_p  r    M_KV(S_c)          <- old KV materialized for append
    """
    w = weights_bytes(model)
    overhead = coeffs.m_runtime + coeffs.m_workspace
    kv_decode = pol.b_d * (1.0 - pol.f) * kv_size(work.s_context, model)
    kv_append = pol.b_p * pol.r * kv_size(work.s_cached, model)
    used = w + overhead + kv_decode + kv_append
    free = hw.m_hbm - used
    return HBMUsage(used=used, free=free, weights=w, kv_decode=kv_decode,
                    kv_append=kv_append, overhead=overhead, fits=used <= hw.m_hbm)


def cpu_capacity(n_resident: float, work: WorkloadConfig,
                 hw: HardwareConfig, model: ModelConfig) -> dict:
    """CPU DRAM usage: M_CPU_used = N_resident * M_KV(S)."""
    per = kv_size(work.s_context, model)
    used = n_resident * per
    return {"used": used, "free": hw.m_cpu - used,
            "fits": used <= hw.m_cpu, "per_session": per,
            "max_resident": int(hw.m_cpu // per) if per > 0 else 0}


def decode_batch_cap(pol: PolicyConfig, work: WorkloadConfig,
                     hw: HardwareConfig, model: ModelConfig,
                     coeffs: Coeffs = Coeffs()) -> float:
    """Max decode batch B_HBM(f) that fits in free HBM (section 5).

        B_HBM(f) = M_HBM_free / ((1-f) M_KV(S))

    Free HBM here excludes weights/overhead and the append reservation.  As
    f -> 1 the hot-KV cost vanishes and capacity -> infinity (returns inf),
    which the throughput model then bounds by other resources.
    """
    w = weights_bytes(model)
    kv_append = pol.b_p * pol.r * kv_size(work.s_cached, model)
    free = hw.m_hbm - w - coeffs.m_runtime - coeffs.m_workspace - kv_append
    if free <= 0:
        return 0.0
    per_seq = (1.0 - pol.f) * kv_size(work.s_context, model)
    if per_seq <= 0:
        return float("inf")
    return free / per_seq


# --------------------------------------------------------------------------
# 6  Append-prefill model
# --------------------------------------------------------------------------
@dataclass
class AppendTime:
    total: float
    old_kv_load: float
    gpu_compute: float
    new_kv_flush: float


def append_time(pol: PolicyConfig, work: WorkloadConfig,
                hw: HardwareConfig, model: ModelConfig,
                coeffs: Coeffs = Coeffs()) -> AppendTime:
    """T_append = T_oldKV_load + T_GPU_append + T_newKV_flush (section 6).

    Modeling rule: if CPU already holds authoritative old KV (cpu_backing),
    only *new* KV is flushed back -- old KV is not re-flushed.  Old KV must
    still be *loaded* to GPU (fraction r) for GPU-side append-prefill.
    """
    # Old KV load: bring r * S_c worth of KV from CPU over C2C (one-directional).
    old_kv_load = pol.b_p * pol.r * kv_size(work.s_cached, model) / hw.bw_c2c_oneway

    # GPU append-prefill: max of compute, attention-compute, HBM traffic.
    flops_linear = 2 * model.p_act * pol.b_p * work.a_append
    flops_attn = (coeffs.gamma * pol.b_p * model.layers * work.a_append
                  * (work.s_cached + work.a_append / 2) * model.d_attn)
    # HBM bytes per append step: weight read (dominates small-A, weight-bound
    # appends) plus new-KV written.  Prefill reads weights once like any forward.
    bytes_append = (weight_read_bytes(model, pol.b_p)
                    + kv_size(work.a_append, model) * pol.b_p)
    t_compute = max(flops_linear / hw.f_gpu,
                    flops_attn / hw.f_gpu,
                    bytes_append / hw.bw_hbm)

    # New KV flush back to CPU backing store (one-directional C2C).
    if pol.cpu_backing:
        new_kv_flush = pol.b_p * kv_size(work.a_append, model) / hw.bw_c2c_oneway
    else:
        new_kv_flush = 0.0

    total = old_kv_load + t_compute + new_kv_flush
    return AppendTime(total=total, old_kv_load=old_kv_load,
                      gpu_compute=t_compute, new_kv_flush=new_kv_flush)


# --------------------------------------------------------------------------
# 7  Decode TPOT model
# --------------------------------------------------------------------------
def gpu_nonattn_time(pol: PolicyConfig, hw: HardwareConfig,
                     model: ModelConfig) -> float:
    """T_QKV + T_Oproj + T_MoE : compute- or weight-read-bound linear layers."""
    flops = 2 * model.p_act * pol.b_d
    t_compute = flops / hw.f_gpu
    # Weight read per decode step: batch-dependent for MoE (expert saturation),
    # batch-independent (= P_act) for dense.  See weight_read_bytes.
    t_weight = weight_read_bytes(model, pol.b_d) / hw.bw_hbm
    return max(t_compute, t_weight)


def gpu_attn_time(pol: PolicyConfig, work: WorkloadConfig,
                  hw: HardwareConfig, model: ModelConfig,
                  coeffs: Coeffs = Coeffs()) -> float:
    """T_GPUattn for the GPU's (1-f) KV share.

    Decode attention is memory-bound (stream KV once), but we take max() with
    the attention-FLOP roofline for symmetry with the CPU/append terms -- it
    only bites at large batch / short context.

    APPLES-TO-APPLES sparsity (audit fix): `pol.sparse` scales the GPU's OWN
    decode attention too, not just the offloaded CPU path. When the attention
    algorithm is block-sparse, the GPU reads only `sparse` of the context per
    token as well; previously this term ignored pol.sparse, so the f=0 baseline
    was denied the discount the offload path got, inflating the legacy sparse
    optimize_f gain (~1.86x vs the fair ~1.76x). concurrent.py already does this.
    """
    share = (1.0 - pol.f) * pol.b_d
    s = pol.sparse
    mem = share * s * kv_size(work.s_context, model) / hw.bw_hbm
    flops = coeffs.gamma * share * s * model.layers * work.s_context * model.d_attn
    decomp = share * s * kv_elems(work.s_context, model) * model.kv_decompress_flops
    return max(mem, (flops + decomp) / hw.f_gpu)


def cpu_attn_time(pol: PolicyConfig, work: WorkloadConfig,
                  hw: HardwareConfig, model: ModelConfig,
                  coeffs: Coeffs = Coeffs()) -> float:
    """T_CPUattn = max(mem-traffic, compute) for the CPU's f-fraction.

    Sparsity (pol.sparse) scales both the KV bytes read and the attention FLOPs
    -- ScoutAttention reads only block-selected KV, so the CPU touches
    `sparse * S` context per token.
    """
    if pol.f <= 0:
        return 0.0
    s = pol.sparse
    mem = coeffs.alpha * pol.f * pol.b_d * s * kv_size(work.s_context, model) / hw.bw_cpu
    # CPU compute = attention FLOPs + KV DECOMPRESSION FLOPs. Decompress is the
    # tax compression imposes; on the compute-starved CPU it can dominate.
    flops_attn = coeffs.beta * pol.f * pol.b_d * model.layers * s * work.s_context * model.d_attn
    flops_decomp = pol.f * pol.b_d * s * kv_elems(work.s_context, model) * model.kv_decompress_flops
    comp = (flops_attn + flops_decomp) / hw.f_cpu
    return max(mem, comp)


def c2c_decode_time(pol: PolicyConfig, hw: HardwareConfig,
                    model: ModelConfig) -> float:
    """T_C2C_decode = L B_d (d_Q+d_O) q_act / BW_C2C_oneway + 2 L t_sync (sec 7.2).

    Only Q out / O back travel per layer when CPU computes attention -- the
    KV itself stays resident in CPU DRAM, which is the whole point.

    Directionality: within a layer Q-out (GPU->CPU) and O-back (CPU->GPU) are
    SEQUENTIAL -- O cannot transfer until the CPU has finished attention -- so
    each leg uses a single C2C direction and is charged at bw_c2c_oneway (half
    the bidirectional aggregate). The bidirectional peak applies only if Q-out of
    layer L+1 overlaps O-back of layer L (a layer-ahead double-buffered pipeline);
    we keep the honest, no-pipeline one-way default. (Previously divided by the
    full bidirectional bw_c2c, making this term ~2x too optimistic.)
    """
    if pol.f <= 0:
        return 0.0
    d_q = model.n_heads * model.head_dim
    d_o = model.n_heads * model.head_dim
    transfer = (model.layers * pol.b_d * (d_q + d_o) * model.q_act) / hw.bw_c2c_oneway
    sync = 2 * model.layers * hw.t_sync
    return transfer + sync


@dataclass
class TPOT:
    optimistic: float
    conservative: float
    gpu_nonattn: float
    gpu_attn: float
    cpu_attn: float
    c2c: float


def tpot(pol: PolicyConfig, work: WorkloadConfig, hw: HardwareConfig,
         model: ModelConfig, coeffs: Coeffs = Coeffs()) -> TPOT:
    """Time-per-output-token under optimistic and conservative overlap (7.3)."""
    g_non = gpu_nonattn_time(pol, hw, model)
    g_attn = gpu_attn_time(pol, work, hw, model, coeffs)
    c_attn = cpu_attn_time(pol, work, hw, model, coeffs)
    c2c = c2c_decode_time(pol, hw, model)
    merge = coeffs.t_merge if pol.f > 0 else 0.0

    # Fixed per-step dispatch/kernel-launch overhead (LIFE eq.4), batch-independent.
    disp = coeffs.t_dispatch
    # Optimistic: CPU attention (+its transfer) overlaps GPU attention.
    optimistic = g_non + max(g_attn, c_attn + c2c) + merge + disp
    # Conservative: everything serializes.
    conservative = g_non + g_attn + c_attn + c2c + merge + disp
    return TPOT(optimistic=optimistic, conservative=conservative,
                gpu_nonattn=g_non, gpu_attn=g_attn, cpu_attn=c_attn, c2c=c2c)


# --------------------------------------------------------------------------
# 7.2  C2C utilization
# --------------------------------------------------------------------------
def c2c_util(pol: PolicyConfig, work: WorkloadConfig, hw: HardwareConfig,
             model: ModelConfig, coeffs: Coeffs = Coeffs()) -> dict:
    """C2C bandwidth utilization, split into decode and append demand.

    decode_util: the per-layer Q-out/O-back exchange (bidirectional) during
      decode.  This traffic is ALREADY on the decode TPOT critical path (via
      c2c_decode_time), so by construction decode_bw <= bw_c2c -- it must not be
      throttled again (that was the old double-count).  Reported for diagnostics.
    append_util: the one-directional old-KV load + new-KV flush bulk, averaged
      over a call.  This shares the link with decode but is NOT in TPOT, so it
      is the only genuinely additive contention the throughput model throttles.
    """
    d_q = model.n_heads * model.head_dim
    d_o = model.n_heads * model.head_dim
    decode_bytes = (model.layers * pol.b_d * (d_q + d_o) * model.q_act
                    if pol.f > 0 else 0.0)
    tp = tpot(pol, work, hw, model, coeffs).optimistic
    decode_bw = decode_bytes / tp if tp > 0 else 0.0

    appt = append_time(pol, work, hw, model, coeffs)
    # times were computed against the one-way link, so recover bytes with it
    append_bytes = (appt.old_kv_load + appt.new_kv_flush) * hw.bw_c2c_oneway
    call_time = appt.total + work.o_output * tp
    append_bw = append_bytes / call_time if call_time > 0 else 0.0

    # Both the decode Q/O exchange and the append bulk are SEQUENTIAL one-way
    # transfers (see c2c_decode_time / append_time), so both utilizations are on
    # the SAME one-way basis -- the old code normalized decode by the
    # bidirectional bw_c2c and append by the one-way bw, summing two different
    # bases. With a common basis `util` (=decode+append) is a meaningful share of
    # the one physical link.
    decode_util = decode_bw / hw.bw_c2c_oneway if hw.bw_c2c > 0 else float("inf")
    append_util = append_bw / hw.bw_c2c_oneway if hw.bw_c2c > 0 else float("inf")
    return {"decode_bw": decode_bw, "append_bw": append_bw,
            "decode_util": decode_util, "append_util": append_util,
            "capacity_bw": hw.bw_c2c_oneway,
            "util": decode_util + append_util}


# --------------------------------------------------------------------------
# 8  Throughput model
# --------------------------------------------------------------------------
@dataclass
class Throughput:
    tps: float                 # min-bound system output tokens/s
    bound: str                 # which resource bound is active
    components: dict           # all candidate bounds
    tpot_s: float              # optimistic TPOT used
    batch: int                 # B_d_eff actually run (capacity-bounded)
    batch_target: int          # B_d requested by policy
    batch_cap: float           # B_HBM(f): max batch that fits HBM
    hbm_fits: bool
    cpu_c2c_util: float


def tps(pol: PolicyConfig, work: WorkloadConfig, hw: HardwareConfig,
        model: ModelConfig, coeffs: Coeffs = Coeffs(),
        overlap: str = "optimistic") -> Throughput:
    """System output-token throughput (section 8).

    The active resource bound is encoded INSIDE TPOT via its internal rooflines
    (gpu_nonattn = max(compute, weight-read); attn = max(GPU-HBM, CPU+C2C)), so
    the served rate is B_d_eff / TPOT -- this already reflects whichever of GPU
    compute / HBM bandwidth / CPU attention / decode-C2C binds.  The `components`
    dict exposes each candidate for diagnostics; it is NOT re-min'd here to avoid
    double-counting terms already in TPOT.

    KEY COUPLING (the whole point of offload): the decode batch actually run is
    capacity-bounded,  B_d_eff = min(B_d_target, floor(B_HBM(f))),  and B_HBM(f)
    grows ~1/(1-f) as CPU takes KV off HBM.  Increasing f raises B_d_eff
    (capacity) but also raises TPOT -- their ratio is the crossover that makes
    Gain(f) concave.  The ONLY extra throttle applied here is append-induced C2C
    contention (append bulk shares the link but isn't in TPOT); decode-C2C is
    already in TPOT and is not charged again.
    """
    from dataclasses import replace as _replace

    cap = decode_batch_cap(pol, work, hw, model, coeffs)
    b_eff = pol.b_d if cap >= pol.b_d else max(1, int(cap))
    pol_eff = _replace(pol, b_d=b_eff)

    tp = tpot(pol_eff, work, hw, model, coeffs)
    tpot_s = tp.optimistic if overlap == "optimistic" else tp.conservative
    base = b_eff / tpot_s if tpot_s > 0 else 0.0

    comp = {
        "served_rate": base,
        "gpu_nonattn_bound": b_eff / tp.gpu_nonattn if tp.gpu_nonattn > 0 else float("inf"),
        "gpu_attn_bound": b_eff / tp.gpu_attn if tp.gpu_attn > 0 else float("inf"),
        "cpu_attn_bound": b_eff / tp.cpu_attn if tp.cpu_attn > 0 else float("inf"),
        "c2c_bound": b_eff / tp.c2c if tp.c2c > 0 else float("inf"),
    }

    hbm = hbm_capacity(pol_eff, work, hw, model, coeffs)
    cu = c2c_util(pol_eff, work, hw, model, coeffs)

    served = base
    bound = "gpu_attn(tpot)" if pol.f == 0 else "tpot_balanced"
    if b_eff < pol.b_d:
        bound = "hbm_capacity"
    # C2C link sharing: decode Q/O traffic already occupies `decode_util` of the
    # one-way link (it is on the TPOT path, so it is NOT re-throttled here). The
    # append bulk competes for the RESIDUAL (1 - decode_util) capacity. It only
    # throttles throughput when the two together oversubscribe the link, i.e.
    # append_util > 1 - decode_util  <=>  decode_util + append_util > 1. The
    # throttle factor scales the served rate by the link demand the append bulk
    # cannot fit -- decode is left untouched (no double-count of TPOT traffic).
    residual = max(0.0, 1.0 - cu["decode_util"])
    if cu["append_util"] > residual:
        served = base * residual / cu["append_util"] if cu["append_util"] > 0 else base
        bound = "c2c_append_bandwidth"
    if not hbm.fits:
        served = 0.0
        bound = "hbm_capacity_oom"

    return Throughput(tps=served, bound=bound, components=comp,
                      tpot_s=tpot_s, batch=b_eff, batch_target=pol.b_d,
                      batch_cap=cap, hbm_fits=hbm.fits, cpu_c2c_util=cu["util"])


def gain(pol: PolicyConfig, baseline: PolicyConfig, work: WorkloadConfig,
         hw: HardwareConfig, model: ModelConfig, coeffs: Coeffs = Coeffs(),
         overlap: str = "optimistic") -> float:
    """Gain(f) = TPS(policy) / TPS(baseline)  (section 8)."""
    num = tps(pol, work, hw, model, coeffs, overlap).tps
    den = tps(baseline, work, hw, model, coeffs, overlap).tps
    return num / den if den > 0 else float("inf")


# --------------------------------------------------------------------------
# 8b  SERVING throughput (decode + prefill) -- the real-world headline metric
# --------------------------------------------------------------------------
def prefill_time(pol: PolicyConfig, work: WorkloadConfig, hw: HardwareConfig,
                 model: ModelConfig, coeffs: Coeffs = Coeffs()) -> float:
    """GPU compute time to prefill one call's UNCACHED (append) tokens (s).

    Pure GPU compute over `A` tokens (cached prefix is a prefix-cache hit -> not
    recomputed). This is the per-call prefill burst that decode-only tps()
    ignores; for a prefill-heavy workload (131:1) it is a first-order term.
    """
    return append_time(pol, work, hw, model, coeffs).gpu_compute


def serving_tps(pol: PolicyConfig, work: WorkloadConfig, hw: HardwareConfig,
                model: ModelConfig, coeffs: Coeffs = Coeffs(),
                overlap: str = "optimistic") -> dict:
    """Unified SERVING output-token throughput: includes the prefill burst.

    GPU-seconds per call = T_prefill + O * TPOT(B_eff)/B_eff   (decode batched).
    serving_tps = O / (GPU-seconds per call).  Converges to serving_roofline as
    batch grows. This is what a real engine sustains (decode-only tps OVERstates
    it on a prefill-heavy workload). Returns both for transparency.
    """
    th = tps(pol, work, hw, model, coeffs, overlap)
    O = work.o_output
    B = max(1, th.batch)
    t_pf = prefill_time(pol, work, hw, model, coeffs)
    gpu_s_per_call = t_pf + O * th.tpot_s / B
    serving = O / gpu_s_per_call if gpu_s_per_call > 0 else 0.0
    if not th.hbm_fits:
        serving = 0.0
    return {"serving_tps": serving, "decode_tps": th.tps,
            "prefill_time": t_pf, "batch": th.batch, "tpot_s": th.tpot_s,
            "hbm_fits": th.hbm_fits, "cpu_c2c_util": th.cpu_c2c_util}


# --------------------------------------------------------------------------
# 11  TTFT and SLO-constrained f* optimization
# --------------------------------------------------------------------------
def ttft(pol: PolicyConfig, work: WorkloadConfig, hw: HardwareConfig,
         model: ModelConfig, coeffs: Coeffs = Coeffs(),
         overlap: str = "optimistic") -> float:
    """Time-to-first-token = T_queue + T_append + first decode step.

    Analytical model assumes no queueing (T_queue=0; the event sim adds it).
    First token costs the append-prefill (old-KV load + GPU append + new-KV
    flush) plus one TPOT step at the capacity-bounded batch.
    """
    from dataclasses import replace as _replace
    th = tps(pol, work, hw, model, coeffs, overlap)
    pol_eff = _replace(pol, b_d=th.batch)
    appt = append_time(pol_eff, work, hw, model, coeffs)
    return appt.total + th.tpot_s


B_D_GRID = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]


@dataclass
class FStar:
    f: float                  # optimal CPU attention fraction
    b_d: int                  # jointly-optimal decode batch
    gain: float               # Gain(f*) over the best feasible gpu_hot baseline
    tps: float
    tpot: float
    ttft: float
    feasible: bool            # was any feasible (f, B_d) found?
    base_f0_tps: float        # best feasible throughput at f=0 (baseline)
    base_f0_b_d: int
    frontier: list            # per-f: (f, best_gain, best_b_d, tpot, ttft, c2c, feasible)
    reason: str


def _best_bd_at_f(f, work, hw, model, slo, coeffs, sparse, r, overlap,
                  objective="decode"):
    """For a fixed f, return (best_feasible_tps, b_d, tpot, ttft, c2c) by
    searching B_d.  `objective`: "decode" ranks by decode-only TPS (back-compat);
    "serving" ranks by the prefill-inclusive serving throughput (real-world)."""
    kind = ("gpu_hot" if f == 0 else
            "full_cpu_attn" if f >= 1.0 else "partial_cpu_attn")
    best = None
    for b_d in B_D_GRID:
        pol = PolicyConfig.policy(kind, f=f, b_d=b_d, r=r, sparse=sparse)
        th = tps(pol, work, hw, model, coeffs, overlap)
        tp = th.tpot_s
        tf = ttft(pol, work, hw, model, coeffs, overlap)
        feasible = (th.hbm_fits and tp <= slo.slo_tpot and tf <= slo.slo_ttft
                    and th.cpu_c2c_util <= slo.max_c2c_util)
        score = (th.tps if objective == "decode"
                 else serving_tps(pol, work, hw, model, coeffs, overlap)["serving_tps"])
        if feasible and (best is None or score > best[0]):
            best = (score, b_d, tp, tf, th.cpu_c2c_util)
    return best


def optimize_f(work: WorkloadConfig, hw: HardwareConfig, model: ModelConfig,
               slo: SLOConfig = SLOConfig(), coeffs: Coeffs = Coeffs(),
               *, sparse: float = 1.0, r: float = 1.0,
               overlap: str = "optimistic", grid: list | None = None,
               objective: str = "decode") -> FStar:
    """Jointly optimize (f, B_d) to maximize throughput s.t. the SLOs (sec 11).

        max_{f, B_d} TPS(f, B_d)
        s.t. TPOT <= SLO_TPOT, TTFT <= SLO_TTFT, HBM fits, C2C util <= cap

    Gain is reported against the best feasible f=0 (gpu_hot) baseline, so it
    answers "does offload beat simply tuning batch size on the GPU?".
    """
    if grid is None:
        grid = [round(x * 0.02, 2) for x in range(51)]   # 0.00 .. 1.00 step .02

    # Baseline: best feasible f=0.
    base = _best_bd_at_f(0.0, work, hw, model, slo, coeffs, sparse, r, overlap,
                         objective)
    base_tps = base[0] if base else 0.0
    base_bd = base[1] if base else 0

    frontier = []
    best = None
    for f in grid:
        b = _best_bd_at_f(f, work, hw, model, slo, coeffs, sparse, r, overlap,
                          objective)
        if b is None:
            frontier.append((f, 0.0, 0, float("inf"), float("inf"), 0.0, False))
            continue
        t, b_d, tp, tf, cu = b
        g = t / base_tps if base_tps > 0 else float("inf")
        frontier.append((f, g, b_d, tp, tf, cu, True))
        if best is None or t > best[0]:
            best = (t, f, b_d, g, tp, tf)

    if best is None:
        return FStar(f=0.0, b_d=0, gain=0.0, tps=0.0, tpot=float("inf"),
                     ttft=float("inf"), feasible=False, base_f0_tps=base_tps,
                     base_f0_b_d=base_bd, frontier=frontier,
                     reason="no (f, B_d) satisfies the SLOs (loosen TPOT/TTFT, "
                            "add sparsity, or add HBM)")
    t, f, b_d, g, tp, tf = best
    reason = ("f=0 (pure GPU batch tuning) is optimal under these SLOs"
              if f == 0 else
              f"offload f={f} with B_d={b_d} beats best GPU-only batch "
              f"(B_d={base_bd}) by {g:.2f}x")
    return FStar(f=f, b_d=b_d, gain=g, tps=t, tpot=tp, ttft=tf, feasible=True,
                 base_f0_tps=base_tps, base_f0_b_d=base_bd,
                 frontier=frontier, reason=reason)


def crossover_ok(pol: PolicyConfig, baseline: PolicyConfig,
                 work: WorkloadConfig, hw: HardwareConfig, model: ModelConfig,
                 coeffs: Coeffs = Coeffs(), overlap: str = "optimistic") -> bool:
    """Crossover condition: B_d(f)/TPOT(f) > B_d(0)/TPOT(0)  (section 8)."""
    a = tps(pol, work, hw, model, coeffs, overlap)
    b = tps(baseline, work, hw, model, coeffs, overlap)
    return (a.batch / a.tpot_s) > (b.batch / b.tpot_s)
