"""Configuration dataclasses for the CPU-GPU attention offload analytical model.

Four orthogonal config objects, matching section 4 of the plan:

    WorkloadConfig  -- x = (S_c, A, O, T_idle)      (section 4.1)
    ModelConfig     -- L, P_act, P_total, d_KV, ...  (section 4.2)
    HardwareConfig  -- R = (F_G, BW_HBM, M_HBM, ...) (section 4.3)
    PolicyConfig    -- pi = (f, r, B_p, B_d)         (section 4.4)

All quantities are in SI base units unless noted:
    tokens      : count
    params      : count
    bytes       : bytes
    FLOPs       : floating point ops
    F_*         : FLOP/s
    BW_*        : bytes/s
    M_*         : bytes
    time        : seconds

The defaults are deliberately editable presets, not ground truth.  Every
number that is an approximation of real hardware is flagged in a comment so
later phases (profiler-driven calibration, KVPR-style) can replace it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

GiB = 1024 ** 3
TB = 10 ** 12
GB = 10 ** 9


# --------------------------------------------------------------------------
# 4.1  Workload
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class WorkloadConfig:
    """A single representative LLM call x = (S_c, A, O, T_idle).

    Values below are the *mean* of Inferact/codex_swebenchpro_traces as quoted
    in the plan (section 2).  The trace parser (Phase 1) emits per-percentile
    variants (P50/P90/P99) that can be dropped in here.
    """

    name: str = "codex_swebench_mean"

    s_cached: float = 64_338      # S_c : cached prefix tokens (mean cached tokens/call)
    a_append: float = 3_991       # A   : append-prefill / uncached compute tokens
    o_output: float = 520         # O   : output tokens
    t_idle: float = 10.5          # T_idle : inter-call delay (s)

    @property
    def s_context(self) -> float:
        """S : total context length attended during decode = S_c + A."""
        return self.s_cached + self.a_append


# --------------------------------------------------------------------------
# 4.2  Model
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    """Transformer architecture parameters.

    Default preset is a dense ~70B GQA model (Llama-3.1-70B class) that fits in
    a single GB200's 192 GB HBM -- the right scope for the single-device v0
    analytical model.  The large MoE (does NOT fit one GPU; needs Phase-5
    parallelism to shard) is available via ModelConfig.preset("moe_large_gqa").
    """

    name: str = "dense_70b"

    layers: int = 80                  # L
    hidden: int = 8192
    n_heads: int = 64
    n_kv_heads: int = 8               # GQA group
    head_dim: int = 128

    p_act: float = 70e9               # P_act  : active params / token (dense)
    p_total: float = 70e9             # P_total: total params

    q_kv: int = 2                     # bytes per KV element (bf16/fp16)
    q_weight: int = 2                 # bytes per weight element
    q_act: int = 2                    # bytes per activation element

    # Attention KV scheme: "gqa" (separate K,V over n_kv_heads) or "mla"
    # (Multi-head Latent Attention -- a single compressed latent per token per
    # layer, e.g. DeepSeek-V3 stores ~512 latent + 64 decoupled-RoPE = 576).
    kv_mode: str = "gqa"
    mla_kv_dim: int = 576             # MLA stored elements / token / layer

    # KV cache compression (on top of the scheme above): the fraction of bytes
    # actually STORED/TRANSFERRED/READ vs uncompressed (e.g. int4 KV = 0.25 of
    # bf16). 1.0 = none. Decompressing to compute attention costs FLOPs, charged
    # per KV element on whichever device runs attention (see kv_decompress_flops).
    kv_compress: float = 1.0
    kv_decompress_flops: float = 0.0  # FLOPs to decompress one KV element

    # MoE structure for Phase-5 sharding: fraction of active / total params that
    # live in experts (sharded by EP), vs attention+shared+dense (sharded by TP).
    # Dense models: 0.0.  The expert fraction is the part EP can distribute.
    expert_frac_act: float = 0.0
    expert_frac_total: float = 0.0

    @property
    def d_kv(self) -> int:
        """d_KV : KV elements per token per layer.

        GQA: 2*n_kv_heads*head_dim (K and V over the KV-head groups).
        MLA: a single compressed latent (mla_kv_dim), NOT doubled -- MLA stores
        one latent vector, not separate K/V, which is its whole memory win.
        """
        if self.kv_mode == "mla":
            return self.mla_kv_dim
        return 2 * self.n_kv_heads * self.head_dim

    @property
    def d_attn(self) -> int:
        """FLOP per (decode-token, layer, context-token) for attention.

        QK^T then A.V, each ~2 * (n_heads * head_dim) FLOP over the query
        projection width.  GQA shares KV but every query head still computes.
        """
        return 4 * self.n_heads * self.head_dim

    def kv_bytes_per_token(self) -> float:
        """M_KV for a single token across all layers."""
        return self.layers * self.d_kv * self.q_kv

    @staticmethod
    def preset(name: Literal["dense_70b", "moe_large_mla",
                             "moe_large_gqa"]) -> "ModelConfig":
        if name == "dense_70b":
            return ModelConfig()
        if name in ("moe_large_mla", "moe_large_gqa"):
            # DeepSeek-V3 class: 671B total / 37B active, 61 layers, and -- the
            # key correction from review -- MLA KV (~576 elems/tok/layer), NOT
            # GQA (which would overstate KV ~3.5x).  The "moe_large_gqa" key is
            # kept as an alias for back-compat but now returns the MLA-correct
            # config.  Does NOT fit one GPU; needs Phase-5 sharding.
            return ModelConfig(
                name="moe_large_mla",
                layers=61, hidden=7168,
                n_heads=128, n_kv_heads=128, head_dim=128,
                p_act=37e9, p_total=671e9,
                kv_mode="mla", mla_kv_dim=576,
                # ~most active compute and the vast majority of weights are experts.
                expert_frac_act=0.73, expert_frac_total=0.96,
            )
        raise ValueError(f"unknown model preset {name!r}")


# --------------------------------------------------------------------------
# 4.3  Hardware
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class HardwareConfig:
    """Per-GPU + attached-CPU resource vector R, with peak values.

    Default preset ~ GB200 (Grace-Blackwell) per-GPU view.  All numbers are
    approximate public specs and are meant to be overridden; efficiency eta is
    applied separately via effective().
    """

    name: str = "gb200_pergpu"

    # System scale
    n_gpus: int = 72                 # total GPUs in the system/rack (NVL72)
    cpus_per_gpu: float = 1.0        # CPU sockets serving each GPU. The f_cpu/
                                     # bw_cpu/m_cpu below are the AGGREGATE CPU
                                     # resources available to ONE GPU's offload;
                                     # use HardwareConfig.system() to build them
                                     # from per-CPU specs x this ratio.

    # GPU (per device)
    f_gpu: float = 2.5e15            # F_G    : BF16 dense FLOP/s  (~2.5 PFLOPS)
    bw_hbm: float = 8.0e12           # BW_HBM : HBM3e ~ 8 TB/s
    m_hbm: float = 192 * GiB         # M_HBM  : 192 GB

    # CPU resources attached to / serving one GPU (aggregate over cpus_per_gpu)
    f_cpu: float = 2.0e12            # F_C    : optimistic CPU BF16 ~ 2 TFLOP/s
    bw_cpu: float = 0.5e12           # BW_CPU : LPDDR5X ~ 500 GB/s
    m_cpu: float = 240 * GiB         # M_CPU  : ~half a 480GB Grace per GPU

    # Interconnect.  NVLink-C2C 900 GB/s is the AGGREGATE bidirectional figure;
    # a one-way transfer (e.g. KV flush GPU->CPU) gets ~half.  Use bw_c2c_oneway
    # for unidirectional bulk transfers and bw_c2c for bidirectional exchange.
    bw_c2c: float = 0.9e12           # BW_C2C   : NVLink-C2C ~ 900 GB/s (bidir)
    bw_nvlink: float = 1.8e12        # BW_NVLink: NVLink5 ~ 1.8 TB/s / GPU

    # Fixed per-layer sync cost for CPU<->GPU handoff (s)
    t_sync: float = 2e-6

    @property
    def bw_c2c_oneway(self) -> float:
        """One-directional C2C bandwidth (half the bidirectional aggregate)."""
        return self.bw_c2c / 2.0

    @staticmethod
    def system(*, name="custom", n_gpus=72, cpus_per_gpu=1.0,
               gpu_flops=2.5e15, hbm_capacity_gb=192.0, hbm_bw_tb_s=8.0,
               nvlink_bw_tb_s=1.8, c2c_bw_gb_s=900.0,
               cpu_flops_per_cpu=2.0e12, cpu_dram_bw_gb_s=500.0,
               cpu_mem_gb_per_cpu=240.0, t_sync=2e-6) -> "HardwareConfig":
        """Build a HardwareConfig from human-friendly per-component specs.

        CPU compute / bandwidth / memory are AGGREGATED over cpus_per_gpu, i.e.
        the resources available to offload from one GPU scale with how many CPUs
        serve it (FastDecode-style bandwidth aggregation).  Pass per-CPU figures;
        this multiplies by cpus_per_gpu.
        """
        return HardwareConfig(
            name=name, n_gpus=n_gpus, cpus_per_gpu=cpus_per_gpu,
            f_gpu=gpu_flops,
            bw_hbm=hbm_bw_tb_s * TB,
            m_hbm=hbm_capacity_gb * GB,
            bw_nvlink=nvlink_bw_tb_s * TB,
            bw_c2c=c2c_bw_gb_s * GB,
            f_cpu=cpu_flops_per_cpu * cpus_per_gpu,
            bw_cpu=cpu_dram_bw_gb_s * GB * cpus_per_gpu,
            m_cpu=cpu_mem_gb_per_cpu * GB * cpus_per_gpu,
            t_sync=t_sync,
        )

    def effective(self, eta: float = None, *, eta_compute: float = None,
                  eta_mem: float = None) -> "HardwareConfig":
        """Apply achievable-efficiency factors to throughput/bandwidth.

        SEPARATE compute and memory efficiencies (LIFE, arXiv:2508.00904: real
        inference efficiency varies "beyond a simple roofline" -- compute units
        and memory bandwidth hit very different fractions of peak).
          eta_compute scales FLOP/s (f_gpu, f_cpu)  -- binds prefill/append.
          eta_mem     scales bandwidths (HBM/CPU/C2C/NVLink) -- binds decode.
        `eta` is a shorthand setting both. Capacities (M_*) and sync latency are
        NOT scaled. Decode (memory-bound) is governed by eta_mem; prefill TTFT
        (compute-bound) by eta_compute -- a single scalar cannot match both.
        """
        ec = eta_compute if eta_compute is not None else (eta if eta is not None else 1.0)
        em = eta_mem if eta_mem is not None else (eta if eta is not None else 1.0)
        return replace(
            self,
            name=f"{self.name}@ec{ec},em{em}",
            f_gpu=self.f_gpu * ec,
            f_cpu=self.f_cpu * ec,
            bw_hbm=self.bw_hbm * em,
            bw_cpu=self.bw_cpu * em,
            bw_c2c=self.bw_c2c * em,
            bw_nvlink=self.bw_nvlink * em,
        )


# --------------------------------------------------------------------------
# 4.4  Policy
# --------------------------------------------------------------------------
PolicyName = Literal["gpu_hot", "cpu_backing", "partial_cpu_attn", "full_cpu_attn"]


@dataclass(frozen=True)
class PolicyConfig:
    """pi = (f, r, B_p, B_d) plus the named design point it represents.

    f   : fraction of decode-attention KV handled by CPU      (1-f on GPU)
    r   : fraction of old KV materialized to GPU for append   (default 1.0)
    b_p : append-prefill batch size
    b_d : decode continuous batch size
    cpu_backing : whether CPU holds the authoritative old-KV copy.  When True,
                  append-prefill flushes only *new* KV, not old KV (section 6).
    sparse : fraction of the context KV the CPU actually reads per token
             (ScoutAttention-style block selection, section 3.4).  1.0 = dense
             CPU attention (pessimistic); <1 = sparse co-attention.  Only scales
             the CPU side; HBM/GPU attention is unaffected.
    """

    name: str = "gpu_hot"
    f: float = 0.0
    r: float = 1.0
    b_p: int = 1
    b_d: int = 32
    cpu_backing: bool = False
    sparse: float = 1.0

    @staticmethod
    def policy(kind: PolicyName, *, f: float = 0.0, r: float = 1.0,
               b_p: int = 1, b_d: int = 32, sparse: float = 1.0) -> "PolicyConfig":
        if kind == "gpu_hot":                       # A
            return PolicyConfig("gpu_hot", f=0.0, r=r, b_p=b_p, b_d=b_d,
                                cpu_backing=False, sparse=sparse)
        if kind == "cpu_backing":                   # B
            return PolicyConfig("cpu_backing", f=0.0, r=r, b_p=b_p, b_d=b_d,
                                cpu_backing=True, sparse=sparse)
        if kind == "partial_cpu_attn":              # C
            return PolicyConfig("partial_cpu_attn", f=f, r=r, b_p=b_p, b_d=b_d,
                                cpu_backing=True, sparse=sparse)
        if kind == "full_cpu_attn":                 # D
            return PolicyConfig("full_cpu_attn", f=1.0, r=r, b_p=b_p, b_d=b_d,
                                cpu_backing=True, sparse=sparse)
        raise ValueError(f"unknown policy {kind!r}")


# --------------------------------------------------------------------------
# Calibration coefficients (efficiency / overlap knobs, section 7)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Coeffs:
    """Tunable coefficients separated out so Phase-3 calibration is one object."""

    alpha: float = 1.0       # CPU attention memory-traffic multiplier
    beta: float = 1.0        # CPU attention compute multiplier
    gamma: float = 1.0       # GPU append-prefill attention-compute multiplier
    t_merge: float = 5e-6    # per-step CPU/GPU partial-attention merge cost (s)
    # Fixed per-decode-step dispatch / kernel-launch / scheduling overhead (s),
    # batch-independent (LIFE eq.4 t_dispatch).  Captures the floor that pure
    # roofline misses at small batch / short context.  Default 0 = neutral.
    t_dispatch: float = 0.0

    # Runtime / workspace HBM overheads (bytes) used by the capacity model.
    m_runtime: float = 2 * GiB
    m_workspace: float = 4 * GiB


# --------------------------------------------------------------------------
# Service-level objectives (section 11)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SLOConfig:
    """Latency SLOs used to constrain the f* optimization (section 11).

    Defaults reflect an interactive-but-lenient long-context agentic target:
    50 ms/output-token (~20 tok/s) and 10 s to first token (prefill of a long
    context is inherently slow).  c2c_util must stay <=1 (no over-subscription).
    """

    slo_tpot: float = 0.05      # max seconds per output token
    slo_ttft: float = 10.0      # max seconds to first token
    max_c2c_util: float = 1.0
