"""Invariant tests for the concurrent multi-scenario fluid model.

These are physics/consistency invariants, not curve-fits: each would FAIL if the
flow-balance bottleneck logic were wrong, so they have discriminating power.
"""

from dataclasses import replace

import pytest

import model.analytical as an
from model.concurrent import (_decode_tpot, best_f, fit_overlap, serving_band,
                              serving_mix)
from model.contention import serving_2res
from model.config import (Coeffs, HardwareConfig, ModelConfig, SLOConfig,
                          WorkloadConfig)
from model.roofline import serving_roofline


def _hw(cpg=1.0, eta=0.5):
    # Base per-GPU hw. CPU-resource scaling by cpus_per_gpu is applied INSIDE
    # serving_mix/best_f now, so callers pass `cpus_per_gpu=cpg` to those, not
    # a pre-scaled hw. (cpg kept in the signature for call-site readability.)
    return HardwareConfig().effective(eta)


LONG = WorkloadConfig(s_cached=64_338, a_append=3_991, o_output=520)
SHORT = WorkloadConfig(s_cached=1_500, a_append=500, o_output=300)
# long input, tiny output -> GPU compute binds, not HBM
PREFILL = WorkloadConfig(s_cached=68_000, a_append=12_000, o_output=20)
MODEL = ModelConfig()


def test_below_roofline_ceiling():
    """f=0 throughput can't beat the infinite-batch serving roofline."""
    hw = _hw()
    tps = serving_mix([(LONG, 1.0)], MODEL, hw, f=0.0)["tps"]
    ceil = serving_roofline(MODEL, HardwareConfig().effective(0.5), LONG)
    assert 0 < tps <= ceil["ceiling_tps_per_gpu"] + 1e-6


def test_fluid_is_upper_bound_on_serialized():
    """Perfect compute∥HBM overlap (mix) >= the serialized serving_2res."""
    hw = _hw(cpg=1.0)
    mix = serving_mix([(LONG, 1.0)], MODEL, hw, f=0.0, cpus_per_gpu=1.0)["tps"]
    ser = serving_2res(MODEL, hw, LONG, f=0.0, cpus_per_gpu=1.0)["tps"]
    assert mix >= ser - 1e-6


def test_weight_scale_invariance():
    """Throughput depends on the relative mix, not absolute weights."""
    hw = _hw()
    a = serving_mix([(LONG, 0.7), (SHORT, 0.3)], MODEL, hw, f=0.0)["tps"]
    b = serving_mix([(LONG, 7.0), (SHORT, 3.0)], MODEL, hw, f=0.0)["tps"]
    assert a == pytest.approx(b, rel=1e-9)


def test_compute_bound_offload_no_help():
    """When GPU compute binds, offloading KV (a non-binding resource) is ~neutral."""
    hw = _hw(cpg=1.0)
    base = serving_mix([(PREFILL, 1.0)], MODEL, hw, f=0.0)
    assert base["binding"] == "gpu_compute"
    _, gain, *_ = best_f([(PREFILL, 1.0)], MODEL, hw)
    assert gain <= 1.02      # essentially no improvement


def test_hbm_bound_offload_helps_with_cpu():
    """Long-context (HBM-bound) gains from offload once CPU has headroom -- but
    only at the OPTIMISTIC (overlap) end; best_f's default is now conservative, so
    this asserts the upper-bound claim explicitly."""
    hw = _hw(cpg=1.0)
    base = serving_mix([(LONG, 1.0)], MODEL, hw, f=0.0)
    assert base["binding"] == "gpu_hbm"
    _, gain, *_ = best_f([(LONG, 1.0)], MODEL, hw, overlap="optimistic")
    assert gain > 1.05


def test_more_cpu_never_hurts_best_gain():
    """Adding CPU bandwidth weakly increases the achievable offload gain (asserted
    at the optimistic end so it has discriminating power; at the conservative
    default both can sit at 1.0x and the check would be trivially true)."""
    g_lo = best_f([(LONG, 1.0)], MODEL, _hw(cpg=0.5), cpus_per_gpu=0.5,
                  overlap="optimistic")[1]
    g_hi = best_f([(LONG, 1.0)], MODEL, _hw(cpg=4.0), cpus_per_gpu=4.0,
                  overlap="optimistic")[1]
    assert g_hi >= g_lo - 1e-6


def test_offload_raises_hbm_capacity_bound():
    """Offloading KV frees HBM capacity -> the capacity bound b_cap grows.
    (The realized B=min(b_cap, b_slo) may NOT grow, because offloading dense
    attention to a slow CPU can raise TPOT and tighten the SLO bound instead --
    that latency penalty is exactly what the model now captures.)"""
    hw = _hw()
    b0 = serving_mix([(LONG, 1.0)], MODEL, hw, f=0.0, cpus_per_gpu=4.0)["b_cap"]
    b5 = serving_mix([(LONG, 1.0)], MODEL, hw, f=0.5, cpus_per_gpu=4.0)["b_cap"]
    assert b5 > b0


def test_sparse_offload_can_grow_realized_batch():
    """When offload is cheap enough (sparse), the freed capacity DOES raise the
    realized resident batch B -- the capacity channel wins over the latency
    penalty."""
    hw = _hw()
    b0 = serving_mix([(LONG, 1.0)], MODEL, hw, f=0.0, sparse=0.1,
                     cpus_per_gpu=4.0)["B"]
    b7 = serving_mix([(LONG, 1.0)], MODEL, hw, f=0.7, sparse=0.1,
                     cpus_per_gpu=4.0)["B"]
    assert b7 > b0


def test_mix_between_pure_extremes():
    """A mix's throughput lies between its pure-class throughputs (shared B)."""
    hw = _hw()
    lo = serving_mix([(LONG, 1.0)], MODEL, hw, f=0.0)["tps"]
    hi = serving_mix([(SHORT, 1.0)], MODEL, hw, f=0.0)["tps"]
    mid = serving_mix([(LONG, 0.5), (SHORT, 0.5)], MODEL, hw, f=0.0)["tps"]
    assert min(lo, hi) - 1e-6 <= mid <= max(lo, hi) + 1e-6


def test_band_ordering():
    """Conservative (no-overlap) <= optimistic (perfect overlap), always."""
    hw = _hw()
    for cls in ([(LONG, 1.0)], [(SHORT, 1.0)], [(LONG, 0.7), (SHORT, 0.3)]):
        con, opt = serving_band(cls, MODEL, hw, f=0.0)
        assert con <= opt + 1e-6


def test_long_context_band_is_tight():
    """HBM dominates at long context -> overlap assumption barely matters."""
    con, opt = serving_band([(LONG, 1.0)], MODEL, _hw(), f=0.0)
    assert con / opt > 0.8       # < 20% spread


def test_slo_caps_batch_and_tpot():
    """A tighter (but feasible) TPOT SLO yields a smaller batch within the SLO."""
    hw = _hw()
    loose = serving_mix([(SHORT, 1.0)], MODEL, hw, f=0.0, slo=SLOConfig(slo_tpot=0.05))
    tight = serving_mix([(SHORT, 1.0)], MODEL, hw, f=0.0, slo=SLOConfig(slo_tpot=0.04))
    assert tight["B"] <= loose["B"]
    assert tight["tpot"] <= 0.04 + 1e-9 and tight["slo_feasible"]
    assert loose["tpot"] <= 0.05 + 1e-9 and loose["slo_feasible"]


def test_overlap_fraction_interpolates_band():
    """A float overlap in (0,1) sits strictly inside [conservative, optimistic]
    and is monotonic; ov=0 hits conservative, ov=1 hits the combined ceiling."""
    hw = _hw()
    cls = [(SHORT, 1.0)]      # short context -> compute≈HBM -> wide band
    con, opt = serving_band(cls, MODEL, hw, f=0.0)
    t0 = serving_mix(cls, MODEL, hw, f=0.0, overlap=0.0)["tps"]
    tm = serving_mix(cls, MODEL, hw, f=0.0, overlap=0.5)["tps"]
    t1 = serving_mix(cls, MODEL, hw, f=0.0, overlap=1.0)["tps"]
    assert t0 == pytest.approx(con, rel=1e-6)
    assert con < tm < opt + 1e-9
    assert t0 < tm < t1 + 1e-9


def test_fit_overlap_roundtrips():
    """fit_overlap recovers the ov that produced a given throughput."""
    hw = _hw()
    cls = [(LONG, 0.6), (SHORT, 0.4)]
    target = serving_mix(cls, MODEL, hw, f=0.0, overlap=0.37)["tps"]
    ov = fit_overlap(cls, MODEL, hw, target, f=0.0)
    assert ov == pytest.approx(0.37, abs=1e-2)


def test_fit_overlap_out_of_band_returns_none():
    """A throughput outside the band can't be explained by overlap alone."""
    hw = _hw()
    cls = [(LONG, 1.0)]
    _, opt = serving_band(cls, MODEL, hw, f=0.0)
    assert fit_overlap(cls, MODEL, hw, opt * 2.0, f=0.0) is None


def test_offload_needs_overlap_under_tight_slo():
    """No-overlap offload of slow-CPU attention adds to the per-token critical
    path; under a tight interactive SLO that cancels the capacity gain (~1.0x),
    while perfect overlap still helps."""
    hw = _hw(cpg=1.0)
    g_opt = best_f([(LONG, 1.0)], MODEL, hw, overlap="optimistic")[1]
    g_con = best_f([(LONG, 1.0)], MODEL, hw, overlap="conservative")[1]
    assert g_opt > 1.05
    assert g_con == pytest.approx(1.0, abs=1e-3)


def test_offload_revives_under_loose_slo_needs_sparse_and_enough_cpu():
    """No-overlap (ov=0) revival of the offload gain via a loose latency budget
    needs cheap (sparse) CPU attention AND enough CPU bandwidth. With the realistic
    Grace f_cpu (DRAM-bound attention), the crossover is ~1 Grace/GPU: a starved
    0.5-Grace still doesn't revive, but a full Grace does, and it grows with CPU
    aggregation. (At the old too-low f_cpu the crossover was ~4 Grace -- correcting
    f_cpu to the real peak moved it down.)"""
    # Starved 0.5 Grace: no conservative revival even with an unbounded SLO.
    s_05 = best_f([(LONG, 1.0)], MODEL, _hw(cpg=0.5), overlap="conservative",
                  sparse=0.1, cpus_per_gpu=0.5, slo=SLOConfig(slo_tpot=1.0))[1]
    assert s_05 == pytest.approx(1.0, abs=1e-2)
    # 1 Grace: revives once the SLO is loose.
    s_1 = best_f([(LONG, 1.0)], MODEL, _hw(cpg=1.0), overlap="conservative",
                 sparse=0.1, cpus_per_gpu=1.0, slo=SLOConfig(slo_tpot=1.0))[1]
    assert s_1 > 1.1
    # More CPU -> more revival (capacity channel scales with CPU bandwidth).
    s_4 = best_f([(LONG, 1.0)], MODEL, _hw(cpg=4.0), overlap="conservative",
                 sparse=0.1, cpus_per_gpu=4.0, slo=SLOConfig(slo_tpot=1.0))[1]
    assert s_4 > s_1 + 0.02
    # Dense stays weaker than sparse at 1 Grace (sparse is still required to win much).
    d_1 = best_f([(LONG, 1.0)], MODEL, _hw(cpg=1.0), overlap="conservative",
                 sparse=1.0, cpus_per_gpu=1.0, slo=SLOConfig(slo_tpot=1.0))[1]
    assert d_1 <= s_1 + 1e-9


def test_offload_raises_tpot_without_overlap():
    """At ov=0 the offloaded path is on the latency critical path, so a fixed
    batch's TPOT rises with f."""
    hw = _hw(cpg=1.0)
    loose = SLOConfig(slo_tpot=10.0)        # don't let the SLO cap B differ
    t0 = serving_mix([(LONG, 1.0)], MODEL, hw, f=0.0, overlap="conservative",
                     slo=loose)["tpot"]
    tf = serving_mix([(LONG, 1.0)], MODEL, hw, f=0.3, overlap="conservative",
                     slo=loose)["tpot"]
    assert tf > t0


def test_oom_class_is_flagged_not_silently_served():
    """A sequence whose KV can't fit free HBM -> tps=0, binding=hbm_capacity_oom."""
    hw = _hw()
    huge = WorkloadConfig(s_cached=5_000_000, a_append=0, o_output=100)
    r = serving_mix([(huge, 1.0)], MODEL, hw, f=0.0)
    assert r["tps"] == 0.0 and not r["fits"] and r["binding"] == "hbm_capacity_oom"


def test_zero_output_mix_self_documents():
    """A pure-prefill (zero-output) mix yields tps=0 with a distinct binding,
    not a silent zero that looks like a failure."""
    hw = _hw()
    pure_prefill = WorkloadConfig(s_cached=2_000, a_append=2_000, o_output=0)
    r = serving_mix([(pure_prefill, 1.0)], MODEL, hw, f=0.0)
    assert r["tps"] == 0.0 and r["binding"] == "no_output_tokens"


def test_infeasible_slo_is_flagged():
    """Dense-70B has a weight-streaming TPOT floor (~35ms at eta=0.5); a 10ms
    SLO is physically infeasible on one GPU and must be flagged, not clamped."""
    hw = _hw()
    r = serving_mix([(SHORT, 1.0)], MODEL, hw, f=0.0, slo=SLOConfig(slo_tpot=0.01))
    assert r["B"] == 1.0           # can't batch down further
    assert not r["slo_feasible"]   # floor exceeds the SLO


def test_decode_tpot_respects_roofline_floor_on_headline_path():
    """HARD physical floor on the CONCURRENT model's own per-token time (not just
    the legacy analytical path): each decode step must at least read the active
    weights once and stream the resident (1-f)*sparse KV from HBM. A future change
    to the overlap/offload terms that undercut physics would FAIL here -- the
    falsification power the audit found missing on the headline path."""
    hw = _hw()
    for f in (0.0, 0.3, 0.5, 0.9):
        for sparse in (1.0, 0.1):
            for B in (1.0, 8.0, 64.0):
                for ov in ("conservative", "optimistic", 0.5):
                    tpot = _decode_tpot(B, f, LONG.s_context, MODEL, hw, sparse,
                                        ov, Coeffs())
                    w = an.weight_read_bytes(MODEL, B)
                    kv = B * (1.0 - f) * sparse * an.kv_size(LONG.s_context, MODEL)
                    floor = max(w, kv) / hw.bw_hbm     # perfect weight∥KV overlap
                    assert tpot >= floor - 1e-12, (f, sparse, B, ov, tpot, floor)


def test_cpu_attention_flops_are_load_bearing():
    """Closes the audit's load-bearing blind spot: the CPU-attention FLOP cost
    (beta) MUST actually affect the offloaded path. The headline gain@con guard is
    SLO-pinned (tautologically 1.0) and CANNOT detect an undercount of the CPU
    cost; this test can. At a CPU-bound operating point, raising beta must
    strictly INCREASE the per-token time (i.e. the FLOPs are wired in, not
    silently dropped -> g_opt cannot be inflated by undercounting them)."""
    # Use a deliberately CPU-COMPUTE-bound point so beta is on the binding term:
    # at the realistic f_cpu the GQA attention is DRAM-bound (AI≈8 < ridge≈28), so
    # beta wouldn't bite there; a weak-FLOP CPU (ridge < AI) makes compute bind.
    hw = replace(_hw(cpg=1.0), f_cpu=1.0e12)     # ridge≈2 < AI=8 -> compute-bound
    t_lo = _decode_tpot(8.0, 0.85, LONG.s_context, MODEL, hw, 1.0, "optimistic",
                        Coeffs(beta=1.0))
    t_hi = _decode_tpot(8.0, 0.85, LONG.s_context, MODEL, hw, 1.0, "optimistic",
                        Coeffs(beta=2.0))
    assert t_hi > t_lo + 1e-9, (t_lo, t_hi)


def test_optimistic_gain_magnitude_is_pinned():
    """CHARACTERIZATION guard on the OPTIMISTIC gain MAGNITUDE -- the audit's
    remaining gap: every other offload test is a one-sided lower bound (>1.05),
    so an UPWARD inflation from undercounting the CPU cost (e.g. beta=0.5 -> sparse
    gain 1.97->2.5) would pass the whole suite undetected. This pins the headline
    1-Grace optimistic gains, so any silent change to the CPU-attention
    coefficients (alpha/beta), C2C, overlap math, or capacity units trips a
    failure. If the change is intentional, update the expected values here."""
    hw = _hw(cpg=1.0)
    _, g_dense, *_ = best_f([(LONG, 1.0)], MODEL, hw, sparse=1.0,
                            cpus_per_gpu=1.0, overlap="optimistic")
    _, g_sparse, *_ = best_f([(LONG, 1.0)], MODEL, hw, sparse=0.1,
                             cpus_per_gpu=1.0, overlap="optimistic")
    assert g_dense == pytest.approx(1.18, abs=0.04), g_dense
    assert g_sparse == pytest.approx(2.48, abs=0.08), g_sparse


def test_offload_step_respects_cpu_roofline_floor():
    """A fully-offloaded (f=1) decode step can never be faster than the CPU's own
    attention roofline (KV stream OR attention FLOPs, whichever binds) -- a hard
    lower bound on the offloaded TPOT term the audit found untested. Catches any
    future change that lets the offloaded path beat physics."""
    hw = _hw(cpg=1.0)
    B = 8.0
    tpot = _decode_tpot(B, 1.0, LONG.s_context, MODEL, hw, 1.0, "optimistic",
                        Coeffs())
    cpu_mem = B * an.kv_size(LONG.s_context, MODEL) / hw.bw_cpu
    cpu_flop = B * MODEL.layers * LONG.s_context * MODEL.d_attn / hw.f_cpu
    cpu_floor = max(cpu_mem, cpu_flop)            # f=1, sparse=1
    assert tpot >= cpu_floor - 1e-9, (tpot, cpu_floor)


def test_cpu_dram_capacity_is_enforced():
    """Offloaded KV must PHYSICALLY fit CPU DRAM (was unchecked -> offload could
    place unbounded KV on the CPU 'for free' and inflate the gain). (a) a sequence
    whose offloaded KV alone exceeds CPU memory is flagged cpu_capacity_oom, not
    silently served; (b) the CPU bound caps the resident batch."""
    hw = _hw()
    # (a) one 2M-token sequence at f=1 needs ~650GB CPU KV >> stock 0.5-Grace CPU.
    huge = replace(LONG, s_cached=2_000_000, a_append=0)
    r = serving_mix([(huge, 1.0)], MODEL, hw, f=1.0, sparse=0.1, cpus_per_gpu=0.5)
    assert not r["fits"] and r["binding"] == "cpu_capacity_oom"
    # (b) at long context + high f, the resident batch's offloaded KV must fit CPU
    # memory: b_cap * f * kv_size <= m_cpu (the bound that was previously missing).
    long128 = replace(LONG, s_cached=128_000, a_append=4_000)
    rr = serving_mix([(long128, 1.0)], MODEL, hw, f=0.75, sparse=0.02,
                     cpus_per_gpu=0.5, overlap="optimistic")
    m_cpu = HardwareConfig().m_cpu * 0.5         # stock 0.5 Grace (m_cpu not eta-scaled)
    assert rr["b_cap"] * 0.75 * an.kv_size(long128.s_context, MODEL) <= m_cpu * 1.05


def test_pool_default_off_is_back_compat():
    # pool defaults False -> identical tps to an explicit pool=False, and the
    # new ttft key is always present (Mooncake critique fix is opt-in).
    hw = _hw(cpg=0.5)
    a = serving_mix([(LONG, 1.0)], MODEL, hw, f=0.0, cpus_per_gpu=0.5,
                    overlap="optimistic")
    b = serving_mix([(LONG, 1.0)], MODEL, hw, f=0.0, cpus_per_gpu=0.5,
                    overlap="optimistic", pool=False)
    assert a["tps"] == b["tps"]
    assert "ttft" in a and a["ttft"] > 0


def test_pool_load_latency_hits_ttft_only_under_conservative_overlap():
    # The cached-prefix KV pool-load (~kv_size(Sc)/C2C) lands on the TTFT
    # critical path when NOT overlapped, and is hidden layer-wise when it is.
    hw = _hw(cpg=0.5)
    def ttft(pool, ovl):
        return serving_mix([(LONG, 1.0)], MODEL, hw, f=0.0, cpus_per_gpu=0.5,
                           overlap=ovl, pool=pool)["ttft"]
    # conservative: pooling adds real latency (the C2C load serializes)
    assert ttft(True, "conservative") > ttft(False, "conservative") + 1e-3
    # optimistic: layer-wise streaming hides it -> ~unchanged
    assert ttft(True, "optimistic") == pytest.approx(ttft(False, "optimistic"), rel=1e-6)


def test_pool_narrows_never_inflates_offload_gain():
    # Measuring offload against a REALISTIC pooled baseline (which already pays
    # the pool C2C and banks the capacity win) must not give a LARGER gain than
    # against the optimistic single-node baseline.
    hw = _hw(cpg=0.5)
    def gain(pool):
        base = serving_mix([(LONG, 1.0)], MODEL, hw, f=0.0, sparse=0.10,
                           cpus_per_gpu=0.5, overlap="optimistic", pool=pool)["tps"]
        off = serving_mix([(LONG, 1.0)], MODEL, hw, f=0.45, sparse=0.10,
                          cpus_per_gpu=0.5, overlap="optimistic", pool=pool)["tps"]
        return off / base
    assert gain(True) <= gain(False) + 1e-6
