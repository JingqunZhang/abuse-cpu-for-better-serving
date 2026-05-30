"""Invariant tests for the concurrent multi-scenario fluid model.

These are physics/consistency invariants, not curve-fits: each would FAIL if the
flow-balance bottleneck logic were wrong, so they have discriminating power.
"""

from dataclasses import replace

import pytest

from model.concurrent import (best_f, fit_overlap, serving_band, serving_mix)
from model.contention import serving_2res
from model.config import (HardwareConfig, ModelConfig, SLOConfig,
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
    """Long-context (HBM-bound) gains from offload once CPU has headroom."""
    hw = _hw(cpg=1.0)
    base = serving_mix([(LONG, 1.0)], MODEL, hw, f=0.0)
    assert base["binding"] == "gpu_hbm"
    _, gain, *_ = best_f([(LONG, 1.0)], MODEL, hw)
    assert gain > 1.05


def test_more_cpu_never_hurts_best_gain():
    """Adding CPU bandwidth weakly increases the achievable offload gain."""
    g_lo = best_f([(LONG, 1.0)], MODEL, _hw(cpg=0.5), cpus_per_gpu=0.5)[1]
    g_hi = best_f([(LONG, 1.0)], MODEL, _hw(cpg=4.0), cpus_per_gpu=4.0)[1]
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


def test_offload_revives_under_loose_slo_only_when_sparse():
    """Relaxing the latency budget lets the capacity channel revive the gain even
    with NO overlap -- but ONLY when the offloaded CPU attention is cheap enough
    (sparse). Under genuine no-overlap (ov=0) the CPU attention is also on the
    call-rate path, so DENSE offload (~178ms/token on a weak CPU) stays
    throughput-negative regardless of the latency budget: a loose SLO grows B but
    the serialized dense CPU attention dominates. SPARSE offload is cheap enough
    that the freed-HBM capacity gain outweighs the serialized cost. (Before the
    rate-path overlap fix, dense appeared to revive too -- an artifact of treating
    CPU/C2C as free fully-overlapping servers even at ov=0.)"""
    hw = _hw(cpg=1.0)
    # Dense: no revival even with an effectively unbounded latency budget.
    d_loose = best_f([(LONG, 1.0)], MODEL, hw, overlap="conservative",
                     sparse=1.0, slo=SLOConfig(slo_tpot=1.0))[1]
    assert d_loose == pytest.approx(1.0, abs=1e-3)
    # Sparse: the capacity channel revives the gain as the SLO loosens.
    s_tight = best_f([(LONG, 1.0)], MODEL, hw, overlap="conservative",
                     sparse=0.1, slo=SLOConfig(slo_tpot=0.05))[1]
    s_loose = best_f([(LONG, 1.0)], MODEL, hw, overlap="conservative",
                     sparse=0.1, slo=SLOConfig(slo_tpot=1.0))[1]
    assert s_loose > s_tight + 0.02


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
