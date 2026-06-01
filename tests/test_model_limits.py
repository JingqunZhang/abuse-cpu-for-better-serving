"""Phase 3 limiting-case tests for the analytical model.

Run:  python -m pytest tests/ -q     (or)     python tests/test_model_limits.py
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import analytical as an
from model.config import (Coeffs, HardwareConfig, ModelConfig, PolicyConfig,
                          SLOConfig, WorkloadConfig)

WORK = WorkloadConfig()
MODEL = ModelConfig()
HW = HardwareConfig().effective(0.5)
C = Coeffs()


def _close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def test_f0_equals_gpu_hot_baseline():
    """f=0 partial-cpu policy reduces to the GPU-hot baseline TPOT."""
    base = PolicyConfig.policy("gpu_hot", b_d=32)
    p0 = replace(base, name="partial_cpu_attn", f=0.0)
    t_base = an.tpot(base, WORK, HW, MODEL, C).optimistic
    t_p0 = an.tpot(p0, WORK, HW, MODEL, C).optimistic
    assert _close(t_base, t_p0), (t_base, t_p0)
    # CPU/C2C terms must be exactly zero at f=0.
    tp = an.tpot(base, WORK, HW, MODEL, C)
    assert tp.cpu_attn == 0.0 and tp.c2c == 0.0


def test_f1_is_full_cpu_attention():
    """f=1 puts all attention on CPU: GPU attention term vanishes."""
    full = PolicyConfig.policy("full_cpu_attn", b_d=32)
    tp = an.tpot(full, WORK, HW, MODEL, C)
    assert _close(tp.gpu_attn, 0.0), tp.gpu_attn
    assert tp.cpu_attn > 0.0


def test_cpu_bandwidth_infinite_removes_cpu_penalty():
    """BW_CPU, F_C -> inf makes CPU attention memory+compute time -> 0."""
    fast = replace(HW, bw_cpu=1e30, f_cpu=1e30)
    pol = PolicyConfig.policy("partial_cpu_attn", f=0.3, b_d=32)
    c_attn = an.cpu_attn_time(pol, WORK, fast, MODEL, C)
    assert c_attn < 1e-9, c_attn


def test_infinite_hbm_reduces_offload_benefit():
    """M_HBM -> inf: baseline already fits a huge batch, so CPU offload's
    capacity advantage shrinks -- decode_batch_cap for f=0 becomes enormous."""
    huge = replace(HW, m_hbm=1e18)
    base = PolicyConfig.policy("gpu_hot", b_d=32)
    cap_normal = an.decode_batch_cap(base, WORK, HW, MODEL, C)
    cap_huge = an.decode_batch_cap(base, WORK, huge, MODEL, C)
    assert cap_huge > cap_normal * 100, (cap_normal, cap_huge)


def test_zero_c2c_makes_cpu_backing_unusable_for_append():
    """BW_C2C -> 0: old-KV load time for append-prefill -> inf."""
    nolink = replace(HW, bw_c2c=1e-30)
    pol = PolicyConfig.policy("cpu_backing", b_d=32)
    appt = an.append_time(pol, WORK, nolink, MODEL, C)
    assert math.isinf(appt.total) or appt.total > 1e6, appt.total


def _gain_curve(sparse, b_d=256):
    base = PolicyConfig.policy("gpu_hot", b_d=b_d, sparse=sparse)
    fs = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    out = []
    for f in fs:
        kind = "full_cpu_attn" if f == 1.0 else (
            "gpu_hot" if f == 0 else "partial_cpu_attn")
        pol = PolicyConfig.policy(kind, f=f, b_d=b_d, sparse=sparse)
        out.append(an.gain(pol, base, WORK, HW, MODEL, C))
    return fs, out


def test_dense_cpu_attention_helps_only_modestly():
    """With the realistic Grace f_cpu (DRAM-bound CPU attention, not the old
    too-low compute-bound 2 TFLOP/s), dense CPU offload CAN help a little via freed
    HBM capacity, but only modestly -- the ~16x CPU/HBM bandwidth gap still caps it
    well below the sparse case. (Old claim 'dense never helps' was an artifact of
    the too-weak f_cpu.)"""
    fs, gains = _gain_curve(sparse=1.0)
    assert gains[0] == 1.0
    assert max(gains) <= 1.35, gains        # modest, bandwidth-gap-capped


def test_sparse_cpu_attention_beats_baseline_and_dense():
    """ScoutAttention-style sparse CPU attention (10% blocks) lets offload exceed
    baseline throughput, and its BEST f beats dense's best (sparse is the regime
    where CPU offload really helps)."""
    fs, gains = _gain_curve(sparse=0.1)
    assert gains[0] == 1.0
    assert max(gains) > 1.05, gains
    # sparse's best operating point beats dense's best (not necessarily pointwise,
    # since with a realistic CPU dense can win at a different f than sparse).
    _, dense = _gain_curve(sparse=1.0)
    assert max(gains) > max(dense) + 1e-9, (gains, dense)


def test_hbm_capacity_monotonic_in_f():
    """Higher f -> less hot KV on GPU -> larger decode batch capacity."""
    caps = []
    for f in [0.0, 0.2, 0.5]:
        kind = "gpu_hot" if f == 0 else "partial_cpu_attn"
        pol = PolicyConfig.policy(kind, f=f, b_d=32)
        caps.append(an.decode_batch_cap(pol, WORK, HW, MODEL, C))
    assert caps[0] < caps[1] < caps[2], caps


def test_cpu_backing_skips_old_kv_reflush():
    """cpu_backing flushes only new KV; gpu_hot (no backing) flushes nothing
    extra but also performs no old-KV load saving -- check flush accounting."""
    backing = PolicyConfig.policy("cpu_backing", b_d=32)
    appt = an.append_time(backing, WORK, HW, MODEL, C)
    # new-KV flush is proportional to A (append tokens), not S_c (cached), and
    # uses the one-directional C2C bandwidth (bulk GPU->CPU transfer).
    expected_flush = backing.b_p * an.kv_size(WORK.a_append, MODEL) / HW.bw_c2c_oneway
    assert _close(appt.new_kv_flush, expected_flush), (appt.new_kv_flush, expected_flush)


def test_ttft_increases_with_append_tokens():
    """More append-prefill tokens -> larger old-KV materialization + GPU append
    -> larger TTFT."""
    pol = PolicyConfig.policy("cpu_backing", b_d=2)
    small = replace(WORK, a_append=500)
    big = replace(WORK, a_append=20000)
    assert an.ttft(pol, big, HW, MODEL, C) > an.ttft(pol, small, HW, MODEL, C)


def test_optimize_f_sparse_beats_baseline_under_slo():
    """Under the TPOT/TTFT SLO, sparse CPU attention yields an interior f*>0
    that beats the best GPU-only batch."""
    fs = an.optimize_f(WORK, HW, MODEL, SLOConfig(), C, sparse=0.1)
    assert fs.feasible
    assert fs.f > 0.0
    assert fs.gain > 1.05, fs.gain


def test_optimize_f_dense_stays_near_baseline_under_slo():
    """Dense CPU attention can't beat GPU-only batch tuning under the SLO."""
    fs = an.optimize_f(WORK, HW, MODEL, SLOConfig(), C, sparse=1.0)
    assert fs.feasible
    assert fs.gain <= 1.05, fs.gain


def test_more_hbm_reduces_offload_gain():
    """As HBM grows, the GPU-only baseline fits the SLO batch on its own, so the
    offload gain shrinks toward ~1."""
    small = an.optimize_f(WORK, HW, MODEL, SLOConfig(), C, sparse=0.1)
    big_hw = replace(HardwareConfig(), m_hbm=768 * (1024 ** 3)).effective(0.5)
    big = an.optimize_f(WORK, big_hw, MODEL, SLOConfig(), C, sparse=0.1)
    assert big.gain <= small.gain + 1e-9, (small.gain, big.gain)


def test_tight_tpot_slo_is_infeasible_at_low_efficiency():
    """At eta=0.3 the 50ms TPOT is unachievable (weight streaming alone)."""
    hw = HardwareConfig().effective(0.3)
    fs = an.optimize_f(WORK, hw, MODEL, SLOConfig(), C, sparse=0.1)
    assert not fs.feasible


def test_mla_kv_much_smaller_than_gqa():
    """MLA stores one compressed latent (~576) vs GQA's 2*n_kv*head_dim."""
    moe = ModelConfig.preset("moe_large_mla")
    assert moe.kv_mode == "mla"
    assert moe.d_kv == 576
    # an equivalent GQA model would be 2*n_kv_heads*head_dim; MLA is far smaller
    gqa_equiv = 2 * 8 * 128  # 2048, a typical GQA-8 head config
    assert moe.d_kv < gqa_equiv / 3


def test_moe_weight_read_saturates_toward_total_with_batch():
    """MoE: weight read ~= P_act at B=1, grows toward P_total as batch saturates
    experts. Dense: batch-independent (= P_act)."""
    moe = ModelConfig.preset("moe_large_mla")
    w1 = an.weight_read_bytes(moe, 1)
    w_big = an.weight_read_bytes(moe, 512)
    assert _close(w1, moe.p_act * moe.q_weight, rel=0.02)
    assert w_big > 5 * w1                       # large batch reads many experts
    assert w_big <= moe.p_total * moe.q_weight + 1e-9
    # dense is flat in batch
    dense = ModelConfig()
    assert _close(an.weight_read_bytes(dense, 1), an.weight_read_bytes(dense, 256))


def test_c2c_not_double_charged():
    """tps must not throttle by the decode-C2C that is already inside TPOT."""
    pol = PolicyConfig.policy("partial_cpu_attn", f=0.3, b_d=8, sparse=0.1)
    cu = an.c2c_util(pol, WORK, HW, MODEL, C)
    # decode-C2C is on the TPOT critical path -> by construction util <= 1
    assert cu["decode_util"] <= 1.0 + 1e-9, cu["decode_util"]


def test_split_efficiency_decode_is_memory_bound():
    """LIFE-style split efficiency: decode TPOT (memory-bound) responds strongly
    to eta_mem and barely to eta_compute; effective(eta) sets both (back-compat)."""
    base = HardwareConfig()
    pol = PolicyConfig.policy("gpu_hot", b_d=1)
    w = replace(WORK, a_append=0.0)
    # halve memory efficiency -> decode TPOT should rise a lot
    slow_mem = base.effective(eta_compute=0.9, eta_mem=0.3)
    slow_cmp = base.effective(eta_compute=0.3, eta_mem=0.9)
    t_mem = an.tpot(pol, w, slow_mem, MODEL, C).optimistic
    t_cmp = an.tpot(pol, w, slow_cmp, MODEL, C).optimistic
    assert t_mem > t_cmp * 1.5, (t_mem, t_cmp)        # memory dominates decode
    # back-compat: effective(0.5) == both 0.5
    a = base.effective(0.5)
    b = base.effective(eta_compute=0.5, eta_mem=0.5)
    assert abs(a.bw_hbm - b.bw_hbm) < 1 and abs(a.f_gpu - b.f_gpu) < 1


def test_dispatch_overhead_adds_fixed_floor():
    """t_dispatch adds a fixed per-step latency to TPOT (LIFE eq.4)."""
    from model.config import Coeffs as _C
    pol = PolicyConfig.policy("gpu_hot", b_d=1)
    w = replace(WORK, a_append=0.0)
    base = an.tpot(pol, w, HW, MODEL, _C()).optimistic
    with_disp = an.tpot(pol, w, HW, MODEL, _C(t_dispatch=5e-3)).optimistic
    assert abs((with_disp - base) - 5e-3) < 1e-9


def test_trace_parser_segmentation_hand_computed():
    """Hand-built 2-call conversation with known char lengths; verify the
    append-only segmentation (input/cached/uncached/output) exactly."""
    from model import trace_parser as tp
    cpt = 4.0
    # human0=400 chars, gpt0=40, human1=80, gpt1=120  (alternating)
    conv = [{"from": "human", "value": "h" * 400},
            {"from": "gpt", "value": "g" * 40},
            {"from": "human", "value": "h" * 80},
            {"from": "gpt", "value": "g" * 120}]
    trials = tp.parse_trials([conv], cpt)
    calls = trials[0]["calls"]
    assert len(calls) == 2
    # call 0: input = h0 = 400/4 = 100; cached=0 (cold); uncached=100; output=40/4=10
    assert calls[0]["input"] == 100 and calls[0]["cached"] == 0
    assert calls[0]["uncached"] == 100 and calls[0]["output"] == 10
    # call 1: input = (400+40+80)/4 = 130; cached = (400+40)/4 = 110 (prev request
    # is a perfect prefix); uncached = new human = 80/4 = 20; output = 120/4 = 30
    assert calls[1]["input"] == 130 and calls[1]["cached"] == 110
    assert calls[1]["uncached"] == 20 and calls[1]["output"] == 30


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
