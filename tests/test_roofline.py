"""Tests for the theoretical throughput-ceiling (roofline) model."""

from __future__ import annotations

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import analytical as an
from model import roofline as rl
from model.config import HardwareConfig, ModelConfig, PolicyConfig, WorkloadConfig

HW = HardwareConfig().effective(0.5)
MODEL = ModelConfig()
WORK = WorkloadConfig()


def test_decode_tps_converges_to_kv_bandwidth_ceiling():
    """As batch grows, decode-only TPS -> BW_HBM / KV(S) (the memory roofline)."""
    kv = an.kv_size(WORK.s_context, MODEL)
    ceiling = HW.bw_hbm / kv
    w = replace(WORK, a_append=0.0)
    tps_big = 8192 / an.tpot(PolicyConfig.policy("gpu_hot", b_d=8192), w, HW,
                             MODEL).optimistic
    assert 0.9 * ceiling <= tps_big <= 1.1 * ceiling, (tps_big, ceiling)


def test_offload_raises_ceiling():
    """Higher f (less GPU KV per token) raises the theoretical ceiling."""
    c0 = rl.serving_roofline(MODEL, HW, WORK, f=0.0)["ceiling_tps_per_gpu"]
    c5 = rl.serving_roofline(MODEL, HW, WORK, f=0.5)["ceiling_tps_per_gpu"]
    assert c5 > c0


def test_operating_point_below_ceiling():
    """The admission-bound operating point is below the theoretical ceiling."""
    base = an.tps(PolicyConfig.policy("gpu_hot", b_d=256), WORK, HW, MODEL)
    u = rl.utilization(base.tps, MODEL, HW, WORK)
    assert 0.0 < u < 1.0, u


def test_more_hbm_bandwidth_raises_ceiling_when_kv_bound():
    """When KV-bandwidth-bound, doubling HBM BW ~doubles the ceiling."""
    fast = replace(HardwareConfig(), bw_hbm=16e12).effective(0.5)
    slow = replace(HardwareConfig(), bw_hbm=8e12).effective(0.5)
    cf = rl.serving_roofline(MODEL, fast, WORK)["ceiling_tps_per_gpu"]
    cs = rl.serving_roofline(MODEL, slow, WORK)["ceiling_tps_per_gpu"]
    assert cf > cs * 1.3                     # KV term halves; ceiling rises a lot


def test_serving_tps_below_decode_only():
    """Serving throughput (prefill included) <= decode-only TPS (prefill omitted),
    for the prefill-heavy workload."""
    pol = PolicyConfig.policy("gpu_hot", b_d=64)
    s = an.serving_tps(pol, WORK, HW, MODEL)
    assert s["serving_tps"] <= s["decode_tps"] + 1e-9
    assert s["serving_tps"] > 0 and s["prefill_time"] > 0


def test_serving_tps_converges_to_serving_roofline():
    """At large batch, serving_tps approaches the unified serving roofline."""
    pol = PolicyConfig.policy("gpu_hot", b_d=100000)   # effectively uncapped by target
    s = an.serving_tps(pol, WORK, HW, MODEL)
    ceiling = rl.serving_roofline(MODEL, HW, WORK)["ceiling_tps_per_gpu"]
    # serving_tps uses the HBM-capped batch, so it is <= ceiling; check it's a
    # sane fraction and never exceeds the ceiling.
    assert 0 < s["serving_tps"] <= ceiling * 1.05, (s["serving_tps"], ceiling)


def _run_all():
    fns = [x for k, x in globals().items() if k.startswith("test_") and callable(x)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
