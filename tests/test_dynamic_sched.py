"""Tests for the dynamic GPU/CPU decode-offload scheduler.

Run:  python tests/test_dynamic_sched.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.config import HardwareConfig, ModelConfig, WorkloadConfig
from sim import dynamic_sched as ds
from sim.event_sim import Call

HW = HardwareConfig().effective(0.5)
MODEL = ModelConfig()


def _sessions(n=16, calls=4):
    w = WorkloadConfig()
    # mixed small/large contexts so some fit, exercising the scheduler
    return [[Call(si, i, (8000 if (si + i) % 2 else w.s_cached),
                  1000.0, 150.0) for i in range(calls)] for si in range(n)]


def test_dynamic_never_worse_than_baseline():
    """Across CPU ratios, dynamic scheduling throughput >= GPU-only baseline
    (the scheduler declines to offload when it would hurt) -- 'no worse'."""
    s = _sessions()
    for cpg in (0.5, 4.0, 16.0):
        base = ds.run_one(s, HW, MODEL, "gpu_only", cpg, arrival_rate=3.0)
        dyn = ds.run_one(s, HW, MODEL, "dynamic", cpg, arrival_rate=3.0)
        assert dyn["throughput_tok_s"] >= base["throughput_tok_s"] * 0.999, (
            cpg, base["throughput_tok_s"], dyn["throughput_tok_s"])


def test_dynamic_at_least_as_good_as_static_all_cpu():
    """Dynamic (spill only overflow) >= naive all-CPU offload."""
    s = _sessions()
    for cpg in (0.5, 16.0):
        dyn = ds.run_one(s, HW, MODEL, "dynamic", cpg, arrival_rate=3.0)
        allc = ds.run_one(s, HW, MODEL, "all_cpu", cpg, arrival_rate=3.0)
        assert dyn["throughput_tok_s"] >= allc["throughput_tok_s"] * 0.999, (
            cpg, dyn["throughput_tok_s"], allc["throughput_tok_s"])


def test_low_cpu_ratio_offloads_little_or_nothing():
    """At the stock NVL72 ratio CPU decode can't meet the SLO, so dynamic spills
    ~nothing -> break-even, not a regression."""
    s = _sessions()
    dyn = ds.run_one(s, HW, MODEL, "dynamic", 0.5, arrival_rate=3.0)
    assert dyn["offloaded_frac"] <= 0.05, dyn["offloaded_frac"]


def test_high_cpu_ratio_offloads_and_gains():
    """With enough CPU, dynamic offloads a real fraction and beats baseline."""
    s = _sessions()
    base = ds.run_one(s, HW, MODEL, "gpu_only", 16.0, arrival_rate=3.0)
    dyn = ds.run_one(s, HW, MODEL, "dynamic", 16.0, arrival_rate=3.0)
    assert dyn["offloaded_frac"] > 0.05
    assert dyn["throughput_tok_s"] > base["throughput_tok_s"]


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
