"""Phase 4 sanity tests for the event-driven simulator.

Run:  python tests/test_event_sim.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.config import HardwareConfig, ModelConfig, WorkloadConfig
from sim import event_sim as es

HW = HardwareConfig().effective(0.5)
MODEL = ModelConfig()


def _synthetic(n_sessions=12, n_calls=4):
    w = WorkloadConfig()
    return [[es.Call(si, i, w.s_cached, w.a_append, w.o_output)
             for i in range(n_calls)] for si in range(n_sessions)]


def test_sim_completes_all_calls():
    s = _synthetic()
    res = es.run_one(s, HW, MODEL, f=0.0, sparse=1.0,
                     arrival_rate=2.0, think_time=0.3)
    assert res["completed"] == sum(len(c) for c in s)
    for k in ("throughput_tok_s", "ttft_mean", "tpot_mean", "gpu_util"):
        assert res[k] > 0, k


def test_sim_seed_changes_arrivals_but_same_seed_reproduces():
    """Non-tautological: different seeds give different Poisson arrival streams
    (=> different makespan), while the same seed reproduces exactly. This
    actually exercises the RNG, not just determinism of a pure function."""
    def run(seed):
        sim = es.Sim(_synthetic(), HW, MODEL, f=0.0, sparse=1.0,
                     arrival_rate=3.0, think_time=0.3, seed=seed)
        return sim.run()
    a, b, a2 = run(1), run(2), run(1)
    assert abs(a["makespan_s"] - a2["makespan_s"]) < 1e-9        # same seed -> identical
    assert abs(a["makespan_s"] - b["makespan_s"]) > 1e-6         # diff seed -> different


def test_sparse_offload_reduces_admission_pressure():
    """Sparse offload shrinks resident hot KV so MORE sessions are admitted ->
    less admission stall, lower TTFT, >= throughput vs the dense GPU-hot baseline.

    Uses a smaller-context workload: at full 64k context the append old-KV
    materialization (~21 GB, independent of f) sets an HBM floor that decode
    offload alone cannot lower, so the admission win only appears once the
    per-session footprint is small enough that f changes how many fit.
    """
    def clone(sessions):
        return [[es.Call(x.sess, x.idx, x.s_cached, x.a_append, x.o_output)
                 for x in row] for row in sessions]
    s = [[es.Call(si, i, 12000.0, 2000.0, 200.0) for i in range(4)]
         for si in range(16)]
    base = es.run_one(clone(s), HW, MODEL, f=0.0, sparse=1.0,
                      arrival_rate=3.0, think_time=0.3)
    off = es.run_one(clone(s), HW, MODEL, f=0.5, sparse=0.1,
                     arrival_rate=3.0, think_time=0.3)
    assert off["admit_stall_total_s"] <= base["admit_stall_total_s"] + 1e-6
    assert off["ttft_mean"] <= base["ttft_mean"] + 1e-6
    assert off["throughput_tok_s"] >= base["throughput_tok_s"] - 1e-6


def test_event_types_logged():
    s = _synthetic(8, 3)
    es_sim = es.Sim(s, HW, MODEL, f=0.3, sparse=0.1, arrival_rate=2.0, think_time=0.3)
    res = es_sim.run()
    ev = res["event_counts"]
    for name in ("ARRIVE", "OLD_KV_LOAD", "APPEND_PREFILL", "NEW_KV_FLUSH",
                 "DECODE_STEP", "CPU_ATTN", "FINISH", "C2C_TRANSFER"):
        assert ev.get(name, 0) > 0, f"missing event {name}: {ev}"


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
